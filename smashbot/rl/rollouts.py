"""Rollout collection: turn N live games into PPO Trajectory batches.

Two layers:
- ChunkAssembler (pure, unit-tested): accumulates per-frame records and
  transition rewards, emits Trajectory chunks with the delay-shifted reward
  alignment the learner expects (reward slot t of a chunk = the game
  transition at sample-time t + delay, mirroring delay_lib's BC slicing).
- DolphinRolloutWorker: thread-per-env Dolphin driving with a sync barrier —
  each frame, all envs' parsed+encoded states are batched into one policy
  forward (BatchedPolicyAgent), controllers fan back out to the env threads.

Rewards are computed directly from gamestate deltas (stocks/percent), zeroed
at game boundaries; slippi-ai's shaping penalties (ledge/stall) can be added
here later via their reward lib.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent, FrameRecord
from smashbot.rl.ppo import ActionData, Trajectory


class ChunkAssembler:
    """Accumulates FrameRecords + rewards, emits [N, T+1] Trajectory chunks.

    push_frame() every frame (with each env's is_resetting flag and the
    agent's hidden snapshot at chunk starts); push_reward() every transition
    (aligned to real time). A chunk covering sample-times [0, T] emits once
    rewards through real-transition T + delay - 1 have arrived. Chunks
    overlap by one frame, per the Frames convention.
    """

    def __init__(self, unroll_length: int, delay: int):
        self.T = unroll_length
        self.delay = delay
        self._records: list[FrameRecord] = []
        self._resets: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._initial_state: tp.Any = None
        self._next_initial: tp.Any = None

    def push_frame(
        self,
        record: FrameRecord,
        is_resetting: torch.Tensor,
        hidden_snapshot=None,
    ) -> None:
        """hidden_snapshot must be provided whenever this frame starts a chunk
        (every `unroll_length` frames, including the very first): it is the
        agent's recurrent state BEFORE stepping this frame."""
        if hidden_snapshot is not None:
            if not self._records:
                self._initial_state = hidden_snapshot
            else:
                self._next_initial = hidden_snapshot
        self._records.append(record)
        self._resets.append(is_resetting)

    def push_reward(self, reward: torch.Tensor) -> None:  # [N]
        self._rewards.append(reward)

    def ready(self) -> bool:
        return (
            len(self._records) >= self.T + 1
            and len(self._rewards) >= self.T + self.delay
        )

    def emit(self) -> Trajectory:
        assert self.ready()
        T, D = self.T, self.delay
        stack = lambda seq: tree.map_structure(
            lambda *xs: torch.stack(xs, dim=1), *seq
        )
        records = self._records[: T + 1]
        traj = Trajectory(
            states=stack([r.state for r in records]),
            name=torch.stack([r.name for r in records], dim=1),
            actions=ActionData(
                controller_state=stack([r.prev_action for r in records]),
                logits=stack([r.logits for r in records]),
            ),
            # reward slot t <- real transition t + D ("rewards that follow
            # actions"), matching the BC value-training alignment.
            rewards=torch.stack(self._rewards[D : T + D], dim=1),
            is_resetting=torch.stack(self._resets[: T + 1], dim=1),
            initial_state=self._initial_state,
        )
        # Keep the overlap frame and the not-yet-consumed reward tail.
        self._records = self._records[T:]
        self._resets = self._resets[T:]
        self._rewards = self._rewards[T:]
        self._initial_state = self._next_initial
        self._next_initial = None
        return traj


def compute_reward(
    prev_stocks: torch.Tensor,  # [N, 2] (own, opp)
    stocks: torch.Tensor,
    prev_percent: torch.Tensor,  # [N, 2]
    percent: torch.Tensor,
    is_resetting: torch.Tensor,  # [N]
    damage_ratio: float = 0.01,
) -> torch.Tensor:
    """Zero-sum reward from the bot's perspective, zeroed at game boundaries.

    death: stock decrease. damage: positive percent delta (percent resets to
    zero on death; negative deltas are ignored).
    """
    own_death = (stocks[:, 0] < prev_stocks[:, 0]).float()
    opp_death = (stocks[:, 1] < prev_stocks[:, 1]).float()
    own_dmg = (percent[:, 0] - prev_percent[:, 0]).clamp(min=0)
    opp_dmg = (percent[:, 1] - prev_percent[:, 1]).clamp(min=0)
    reward = (opp_death - own_death) + damage_ratio * (opp_dmg - own_dmg)
    return torch.where(is_resetting, torch.zeros_like(reward), reward)


class WinTracker:
    """Rolling win rate vs the CURRENT training opponent (teacher now,
    snapshot pool later). Fixed-yardstick evals stay in the M8 batteries."""

    def __init__(self, window: int = 100):
        import collections

        self.recent = collections.deque(maxlen=window)
        self.wins = self.losses = self.draws = 0

    def add(self, result: int) -> None:
        self.recent.append(result)
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1

    def stats(self) -> dict:
        games = self.wins + self.losses + self.draws
        decided = [r for r in self.recent if r != 0]
        return {
            "games_played": games,
            "win_rate_recent": (
                sum(1 for r in decided if r > 0) / len(decided) if decided else 0.5
            ),
        }


def outcome_stats(rewards: torch.Tensor) -> dict:
    """Kills/deaths/damage rates from a chunk's reward tensor [N, T].
    Kills/deaths are the +-1 events (threshold 0.5: damage terms are
    0.01/percent and cannot reach it between deaths in one frame)."""
    frames = rewards.numel()
    minutes = frames / 3600.0
    kills = (rewards > 0.5).sum().item()
    deaths = (rewards < -0.5).sum().item()
    net_damage = (rewards.sum().item() - (kills - deaths)) / 0.01
    return {
        "kills_per_min": kills / minutes,
        "deaths_per_min": deaths / minutes,
        "net_damage_per_min": net_damage / minutes,
    }


@dataclasses.dataclass
class RolloutConfig:
    num_envs: int = 8
    unroll_length: int = 240  # 4s, slippi-ai's RL rollout length
    batch_steps: int = 1  # frames per inference flush; measured best on this rig (see docs)
    opponent: str = "teacher"  # "teacher" | "cpu:<level>"
    bot_char: str = "FOX"
    opponent_char: str = "FOX"
    stage: str = "FINAL_DESTINATION"
    games_per_dolphin: int = 20


def _env_process_main(idx: int, cfg: "RolloutConfig", conn) -> None:
    """One Dolphin per PROCESS (multiple libmelee Consoles cannot share a
    process — the vendor's envs.py reaches the same conclusion). Speaks over
    a Pipe: sends per-frame payloads, receives {port: Controller} commands
    (None = shut down)."""
    import melee
    import numpy as np
    import tree as tree_lib

    from slippi_ai import controller_lib
    from slippi_ai import dolphin as dolphin_lib
    from slippi_db.parse_libmelee import Parser

    from smashbot import embed as embed_lib
    from smashbot.eval import game as game_lib

    # Encode (from_state: pure numpy typing/bucketing, no NN) worker-side so
    # the 32 env processes parallelize it instead of the main loop. Must match
    # the policy's embed schema — both use the default EmbedConfig (verified
    # by test_worker_side_encode_matches_policy_encode).
    embed_game = embed_lib.EmbedConfig().make_game_embedding()

    opp = game_lib.Opponent.parse(
        cfg.opponent if cfg.opponent.startswith("cpu")
        else f"cpu:9:{cfg.opponent_char}"  # placeholder; ckpt/teacher use AI
    )
    players = {
        1: dolphin_lib.AI(character=melee.Character[cfg.bot_char.upper()]),
        2: dolphin_lib.AI(character=melee.Character[cfg.opponent_char.upper()])
        if cfg.opponent == "teacher" else opp.make_player(),
    }
    try:
        while True:
            dolphin = game_lib.make_dolphin(players, headless=True, stage=cfg.stage)
            parser = Parser(ports=[1, 2])
            games = 0
            last_frame = None
            last_stocks = None
            try:
                for gs in dolphin.iter_gamestates(skip_menu_frames=True):
                    resetting = last_frame is not None and gs.frame < last_frame
                    result = None
                    if resetting:
                        parser = Parser(ports=[1, 2])
                        games += 1
                        if last_stocks is not None:
                            a, b = last_stocks
                            result = 1 if a > b else (-1 if b > a else 0)
                        if games >= cfg.games_per_dolphin:
                            break
                    last_frame = gs.frame
                    raw = tree_lib.map_structure(
                        np.asarray, parser.get_game(gs)
                    )
                    game = embed_game.from_state(raw)
                    p1, p2 = gs.players[1], gs.players[2]
                    last_stocks = (int(p1.stock), int(p2.stock))
                    conn.send(
                        dict(
                            game=game,
                            resetting=resetting,
                            result=result,  # +1 bot won / -1 lost / 0 draw; None mid-game
                            stocks=last_stocks,
                            percent=(float(p1.percent), float(p2.percent)),
                        )
                    )
                    controllers = conn.recv()
                    if controllers is None:
                        return
                    for port, controller_state in controllers.items():
                        controller_lib.send_controller(
                            dolphin.controllers[port], controller_state
                        )
            finally:
                dolphin.stop()
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass


class DolphinRolloutWorker:
    """N Dolphins, one batched student agent (port 1), one batched opponent
    agent (port 2) when self-playing; sync-barrier frame loop."""

    def __init__(
        self,
        config: RolloutConfig,
        student: BatchedPolicyAgent,
        opponent: BatchedPolicyAgent | None,  # None -> in-game CPU opponent
    ):
        # Imported lazily: this class needs Dolphin, the rest of the module
        # doesn't.
        from smashbot.eval import game as game_lib

        self.config = config
        self.student = student
        self.opponent = opponent
        self.game_lib = game_lib
        self.assembler = ChunkAssembler(config.unroll_length, student.delay)
        self.win_tracker = WinTracker()
        self._procs: list = []
        self._conns: list = []

    def _ensure_started(self) -> None:
        if self._procs:
            return
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for i in range(self.config.num_envs):
            parent, child = ctx.Pipe()
            # non-daemon: libmelee's slippstream forks its own child
            p = ctx.Process(
                target=_env_process_main, args=(i, self.config, child)
            )
            p.start()
            self._procs.append(p)
            self._conns.append(parent)
        self._frame_count = 0
        n = self.config.num_envs
        self._prev_stocks = torch.full((n, 2), 4.0)
        self._prev_percent = torch.zeros(n, 2)

    def _gather_all(self) -> list[dict]:
        payloads = []
        for i, conn in enumerate(self._conns):
            try:
                payloads.append(conn.recv())
            except (EOFError, BrokenPipeError) as e:
                raise RuntimeError(f"env {i} died") from e
        return payloads

    def _encode(self, games: list) -> tp.Any:
        """games: per-env structs ALREADY encoded by the env processes
        (from_state runs worker-side); here we only stack and torch-ify."""
        import numpy as np

        device = self.student.device
        batched = tree.map_structure(lambda *xs: np.stack(xs), *games)
        return tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x)
            ).to(device),
            batched,
        )

    @staticmethod
    def _swap_perspective(game):
        return game._replace(p0=game.p1, p1=game.p0)

    def collect(self, num_trajectories: int) -> list[Trajectory]:
        """Run the sync-barrier loop until N trajectory chunks are assembled."""
        self._ensure_started()
        cfg = self.config
        out: list[Trajectory] = []

        assert cfg.unroll_length % self.student.batch_steps == 0, (
            "unroll_length must be a multiple of batch_steps so chunk "
            "boundaries land on flush boundaries"
        )
        device = self.student.device
        if not hasattr(self, "_pending_resets"):
            self._pending_resets: list[torch.Tensor] = []
        pending_resets = self._pending_resets
        records_pushed = getattr(self, "_records_pushed", 0)

        while len(out) < num_trajectories:
            payloads = self._gather_all()
            for p in payloads:
                if p.get("result") is not None:
                    self.win_tracker.add(p["result"])
            resets = torch.tensor([p["resetting"] for p in payloads])

            stocks = torch.tensor([p["stocks"] for p in payloads], dtype=torch.float32)
            percent = torch.tensor([p["percent"] for p in payloads], dtype=torch.float32)
            if self._frame_count > 0:
                self.assembler.push_reward(
                    compute_reward(
                        self._prev_stocks, stocks,
                        self._prev_percent, percent, resets,
                    ).to(device)
                )
            self._prev_stocks, self._prev_percent = stocks, percent

            games = [p["game"] for p in payloads]
            states = self._encode(games)
            resets_dev = resets.to(device)
            pending_resets.append(resets_dev)
            controllers1, records, hidden_before = self.student.step(
                states, resets_dev
            )

            controllers2 = None
            if self.opponent is not None:
                # perspective swap commutes with encoding: build the opponent
                # view from the encoded struct (pointer swap, no second pass)
                opp_states = states._replace(p0=states.p1, p1=states.p0)
                controllers2, _, _ = self.opponent.step(opp_states, resets_dev)

            for i, conn in enumerate(self._conns):
                cmd = {1: controllers1[i]}
                if controllers2 is not None:
                    cmd[2] = controllers2[i]
                conn.send(cmd)

            for j, record in enumerate(records):
                snap = None
                if records_pushed % cfg.unroll_length == 0:
                    # chunk boundary: the recurrent state before this flush is
                    # the state before its first frame (j == 0 always, given
                    # the unroll/batch_steps divisibility assert)
                    snap = hidden_before
                self.assembler.push_frame(record, pending_resets[j], snap)
                records_pushed += 1
            if records:
                del pending_resets[: len(records)]

            self._frame_count += 1
            if self.assembler.ready():
                out.append(self.assembler.emit())
        self._records_pushed = records_pushed
        return out

    def stop(self) -> None:
        for conn in self._conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
