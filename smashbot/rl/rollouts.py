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

NOTE: env processes use the multiprocessing 'spawn' context, which re-imports
the caller's __main__ — any script that builds a DolphinRolloutWorker MUST
guard its entrypoint with `if __name__ == "__main__":`.
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


class GameTracker:
    """Game-outcome metrics vs the CURRENT training opponent (teacher now,
    snapshot pool later); fixed-yardstick evals stay in the M8 batteries.

    Time-free by design: a win is a win at 2 minutes or 7. Tracks rolling
    win rate, average final stock differential (-4..+4 dominance scale),
    average opponent percent at our kills (low = early kills, strong punish
    game), and average own percent at our deaths (high = hard to kill)."""

    def __init__(self, window: int = 100, event_window: int = 200):
        import collections

        self.diffs = collections.deque(maxlen=window)  # per finished game
        self.kill_percents = collections.deque(maxlen=event_window)
        self.death_percents = collections.deque(maxlen=event_window)
        self.wins = self.losses = self.draws = 0

    def add_game(self, final_stocks: tuple[int, int]) -> None:
        bot, opp = final_stocks
        self.diffs.append(bot - opp)
        if bot > opp:
            self.wins += 1
        elif opp > bot:
            self.losses += 1
        else:
            self.draws += 1

    def add_kill(self, opp_percent: float) -> None:
        self.kill_percents.append(opp_percent)

    def add_death(self, own_percent: float) -> None:
        self.death_percents.append(own_percent)

    def stats(self) -> dict:
        mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
        decided = [d for d in self.diffs if d != 0]
        return {
            "games_played": self.wins + self.losses + self.draws,
            "win_rate_recent": (
                sum(1 for d in decided if d > 0) / len(decided) if decided else 0.5
            ),
            "avg_stock_diff": mean(self.diffs),
            "avg_percent_at_kill": mean(self.kill_percents),
            "avg_percent_at_death": mean(self.death_percents),
        }


@dataclasses.dataclass
class RolloutConfig:
    num_envs: int = 8
    unroll_length: int = 240  # 4s, slippi-ai's RL rollout length
    batch_steps: int = 1  # frames per inference flush; measured best on this rig (see docs)
    bot_char: str = "FOX"
    stage: str = "FINAL_DESTINATION"
    games_per_dolphin: int = 20
    # Opponent pool partition (see rl/pool.py). Defaults replicate the
    # simple all-teacher setup; production: cpu_envs=8, teacher_envs=16,
    # snapshot_slots=5 at num_envs=64.
    cpu_envs: int = 0
    teacher_envs: int = -1  # -1 = all envs not assigned to cpu/snapshots
    snapshot_slots: int = 0
    main12_prob: float = 0.6
    snapshot_interval: int = 500  # learner steps between student snapshots
    partition_seed: int = 0
    headless: bool = True  # False: rendered window at normal speed (watch mode)
    log_tag: str = ""  # namespaces /tmp/smashbot-env-*.log between runs
    # Reference opponent (slippi-ai medium-v2 via venv-ref subprocess).
    ref_envs: int = 0
    ref_ckpt: str = "/home/kage/drive2/ShineBot/models/medium-v2"


def _env_process_main(idx: int, cfg: "RolloutConfig", spec, conn) -> None:
    """One Dolphin per PROCESS (multiple libmelee Consoles cannot share a
    process — the vendor's envs.py reaches the same conclusion). Speaks over
    a Pipe: sends per-frame payloads, receives {port: Controller} commands
    (None = shut down)."""
    import os
    import sys

    # Dolphin banners/spam would hit the parent terminal on every boot and
    # recycle; redirect this process (and its Dolphin child, via fd
    # inheritance) to a per-env log where real errors remain findable.
    _tag = f"-{cfg.log_tag}" if cfg.log_tag else ""
    _log = open(f"/tmp/smashbot-env{_tag}-{idx}.log", "a", buffering=1)
    os.dup2(_log.fileno(), sys.stdout.fileno())
    os.dup2(_log.fileno(), sys.stderr.fileno())

    import melee
    import numpy as np
    import tree as tree_lib

    from slippi_ai import controller_lib
    from slippi_ai import dolphin as dolphin_lib
    from slippi_ai.dolphin import WrongCharacterSelected
    from slippi_db.parse_libmelee import Parser

    from smashbot import embed as embed_lib
    from smashbot.eval import game as game_lib

    # Encode (from_state: pure numpy typing/bucketing, no NN) worker-side so
    # the 32 env processes parallelize it instead of the main loop. Must match
    # the policy's embed schema — both use the default EmbedConfig (verified
    # by test_worker_side_encode_matches_policy_encode).
    embed_game = embed_lib.EmbedConfig().make_game_embedding()

    opp_port = 3 - spec.student_port
    if spec.kind == "cpu":
        opponent_player = dolphin_lib.CPU(
            character=melee.Character[spec.opponent_char.upper()],
            level=spec.cpu_level,
        )
    else:
        opponent_player = dolphin_lib.AI(
            character=melee.Character[spec.opponent_char.upper()]
        )
    players = {
        spec.student_port: dolphin_lib.AI(
            character=melee.Character[cfg.bot_char.upper()]
        ),
        opp_port: opponent_player,
    }
    # Carried across Dolphin recycles: the new instance's first frame must
    # still announce the game boundary (fresh recurrent state, zeroed reward)
    # and deliver the final game's result — otherwise two different games
    # would silently splice into one stream.
    pending_reset = False
    pending_result = None
    consecutive_misselects = 0
    try:
        while True:
            dolphin = game_lib.make_dolphin(
                players, headless=cfg.headless, stage=cfg.stage
            )
            parser = Parser(ports=[1, 2])
            games = 0
            last_frame = None
            last_stocks = None
            try:
              try:
                for gs in dolphin.iter_gamestates(skip_menu_frames=True):
                    boundary = last_frame is not None and gs.frame < last_frame
                    resetting = boundary or pending_reset
                    result = pending_result if pending_reset else None
                    pending_reset, pending_result = False, None
                    if boundary:
                        games += 1
                        result = last_stocks  # ended game's final (bot, opp)
                        if games >= cfg.games_per_dolphin:
                            pending_reset, pending_result = True, result
                            break
                        parser = Parser(ports=[1, 2])
                    last_frame = gs.frame
                    raw = tree_lib.map_structure(
                        np.asarray, parser.get_game(gs)
                    )
                    game = embed_game.from_state(raw)
                    # reference opponents (slippi-ai models) do their own
                    # encoding server-side and need the RAW struct, seen
                    # from THEIR side (p0 = the reference agent's player)
                    raw_for_ref = None
                    if spec.kind == "reference":
                        raw_for_ref = (
                            raw if spec.student_port == 2
                            else raw._replace(p0=raw.p1, p1=raw.p0)
                        )
                    p1, p2 = gs.players[1], gs.players[2]
                    last_stocks = (int(p1.stock), int(p2.stock))
                    conn.send(
                        dict(
                            game=game,
                            resetting=resetting,
                            final_stocks=result,  # ended game's (bot, opp); None mid-game
                            raw_game=raw_for_ref,
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
              except WrongCharacterSelected as e:
                # menu cursor race under fast-forward (notably the Sheik/
                # Zelda slot): scrap this Dolphin and retry with a fresh one.
                # BOUNDED: persistent misselection means the character is
                # mechanically unpickable — die loudly, not loop forever
                # (learned via 362 consecutive CPU-Sheik retries).
                consecutive_misselects += 1
                if consecutive_misselects >= 3:
                    raise
                print(f"menu misselection, retrying: {e}", flush=True)
              else:
                consecutive_misselects = 0
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
        opponents: dict | None = None,  # {"teacher": agent, ("slot", i): agent}
        specs: list | None = None,  # per-env EnvSpec; default from make_partition
    ):
        # Imported lazily: this class needs Dolphin, the rest of the module
        # doesn't.
        from smashbot.eval import game as game_lib

        from smashbot.rl.pool import make_partition

        self.config = config
        self.student = student
        self.specs = specs or make_partition(
            config.num_envs, config.cpu_envs, config.teacher_envs,
            config.snapshot_slots, config.main12_prob, config.partition_seed,
            ref_envs=config.ref_envs,
        )
        assert len(self.specs) == config.num_envs
        self.opponents = opponents or {}
        # group name -> env index list (fixed membership = stable batch shapes)
        self.groups: dict = {}
        self.ref_idx: list[int] = []
        for i, spec in enumerate(self.specs):
            if spec.kind == "teacher":
                self.groups.setdefault("teacher", []).append(i)
            elif spec.kind == "snapshot":
                self.groups.setdefault(("slot", spec.group), []).append(i)
            elif spec.kind == "reference":
                self.ref_idx.append(i)
        for name, idx in self.groups.items():
            assert name in self.opponents, f"no agent supplied for group {name}"
            assert self.opponents[name].num_envs == len(idx)
        self.ref_bridge = None
        self.ref_step_ms = 0.0
        if self.ref_idx:
            from smashbot.rl.ref_bridge import RefBridge

            self.ref_bridge = RefBridge(
                batch_size=len(self.ref_idx), ckpt=config.ref_ckpt
            )
        self.seat2 = torch.tensor(
            [sp.student_port == 2 for sp in self.specs]
        )
        self.game_lib = game_lib
        self.assembler = ChunkAssembler(config.unroll_length, student.delay)
        self.trackers = {
            k: GameTracker() for k in ("cpu", "teacher", "snapshot", "reference")
        }
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
                target=_env_process_main, args=(i, self.config, self.specs[i], child)
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
            for i, p in enumerate(payloads):
                if p.get("final_stocks") is not None:
                    a, b = p["final_stocks"]  # (port1, port2)
                    if self.specs[i].student_port == 2:
                        a, b = b, a
                    self.trackers[self.specs[i].kind].add_game((a, b))
            resets = torch.tensor([p["resetting"] for p in payloads])

            stocks = torch.tensor([p["stocks"] for p in payloads], dtype=torch.float32)
            percent = torch.tensor([p["percent"] for p in payloads], dtype=torch.float32)
            # payloads are (port1, port2); flip seat-2 rows -> (student, opp)
            flip = self.seat2
            stocks[flip] = stocks[flip].flip(-1)
            percent[flip] = percent[flip].flip(-1)
            if self._frame_count > 0:
                self.assembler.push_reward(
                    compute_reward(
                        self._prev_stocks, stocks,
                        self._prev_percent, percent, resets,
                    ).to(device)
                )
            if self._frame_count > 0:
                for i in range(len(payloads)):
                    if resets[i]:
                        continue  # boundary artifacts belong to no game
                    tracker = self.trackers[self.specs[i].kind]
                    if stocks[i, 0] < self._prev_stocks[i, 0]:
                        tracker.add_death(float(self._prev_percent[i, 0]))
                    if stocks[i, 1] < self._prev_stocks[i, 1]:
                        tracker.add_kill(float(self._prev_percent[i, 1]))
            self._prev_stocks, self._prev_percent = stocks, percent

            games = [p["game"] for p in payloads]
            encoded = self._encode(games)
            # perspective swap commutes with encoding: both seat views are
            # pointer swaps of one encoded struct. The parser fixes p0=port1;
            # each agent must see ITSELF as p0, so seat-2 envs get the
            # swapped view (per-leaf where over the seat mask).
            swapped = encoded._replace(p0=encoded.p1, p1=encoded.p0)
            seat2 = self.seat2.to(device)

            def mix(a, b):  # a where seat2 else b, per leaf
                return tree.map_structure(
                    lambda x, y: torch.where(
                        seat2.view(-1, *([1] * (x.dim() - 1))), x, y
                    ),
                    a, b,
                )

            student_view = mix(swapped, encoded)
            opponent_view = mix(encoded, swapped)
            resets_dev = resets.to(device)
            pending_resets.append(resets_dev)
            # Fire the reference request first: the TF subprocess computes
            # while we run student + opponent inference on the GPU, so only
            # the un-hidden remainder shows up as stall in recv() below.
            ref_stall = 0.0
            if self.ref_bridge is not None:
                import time as time_lib

                t0 = time_lib.perf_counter()
                ref_games = [payloads[i]["raw_game"] for i in self.ref_idx]
                ref_resets = [bool(resets[i]) for i in self.ref_idx]
                self.ref_bridge.send(ref_games, ref_resets)
                ref_stall += time_lib.perf_counter() - t0

            controllers1, records, hidden_before = self.student.step(
                student_view, resets_dev
            )

            opp_controllers: dict[int, tp.Any] = {}
            for name, idx in self.groups.items():
                agent = self.opponents[name]
                sel = torch.tensor(idx, device=device)
                group_view = tree.map_structure(
                    lambda x: x.index_select(0, sel), opponent_view
                )
                ctrls, _, _ = agent.step(group_view, resets_dev[sel])
                for j, env_i in enumerate(idx):
                    opp_controllers[env_i] = ctrls[j]

            if self.ref_bridge is not None:
                import time as time_lib

                t0 = time_lib.perf_counter()
                ref_ctrls = self.ref_bridge.recv()
                ref_stall += time_lib.perf_counter() - t0
                # ref_step_ms = time actually stalled, not TF's compute time
                self.ref_step_ms = 0.9 * self.ref_step_ms + 0.1 * (
                    1e3 * ref_stall
                )
                for j, env_i in enumerate(self.ref_idx):
                    opp_controllers[env_i] = tree.map_structure(
                        lambda x: x[j], ref_ctrls
                    )

            for i, conn in enumerate(self._conns):
                port = self.specs[i].student_port
                cmd = {port: controllers1[i]}
                if i in opp_controllers:
                    cmd[3 - port] = opp_controllers[i]
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
        if self.ref_bridge is not None:
            self.ref_bridge.stop()
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
