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
import queue as queue_lib
import threading
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


@dataclasses.dataclass
class RolloutConfig:
    num_envs: int = 8
    unroll_length: int = 240  # 4s, slippi-ai's RL rollout length
    opponent: str = "teacher"  # "teacher" | "cpu:<level>"
    bot_char: str = "FOX"
    opponent_char: str = "FOX"
    stage: str = "FINAL_DESTINATION"
    games_per_dolphin: int = 20


class _EnvThread(threading.Thread):
    """Owns one Dolphin; parses/encodes frames in, controllers out."""

    def __init__(self, idx: int, make_env, out_queue: queue_lib.Queue):
        super().__init__(daemon=True)
        self.idx = idx
        self.make_env = make_env
        self.out = out_queue
        self.inbox: queue_lib.Queue = queue_lib.Queue(maxsize=1)
        self.stop_flag = threading.Event()

    def run(self) -> None:
        try:
            for payload in self.make_env(self.idx, self.inbox, self.stop_flag):
                self.out.put((self.idx, payload))
        except Exception as e:  # surfaced by the worker's gather loop
            self.out.put((self.idx, e))


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
        self._threads: list[_EnvThread] = []
        self._gather: queue_lib.Queue = queue_lib.Queue()

    def _make_env(self, idx: int, inbox: queue_lib.Queue, stop: threading.Event):
        """Generator: yields (gamestate-derived payloads); receives controllers."""
        import melee

        from slippi_ai import controller_lib
        from slippi_ai import dolphin as dolphin_lib
        from slippi_db.parse_libmelee import Parser

        cfg = self.config
        opp = self.game_lib.Opponent.parse(
            cfg.opponent if cfg.opponent.startswith("cpu")
            else f"ckpt:_:{cfg.opponent_char}"
        )
        players = {
            1: dolphin_lib.AI(character=melee.Character[cfg.bot_char.upper()]),
            2: opp.make_player(),
        }
        while not stop.is_set():
            dolphin = self.game_lib.make_dolphin(players, headless=True, stage=cfg.stage)
            parser = Parser(ports=[1, 2])
            games = 0
            last_frame = None
            try:
                for gs in dolphin.iter_gamestates(skip_menu_frames=True):
                    resetting = last_frame is not None and gs.frame < last_frame
                    if resetting:
                        parser = Parser(ports=[1, 2])
                        games += 1
                        if games >= cfg.games_per_dolphin:
                            break
                    last_frame = gs.frame
                    game = parser.get_game(gs)
                    p1, p2 = gs.players[1], gs.players[2]
                    payload = dict(
                        game=game,
                        resetting=resetting,
                        stocks=(int(p1.stock), int(p2.stock)),
                        percent=(float(p1.percent), float(p2.percent)),
                    )
                    yield payload
                    controllers = inbox.get()  # barrier: wait for batched step
                    if controllers is None:
                        return
                    for port, controller_state in controllers.items():
                        controller_lib.send_controller(
                            dolphin.controllers[port], controller_state
                        )
            finally:
                dolphin.stop()

    def _ensure_started(self) -> None:
        if self._threads:
            return
        import numpy as np  # noqa: F401  (env threads use numpy via parser)

        for i in range(self.config.num_envs):
            t = _EnvThread(i, lambda idx, inbox, stop: self._make_env(idx, inbox, stop), self._gather)
            self._threads.append(t)
            t.start()
        self._frame_count = 0
        n = self.config.num_envs
        self._prev_stocks = torch.full((n, 2), 4.0)
        self._prev_percent = torch.zeros(n, 2)

    def _gather_all(self) -> list[dict]:
        n = self.config.num_envs
        payloads: dict[int, dict] = {}
        while len(payloads) < n:
            idx, payload = self._gather.get()
            if isinstance(payload, Exception):
                raise RuntimeError(f"env {idx} died") from payload
            payloads[idx] = payload
        return [payloads[i] for i in range(n)]

    def _encode(self, games: list) -> tp.Any:
        import numpy as np

        batched = tree.map_structure(lambda *xs: np.stack(xs), *games)
        encoded = self.student.policy.network.encode_game(batched)
        return tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x)
            ),
            encoded,
        )

    @staticmethod
    def _swap_perspective(game):
        return game._replace(p0=game.p1, p1=game.p0)

    def collect(self, num_trajectories: int) -> list[Trajectory]:
        """Run the sync-barrier loop until N trajectory chunks are assembled."""
        self._ensure_started()
        cfg = self.config
        out: list[Trajectory] = []

        while len(out) < num_trajectories:
            payloads = self._gather_all()
            resets = torch.tensor([p["resetting"] for p in payloads])
            for i in torch.nonzero(resets).flatten().tolist():
                self.student.reset_env(i)
                if self.opponent is not None:
                    self.opponent.reset_env(i)

            stocks = torch.tensor([p["stocks"] for p in payloads], dtype=torch.float32)
            percent = torch.tensor([p["percent"] for p in payloads], dtype=torch.float32)
            if self._frame_count > 0:
                self.assembler.push_reward(
                    compute_reward(
                        self._prev_stocks, stocks,
                        self._prev_percent, percent, resets,
                    )
                )
            self._prev_stocks, self._prev_percent = stocks, percent

            snap = None
            if self._frame_count % cfg.unroll_length == 0:
                snap = self.student.hidden_snapshot()

            games = [p["game"] for p in payloads]
            states = self._encode(games)
            controllers1, record = self.student.step(states)

            controllers2 = None
            if self.opponent is not None:
                opp_states = self._encode(
                    [self._swap_perspective(g) for g in games]
                )
                controllers2, _ = self.opponent.step(opp_states)

            for i, thread in enumerate(self._threads):
                cmd = {1: controllers1[i]}
                if controllers2 is not None:
                    cmd[2] = controllers2[i]
                thread.inbox.put(cmd)

            self.assembler.push_frame(record, resets, snap)
            self._frame_count += 1
            if self.assembler.ready():
                out.append(self.assembler.emit())
        return out

    def stop(self) -> None:
        for t in self._threads:
            t.stop_flag.set()
            try:
                t.inbox.put_nowait(None)
            except queue_lib.Full:
                pass
