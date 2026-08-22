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

import typing as tp

import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent, FrameRecord
from smashbot.rl.config import RolloutConfig  # noqa: F401  (re-export)
from smashbot.rl.env_process import (  # noqa: F401  (re-export)
    _env_process_main, next_opponent_char,
)
from smashbot.rl.ppo import ActionData, Trajectory, slice_trajectory_rows


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

    # ema_alpha 0.008 ~ a 250-game horizon (user-dialed): 2-3 full waves of
    # concurrent games across the fleet, so the ticker EMA reflects several
    # rounds rather than single-batch luck (~±3pp wobble vs ±5pp at the old
    # 100-game horizon). Persisted tracker state stores the EMA VALUES
    # only, so restored checkpoints pick up the new alpha automatically.
    def __init__(self, window: int = 100, event_window: int = 200,
                 ema_alpha: float = 0.008):
        import collections

        self.diffs = collections.deque(maxlen=window)  # per finished game
        self.kill_percents = collections.deque(maxlen=event_window)
        self.death_percents = collections.deque(maxlen=event_window)
        self.wins = self.losses = self.draws = 0
        # EMA companion to the window: smoother (no window-exit jumps) and
        # persistable across restarts via state()/load_state — the window
        # resets every boot; the EMA rides in the RL checkpoint.
        self.ema_alpha = ema_alpha
        self.win_ema: float | None = None
        self.diff_ema: float | None = None

    def add_game(self, final_stocks: tuple[int, int]) -> None:
        bot, opp = final_stocks
        diff = bot - opp
        self.diffs.append(diff)
        if bot > opp:
            self.wins += 1
        elif opp > bot:
            self.losses += 1
        else:
            self.draws += 1
        if diff != 0:  # EMA over decided games, matching win_rate_recent
            outcome = 1.0 if diff > 0 else 0.0
            a = self.ema_alpha
            # seed at the 0.5 prior, not the first outcome: an extreme seed
            # takes ~200 games to wash out at this alpha (live-caught: SP:
            # read 0% for hours after its first game happened to be a loss)
            prev = 0.5 if self.win_ema is None else self.win_ema
            self.win_ema = (1 - a) * prev + a * outcome
        a = self.ema_alpha
        prev_d = 0.0 if self.diff_ema is None else self.diff_ema
        self.diff_ema = (1 - a) * prev_d + a * diff

    def state(self) -> dict:
        """Persistable summary state (EMA VALUES + lifetime counters); the
        raw windows are boot-local by design. ema_alpha is deliberately NOT
        persisted: the horizon is a code-level tuning knob, so restored
        checkpoints pick up the current default automatically."""
        return {"win_ema": self.win_ema, "diff_ema": self.diff_ema,
                "wins": self.wins, "losses": self.losses,
                "draws": self.draws}

    def load_state(self, st: dict) -> None:
        self.win_ema = st.get("win_ema")
        self.diff_ema = st.get("diff_ema")
        self.wins = st.get("wins", 0)
        self.losses = st.get("losses", 0)
        self.draws = st.get("draws", 0)

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
            "win_rate_ema": self.win_ema if self.win_ema is not None else 0.5,
            "stock_diff_ema": self.diff_ema if self.diff_ema is not None else 0.0,
        }


class _PhaseProfiler:
    """Opt-in per-frame phase timing for the worker loop (SMASHBOT_PROFILE=1):
    prints averaged ms per phase every `every` frames after a warm-up."""

    def __init__(self, every: int = 200, warmup_frames: int = 600):
        import time as _time

        self._time = _time
        self.every = every
        self.warmup = warmup_frames
        self.acc: dict[str, float] = {}
        self.n = 0

    def t(self) -> float:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return self._time.perf_counter()

    def lap(self, key: str, t0: float) -> None:
        self.acc[key] = self.acc.get(key, 0.0) + (self.t() - t0) * 1e3

    def frame(self, frame_count: int) -> None:
        if frame_count <= self.warmup:
            self.acc = {}
            return
        self.n += 1
        if self.n % self.every == 0:
            total = sum(self.acc.values())
            parts = "  ".join(f"{k} {v / self.n:6.1f}" for k, v in self.acc.items())
            print(f"[profile] ms/frame total {total / self.n:6.1f} | {parts}",
                  flush=True)


class _HarvestGroup:
    """Imitation harvest for opponent seats of one model config (delay +
    controller encoding): a ChunkAssembler over a fixed row set (all slot
    envs) plus a per-frame eligibility mask; emit keeps only rows eligible
    for the entire chunk."""

    def __init__(self, key, rows, unroll, delay, reencode, device):
        self.key = key
        self.rows = list(rows)
        self.rows_cpu = torch.tensor(self.rows)
        self.rows_t = torch.tensor(self.rows, device=device)
        self.unroll = unroll
        self.assembler = ChunkAssembler(unroll, delay)
        # trajectory -> student-schema trajectory (applied once per emitted
        # chunk, on the sliced rows only), or None
        self.reencode = reencode
        self.pending: list = []  # (resets[R], elig[R]) per frame
        self.elig: list[torch.Tensor] = []  # [R] per pushed record
        self.device = device

    def push_reward(self, reward_rows: torch.Tensor) -> None:
        self.assembler.push_reward(reward_rows)

    def step(self, resets_rows, elig_rows, records, imit_out) -> None:
        self.pending.append((resets_rows, elig_rows))
        for j, rec in enumerate(records):
            frame_resets, frame_elig = self.pending[j]
            self.assembler.push_frame(rec, frame_resets.to(self.device), None)
            self.elig.append(frame_elig)
        if records:
            del self.pending[: len(records)]
        if self.assembler.ready():
            traj = self.assembler.emit()._replace(kind="imitation")
            T = self.unroll
            window = torch.stack(self.elig[: T + 1], dim=1)  # [R, T+1]
            self.elig = self.elig[T:]
            rows = window.all(dim=1).nonzero().flatten().tolist()
            if rows:
                traj = slice_trajectory_rows(traj, rows)
                if self.reencode is not None:
                    traj = self.reencode(traj)
                imit_out.append(traj)


class _Seat:
    """A slot role (`current` or `outgoing`): member key, the agent serving
    it, and the agent's model-config label (for harvest grouping)."""

    __slots__ = ("member", "agent", "config")

    def __init__(self, member: str, agent, config: str):
        self.member = member
        self.agent = agent
        self.config = config


class _SlotPool:
    """Agents a slot's seats borrow from — the only place model configs are
    named: `ours_main` (the slot policy), `ours_spare` (parks outgoing
    weights), `phillip` (wrapper over the shared Phillip module)."""

    __slots__ = ("ours_main", "ours_spare", "phillip")

    def __init__(self):
        self.ours_main = None
        self.ours_spare = None
        self.phillip = None


class DolphinRolloutWorker:
    """N Dolphins, one batched student agent covering every student-driven
    seat (each env's student seat + BOTH seats of self-play envs — one wide
    forward, no second policy copy), plus batched opponent agents; sync-
    barrier frame loop.

    Learner-row layout: rows 0..D-1 are the D dolphins' primary (student)
    seats; rows D.. are the second seats of self-play dolphins. Row count is
    always config.num_envs (= the trajectory/memory budget), while the
    dolphin count is num_envs - self_envs."""

    def __init__(
        self,
        config: RolloutConfig,
        student: BatchedPolicyAgent,
        opponents: dict | None = None,  # {"teacher": agent, ("slot", i): agent}
        specs: list | None = None,  # per-env EnvSpec; default from make_partition
        harvest_imitation: bool = False,  # collect whitelisted ref seats
    ):
        # Imported lazily: this class needs Dolphin, the rest of the module
        # doesn't.
        from smashbot.eval import game as game_lib

        from smashbot.rl.pool import make_partition, student_whitelist

        self.config = config
        self.student = student
        # validates the league flags (env counts must be 0, pfsp required)
        self._league = config.league_members()
        self._league_cpu = config.league_cpu
        # Deferred adoption: an env changes opponent (brain, char, payoff
        # label) only at its own game boundary. The auction announces a
        # slot's incoming member; each env adopts it when its game ends.
        # Seats: `current` / `outgoing` per slot, each a full-batch agent
        # (fixed shape; controllers routed per row). CPU is the exception:
        # no seat, adopted at a Dolphin recycle via slot_desired.
        #   env_member[i]    member env i is fighting now
        #   slot_incoming[k] announced member
        #   _seats[k]        {"current", "outgoing"} -> _Seat | None
        #   slot_pending[k]  envs yet to adopt the incoming
        self.env_member: dict[int, str] = {}
        self.slot_incoming: dict[int, str] = {}
        self.slot_pending: dict[int, set] = {}
        self._seats: dict[int, dict[str, "_Seat | None"]] = {}
        self._pool: dict[int, _SlotPool] = {}
        # n -> agent over a fresh our-config module (the per-slot spare)
        self.outgoing_factory: tp.Optional[tp.Callable[[int], tp.Any]] = None
        self.slot_desired: dict[int, str] = {}
        # slot -> char lock ("FOX"/... or None) while the slot serves a
        # char-locked import; written by pool.apply_assignments and
        # piggybacked to the slot's envs ("opp_char_lock") so they pin the
        # member's character instead of redrawing per game. Empty / all-None
        # outside league_imports (the key is then never sent at all).
        self.slot_char_lock: dict[int, str | None] = {}
        self._has_imports = bool(config.league_imports)
        whitelist = student_whitelist(config.char_whitelist, config.bot_char)
        self._whitelist = set(whitelist)
        self.specs = specs or make_partition(
            config.num_envs, config.cpu_envs, config.teacher_envs,
            config.snapshot_slots, config.main12_prob, config.partition_seed,
            ref_envs=config.ref_envs, self_envs=config.self_envs,
            char_whitelist=whitelist,
        )
        # Memory-neutral arithmetic (hard OOM constraint): each self-play
        # dolphin feeds TWO learner rows, so dolphins = num_envs - self_envs
        # and the learner batch stays exactly num_envs trajectories.
        self.self_idx = [
            i for i, sp in enumerate(self.specs) if sp.kind == "self"
        ]
        self.num_dolphins = len(self.specs)
        self.num_rows = self.num_dolphins + len(self.self_idx)
        assert self.num_rows == config.num_envs, (
            f"specs must cover num_envs learner rows: {self.num_dolphins} "
            f"dolphins + {len(self.self_idx)} self seats != {config.num_envs}"
        )
        if specs is None:
            assert len(self.self_idx) == config.self_envs
        assert student.num_envs == self.num_rows, (
            "student agent must cover every learner row"
        )
        self.opponents = opponents or {}
        # controllers travel to the envs as 13-float rows (encode.controller_*)
        student.set_flat_controllers(True)
        for ag in self.opponents.values():
            ag.set_flat_controllers(True)
        # group name -> env index list (fixed membership = stable batch shapes)
        self.groups: dict = {}
        self.ref_idx: list[int] = []
        for i, spec in enumerate(self.specs):
            if spec.kind == "teacher":
                self.groups.setdefault("teacher", []).append(i)
            elif spec.kind == "snapshot":
                self.groups.setdefault(("slot", spec.group), []).append(i)
            elif spec.kind == "reference":
                # served in-process by the ported torch checkpoint (see
                # scripts/port_ref_model.py) — same path as teacher/slots.
                # The TF RefBridge remains available for eval batteries.
                self.groups.setdefault("reference", []).append(i)
                self.ref_idx.append(i)
        for name, idx in self.groups.items():
            assert name in self.opponents, f"no agent supplied for group {name}"
            assert self.opponents[name].num_envs == len(idx)
            # contiguous env ranges per group: group views are plain slices
            # (no index_select launches — ~150 leaves x groups per frame)
            assert idx == list(range(idx[0], idx[0] + len(idx))), (
                f"group {name} envs not contiguous: {idx}"
            )

        # all same-config slots are stepped by ONE LeagueAgent (one Python
        # pass per frame); seats hold slot refs into it. Spares and Phillip
        # keep their own agents.
        slot_names = sorted(n for n in self.groups if isinstance(n, tuple))
        self._league_slots = [n[1] for n in slot_names]
        self._league_agent = None
        if slot_names:
            from smashbot.rl.agent import LeagueAgent

            first = self.opponents[slot_names[0]]
            self._league_agent = LeagueAgent(
                [self.opponents[n].policy for n in slot_names],
                len(self.groups[slot_names[0]]),
                name_code=int(first._name[0].item()), device=first.device,
                temperature=first.temperature,
            )
        # dolphin-level seat mask (for the opponent-view mix)
        self.seat2 = torch.tensor(
            [sp.student_port == 2 for sp in self.specs]
        )
        # row-level maps: which dolphin, which port, which kind per row
        self._row_dolphin = torch.tensor(
            list(range(self.num_dolphins)) + self.self_idx
        )
        row_ports = [sp.student_port for sp in self.specs] + [
            3 - self.specs[i].student_port for i in self.self_idx
        ]
        self.row_seat2 = torch.tensor([p == 2 for p in row_ports])
        self._self_row_of = {
            d: self.num_dolphins + j for j, d in enumerate(self.self_idx)
        }
        self.row_kinds = [sp.kind for sp in self.specs] + (
            ["self"] * len(self.self_idx)
        )
        self.game_lib = game_lib
        self.assembler = ChunkAssembler(config.unroll_length, student.delay)
        self.trackers = {
            k: GameTracker()
            for k in ("cpu", "teacher", "snapshot", "reference", "self")
        }
        # slot-game callback for PFSP payoff attribution: (slot, student_won,
        # actual_kind) per decided game on a snapshot-slot env — actual_kind
        # is "snapshot"/"teacher"/"cpu", following what the env really served
        # (league members share this pathway). Wired by train_rl.
        self.on_snapshot_game: tp.Optional[
            tp.Callable[[int, bool, str], None]
        ] = None
        # n -> wrapper over the shared Phillip module (set by train_rl)
        self.phillip_factory: tp.Optional[tp.Callable[[int], tp.Any]] = None
        # Imitation harvest of opponent seats: the fixed reference group
        # (ref_envs mode) and/or every league policy seat, grouped by model
        # config into _HarvestGroups.
        self.harvest_imitation = harvest_imitation and (
            bool(self.ref_idx) or bool(self._league)
        )
        self._harvest_groups: dict[str, _HarvestGroup] = {}
        if self.harvest_imitation:
            self._stu_embed = student._embed_controller
            self._student_name_code = int(student._name[0].item())
            self._slot_order = sorted(
                {sp.group for sp in self.specs if sp.kind == "snapshot"}
            )
            self._slot_rows = [
                i for k in self._slot_order for i in self.groups[("slot", k)]
            ]
            assert len({len(self.groups[("slot", k)]) for k in self._slot_order}) <= 1
            if self.ref_idx:
                ref_agent = self.opponents["reference"]
                self._imit_elig: list[torch.Tensor] = []
                self._imit_pending: list = []
                self._imit_assembler = ChunkAssembler(
                    config.unroll_length, ref_agent.delay
                )
                self._ref_embed = ref_agent._embed_controller
        self._procs: list = []
        self._conns: list = []
        import os as _os

        self._prof = _PhaseProfiler() if _os.environ.get("SMASHBOT_PROFILE") else None

    def _ensure_started(self) -> None:
        if self._procs:
            return
        import multiprocessing as mp

        # forkserver, preloading ONLY the torch-free env module: a spawned
        # child would re-import __main__ (train_rl -> torch, ~0.26 GB private
        # per env); forked-from-server envs stay ~10 MB and share its pages
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["smashbot.rl.env_process"])
        # env processes encode frames with a torch-free numpy encoder rebuilt
        # from this spec (pure data; see smashbot.encode)
        from smashbot import embed as embed_lib

        encoder_spec = embed_lib.EmbedConfig().make_game_embedding().spec()
        for i in range(self.num_dolphins):
            parent, child = ctx.Pipe()
            # non-daemon: libmelee's slippstream forks its own child
            p = ctx.Process(
                target=_env_process_main,
                args=(i, self.config, self.specs[i], child, encoder_spec),
            )
            p.start()
            self._procs.append(p)
            self._conns.append(parent)
        self._frame_count = 0
        n = self.num_rows
        self._prev_stocks = torch.full((n, 2), 4.0)
        self._prev_percent = torch.zeros(n, 2)

    def _gather_all(self) -> list[dict]:
        """Barrier recv with a watchdog: one silent env must crash the run
        loudly (env index + spec + log path), never hang it. Learned the hard
        way — a wedged Dolphin boot froze a 128-env run overnight at step 36
        with zero symptoms beyond a stopped ticker."""
        import time as time_lib

        deadline = time_lib.monotonic() + self.config.env_timeout
        payloads = []
        for i, conn in enumerate(self._conns):
            try:
                if not conn.poll(max(0.0, deadline - time_lib.monotonic())):
                    tag = f"-{self.config.log_tag}" if self.config.log_tag else ""
                    # capture WHERE it is stuck: SIGUSR1 -> faulthandler dumps
                    # the env's python stack into its log before we die
                    import os as os_lib
                    import signal as sig_lib

                    try:
                        os_lib.kill(self._procs[i].pid, sig_lib.SIGUSR1)
                        time_lib.sleep(2.0)
                    except (OSError, IndexError):
                        pass
                    raise RuntimeError(
                        f"env {i} silent for {self.config.env_timeout}s "
                        f"(spec={self.specs[i]}); its python stack was just "
                        f"dumped to /tmp/smashbot-env{tag}-{i}.log"
                    )
                payloads.append(conn.recv())
            except (EOFError, BrokenPipeError) as e:
                raise RuntimeError(f"env {i} died") from e
        return payloads

    def _encode(self, games: list) -> tp.Any:
        """games: per-env (bools, ints, floats) flat vectors of the ALREADY
        encoded frame (encode.flatten_typed, env-side). Three stacks, three
        host->GPU copies, then split back into the struct on the GPU."""
        import numpy as np

        from smashbot import encode

        device = self.student.device
        if not hasattr(self, "_game_template"):
            from smashbot import embed as embed_lib

            # building the embedding constructs nn.Modules (weight init
            # draws from the global RNG): fork it so encoding leaves the
            # sampling stream untouched
            with torch.random.fork_rng(devices=[]):
                self._game_template = embed_lib.EmbedConfig().make_game_embedding().dummy()
            self._game_layout = encode.layout_of(self._game_template)
        b, i, f = (
            torch.from_numpy(np.stack([g[k] for g in games])).to(device, non_blocking=True)
            for k in range(3)
        )
        return encode.unflatten_typed_torch(self._game_template, self._game_layout, b, i, f)

    def _group_view(self, name, opponent_view):
        """The group's rows of the opponent view, as zero-copy slices."""
        idx = self.groups[name]
        lo, hi = idx[0], idx[-1] + 1
        return tree.map_structure(lambda x: x[lo:hi], opponent_view)

    @staticmethod
    def _swap_perspective(game):
        return game._replace(p0=game.p1, p1=game.p0)

    def _reencode_record(self, rec: FrameRecord, embed=None) -> FrameRecord:
        """Opponent-seat record -> student schema: actions re-encoded through
        the student's controller embedding, name set to the student's
        code. `embed` = the opponent's embedding (default: reference)."""
        import numpy as np

        embed = embed if embed is not None else self._ref_embed
        # records store actions widened to int64/bool; decode expects each
        # leaf embedding's native dtype (uint8/int32) back
        encoded_np = embed.map(
            lambda e, x: x.astype(getattr(e, "dtype", x.dtype)),
            tree.map_structure(lambda x: x.cpu().numpy(), rec.prev_action),
        )
        raw = embed.decode(encoded_np)
        prev = tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(
                    x.astype(np.int64) if x.dtype.kind in "iu" else x
                )
            ).to(self.student.device),
            self._stu_embed.from_state(raw),
        )
        return FrameRecord(
            state=rec.state,
            prev_action=prev,
            logits=rec.logits,  # ref-schema; unused by the imitation loss
            name=torch.full_like(rec.name, self._student_name_code),
        )

    # league serving kinds -> GameTracker keys (phillip games keep the
    # "reference" tracker for ticker R:/rl/reference/* continuity)
    _TRACKER_KIND = {"phillip": "reference"}

    @staticmethod
    def member_kind(key: str | None) -> str:
        """teacher/cpu/phillip as themselves; snapshot paths and imports
        are "snapshot"."""
        return key if key in ("teacher", "cpu", "phillip") else "snapshot"

    def _actual_kind(self, i: int, serving: str | None) -> str:
        """Kind actually serving env i's opponent seat (tracker key)."""
        sp = self.specs[i]
        if sp.kind != "snapshot":
            return sp.kind
        if serving == "cpu":
            return "cpu"
        return self.member_kind(self.env_member.get(i))

    def _result_key(self, i: int, result_serving: str | None) -> str | None:
        """Payoff-row key for a game that ended on slot env i."""
        if result_serving == "cpu":
            return "cpu"
        return self.env_member.get(i)

    _POLICY_KINDS = ("snapshot", "teacher")

    def _pool_agent(self, slot: int, member: str):
        """(agent, config label) serving `member`, created lazily."""
        pool = self._pool.setdefault(slot, _SlotPool())
        n = len(self.groups[("slot", slot)])
        if member == "phillip":
            if pool.phillip is None:
                assert self.phillip_factory is not None, (
                    "a slot is assigned to phillip but worker.phillip_factory "
                    "was never set (train_rl sets it at startup under "
                    "league_phillip)"
                )
                pool.phillip = self.phillip_factory(n)
                pool.phillip.set_flat_controllers(True)
            return pool.phillip, "phillip"
        if pool.ours_main is None:
            pool.ours_main = self._league_agent.slot_ref(self._league_slots.index(slot))
        return pool.ours_main, "ours"

    def slot_weights_changed(self, slot: int) -> None:
        if self._league_agent is not None and slot in self._league_slots:
            self._league_agent.slot_weights_changed(self._league_slots.index(slot))

    def _rows_on(self, slot: int, key: str | None) -> list[int]:
        if key is None:
            return []
        return [
            i for i in self.groups.get(("slot", slot), [])
            if self.env_member.get(i) == key
        ]

    def _release_outgoing(self, slot: int) -> None:
        seats = self._seats.get(slot)
        if seats and seats["outgoing"] is not None:
            if not self._rows_on(slot, seats["outgoing"].member):
                seats["outgoing"] = None

    def _needs_slot_policy(self, member: str) -> bool:
        """Does `member` live in the slot policy module? (cpu has none,
        phillip has his own.)"""
        return member != "cpu" and member != "phillip"

    def _seat_for(self, slot: int, member: str) -> "_Seat | None":
        if member == "cpu":
            return None  # engine AI: no brain, no seat
        return _Seat(member, *self._pool_agent(slot, member))

    def _park(self, slot: int, seat: _Seat, slot_policy) -> _Seat:
        """Copy the slot policy's weights into the spare and re-seat the
        occupant there (the slot policy is about to be overwritten)."""
        assert self.outgoing_factory is not None and slot_policy is not None, (
            "deferred adoption needs a spare-brain factory (train_rl sets "
            "worker.outgoing_factory) and the slot policy to park"
        )
        pool = self._pool[slot]
        if pool.ours_spare is None:
            pool.ours_spare = self.outgoing_factory(len(self.groups[("slot", slot)]))
            pool.ours_spare.set_flat_controllers(True)
        pool.ours_spare.policy.load_state_dict(slot_policy.state_dict())
        return _Seat(seat.member, pool.ours_spare, seat.config)

    def _evict_outgoing(self, slot: int, onto: str) -> None:
        """Free the outgoing seat; any rows still on it take one mid-game
        swap (only when a game outlasts a whole snapshot_interval)."""
        out = self._seats[slot]["outgoing"]
        if out is None:
            return
        rows = self._rows_on(slot, out.member)
        if rows:
            print(f"slot {slot}: {len(rows)} env(s) still on {out.member} "
                  f"when the next member arrived — forcing onto {onto} "
                  f"(one mid-game swap)", flush=True)
            for i in rows:
                self.env_member[i] = onto
        self._seats[slot]["outgoing"] = None

    def begin_transition(
        self, slot: int, new_key: str, char_lock: str | None,
        slot_policy=None,
    ) -> None:
        """Announce the slot's next member (before apply_assignments loads
        any weights). Boot: envs adopt immediately and a char lock goes
        into the env specs. Later: the current occupant becomes outgoing
        if envs still fight it (parked when the newcomer needs the slot
        policy), and every env is pending until its own game boundary."""
        idx = self.groups.get(("slot", slot), [])
        seats = self._seats.setdefault(slot, {"current": None, "outgoing": None})
        old = self.slot_incoming.get(slot)
        if old == new_key:
            return
        if old is None:  # boot
            for i in idx:
                self.env_member[i] = new_key
                if char_lock is not None and new_key != "cpu":
                    self.specs[i].opponent_char = char_lock
        else:
            cur = seats["current"]
            if cur is not None and self._rows_on(slot, cur.member):
                self._evict_outgoing(slot, onto=cur.member)
                seats["outgoing"] = (
                    self._park(slot, cur, slot_policy)
                    if self._needs_slot_policy(cur.member)
                    and self._needs_slot_policy(new_key)
                    else cur  # agent undisturbed: serves in place
                )
        seats["current"] = self._seat_for(slot, new_key)
        self.slot_incoming[slot] = new_key
        pend = {i for i in idx if self.env_member.get(i) != new_key}
        if pend:
            self.slot_pending[slot] = pend
        else:
            self.slot_pending.pop(slot, None)
        self._release_outgoing(slot)

    def _adopt_pending(self, payloads: list[dict], resets_d) -> None:
        """Flip pending envs at their game boundary (resetting frame). A
        char-locked incoming waits until the new game's opp_char is the
        lock; cpu waits for the env's recycle report. Runs after the
        frame's results are attributed."""
        for slot, pend in list(self.slot_pending.items()):
            inc = self.slot_incoming[slot]
            lock = self.slot_char_lock.get(slot)
            for i in list(pend):
                p = payloads[i]
                if inc == "cpu":
                    ok = p.get("opp_serving") == "cpu"
                else:
                    ok = (
                        bool(resets_d[i])
                        and p.get("opp_serving") != "cpu"
                        and (lock is None or p.get("opp_char") == lock)
                    )
                if ok:
                    self.env_member[i] = inc
                    pend.discard(i)
            if not pend:
                del self.slot_pending[slot]
            self._release_outgoing(slot)

    def collect(self, num_trajectories: int) -> list[Trajectory]:
        """Run the sync-barrier loop until N PPO trajectory chunks are
        assembled; any imitation chunks harvested along the way (reference
        seats with whitelisted chars) are appended after them."""
        self._ensure_started()
        cfg = self.config
        out: list[Trajectory] = []
        imit_out: list[Trajectory] = []

        assert cfg.unroll_length % self.student.batch_steps == 0, (
            "unroll_length must be a multiple of batch_steps so chunk "
            "boundaries land on flush boundaries"
        )
        device = self.student.device
        if not hasattr(self, "_pending_resets"):
            self._pending_resets: list[torch.Tensor] = []
        pending_resets = self._pending_resets
        records_pushed = getattr(self, "_records_pushed", 0)
        row_dolphin = self._row_dolphin

        prof = self._prof  # opt-in per-phase timing (SMASHBOT_PROFILE=1)
        while len(out) < num_trajectories:
            t0 = prof.t() if prof else None
            payloads = self._gather_all()
            prof and prof.lap("gather", t0)
            # envs whose opponent seat is engine-AI-driven THIS frame (league
            # cpu adoption is lazy at recycle, so this follows each env's own
            # report, never the desired assignment). Empty outside league_cpu.
            cpu_now = {
                i for i, p in enumerate(payloads)
                if p.get("opp_serving") == "cpu"
            }
            for i, p in enumerate(payloads):
                if p.get("final_stocks") is not None:
                    a, b = p["final_stocks"]  # (port1, port2)
                    sp = self.specs[i]
                    if sp.kind == "self":
                        # both seats are the student: track the PORT-1 seat's
                        # win rate (a ~50% health metric, not a skill signal)
                        self.trackers["self"].add_game((a, b))
                        continue
                    if sp.student_port == 2:
                        a, b = b, a
                    # attribute to the kind that PLAYED the ended game
                    # (result_serving: carried alongside the result so a
                    # recycle-boundary kind flip can't misattribute it)
                    kind = self._actual_kind(i, p.get("result_serving"))
                    self.trackers[
                        self._TRACKER_KIND.get(kind, kind)
                    ].add_game((a, b))
                    if (
                        sp.kind == "snapshot"
                        and self.on_snapshot_game is not None
                        and a != b
                    ):
                        key = self._result_key(i, p.get("result_serving"))
                        if key is not None:
                            self.on_snapshot_game(key, a > b)
            resets_d = torch.tensor([p["resetting"] for p in payloads])
            if self.slot_pending:  # after results: ended games credit the old member
                self._adopt_pending(payloads, resets_d)

            resets = resets_d[row_dolphin]  # row-level
            resets_cpu = resets_d.tolist()

            stocks_d = torch.tensor([p["stocks"] for p in payloads], dtype=torch.float32)
            percent_d = torch.tensor([p["percent"] for p in payloads], dtype=torch.float32)
            # payloads are (port1, port2) per dolphin; expand to learner rows
            # and flip seat-2 rows -> (own seat, other seat)
            stocks = stocks_d[row_dolphin]
            percent = percent_d[row_dolphin]
            flip = self.row_seat2
            stocks[flip] = stocks[flip].flip(-1)
            percent[flip] = percent[flip].flip(-1)
            if self._frame_count > 0:
                reward = compute_reward(
                    self._prev_stocks, stocks,
                    self._prev_percent, percent, resets,
                ).to(device)
                self.assembler.push_reward(reward)
                if self.harvest_imitation:
                    # the reference seat's reward is the zero-sum mirror of
                    # the student seat's (both terms are antisymmetric)
                    if self.ref_idx:
                        ref_rows = torch.tensor(self.ref_idx, device=device)
                        self._imit_assembler.push_reward(-reward[ref_rows])
                    for g in self._harvest_groups.values():
                        g.push_reward(-reward[g.rows_t])
            if self._frame_count > 0:
                for i in range(self.num_dolphins):
                    if resets[i]:
                        continue  # boundary artifacts belong to no game
                    kind = self._actual_kind(
                        i, payloads[i].get("opp_serving")
                    )
                    tracker = self.trackers[self._TRACKER_KIND.get(kind, kind)]
                    if stocks[i, 0] < self._prev_stocks[i, 0]:
                        tracker.add_death(float(self._prev_percent[i, 0]))
                    if stocks[i, 1] < self._prev_stocks[i, 1]:
                        tracker.add_kill(float(self._prev_percent[i, 1]))
            self._prev_stocks, self._prev_percent = stocks, percent

            games = [p["game"] for p in payloads]
            t0 = prof.t() if prof else None
            encoded = self._encode(games)
            prof and prof.lap("encode", t0)
            # perspective swap commutes with encoding: both seat views are
            # pointer swaps of one encoded struct. The parser fixes p0=port1;
            # each agent must see ITSELF as p0, so seat-2 envs get the
            # swapped view (per-leaf where over the seat mask).
            swapped = encoded._replace(p0=encoded.p1, p1=encoded.p0)
            seat2 = self.seat2.to(device)

            def mix(a, b, mask):  # a where mask else b, per leaf
                return tree.map_structure(
                    lambda x, y: torch.where(
                        mask.view(-1, *([1] * (x.dim() - 1))), x, y
                    ),
                    a, b,
                )

            opponent_view = mix(encoded, swapped, seat2)
            # learner rows: primary seats of every dolphin + the second seat
            # of each self-play dolphin, all served by ONE student forward
            rows_dev = row_dolphin.to(device)
            rowsel = lambda s: tree.map_structure(
                lambda x: x.index_select(0, rows_dev), s
            )
            row_seat2 = self.row_seat2.to(device)
            student_view = mix(rowsel(swapped), rowsel(encoded), row_seat2)
            resets_dev = resets.to(device)
            pending_resets.append(resets_dev)
            t0 = prof.t() if prof else None
            controllers1, records, hidden_before = self.student.step(
                student_view, resets_dev,
                reset_indices=resets.nonzero().flatten().tolist(),
                # the snapshot is only consumed at a chunk boundary
                want_snapshot=(records_pushed % cfg.unroll_length == 0),
            )
            prof and prof.lap("student_step", t0)
            t0 = prof.t() if prof else None

            opp_controllers: dict[int, tp.Any] = {}
            ref_records: list[FrameRecord] = []
            # config -> slot -> [(row mask over the slot batch, records,
            # agent)] per occupied seat of that config
            harvest_parts: dict[str, dict[int, list]] = {}
            league_rows: dict = {}
            league_recs: dict = {}
            if self._league_agent is not None and self._seats:
                views, rsts, ridx = [], [], []
                for pos, k in enumerate(self._league_slots):
                    idx_k = self.groups[("slot", k)]
                    views.append(self._group_view(("slot", k), opponent_view))
                    rsts.append(resets_dev[idx_k[0]:idx_k[-1] + 1])
                    ridx.extend((pos, j) for j, i in enumerate(idx_k) if resets_cpu[i])
                rows_by_pos, recs_by_pos = self._league_agent.step(
                    views, torch.stack(rsts), ridx
                )
                for pos, k in enumerate(self._league_slots):
                    league_rows[k], league_recs[k] = rows_by_pos[pos], recs_by_pos[pos]
            for name, idx in self.groups.items():
                if isinstance(name, tuple):
                    # slot group: each occupied seat steps the full batch;
                    # controllers routed per row. No seats yet (pre-auction)
                    # = the slot policy serves every row, unlabeled.
                    k = name[1]
                    seats = self._seats.get(k)
                    view = self._group_view(name, opponent_view)
                    g_resets = resets_dev[idx[0]:idx[-1] + 1]
                    g_reset_idx = [j for j, i in enumerate(idx) if resets_cpu[i]]
                    if not seats:
                        live = [i for i in idx if i not in cpu_now]
                        if live:
                            ctrls, _, _ = self.opponents[name].step(
                                view, g_resets, reset_indices=g_reset_idx,
                                want_snapshot=False,
                            )
                            for j, env_i in enumerate(idx):
                                if env_i in live:
                                    opp_controllers[env_i] = ctrls[j]
                        continue
                    for seat in (seats["current"], seats["outgoing"]):
                        if seat is None:
                            continue
                        live = [
                            i for i in idx
                            if self.env_member.get(i) == seat.member
                            and i not in cpu_now
                        ]
                        if not live:
                            continue
                        if k in league_rows and getattr(seat.agent, "league", None) is self._league_agent:
                            ctrls, recs = list(league_rows[k]), [league_recs[k]]
                        else:
                            ctrls, recs, _ = seat.agent.step(
                                view, g_resets, reset_indices=g_reset_idx,
                                want_snapshot=False,
                            )
                        for j, env_i in enumerate(idx):
                            if env_i in live:
                                opp_controllers[env_i] = ctrls[j]
                        if self.harvest_imitation:
                            mask = torch.tensor(
                                [env_i in live for env_i in idx], device=device
                            )
                            harvest_parts.setdefault(seat.config, {}).setdefault(
                                k, []
                            ).append((mask, recs, seat.agent))
                    continue
                if cpu_now and all(i in cpu_now for i in idx):
                    # whole slot serving CPU lvl-9: no brain to run — skip
                    # the group's inference entirely. Safe for the agent's
                    # recurrent state: an env only returns to policy serving
                    # via a recycle, whose first frame is resetting=True.
                    # (Mixed groups mid-adoption still run the FULL batch —
                    # stable shapes for compile — and routing below drops
                    # the cpu rows' controllers.)
                    continue
                agent = self.opponents[name]
                group_view = self._group_view(name, opponent_view)
                ctrls, g_records, _ = agent.step(
                    group_view, resets_dev[idx[0]:idx[-1] + 1],
                    reset_indices=[j for j, i in enumerate(idx) if resets_cpu[i]],
                    want_snapshot=False,
                )
                if name == "reference" and self.harvest_imitation:
                    ref_records = g_records
                for j, env_i in enumerate(idx):
                    opp_controllers[env_i] = ctrls[j]

            prof and prof.lap("seat_steps", t0)
            t0 = prof.t() if prof else None
            for i, conn in enumerate(self._conns):
                port = self.specs[i].student_port
                cmd = {port: controllers1[i]}
                if i in self._self_row_of:
                    cmd[3 - port] = controllers1[self._self_row_of[i]]
                elif i in opp_controllers and i not in cpu_now:
                    # cpu_now rows have no policy seat: the engine AI drives
                    # the opponent port (like dedicated cpu envs today)
                    cmd[3 - port] = opp_controllers[i]
                if self._league_cpu and self.specs[i].kind == "snapshot":
                    # piggyback the desired serving kind; the env adopts a
                    # policy<->cpu change at its next recycle boundary
                    cmd["opp_kind"] = self.slot_desired.get(
                        self.specs[i].group, "policy"
                    )
                if self._has_imports and self.specs[i].kind == "snapshot":
                    # piggyback the slot's char lock (None = unlocked): the
                    # env pins a locked import's character at its next game
                    # boundary and resumes redraws once the lock clears
                    cmd["opp_char_lock"] = self.slot_char_lock.get(
                        self.specs[i].group
                    )
                conn.send(cmd)

            prof and prof.lap("send", t0)
            t0 = prof.t() if prof else None
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

            if self.harvest_imitation:
                if self.ref_idx:
                    self._harvest_step(
                        payloads, resets_d, self.ref_idx, ref_records,
                        imit_out,
                    )
                else:
                    self._league_harvest(
                        payloads, resets_d, harvest_parts, imit_out
                    )

            prof and prof.lap("assemble+harvest", t0)
            self._frame_count += 1
            prof and prof.frame(self._frame_count)
            if self.assembler.ready():
                out.append(self.assembler.emit())
        self._records_pushed = records_pushed
        return out + imit_out

    def _league_harvest(
        self,
        payloads: list[dict],
        resets_d: torch.Tensor,
        parts: dict[str, dict[int, list]],
        imit_out: list[Trajectory],
    ) -> None:
        """Per config group: one record per flushed frame over the fixed row
        set (all slot envs, slot order) = the slots' full-batch seat records
        concatenated (zeros for a slot with no seat of this config; a
        per-row select when a slot's two seats share the config), plus the
        eligibility mask (row on such a seat AND opponent char whitelisted).
        Groups with no live seat are dropped (their assembler would take
        rewards without records)."""
        T = self.config.unroll_length
        for cfg_key in [k for k in self._harvest_groups if k not in parts]:
            del self._harvest_groups[cfg_key]
        for cfg_key, by_slot in parts.items():
            group = self._harvest_groups.get(cfg_key)
            if group is None:
                agent = next(iter(by_slot.values()))[0][2]
                group = _HarvestGroup(
                    cfg_key, self._slot_rows, T, agent.delay,
                    self._traj_reencoder(agent) if cfg_key != "ours" else None,
                    self.student.device,
                )
                self._harvest_groups[cfg_key] = group
            # eligibility over the fixed row set
            elig = []
            for k in self._slot_order:
                idx = self.groups[("slot", k)]
                seats = by_slot.get(k)
                if not seats:
                    elig.extend([False] * len(idx))
                    continue
                on = torch.stack([m for m, _, _ in seats]).any(0).tolist()
                elig.extend(
                    o and payloads[i].get("opp_char") in self._whitelist
                    for o, i in zip(on, idx)
                )
            nrec = {len(recs) for seats in by_slot.values() for _, recs, _ in seats}
            assert len(nrec) == 1, (
                f"seats of config {cfg_key} flushed unevenly {nrec}: "
                "wrappers must share flush cadence (batch_steps)"
            )
            merged = []
            for j in range(nrec.pop()):
                pieces = []
                template = None
                for k in self._slot_order:
                    seats = by_slot.get(k)
                    if seats:
                        rec = seats[0][1][j]
                        for mask, recs, _ in seats[1:]:
                            mk = mask
                            rec = tree.map_structure(
                                lambda a, b: torch.where(
                                    mk.view(-1, *([1] * (a.dim() - 1))), b, a
                                ),
                                rec, recs[j],
                            )
                        template = rec
                        pieces.append(rec)
                    else:
                        pieces.append(None)
                assert template is not None
                n = len(self.groups[("slot", self._slot_order[0])])
                zeros = self._zero_record(cfg_key, template, n)
                pieces = [zeros if p is None else p for p in pieces]
                merged.append(
                    tree.map_structure(lambda *xs: torch.cat(xs, 0), *pieces)
                )
            group.step(
                resets_d[group.rows_cpu],
                torch.tensor(elig, dtype=torch.bool), merged, imit_out,
            )

    def _zero_record(self, cfg_key: str, template, n: int):
        """Cached zero record of one slot batch for a config (placeholder
        for slots with no seat of that config)."""
        cache = self.__dict__.setdefault("_zero_records", {})
        if cfg_key not in cache:
            cache[cfg_key] = tree.map_structure(
                lambda x: x.new_zeros((n,) + tuple(x.shape[1:])), template
            )
        return cache[cfg_key]

    def _traj_reencoder(self, agent):
        """Trajectory-level version of _reencode_record for an opponent
        config: re-encode the [R, T+1] action stream through the student's
        controller embedding and recondition the name — once per chunk."""
        embed = agent._embed_controller

        def reencode(traj: Trajectory) -> Trajectory:
            import numpy as np

            encoded_np = embed.map(
                lambda e, x: x.astype(getattr(e, "dtype", x.dtype)),
                tree.map_structure(
                    lambda x: x.cpu().numpy(), traj.actions.controller_state
                ),
            )
            raw = embed.decode(encoded_np)
            prev = tree.map_structure(
                lambda x: torch.from_numpy(
                    np.ascontiguousarray(
                        x.astype(np.int64) if x.dtype.kind in "iu" else x
                    )
                ).to(self.student.device),
                self._stu_embed.from_state(raw),
            )
            return traj._replace(
                actions=traj.actions._replace(controller_state=prev),
                name=torch.full_like(traj.name, self._student_name_code),
            )

        return reencode

    def _harvest_step(
        self,
        payloads: list[dict],
        resets_d: torch.Tensor,
        rows: list[int],
        records: list[FrameRecord],
        imit_out: list[Trajectory],
    ) -> None:
        """Per-frame imitation bookkeeping: buffer the harvested seats'
        eligibility (char whitelisted this game?) and resets, feed the
        harvested agent's flushed records into the imitation assembler, and
        emit whole-chunk-eligible rows as kind="imitation" trajectories.
        `rows` = the fixed reference group (ref_envs mode) or Phillip's
        current serving rows (league_phillip)."""
        device = self.student.device
        elig = torch.tensor(
            [payloads[i].get("opp_char") in self._whitelist
             for i in rows]
        )
        self._imit_pending.append(
            (resets_d[torch.tensor(rows)], elig)
        )
        for j, rec in enumerate(records):
            frame_resets, frame_elig = self._imit_pending[j]
            self._imit_assembler.push_frame(
                self._reencode_record(rec), frame_resets.to(device), None
            )
            self._imit_elig.append(frame_elig)
        if records:
            del self._imit_pending[: len(records)]
        if self._imit_assembler.ready():
            traj = self._imit_assembler.emit()._replace(kind="imitation")
            T = self.config.unroll_length
            window = torch.stack(self._imit_elig[: T + 1], dim=1)  # [R, T+1]
            self._imit_elig = self._imit_elig[T:]
            # conservative gate: harvest a row only if the reference seat's
            # char was whitelisted for the ENTIRE chunk
            rows = window.all(dim=1).nonzero().flatten().tolist()
            if rows:
                imit_out.append(slice_trajectory_rows(traj, rows))

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
