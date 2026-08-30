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

from smashbot.rl.agent import BatchedPolicyAgent, FrameRecord, LeagueAgent
from smashbot.rl.config import RolloutConfig  # noqa: F401  (re-export)
from smashbot.rl.env_process import (  # noqa: F401  (re-export)
    _env_process_main, next_opponent_char,
)
from smashbot.rl.league import League
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
    # Percent is a raw libmelee read; the state path wraps+clamps it but
    # the reward path would pass garbage straight through. Nothing deals
    # 100% in one frame, so the delta cap keeps |reward| <= 2.
    own_dmg = (percent[:, 0] - prev_percent[:, 0]).clamp(min=0, max=100)
    opp_dmg = (percent[:, 1] - prev_percent[:, 1]).clamp(min=0, max=100)
    reward = (opp_death - own_death) + damage_ratio * (opp_dmg - own_dmg)
    return torch.where(is_resetting, torch.zeros_like(reward), reward)


class GameTracker:
    """Game-outcome metrics vs the CURRENT training opponent (teacher now,
    snapshot pool later); fixed-yardstick evals stay in the M8 batteries.

    Time-free by design: a win is a win at 2 minutes or 7. Tracks rolling
    win rate, average final stock differential (-4..+4 dominance scale),
    average opponent percent at our kills (low = early kills, strong punish
    game), and average own percent at our deaths (high = hard to kill)."""

    # ema_alpha 0.008 ~ a 250-game horizon: several full fleet waves, so
    # the EMA reflects rounds rather than single-batch luck. Restored
    # checkpoints store EMA values only, so alpha changes apply cleanly.
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
        self.by_char: dict[str, tuple[int, int]] = {}

    def add_game(self, final_stocks: tuple[int, int],
                 opp_char: str | None = None) -> None:
        bot, opp = final_stocks
        diff = bot - opp
        # Winrate by OPPONENT character (locked members excluded at the
        # call site: their identity would pollute their char's column).
        if opp_char and diff != 0:
            w, g = self.by_char.get(opp_char, (0, 0))
            self.by_char[opp_char] = (w + (1 if diff > 0 else 0), g + 1)
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


class LeagueRuntime(tp.NamedTuple):
    """Everything the worker needs to serve the league (built by train_rl):
    the per-match protocol, the S x N grid, and Phillip's grid — a 1-slice
    LeagueAgent over his own architecture (None outside league_phillip).
    Both grids share one code path: execute() / infer() over fixed cells."""

    league: League
    agent: LeagueAgent
    phillip: LeagueAgent | None
    # Static import agent (v9): one slice per import member, cells
    # permanently assigned to dedicated envs — weights loaded once at
    # boot, allocator never involved.
    imports_agent: LeagueAgent | None = None


class DolphinRolloutWorker:
    """N Dolphins, one batched student agent covering every student-driven
    seat (each env's student seat + BOTH seats of self-play envs — one wide
    forward, no second policy copy), plus the opponent side; sync-barrier
    frame loop.

    Opponent side: fixed-kind envs (cpu / teacher / reference / import —
    imports route to the static imports_agent, not self.opponents) keep their
    own agents; LEAGUE envs draw an opponent per match and are ROUTED to a
    cell of the league grid (or a row of Phillip's agent) — see league.py.

    Learner-row layout: rows 0..D-1 are the D dolphins' primary (student)
    seats; rows D.. are the second seats of self-play dolphins. Row count is
    always config.num_envs (= the trajectory/memory budget), while the
    dolphin count is num_envs - self_envs."""

    def __init__(
        self,
        config: RolloutConfig,
        student: BatchedPolicyAgent,
        opponents: dict | None = None,  # {"teacher": agent, "reference": agent}
        specs: list | None = None,  # per-env EnvSpec; default from make_partition
        harvest_imitation: bool = False,  # collect whitelisted opponent seats
        league: LeagueRuntime | None = None,
    ):
        # Imported lazily: this class needs Dolphin, the rest of the module
        # doesn't.
        from smashbot.eval import game as game_lib

        from smashbot.rl.pool import make_partition, student_whitelist

        self.config = config
        self.student = student
        # validates the league flags (env counts must be 0, pfsp required)
        self._league_keys = config.league_members()
        self._runtime = league
        whitelist = student_whitelist(config.char_whitelist, config.bot_char)
        self._whitelist = set(whitelist)
        self.specs = specs or make_partition(
            config.num_envs, config.cpu_envs, config.teacher_envs,
            config.main12_prob, config.partition_seed,
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
        # fixed-kind group name -> env index list (stable batch shapes)
        self.groups: dict = {}
        self.ref_idx: list[int] = []
        self.league_idx: list[int] = []
        # dedicated import envs, in cell order (partition emits them
        # member-major, so index k sits at cell (k // N, k % N))
        self.import_idx: list[int] = []
        _imp = league.imports_agent if league is not None else None
        for i, spec in enumerate(self.specs):
            if spec.kind == "teacher":
                self.groups.setdefault("teacher", []).append(i)
            elif spec.kind == "import":
                self.import_idx.append(i)
            elif spec.kind == "snapshot":
                self.league_idx.append(i)
            elif spec.kind == "reference":
                # served in-process by the ported torch checkpoint (see
                # scripts/port_ref_model.py)
                self.groups.setdefault("reference", []).append(i)
                self.ref_idx.append(i)
        if self.import_idx:
            # LOUD pairing check, unconditional: without the static agent
            # the import envs' opponent seats would silently idle at
            # neutral and every game would be a free win
            assert _imp is not None and len(self.import_idx) == _imp.S * _imp.N, (
                f"{len(self.import_idx)} dedicated import envs but "
                + ("no imports agent" if _imp is None else
                   f"an {_imp.S}x{_imp.N} imports agent")
            )
        for name, idx in self.groups.items():
            assert name in self.opponents, f"no agent supplied for group {name}"
            assert self.opponents[name].num_envs == len(idx)
            # contiguous env ranges per group: group views are plain slices
            # (no index_select launches — ~150 leaves x groups per frame)
            assert idx == list(range(idx[0], idx[0] + len(idx))), (
                f"group {name} envs not contiguous: {idx}"
            )
        if self.league_idx:
            assert league is not None, "league envs need a LeagueRuntime"
            grid = league.agent
            self._grid_cells = grid.S * grid.N
            assert self._grid_cells >= len(self.league_idx), (
                f"league grid {grid.S}x{grid.N} cannot seat "
                f"{len(self.league_idx)} league envs"
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
            for k in ("cpu", "teacher", "snapshot", "reference", "self",
                      "import")
        }
        # Imitation harvest of opponent seats: the fixed reference group
        # (ref_envs mode) and/or every league seat, grouped by model config
        # ("ours" = the grid, "phillip" = his agent) into _HarvestGroups.
        self.harvest_imitation = harvest_imitation and (
            bool(self.ref_idx) or bool(self.league_idx)
            or bool(self.import_idx)
        )
        self._harvest_groups: dict[str, _HarvestGroup] = {}
        if self.harvest_imitation:
            self._stu_embed = student._embed_controller
            self._student_name_code = int(student._name[0].item())
            if self.ref_idx:
                ref_agent = self.opponents["reference"]
                self._imit_elig: list[torch.Tensor] = []
                self._imit_pending: list = []
                self._imit_assembler = ChunkAssembler(
                    config.unroll_length, ref_agent.delay
                )
                self._ref_embed = ref_agent._embed_controller
            if self.league_idx:
                T, dev = config.unroll_length, student.device
                self._harvest_groups["ours"] = _HarvestGroup(
                    "ours", range(self._grid_cells), T, league.agent.delay, None, dev,
                )
                if league.phillip is not None:
                    ph = league.phillip
                    if ph.delay != student.delay:
                        # Harvested chunks keep the OPPONENT's delay but are
                        # trained under the student's convention — a known,
                        # accepted approximation. Announce it.
                        print(
                            f"NOTE: imitation harvest delay mismatch — "
                            f"phillip {ph.delay} vs student {student.delay} "
                            f"({ph.delay - student.delay:+d} frames); his "
                            "imitation targets are timed to his reaction, "
                            "not ours", flush=True,
                        )
                    self._harvest_groups["phillip"] = _HarvestGroup(
                        "phillip", range(ph.S * ph.N), T, ph.delay,
                        self._traj_reencoder(ph), dev,
                    )
            imp = league.imports_agent if league is not None else None
            if imp is not None and self.import_idx:
                T, dev = config.unroll_length, student.device
                self._harvest_groups["imports"] = _HarvestGroup(
                    "imports", range(imp.S * imp.N), T, imp.delay, None, dev,
                )
        self._procs: list = []
        self._conns: list = []
        import os as _os

        self._prof = _PhaseProfiler() if _os.environ.get("SMASHBOT_PROFILE") else None
        # stream-parallel serving state (see _run_serving)
        self._serve_pool = None
        self._serve_streams: dict[str, tp.Any] = {}
        self._serve_frames = 0
        self._serve_parallel = (
            self.config.parallel_serving
            and torch.device(self.student.device).type == "cuda"
        )
        # the grid sub-timers device-sync between laps, which would
        # serialize the streams — serial mode only
        if (self._prof is not None and league is not None
                and not self._serve_parallel):
            # grid sub-phases (forward / record+to_cpu / decode+queues)
            t = {"t0": None}

            def timer(name):
                now = self._prof.t()
                if t["t0"] is not None:
                    self._prof.acc[f"grid:{name}"] = self._prof.acc.get(f"grid:{name}", 0.0) + (now - t["t0"]) * 1e3
                t["t0"] = now
            league.agent._timer = timer
            league.agent._timer_reset = lambda: t.__setitem__("t0", self._prof.t())

    @property
    def league(self) -> League | None:
        return self._runtime.league if self._runtime is not None else None

    def _ensure_started(self) -> None:
        if self._procs:
            return
        import multiprocessing as mp

        if self.league_idx:
            # first opponents (every member once, then draws); a char-locked
            # import's character goes into the cold-boot spec
            locks = self.league.boot(self.league_idx)
            for i, lock in locks.items():
                if lock is not None:
                    self.specs[i].opponent_char = lock
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

    def _encode_flats(self, games: list) -> tuple:
        """games: per-env (bools, ints, floats) flat vectors of the ALREADY
        encoded frame (encode.flatten_typed, env-side). Three stacks, three
        host->GPU copies; view construction stays at the flat level (see
        _flat_view) so perspective swaps, seat mixes and row gathers are a
        handful of whole-tensor kernels instead of ~120 per-leaf launches."""
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
            self._swap_perm = {
                k: (None if p is None else torch.from_numpy(p).to(device))
                for k, p in encode.swap_perm(
                    self._game_template, self._game_layout
                ).items()
            }
        return tuple(
            torch.from_numpy(np.stack([g[k] for g in games])).to(device, non_blocking=True)
            for k in range(3)
        )

    _KINDS = ("bool", "int", "float")

    def _swap_flats(self, flats: tuple) -> tuple:
        """p0 <-> p1 perspective swap as a column permutation (bit-exact)."""
        return tuple(
            t if self._swap_perm[k] is None else t.index_select(-1, self._swap_perm[k])
            for k, t in zip(self._KINDS, flats)
        )

    def _flat_view(self, flats: tuple, rows=None, lead=None) -> tp.Any:
        """Struct view of the flat tensors: optional row gather (index
        tensor) and leading reshape, then one unflatten (views)."""
        from smashbot import encode

        if rows is not None:
            flats = tuple(t.index_select(0, rows) for t in flats)
        if lead is not None:
            flats = tuple(t.view(*lead, t.shape[-1]) for t in flats)
        return encode.unflatten_typed_torch(
            self._game_template, self._game_layout, *flats
        )

    def _reencode_record(self, rec: FrameRecord, embed=None) -> FrameRecord:
        """Opponent-seat record -> student schema: actions re-encoded through
        the student's controller embedding, name set to the student's
        code. `embed` = the opponent's embedding (default: reference).
        Only ever called for the ref_envs group (league harvests go
        through _HarvestGroup.step)."""
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
        return self.member_kind(self.league.member_now.get(i))

    def _serve_submit(self, name: str, fn):
        """Queue one opponent-grid forward on a dedicated CUDA stream in a
        worker thread. The stream first waits on the default stream (this
        frame's encoded views are produced there, and the wait is recorded
        before the caller enqueues its own student forward, so the grids
        never wait on the student); the caller makes the default stream
        wait on every serving stream before consuming the outputs. Those
        two ordering edges also make the caching allocator's cross-stream
        memory reuse safe: freed memory is only handed to work enqueued
        after the corresponding wait."""
        if self._serve_pool is None:
            import concurrent.futures

            self._serve_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="serve"
            )
        stream = self._serve_streams.get(name)
        if stream is None:
            stream = self._serve_streams[name] = torch.cuda.Stream()
        prof = self._prof

        def run():
            import time as _time

            stream.wait_stream(torch.cuda.default_stream())
            t0 = _time.perf_counter() if prof else None
            with torch.cuda.stream(stream):
                fn()
            if prof is not None:
                # wall time without a device sync (a sync here would
                # serialize the streams); the .cpu() inside fn makes this
                # track the real latency closely
                key = f"{name}_wall"
                prof.acc[key] = prof.acc.get(key, 0.0) + (
                    _time.perf_counter() - t0
                ) * 1e3

        return self._serve_pool.submit(run)

    def _seat_tables(self):
        """Row -> env maps for the grid cells and Phillip's rows (None =
        idle), from the current seating."""
        seats = self.league.seats
        grid = seats.S
        cells: list[int | None] = []
        for s in range(grid):
            cells += seats.env_of_rows(s)
        phillip = seats.env_of_rows(grid) if len(seats.pools) > grid else []
        return cells, phillip

    def collect(self, num_trajectories: int) -> list[Trajectory]:
        """Run the sync-barrier loop until N PPO trajectory chunks are
        assembled; any imitation chunks harvested along the way (opponent
        seats with whitelisted chars) are appended after them.

        Per frame: gather -> bookkeeping (results, seats) -> EXECUTE (pop
        every agent's delay queue) -> SEND -> infer (encode + every forward,
        appending to the queues) -> assemble/harvest. Sending before
        inferring lets the Dolphins step the next frame while the GPU works:
        the controller executed now was sampled `delay` frames ago, so it is
        already in the queue (see BatchedPolicyAgent.execute)."""
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
        league = self.league
        rt = self._runtime
        grid = rt.agent if rt is not None else None
        ph = rt.phillip if rt is not None else None
        imp = rt.imports_agent if rt is not None else None
        imp_rows = self.import_idx
        imp_rows_t = (
            torch.tensor(imp_rows, dtype=torch.int64,
                         device=self.student.device)
            if imp is not None and imp_rows else None
        )

        prof = self._prof  # opt-in per-phase timing (SMASHBOT_PROFILE=1)
        while len(out) < num_trajectories:
            t0 = prof.t() if prof else None
            payloads = self._gather_all()
            prof and prof.lap("gather", t0)
            t0 = prof.t() if prof else None
            # envs whose opponent seat is engine-AI-driven THIS frame
            # (reported by the env itself, never the desired assignment)
            cpu_now = {
                i for i, p in enumerate(payloads)
                if p.get("opp_serving") == "cpu"
            }
            for i, p in enumerate(payloads):
                if p.get("final_stocks") is None:
                    continue
                a, b = p["final_stocks"]  # (port1, port2)
                sp = self.specs[i]
                if sp.kind == "self":
                    # both seats are the student: track the PORT-1 seat's
                    # win rate (a ~50% health metric, not a skill signal)
                    self.trackers["self"].add_game((a, b), p.get("opp_char"))
                    continue
                if sp.student_port == 2:
                    a, b = b, a
                # attribute to the kind that PLAYED the ended game
                # (result_serving: carried alongside the result so a
                # recycle-boundary kind flip can't misattribute it)
                kind = self._actual_kind(i, p.get("result_serving"))
                # Char-LOCKED members are excluded from by_char: a locked
                # member ties its character's column to its own strength.
                mem = (
                    self.league.member_now.get(i)
                    if self.league is not None else None
                )
                locked = (
                    mem is not None
                    and self.league.lock_of(mem) is not None
                ) or sp.char_lock is not None
                self.trackers[
                    self._TRACKER_KIND.get(kind, kind)
                ].add_game((a, b), None if locked else p.get("opp_char"))
                if (sp.kind == "reference" and self.league is not None
                        and self.league.on_result is not None):
                    # dedicated Phillip (ref_envs mode): keep his ledger row
                    # alive so R:/rl/phillip metrics survive leaving the draw
                    if a != b:
                        self.league.on_result("phillip", a > b)
                if (sp.kind == "import" and self.league is not None
                        and self.league.on_result is not None):
                    # dedicated envs are outside the draw, but their games
                    # feed the SAME payoff ledger so rl/imports/* metrics
                    # and the ticker read identically to the league era
                    if a != b:
                        self.league.on_result(sp.member, a > b)
                if sp.kind == "snapshot":
                    # credit the ended game, take a seat for the drawn
                    # member, draw the one after
                    league.on_boundary(
                        i, p.get("opp_serving"), p.get("opp_char"),
                        (a > b) if a != b else None,
                    )
            # fresh envs resolve to seats AFTER all boundaries/compactions
            # have applied, so the coordinates cannot be stale
            fresh: set = set()
            if league is not None:
                fresh, league.fresh_envs = set(league.fresh_envs), []
            resets_d = torch.tensor([p["resetting"] for p in payloads])
            resets = resets_d[row_dolphin]  # row-level
            resets_cpu = resets_d.tolist()
            reset_rows = resets.nonzero().flatten().tolist()

            # ---- EXECUTE: pop this frame's controllers from every queue
            controllers1 = self.student.execute(reset_rows)
            opp_controllers: dict[int, tp.Any] = {}
            cells, ph_rows = self._seat_tables() if self.league_idx else ([], [])
            if self.league_idx:
                cell_env, cell_reset = self._route(cells, resets_cpu, fresh)
                for e_ in fresh:
                    seat_ = league.seats.seat_of(e_)
                    if seat_ is not None and seat_[0] < grid.S:
                        grid.reset_cell(*seat_)
                rows = grid.execute()
                for r, env in enumerate(cells):
                    if env is not None:
                        opp_controllers[env] = rows[r]
                if ph is not None:
                    ph_env, ph_reset = self._route(ph_rows, resets_cpu, fresh)
                    for e_ in fresh:
                        seat_ = league.seats.seat_of(e_)
                        if seat_ is not None and seat_[0] == grid.S:
                            ph.reset_cell(0, seat_[1])
                    ctrls = ph.execute()
                    for r, env in enumerate(ph_rows):
                        if env is not None:
                            opp_controllers[env] = ctrls[r]
            if imp is not None and imp_rows:
                imp_reset = torch.tensor(
                    [resets_cpu[e] for e in imp_rows], dtype=torch.bool,
                    device=self.student.device,
                )
                for r, e_ in enumerate(imp_rows):
                    if resets_cpu[e_]:
                        imp.reset_cell(r // imp.N, r % imp.N)
                ictrls = imp.execute()
                for r, env in enumerate(imp_rows):
                    opp_controllers[env] = ictrls[r]
            group_live = {}
            for name, idx in self.groups.items():
                if cpu_now and all(i in cpu_now for i in idx):
                    continue  # whole group engine-driven: no brain to run
                group_live[name] = idx
                ctrls = self.opponents[name].execute(
                    [j for j, i in enumerate(idx) if resets_cpu[i]]
                )
                for j, env_i in enumerate(idx):
                    opp_controllers[env_i] = ctrls[j]
            prof and prof.lap("bookkeeping+execute", t0)
            t0 = prof.t() if prof else None

            # ---- SEND: the envs step the next frame while we infer below
            for i, conn in enumerate(self._conns):
                port = self.specs[i].student_port
                cmd = {port: controllers1[i]}
                if i in self._self_row_of:
                    cmd[3 - port] = controllers1[self._self_row_of[i]]
                elif i in opp_controllers and i not in cpu_now:
                    # cpu_now rows have no policy seat: the engine AI drives
                    # the opponent port (like dedicated cpu envs today)
                    cmd[3 - port] = opp_controllers[i]
                if self.specs[i].kind == "snapshot":
                    # what the opponent seat should be for the env's NEXT
                    # game (kind + char lock), drawn one game ahead
                    cmd["opp_next"] = league.next_command(i)
                conn.send(cmd)
            prof and prof.lap("send", t0)
            t0 = prof.t() if prof else None

            # ---- rewards / trackers (this frame's transition)
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
                    # the opponent seat's reward is the zero-sum mirror of
                    # the student seat's (both terms are antisymmetric)
                    if self.ref_idx:
                        ref_rows = torch.tensor(self.ref_idx, device=device)
                        self._imit_assembler.push_reward(-reward[ref_rows])
                    for key, rows_ in (
                        ("ours", cells), ("phillip", ph_rows),
                        ("imports", imp_rows),
                    ):
                        g = self._harvest_groups.get(key)
                        if g is not None:
                            g.push_reward(self._rows_of(-reward, rows_))
                # stock events are rare: find them with tensor ops and only
                # loop over the hits (a per-dolphin Python loop with element
                # indexing cost ~10 ms/frame at 176 Dolphins)
                D = self.num_dolphins
                live = ~resets[:D]
                lost = (stocks[:D] < self._prev_stocks[:D]) & live[:, None]
                if bool(lost.any()):
                    prev_pct = self._prev_percent[:D]
                    for i, seat in lost.nonzero().tolist():
                        kind = self._actual_kind(
                            i, payloads[i].get("opp_serving")
                        )
                        tracker = self.trackers[self._TRACKER_KIND.get(kind, kind)]
                        if seat == 0:
                            tracker.add_death(float(prev_pct[i, 0]))
                        else:
                            tracker.add_kill(float(prev_pct[i, 1]))
            self._prev_stocks, self._prev_percent = stocks, percent

            # ---- INFER: encode, every forward, append to the queues
            games = [p["game"] for p in payloads]
            flats = self._encode_flats(games)
            prof and prof.lap("rewards+encode", t0)
            # perspective swap commutes with encoding: the parser fixes
            # p0=port1; each agent must see ITSELF as p0, so seat-2 envs
            # get the swapped columns. All view construction happens on the
            # THREE flat tensors (a few whole-tensor kernels), structs are
            # built as views at the end (_flat_view).
            swapped = self._swap_flats(flats)
            seat2 = self.seat2.to(device)[:, None]
            opp_flats = tuple(
                torch.where(seat2, a, b) for a, b in zip(flats, swapped)
            )
            # learner rows: primary seats of every dolphin + the second seat
            # of each self-play dolphin, all served by ONE student forward
            rows_dev = row_dolphin.to(device)
            row_seat2 = self.row_seat2.to(device)[:, None]
            student_view = self._flat_view(tuple(
                torch.where(row_seat2, a.index_select(0, rows_dev), b.index_select(0, rows_dev))
                for a, b in zip(swapped, flats)
            ))
            resets_dev = resets.to(device)
            pending_resets.append(resets_dev)

            # first frames of a boot stay serial: torch.compile and the
            # manual CUDA-graph captures must run single-threaded
            use_par = (
                self._serve_parallel and self._serve_frames >= 40
                and (bool(self.league_idx) or (imp is not None and bool(imp_rows)))
            )
            self._serve_frames += 1
            ref_records: list[FrameRecord] = []
            harvest: dict[str, tuple] = {}  # key -> (rows, resets, records)
            if not use_par:
                t0 = prof.t() if prof else None
                records, hidden_before = self.student.infer(
                    student_view, resets_dev,
                    # the snapshot is only consumed at a chunk boundary
                    want_snapshot=(records_pushed % cfg.unroll_length == 0),
                )
                prof and prof.lap("student_infer", t0)
                t0 = prof.t() if prof else None

                if self.league_idx:
                    # ---- the grid: gather every cell's env view, one forward
                    gv = self._flat_view(opp_flats, rows=cell_env, lead=(grid.S, grid.N))
                    prof and prof.lap("grid_gather", t0)
                    t0 = prof.t() if prof else None
                    prof and getattr(grid, "_timer_reset", lambda: None)()
                    record = grid.infer(gv, cell_reset.view(grid.S, grid.N))
                    harvest["ours"] = (cells, cell_reset, [record])
                    prof and prof.lap("grid_infer", t0)
                    t0 = prof.t() if prof else None
                    if ph is not None:
                        pv = self._flat_view(opp_flats, rows=ph_env, lead=(1, ph.N))
                        rec = ph.infer(pv, ph_reset.view(1, ph.N))
                        harvest["phillip"] = (ph_rows, ph_reset, [rec])
                    prof and prof.lap("phillip_infer", t0)
                if imp is not None and imp_rows:
                    t0 = prof.t() if prof else None
                    iv = self._flat_view(
                        opp_flats, rows=imp_rows_t, lead=(imp.S, imp.N)
                    )
                    irec = imp.infer(iv, imp_reset.view(imp.S, imp.N))
                    harvest["imports"] = (imp_rows, imp_reset, [irec])
                    prof and prof.lap("imports_infer", t0)
                t0 = prof.t() if prof else None
                for name, idx in group_live.items():
                    group_view = self._flat_view(
                        tuple(t[idx[0]:idx[-1] + 1] for t in opp_flats)
                    )
                    g_records, _ = self.opponents[name].infer(
                        group_view, resets_dev[idx[0]:idx[-1] + 1], want_snapshot=False,
                    )
                    if name == "reference" and self.harvest_imitation:
                        ref_records = g_records
                prof and prof.lap("groups_infer", t0)
            else:
                # ---- stream-parallel serving: the grid and imports
                # forwards (manual CUDA graphs, replayable on any stream)
                # run on their own streams from worker threads while this
                # thread serves student+groups (torch.compile'd -> default
                # stream). The serving streams wait on the default stream
                # (views built there); the default stream waits on them
                # before the outputs are consumed below.
                t0 = prof.t() if prof else None
                futs = []
                if self.league_idx:
                    def _grid_task():
                        gv = self._flat_view(
                            opp_flats, rows=cell_env, lead=(grid.S, grid.N)
                        )
                        record = grid.infer(gv, cell_reset.view(grid.S, grid.N))
                        harvest["ours"] = (cells, cell_reset, [record])
                        if ph is not None:
                            pv = self._flat_view(
                                opp_flats, rows=ph_env, lead=(1, ph.N)
                            )
                            rec = ph.infer(pv, ph_reset.view(1, ph.N))
                            harvest["phillip"] = (ph_rows, ph_reset, [rec])
                    futs.append(self._serve_submit("grid", _grid_task))
                if imp is not None and imp_rows:
                    def _imports_task():
                        iv = self._flat_view(
                            opp_flats, rows=imp_rows_t, lead=(imp.S, imp.N)
                        )
                        irec = imp.infer(iv, imp_reset.view(imp.S, imp.N))
                        harvest["imports"] = (imp_rows, imp_reset, [irec])
                    futs.append(self._serve_submit("imports", _imports_task))
                records, hidden_before = self.student.infer(
                    student_view, resets_dev,
                    want_snapshot=(records_pushed % cfg.unroll_length == 0),
                )
                for name, idx in group_live.items():
                    group_view = self._flat_view(
                        tuple(t[idx[0]:idx[-1] + 1] for t in opp_flats)
                    )
                    g_records, _ = self.opponents[name].infer(
                        group_view, resets_dev[idx[0]:idx[-1] + 1],
                        want_snapshot=False,
                    )
                    if name == "reference" and self.harvest_imitation:
                        ref_records = g_records
                for f in futs:
                    f.result()  # re-raises worker-thread errors
                if self._serve_streams:
                    cur = torch.cuda.current_stream()
                    for s in self._serve_streams.values():
                        cur.wait_stream(s)
                prof and prof.lap("serve(par)", t0)
            t0 = prof.t() if prof else None

            # ---- assemble / harvest
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
                for key, (rows_, rst, recs) in harvest.items():
                    g = self._harvest_groups.get(key)
                    if g is not None:
                        elig = torch.tensor([
                            env is not None
                            and payloads[env].get("opp_char") in self._whitelist
                            for env in rows_
                        ], dtype=torch.bool)
                        g.step(rst.cpu(), elig, recs, imit_out)

            prof and prof.lap("assemble+harvest", t0)
            self._frame_count += 1
            prof and prof.frame(self._frame_count)
            if self.assembler.ready():
                out.append(self.assembler.emit())
        self._records_pushed = records_pushed
        return out + imit_out

    def _route(self, rows, resets_cpu, fresh):
        """Per-row env index (idle rows read env 0: harmless, always reset)
        and reset flags for one seat pool: an env's own reset, a fresh seat
        (new game in this row), or idle. `fresh` is a set of ENVS, so this
        is pool-agnostic — the old (pool, row) form silently matched only
        one pool and contributed nothing on every other slice."""
        device = self.student.device
        env_idx = torch.tensor(
            [0 if e is None else e for e in rows], dtype=torch.int64, device=device
        )
        reset = torch.tensor([
            e is None or resets_cpu[e] or e in fresh
            for e in rows
        ], dtype=torch.bool, device=device)
        return env_idx, reset

    def _rows_of(self, per_env: torch.Tensor, rows) -> torch.Tensor:
        """Gather a per-learner-row vector onto seat rows (0 when idle)."""
        idx = torch.tensor(
            [0 if e is None else e for e in rows], dtype=torch.int64,
            device=per_env.device,
        )
        keep = torch.tensor([e is not None for e in rows], device=per_env.device)
        return torch.where(keep, per_env.index_select(0, idx), torch.zeros_like(idx, dtype=per_env.dtype))

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
