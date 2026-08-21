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
    # Redraw the opponent character at each Dolphin recycle. Flag-gated:
    # its first production firing correlated with the step-706 NaN crash
    # (under investigation); off = the proven fixed-char recycle path.
    redraw_chars: bool = True
    # Boot each Dolphin's replacement in the background during its final game
    # (recycle hot-swap); the spare parks at intro menus until swapped in.
    # OFF by default: worth only ~5% at pool-era fps, and its async teardown
    # caused the 2026-08-12 overnight hang (port race on the misselect-retry
    # path, since fixed via _drain_old_stops). Opt in for short/tended runs;
    # re-earn trust before any week-long run relies on it.
    double_buffer: bool = False
    # (historical: ref_shard_size tuned the retired in-worker TF bridge;
    # kept so old commands don't break. The worker ignores it now.)
    ref_shard_size: int = 16
    # Watchdog: max seconds the barrier waits on one env's payload before
    # crashing loudly (a supervisor/--runtime.restore auto turns that crash
    # into a ~20min self-heal instead of a silent overnight hang).
    env_timeout: float = 300.0
    # Slippi replay recording (.slp per game; headless included — run fast,
    # watch later in Slippi at 60fps). Empty replay_dir = Slippi default.
    save_replays: bool = False
    replay_dir: str = ""
    # Reference opponent (slippi-ai medium-v2 via venv-ref subprocess).
    ref_envs: int = 0
    ref_ckpt: str = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
    # Student character whitelist: the student seat's character is drawn
    # per-game uniformly from this list (len==1 = exactly the fixed-char
    # behavior; the default ["FOX"] defers to the legacy bot_char flag —
    # see pool.student_whitelist). It also gates second-seat harvesting:
    # an opponent seat is harvested for imitation only while its current
    # character is whitelisted.
    char_whitelist: list[str] = dataclasses.field(
        default_factory=lambda: ["FOX"]
    )
    # Self-play envs: BOTH seats driven by the current student policy (one
    # batched forward — no second policy copy). Each contributes 2 on-policy
    # PPO trajectories, so it costs 2 units of the num_envs trajectory
    # budget while booting ONE Dolphin: dolphins = num_envs - self_envs.
    # 0 = dormant (today's behavior).
    self_envs: int = 0
    # PFSP opponent prioritization (AlphaStar f_hard) for snapshot slot
    # assignment; False = the original recency-biased sampling. Selection
    # only — zero effect on losses or memory.
    pfsp: bool = True
    pfsp_p: float = 2.0  # squared f_hard (AlphaStar mains): see pool.py
    pfsp_hard_frac: float = 1.0  # f_hard share of weighted draws: pool.py
    pfsp_explore: float = 0.0  # uniform probe fraction: see pool.py
    # League membership for the teacher / CPU lvl-9 / Phillip (dormant by
    # default): instead of fixed teacher_envs/cpu_envs/ref_envs partitions,
    # the member joins the PFSP class-weighted candidate set and competes
    # for non-latest snapshot slots — it serves only while the payoff table
    # says it's worth serving. Requires that kind's fixed env count be 0
    # (move the envs into snapshot slots) and pfsp=True. Phillip serves by
    # ROUTING to his own agent (different architecture — never loaded into
    # a slot policy). See league_members().
    league_teacher: bool = False
    league_cpu: bool = False
    league_phillip: bool = False
    # Imported league members (dormant by default): frozen checkpoints from
    # a PREVIOUS run join the league as PERMANENT members — RL-strong
    # opponents that pressure the new student and serve as a live
    # cross-generation benchmark (the payoff row vs an import = "are we
    # beating the old model yet"). Entries are "NAME=/path/to/state_dict.pt"
    # (bare policy state_dict, snapshot-pool format, same architecture as
    # the student — loaded exactly like a ghost slot load), optionally with
    # a per-import character lock "NAME=PATH@CHAR" (default lock: FOX —
    # these are trained-fox opponents). While a slot serves an import, its
    # envs pin the locked character instead of redrawing per game.
    # Requires pfsp=True and snapshot_slots > 0.
    league_imports: list[str] = dataclasses.field(default_factory=list)

    def import_members(self) -> dict[str, tuple[str, str]]:
        """Parsed league_imports: {NAME: (path, char_lock)}. Bad entries
        fail loudly (a silently dropped import would serve nothing and skew
        the auction)."""
        out: dict[str, tuple[str, str]] = {}
        for entry in self.league_imports:
            name, eq, rest = entry.partition("=")
            assert eq and name and rest, (
                f"bad league_imports entry {entry!r}: want "
                f"NAME=/path/to/state_dict.pt or NAME=PATH@CHAR"
            )
            assert all(c.isalnum() or c in "-_." for c in name), (
                f"bad league_imports name {name!r}: names key payoff rows "
                f"and wandb metrics — alphanumeric/-/_/. only"
            )
            before, at, after = rest.rpartition("@")
            if at:
                path, char = before, after.upper()
            else:
                path, char = rest, "FOX"
            assert path, f"bad league_imports entry {entry!r}: empty path"
            from smashbot.rl.pool import MAIN_12

            assert char in MAIN_12, (
                f"league import {name!r}: char lock {char!r} not in the "
                f"policy-opponent roster {MAIN_12}"
            )
            assert name not in out, (
                f"duplicate league_imports name {name!r}"
            )
            out[name] = (path, char)
        return out

    def league_members(self) -> list[str]:
        """Special league member keys enabled by the flags; validates the
        config (loud asserts — a silently ignored flag would strand envs)."""
        members = []
        if self.league_teacher:
            assert self.teacher_envs == 0, (
                f"league_teacher folds the teacher into the PFSP league — "
                f"set teacher_envs=0 (got {self.teacher_envs}) and move "
                f"those envs into the snapshot slots"
            )
            members.append("teacher")
        if self.league_cpu:
            assert self.cpu_envs == 0, (
                f"league_cpu folds the lvl-9 CPU into the PFSP league — "
                f"set cpu_envs=0 (got {self.cpu_envs}) and move those envs "
                f"into the snapshot slots"
            )
            members.append("cpu")
        if self.league_phillip:
            assert self.ref_envs == 0, (
                f"league_phillip folds Phillip into the PFSP league — set "
                f"ref_envs=0 (got {self.ref_envs}) and move those envs into "
                f"the snapshot slots"
            )
            members.append("phillip")
        imports = self.import_members()
        if imports:
            assert self.snapshot_slots > 0, (
                f"league_imports serve through snapshot slots — set "
                f"snapshot_slots > 0 (got {self.snapshot_slots})"
            )
            members += [f"import:{name}" for name in imports]
        if members:
            assert self.pfsp, (
                "league_teacher/league_cpu require pfsp=True: league members "
                "earn/lose serving time through the payoff table, which the "
                "recency sampler never consults"
            )
        return members


def next_opponent_char(
    cur_kind: str,
    char_lock: str | None,
    redraw_chars: bool,
    armed_char: str | None,
    draw: tp.Callable[[], str],
) -> str | None:
    """Opponent-seat character for the NEXT game (game boundary or Dolphin
    recycle). Returns the character to arm, or None to keep the sitting one.

    Char-locked league members (imports) pin their lock: while locked, NO
    rng draw is consumed (the redraw stream simply pauses) and the seat pins
    the lock; when the lock clears, normal redraws resume. CPU serving
    ignores the lock — locks belong to policy imports only, and an env that
    lazily adopted cpu must draw from the cpu roster. Default path (lock
    None) is byte-identical to the pre-import redraw behavior."""
    if char_lock is not None and cur_kind != "cpu":
        return char_lock if armed_char != char_lock else None
    if redraw_chars:
        return draw()
    return None


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

    import faulthandler
    import signal as _sig

    # On SIGUSR1 (sent by the worker watchdog before it gives up), dump this
    # process's exact python stack to the env log — no more guessing where a
    # silent env is stuck.
    faulthandler.register(_sig.SIGUSR1, file=_log, all_threads=True)

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


    import random as random_lib

    from smashbot.rl.pool import (
        CPU_CHARS, OFF_ROSTER, OPPONENT_CHARS, student_whitelist,
    )

    opp_port = 3 - spec.student_port
    # Per-env deterministic RNG for opponent-character redraws at each
    # Dolphin recycle: matchups rotate over the run instead of being frozen
    # by the boot-time draw (the first Dolphin still uses the partition's
    # stratified char, preserving guaranteed full-roster coverage at boot).
    # Per-env replay subdir: parallel Dolphins start games in the same
    # second, and Slippi names files Game_<timestamp>.slp — a shared dir
    # would collide/overwrite across envs.
    _replay_dir = f"{cfg.replay_dir}/env-{idx}" if cfg.replay_dir else ""

    char_rng = random_lib.Random((cfg.partition_seed << 16) ^ idx)
    # Student-seat whitelist draws use their OWN rng stream so a multi-char
    # whitelist never perturbs the opponent redraw sequence (and vice versa).
    whitelist = student_whitelist(cfg.char_whitelist, cfg.bot_char)
    student_rng = random_lib.Random(((cfg.partition_seed + 0x51EC7) << 16) ^ idx)

    # Kind ACTUALLY serving on the opponent seat. Normally fixed (= spec.kind
    # for the whole run); under league_cpu a snapshot-slot env can be asked
    # (via the "opp_kind" command key) to flip policy<->cpu — Dolphin players
    # are only built at (re)boot and a CPU port cannot hot-swap mid-game, so
    # the flip is adopted LAZILY at the next recycle boundary. Until then the
    # env keeps serving its previous kind and reports what it serves, so
    # attribution follows reality, not the desired assignment.
    cur_kind = spec.kind
    desired_kind = None  # latest "policy"/"cpu" wish from the worker
    # Character lock piggybacked by the worker while this env's slot serves
    # a char-locked import ("opp_char_lock" command key; None = unlocked).
    # While locked, the opponent seat pins this char instead of redrawing
    # per game — see next_opponent_char. Never sent outside league_imports.
    char_lock = None

    def _draw_char() -> str:
        if cur_kind == "cpu":
            # CPU opponents draw 60/40 main12/off-roster; CPU-Sheik is
            # impossible (engine ignores the transform on CPU ports), hence
            # CPU_CHARS
            pool = (CPU_CHARS if char_rng.random() < cfg.main12_prob
                    else OFF_ROSTER)
            return char_rng.choice(pool)
        if cur_kind == "self":  # second student seat: whitelist only
            return char_rng.choice(whitelist)
        return char_rng.choice(OPPONENT_CHARS)

    def _draw_student_char() -> str:
        # len==1: exactly the fixed-character behavior, zero rng draws.
        if len(whitelist) == 1:
            return whitelist[0]
        return student_rng.choice(whitelist)

    def _build_players(opp_char: str, student_char: str) -> dict:
        if cur_kind == "cpu":
            opponent_player = dolphin_lib.CPU(
                character=melee.Character[opp_char.upper()],
                level=spec.cpu_level,
            )
        else:
            opponent_player = dolphin_lib.AI(
                character=melee.Character[opp_char.upper()]
            )
        return {
            spec.student_port: dolphin_lib.AI(
                character=melee.Character[student_char.upper()]
            ),
            opp_port: opponent_player,
        }

    players = _build_players(spec.opponent_char, _draw_student_char())
    # Character actually playing the CURRENT game on the opponent seat.
    # Per-game redraws mutate `players` BETWEEN games and take effect at the
    # NEXT rematch CSS pass, so the char is "armed" one boundary before it
    # plays: cur <- armed at each boundary, then armed <- fresh draw.
    cur_opp_char = spec.opponent_char
    armed_opp_char = spec.opponent_char
    # Carried across Dolphin recycles: the new instance's first frame must
    # still announce the game boundary (fresh recurrent state, zeroed reward)
    # and deliver the final game's result — otherwise two different games
    # would silently splice into one stream.
    pending_reset = False
    pending_result = None
    # serving label ("policy"/"cpu") of the game that produced
    # pending_result: a result carried across a recycle must attribute to
    # the kind that PLAYED it, even if the recycle just adopted a new kind
    pending_result_kind = None
    consecutive_misselects = 0

    # Double buffering: during a Dolphin's LAST game, boot its replacement in
    # a background thread. The spare is only CONSTRUCTED (process up, ISO
    # loaded, idle at intro menus — menus don't advance without inputs, so
    # nothing progresses unattended); menu navigation still happens at swap,
    # via the normal iter_gamestates path with its misselect guard. Hides the
    # 10-15s boot that otherwise stalls the whole worker barrier per recycle.
    import signal
    import threading

    spare = {"dolphin": None, "thread": None}
    old_stops: list = []  # (thread, dolphin_pid) pairs

    def _dolphin_pid(d) -> int | None:
        try:
            return d.console._process.pid
        except AttributeError:
            return None

    def _boot_spare() -> None:
        try:
            spare["dolphin"] = game_lib.make_dolphin(
                players, headless=cfg.headless, stage=cfg.stage,
                save_replays=cfg.save_replays, replay_dir=_replay_dir,
            )
        except Exception as e:  # fall back to a cold boot at swap time
            print(f"spare boot failed (cold boot at swap): {e}", flush=True)
            spare["dolphin"] = None

    def _start_spare() -> None:
        if cfg.double_buffer and spare["thread"] is None:
            spare["thread"] = threading.Thread(target=_boot_spare, daemon=True)
            spare["thread"].start()

    def _take_spare():
        if spare["thread"] is None:
            return None
        spare["thread"].join(timeout=120)
        if spare["thread"].is_alive():
            # wedged spare boot: abandon it (daemon thread) and cold-boot
            print("WARNING: spare boot wedged; abandoning it", flush=True)
            spare["thread"] = None
            spare["dolphin"] = None
            return None
        d, spare["dolphin"], spare["thread"] = spare["dolphin"], None, None
        if d is not None:
            print("recycle: swapped to pre-booted spare", flush=True)
        return d

    def _drain_old_stops(timeout: float = 20.0) -> None:
        """Cold boots must not overlap a dying Dolphin: the previous instance
        can still hold ports (seen live: retry Dolphin failed its spectator-
        server bind and wedged mid-boot, hanging the whole worker barrier).
        Wait for pending stops; SIGKILL any Dolphin whose stop() is stuck."""
        for t, pid in old_stops:
            t.join(timeout)
            if t.is_alive() and pid is not None:
                print(f"stop() stuck; SIGKILL dolphin pid {pid}", flush=True)
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                t.join(timeout=5)
        old_stops[:] = [(t, p) for t, p in old_stops if t.is_alive()]

    class AlarmTimeout(Exception):
        pass

    def _alarm_handler(*_):
        raise AlarmTimeout()

    signal.signal(signal.SIGALRM, _alarm_handler)

    def _cold_boot():
        """Boot with a hard deadline: a Dolphin that wedges during startup
        (port collision, dead handshake) must become a bounded retry, not an
        infinite hang. SIGALRM is safe here: env-process main thread."""
        _drain_old_stops()
        signal.alarm(180)
        try:
            return game_lib.make_dolphin(
                players, headless=cfg.headless, stage=cfg.stage,
                save_replays=cfg.save_replays, replay_dir=_replay_dir,
            )
        finally:
            signal.alarm(0)

    consecutive_boot_failures = 0
    consecutive_wedges = 0
    first_boot = True
    try:
        while True:
            # Recycle boundary = the ONLY place a desired policy<->cpu flip
            # is adopted: players are built fresh right below, exactly like a
            # cold boot. A kind change forces a redraw even with redraw_chars
            # off — the sitting char may be illegal for the new kind
            # (CPU-Sheik) and the player object type (AI vs CPU) changes.
            kind_changed = False
            if desired_kind is not None:
                want = "cpu" if desired_kind == "cpu" else spec.kind
                if want != cur_kind:
                    cur_kind = want
                    kind_changed = True
                    print(f"recycle: opponent kind adopted -> {cur_kind}",
                          flush=True)
            if not first_boot and (cfg.redraw_chars or kind_changed):
                # armed_char=None forces the lock through even when already
                # armed (players are rebuilt from scratch below)
                new_char = next_opponent_char(
                    cur_kind, char_lock, True, None, _draw_char
                )
                players.clear()
                players.update(_build_players(new_char, _draw_student_char()))
                # fresh Dolphin: its first CSS pass uses the new draw directly
                cur_opp_char = armed_opp_char = new_char
                print(f"recycle: opponent redrawn -> {new_char}", flush=True)
            first_boot = False
            try:
                dolphin = _take_spare() or _cold_boot()
                consecutive_boot_failures = 0
            except (AlarmTimeout, dolphin_lib.ConnectFailed) as e:
                # transient boot flakes (slow boot, console connect refusal
                # during a 128-wide boot storm) are retriable, not fatal
                consecutive_boot_failures += 1
                print(f"BOOT FAILURE ({consecutive_boot_failures}/3): {e}",
                      flush=True)
                if consecutive_boot_failures >= 3:
                    raise
                continue
            parser = Parser(ports=[1, 2])
            games = 0
            last_frame = None
            last_stocks = None
            # New-game gate: a pre-booted spare idles at the title screen,
            # where Melee's ATTRACT-MODE DEMO auto-plays after a timeout —
            # demo frames are "in-game" frames with garbage ports/fields
            # (live-caught: NaN states -> multinomial assert at the first
            # 128-env swap wave). Drop frames until a REAL game start
            # (frame counter resets to INITIAL_FRAME=-123, i.e. < 0).
            # Cold boots' first frame IS -123, so this is a no-op for them.
            game_started = False
            try:
              try:
                # Dolphin can freeze silently mid-game (live-caught: env 17,
                # zero log output, console.step never returned). Guard every
                # frame fetch with an alarm: legit silent stretches (boot +
                # menus + rematch) stay under ~60s, so 120s = wedged. On
                # trip: SIGKILL this Dolphin, mark the game aborted (reset,
                # no result), and boot a fresh one. Bounded: 3 wedges with
                # no completed game in between = something systemic, die
                # loudly (the worker watchdog is the outer backstop).
                gs_iter = iter(dolphin.iter_gamestates(skip_menu_frames=True))
                while True:
                    signal.alarm(120)
                    try:
                        gs = next(gs_iter)
                    except AlarmTimeout:
                        consecutive_wedges += 1
                        print(f"DOLPHIN WEDGED mid-stream "
                              f"({consecutive_wedges}/3); killing it",
                              flush=True)
                        if consecutive_wedges >= 3:
                            raise RuntimeError(
                                "dolphin wedged 3x without a completed game"
                            )
                        pid = _dolphin_pid(dolphin)
                        if pid is not None:
                            # forensics before the kill: what was Dolphin
                            # stuck on? (kernel wait-channel per thread)
                            try:
                                for tid in os.listdir(f"/proc/{pid}/task"):
                                    base = f"/proc/{pid}/task/{tid}"
                                    with open(f"{base}/wchan") as fh:
                                        wchan = fh.read().strip()
                                    with open(f"{base}/stat") as fh:
                                        state = fh.read().split()[2]
                                    print(f"  wedged tid {tid}: "
                                          f"state={state} wchan={wchan}",
                                          flush=True)
                            except OSError:
                                pass
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except OSError:
                                pass
                        pending_reset, pending_result = True, None
                        pending_result_kind = None
                        break
                    except StopIteration:
                        break
                    finally:
                        signal.alarm(0)
                    if not game_started:
                        if gs.frame > 0:
                            continue  # attract-mode demo frame: discard
                        game_started = True
                    serving = "cpu" if cur_kind == "cpu" else "policy"
                    boundary = last_frame is not None and gs.frame < last_frame
                    resetting = boundary or pending_reset
                    result = pending_result if pending_reset else None
                    result_kind = pending_result_kind if pending_reset else None
                    pending_reset, pending_result = False, None
                    pending_result_kind = None
                    if boundary:
                        games += 1
                        consecutive_wedges = 0
                        result = last_stocks  # ended game's final (bot, opp)
                        result_kind = serving  # kind is fixed within a dolphin
                        # the game starting NOW plays the previously armed char
                        cur_opp_char = armed_opp_char
                        if games >= cfg.games_per_dolphin:
                            pending_reset, pending_result = True, result
                            pending_result_kind = serving
                            break
                        # per-GAME character rotation: the vendor's menu
                        # helper and misselect guard both read
                        # player.character LIVE each menu pass, so
                        # mutating it between games retargets the next
                        # rematch CSS pick — no recycle needed. A char lock
                        # (import serving) pins the seat instead — no rng
                        # draw consumed; unlock resumes normal redraws.
                        nc = next_opponent_char(
                            cur_kind, char_lock, cfg.redraw_chars,
                            armed_opp_char, _draw_char,
                        )
                        if nc is not None:
                            players[opp_port].character = (
                                melee.Character[nc.upper()]
                            )
                            armed_opp_char = nc
                            verb = (
                                "pinned (char lock)"
                                if char_lock is not None and cur_kind != "cpu"
                                else "redrawn"
                            )
                            print(f"game end: opponent {verb} -> {nc}",
                                  flush=True)
                        if cfg.redraw_chars and len(whitelist) > 1:
                            # student-seat whitelist draws are unaffected by
                            # any opponent char lock (own rng stream)
                            sc = _draw_student_char()
                            players[spec.student_port].character = (
                                melee.Character[sc.upper()]
                            )
                            print(f"game end: student redrawn -> {sc}",
                                  flush=True)
                        parser = Parser(ports=[1, 2])
                    if games >= cfg.games_per_dolphin - 1:
                        _start_spare()  # entering this Dolphin's final game
                    last_frame = gs.frame
                    raw = tree_lib.map_structure(
                        np.asarray, parser.get_game(gs)
                    )
                    game = embed_game.from_state(raw)
                    # armor at the source: never ship a nonfinite frame
                    finite = all(
                        np.all(np.isfinite(leaf))
                        for leaf in tree_lib.flatten(game)
                        if np.issubdtype(np.asarray(leaf).dtype, np.floating)
                    )
                    if not finite:
                        print(f"nonfinite frame dropped (frame {gs.frame})",
                              flush=True)
                        continue
                    p1, p2 = gs.players[1], gs.players[2]
                    last_stocks = (int(p1.stock), int(p2.stock))
                    conn.send(
                        dict(
                            game=game,
                            resetting=resetting,
                            final_stocks=result,  # ended game's (port1, port2); None mid-game
                            stocks=last_stocks,
                            percent=(float(p1.percent), float(p2.percent)),
                            # opponent seat's char in the CURRENT game — the
                            # worker's imitation-harvest whitelist gate
                            opp_char=cur_opp_char,
                            # what the opponent seat ACTUALLY serves right
                            # now / served in the game `result` came from
                            # (league_cpu lazy adoption: may differ from the
                            # worker's desired kind until the next recycle)
                            opp_serving=serving,
                            result_serving=result_kind,
                        )
                    )
                    controllers = conn.recv()
                    if controllers is None:
                        return
                    # league_cpu: the worker piggybacks its desired serving
                    # kind on the command dict; stash it for the next recycle
                    desired = controllers.pop("opp_kind", None)
                    if desired is not None:
                        desired_kind = desired
                    # league_imports: piggybacked char lock (None = unlock);
                    # absent key (imports not configured) keeps the current
                    # value — i.e. stays None forever, today's behavior
                    char_lock = controllers.pop("opp_char_lock", char_lock)
                    for port, controller_state in controllers.items():
                        if cur_kind == "cpu" and port == opp_port:
                            continue  # engine AI drives this port; no inputs
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
                # Recycle path: stop the old Dolphin off-thread so a SPARE
                # swap isn't gated on teardown. Cold boots drain these first
                # (_drain_old_stops), so teardown/boot never overlap except
                # in the validated healthy-spare case. Non-daemon: shutdown
                # waits for the kills (no zombie Dolphins).
                pid = _dolphin_pid(dolphin)
                t = threading.Thread(target=dolphin.stop)
                t.start()
                old_stops.append((t, pid))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if spare["thread"] is not None:
            spare["thread"].join(timeout=60)
            if spare["dolphin"] is not None:
                spare["dolphin"].stop()
        _drain_old_stops()


class _HarvestGroup:
    """Imitation harvest state for opponent seats sharing one model config
    (delay + controller encoding): a ChunkAssembler over a FIXED row set
    plus the per-frame eligibility stream. Records arrive merged over the
    whole row set (zero placeholders for rows not on this config); the
    whole-chunk eligibility gate at emit slices out only rows that were
    real and whitelisted for the ENTIRE chunk."""

    def __init__(self, key, rows, unroll, delay, reencode, device):
        self.key = key
        self.rows = list(rows)
        self.rows_cpu = torch.tensor(self.rows)
        self.rows_t = torch.tensor(self.rows, device=device)
        self.unroll = unroll
        self.assembler = ChunkAssembler(unroll, delay)
        self.reencode = reencode  # record -> student-schema record, or None
        self.pending: list = []  # (resets[R], elig[R]) per frame
        self.elig: list[torch.Tensor] = []  # [R] per pushed record
        self.device = device

    def push_reward(self, reward_rows: torch.Tensor) -> None:
        self.assembler.push_reward(reward_rows)

    def step(self, resets_rows, elig_rows, records, imit_out) -> None:
        self.pending.append((resets_rows, elig_rows))
        for j, rec in enumerate(records):
            frame_resets, frame_elig = self.pending[j]
            if self.reencode is not None:
                rec = self.reencode(rec)
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
                imit_out.append(slice_trajectory_rows(traj, rows))


class _Seat:
    """A role in a slot's transition — `current` (the announced member) or
    `outgoing` (the previous member, still fought by envs mid-game):
    member key + the agent serving it + the agent's model config label
    (groups the imitation harvest by delay/controller encoding).
    Model-agnostic: the agent is whatever the member needs, borrowed from
    the slot's _SlotPool, which is what labels it."""

    __slots__ = ("member", "agent", "config")

    def __init__(self, member: str, agent, config: str):
        self.member = member
        self.agent = agent
        self.config = config


class _SlotPool:
    """The physical modules a slot's seats borrow from (the ONLY place
    "ours vs phillip" exists): two our-config batch-8 agents — `ours_main`
    (the slot policy, compiled) and `ours_spare` (eager; parks weights) —
    and a batch-8 wrapper over the ONE shared Phillip module. All are
    created lazily."""

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
        # League serving state, written by the slot refresh
        # (pool.apply_assignments -> begin_transition); empty = no auction
        # yet (slots serve their init weights unlabeled).
        #
        # DEFERRED ADOPTION (user-designed, v4): an env only ever changes
        # opponent BETWEEN games, never during — brain, body (char) and
        # payoff label move together at the env's own next game boundary.
        # The auction only announces a slot's INCOMING member; each env
        # flips to it when its current game ends (for a char-locked
        # import: when the lock has actually reached the CSS, i.e. the new
        # game's opp_char == lock). Until then it keeps playing its old
        # brain. Never any waiting: the first env to finish is the first to
        # adopt.
        #
        # SEATS (user-designed, v4): every slot has two model-agnostic
        # role seats — `current` (the announced member) and `outgoing` (the
        # previous member while envs still play it). Each seat's agent is a
        # batch-8 wrapper stepping the slot's FULL env batch every frame
        # (one fixed shape -> one compiled graph, per-row recurrent state
        # never disturbed; rows not on that seat just drop their
        # controller). Seats borrow agents from the slot's _SlotPool — the
        # only place "ours vs phillip" is ever mentioned. Ghost, import,
        # teacher and Phillip transitions are one code path. CPU remains
        # the one true exception: not a brain (Dolphin's engine AI),
        # adopted at a Dolphin recycle (slot_desired + the env's own
        # opp_serving report) — it occupies no seat.
        #   env_member[i]    member key env i is ACTUALLY fighting now
        #   slot_incoming[k] the slot's announced member
        #   _seats[k]        {"current": _Seat|None, "outgoing": _Seat|None}
        #   slot_pending[k]  envs of slot k yet to adopt the incoming
        self.env_member: dict[int, str] = {}
        self.slot_incoming: dict[int, str] = {}
        self.slot_pending: dict[int, set] = {}
        self._seats: dict[int, dict[str, "_Seat | None"]] = {}
        self._pool: dict[int, _SlotPool] = {}
        # n -> BatchedPolicyAgent over a FRESH (eager) our-config module:
        # the per-slot spare. train_rl sets it; None = transitions fall
        # back to instant swaps (tests without a factory).
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
        # Phillip as a league member: n -> BatchedPolicyAgent wrapper over
        # the ONE shared Phillip module (his own config). Each slot that
        # draws him gets its own batch-8 wrapper (see _SlotPool); train_rl
        # sets the factory under league_phillip.
        self.phillip_factory: tp.Optional[tp.Callable[[int], tp.Any]] = None
        # Imitation harvest of OPPONENT seats (user-designed, v4: agnostic
        # to who the opponent is). Two sources:
        #  - ref_envs mode: the fixed reference group (one assembler).
        #  - league slots: EVERY policy seat (ghost, import, teacher copy,
        #    Phillip — cpu has no seat). Harvested rows are grouped by the
        #    seat agent's model config (delay + controller encoding) into
        #    _HarvestGroups, each with a FIXED row set (all slot envs) and
        #    a per-row per-frame eligibility mask (row on a seat of this
        #    config AND opponent char whitelisted). Membership changes
        #    move through the mask, never by resetting an assembler, so
        #    transitions drop nothing; the whole-chunk eligibility gate at
        #    emit keeps every harvested row homogeneous.
        self.harvest_imitation = harvest_imitation and (
            bool(self.ref_idx) or bool(self._league)
        )
        self._harvest_groups: dict[str, _HarvestGroup] = {}
        if self.harvest_imitation:
            self._stu_embed = student._embed_controller
            self._student_name_code = int(student._name[0].item())
            self._slot_rows = [
                i for i, sp in enumerate(self.specs) if sp.kind == "snapshot"
            ]
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

    def _ensure_started(self) -> None:
        if self._procs:
            return
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for i in range(self.num_dolphins):
            parent, child = ctx.Pipe()
            # non-daemon: libmelee's slippstream forks its own child
            p = ctx.Process(
                target=_env_process_main, args=(i, self.config, self.specs[i], child)
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

    def _reencode_record(self, rec: FrameRecord, embed=None) -> FrameRecord:
        """Opponent-seat record -> student-schema record: actions re-encoded
        through the STUDENT's controller embedding (the opponent checkpoint
        may discretize differently) and the name conditioned on the
        student's code (its name codes index a different vocabulary).
        `embed` = the opponent's controller embedding (default: the fixed
        reference group's)."""
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
        """Serving kind of a league member key: teacher/cpu/phillip are
        themselves; snapshot paths and imports are "snapshot"."""
        return key if key in ("teacher", "cpu", "phillip") else "snapshot"

    def _actual_kind(self, i: int, serving: str | None) -> str:
        """Kind ACTUALLY serving env i's opponent seat, given the env's
        reported serving label ("policy"/"cpu"/None). Non-slot envs keep
        their fixed spec kind; slot envs resolve "cpu" directly and
        otherwise follow env_member (deferred adoption: what the env is
        really fighting, not what the slot was last assigned). Mapped
        through _TRACKER_KIND this keys the per-kind GameTrackers so
        teacher/cpu/phillip games keep their ticker/wandb continuity even
        when served via league slots."""
        sp = self.specs[i]
        if sp.kind != "snapshot":
            return sp.kind
        if serving == "cpu":
            return "cpu"
        return self.member_kind(self.env_member.get(i))

    def _result_key(self, i: int, result_serving: str | None) -> str | None:
        """Payoff-row key for a game that just ended on slot env i: "cpu"
        if the engine AI played it, else the member the env was fighting
        (None = unlabeled init serving before any auction)."""
        if result_serving == "cpu":
            return "cpu"
        return self.env_member.get(i)

    _POLICY_KINDS = ("snapshot", "teacher")

    def _pool_agent(self, slot: int, member: str):
        """Borrow the agent a member needs from the slot's pool (lazily
        created) as (agent, config label): Phillip -> the shared-module
        wrapper, "phillip"; anything else -> the slot policy (ours_main),
        "ours". Parking onto ours_spare is done by _park."""
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
            return pool.phillip, "phillip"
        if pool.ours_main is None:
            pool.ours_main = self.opponents[("slot", slot)]
        return pool.ours_main, "ours"

    def _rows_on(self, slot: int, key: str | None) -> list[int]:
        if key is None:
            return []
        return [
            i for i in self.groups.get(("slot", slot), [])
            if self.env_member.get(i) == key
        ]

    def _release_outgoing(self, slot: int) -> None:
        """The outgoing seat empties once no env fights its member."""
        seats = self._seats.get(slot)
        if seats and seats["outgoing"] is not None:
            if not self._rows_on(slot, seats["outgoing"].member):
                seats["outgoing"] = None

    def _needs_slot_policy(self, member: str) -> bool:
        """Does this member live in the slot policy module? (ghosts,
        imports, teacher copies do; phillip has his own module; cpu has
        none). The ONE model-specific predicate in the transition logic."""
        return member != "cpu" and member != "phillip"

    def _seat_for(self, slot: int, member: str) -> "_Seat | None":
        if member == "cpu":
            return None  # engine AI: no brain, no seat
        return _Seat(member, *self._pool_agent(slot, member))

    def _park(self, slot: int, seat: _Seat, slot_policy) -> _Seat:
        """The slot policy is about to be overwritten while envs still
        fight its occupant: copy its weights into the pool's spare module
        and re-seat the occupant there."""
        assert self.outgoing_factory is not None and slot_policy is not None, (
            "deferred adoption needs a spare-brain factory (train_rl sets "
            "worker.outgoing_factory) and the slot policy to park"
        )
        pool = self._pool[slot]
        if pool.ours_spare is None:
            pool.ours_spare = self.outgoing_factory(len(self.groups[("slot", slot)]))
        pool.ours_spare.policy.load_state_dict(slot_policy.state_dict())
        return _Seat(seat.member, pool.ours_spare, seat.config)

    def _evict_outgoing(self, slot: int, onto: str) -> None:
        """Rare: a transition still open when the next auction lands (a
        game longer than a whole snapshot_interval). The outgoing seat must
        be freed, so its remaining rows take ONE mid-game swap. Loud."""
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
        """Announce slot `slot`'s next member (called by
        pool.apply_assignments BEFORE it loads any weights into the slot
        policy).

        Boot (no member yet): every env adopts immediately — nothing is in
        flight — and a char lock is written into the env specs so the very
        FIRST game already pins it.
        Later: the occupant of `current` moves to `outgoing` if any env is
        still fighting it (parked onto the spare module when the newcomer
        needs the slot policy), `current` becomes the newcomer, and every
        env of the slot is pending until its own game boundary
        (_adopt_pending)."""
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
        """Flip pending envs onto their slot's incoming member at their game
        boundary (the payload's resetting frame = first frame of the new
        game). A char-locked incoming additionally waits until the new
        game's opp_char IS the lock (the CSS pick lags one game behind the
        arming); "cpu" incoming waits for the env's own recycle adoption
        (opp_serving == "cpu"); a row currently on cpu waits for its
        recycle back to a policy seat. Must run AFTER the frame's results
        were attributed (the ended game belongs to the previous member)."""
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

        while len(out) < num_trajectories:
            payloads = self._gather_all()
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
            # deferred adoption flips happen AFTER the ended games above were
            # credited to the member that actually played them
            if self.slot_pending:
                self._adopt_pending(payloads, resets_d)

            resets = resets_d[row_dolphin]  # row-level

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
            encoded = self._encode(games)
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
            controllers1, records, hidden_before = self.student.step(
                student_view, resets_dev
            )

            opp_controllers: dict[int, tp.Any] = {}
            ref_records: list[FrameRecord] = []
            # per config: [(env rows live on a seat, their local idx in
            # the slot batch, the seat's flushed records, the seat agent)]
            harvest_parts: dict[str, list] = {}
            for name, idx in self.groups.items():
                if isinstance(name, tuple):
                    # slot group: each occupied seat (current, outgoing)
                    # with live rows steps the slot's FULL batch (stable
                    # compile shape, per-row recurrent state undisturbed;
                    # unused rows' state is junk that re-zeros at their
                    # next boundary) and the controllers are routed per
                    # row. A Phillip seat also feeds the imitation harvest.
                    # Before any auction (no seats yet) the slot policy
                    # serves every row, unlabeled.
                    k = name[1]
                    seats = self._seats.get(k)
                    sel = torch.tensor(idx, device=device)
                    if not seats:
                        live = [i for i in idx if i not in cpu_now]
                        if live:
                            group_view = tree.map_structure(
                                lambda x: x.index_select(0, sel), opponent_view
                            )
                            ctrls, _, _ = self.opponents[name].step(
                                group_view, resets_dev[sel]
                            )
                            for j, env_i in enumerate(idx):
                                if env_i in live:
                                    opp_controllers[env_i] = ctrls[j]
                        continue
                    group_view = None
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
                        if group_view is None:
                            group_view = tree.map_structure(
                                lambda x: x.index_select(0, sel), opponent_view
                            )
                        ctrls, recs, _ = seat.agent.step(group_view, resets_dev[sel])
                        for j, env_i in enumerate(idx):
                            if env_i in live:
                                opp_controllers[env_i] = ctrls[j]
                        if self.harvest_imitation:
                            local = [j for j, env_i in enumerate(idx) if env_i in live]
                            harvest_parts.setdefault(seat.config, []).append(
                                (live, local, recs, seat.agent)
                            )
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
                sel = torch.tensor(idx, device=device)
                group_view = tree.map_structure(
                    lambda x: x.index_select(0, sel), opponent_view
                )
                ctrls, g_records, _ = agent.step(group_view, resets_dev[sel])
                if name == "reference" and self.harvest_imitation:
                    ref_records = g_records
                for j, env_i in enumerate(idx):
                    opp_controllers[env_i] = ctrls[j]

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

            self._frame_count += 1
            if self.assembler.ready():
                out.append(self.assembler.emit())
        self._records_pushed = records_pushed
        return out + imit_out

    def _league_harvest(
        self,
        payloads: list[dict],
        resets_d: torch.Tensor,
        parts: dict[str, list],
        imit_out: list[Trajectory],
    ) -> None:
        """Feed every config group one merged record per flushed frame
        over its FIXED row set (all slot envs): rows live on a seat of that
        config get their real record (re-encoded to the student's schema
        when the config differs), other rows a zero placeholder, and the
        eligibility mask marks which rows were real AND whitelisted. A
        group is created at its config's first appearance (delay from its
        agent); a frame with no seat of that config pushes only an
        all-ineligible mask so rewards/resets/records stay aligned."""
        slot_rows = self._slot_rows
        pos = {r: k for k, r in enumerate(slot_rows)}
        T = self.config.unroll_length
        for cfg_key, plist in parts.items():
            group = self._harvest_groups.get(cfg_key)
            if group is None:
                agent = plist[0][3]
                reencode = None
                if cfg_key != "ours":
                    embed = agent._embed_controller
                    reencode = lambda rec, e=embed: self._reencode_record(rec, e)
                group = _HarvestGroup(
                    cfg_key, slot_rows, T, agent.delay, reencode,
                    self.student.device,
                )
                self._harvest_groups[cfg_key] = group
        for cfg_key, group in self._harvest_groups.items():
            plist = parts.get(cfg_key, [])
            elig = torch.zeros(len(slot_rows), dtype=torch.bool)
            for live, _, _, _ in plist:
                for r in live:
                    if payloads[r].get("opp_char") in self._whitelist:
                        elig[pos[r]] = True
            nrec = {len(recs) for _, _, recs, _ in plist} or {0}
            assert len(nrec) == 1, (
                f"seats of config {cfg_key} flushed unevenly {nrec}: "
                "wrappers must share flush cadence (batch_steps)"
            )
            merged = []
            for j in range(nrec.pop()):
                full = None
                for live, local, recs, _ in plist:
                    rec = recs[j]
                    dev = rec.name.device
                    if full is None:
                        full = tree.map_structure(
                            lambda x: x.new_zeros(
                                (len(slot_rows),) + tuple(x.shape[1:])
                            ),
                            rec,
                        )
                    dst = torch.tensor([pos[r] for r in live], device=dev)
                    src = torch.tensor(local, device=dev)
                    full = tree.map_structure(
                        lambda f, x: f.index_copy(0, dst, x.index_select(0, src)),
                        full, rec,
                    )
                merged.append(full)
            group.step(resets_d[group.rows_cpu], elig, merged, imit_out)

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
