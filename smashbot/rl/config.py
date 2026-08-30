"""Torch-free rollout configuration shared by the worker and the spawned
env processes (which must never import torch: ~0.26 GB private RSS each).
"""

from __future__ import annotations

import dataclasses
import typing as tp

MAIN_12 = [
    "FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "CPTFALCON",
    "PEACH", "YOSHI", "POPO", "LUIGI", "PIKACHU", "SAMUS",
]
# Policy opponents can be any of the 12: Sheik works via the netplay CSS
# Zelda slot (its Sheik/Zelda toggle defaults to Sheik); occasional menu
# races are survived by the env-process retry guard. CPU opponents cannot
# be Sheik (libmelee cannot force a CPU to transform), and Zelda is
# unpickable on the netplay CSS entirely.
OPPONENT_CHARS = list(MAIN_12)
# CPU Sheik is IMPOSSIBLE (tested live: 362/362 attempts spawned Zelda —
# the engine ignores held A on CPU-status ports, so the Zelda->Sheik
# transform never triggers; libmelee's guard was right). Sheik matchup
# coverage flows through the policy-opponent envs instead.
CPU_CHARS = [c for c in MAIN_12 if c != "SHEIK"]
# Rest of the CSS cast reachable by simple menuing (SHEIK reached via ZELDA
# is already in MAIN_12 through the parser's lens; ZELDA herself included).
OFF_ROSTER = [
    "MARIO", "DOC", "LINK", "YLINK", "NESS", "BOWSER", "DK",
    "GANONDORF", "GAMEANDWATCH", "KIRBY", "MEWTWO", "PICHU",
    "ROY",
]


def student_whitelist(
    char_whitelist: tp.Sequence[str], bot_char: str = "FOX"
) -> list[str]:
    """Effective student-character whitelist.

    The default whitelist ["FOX"] defers to the legacy bot_char flag (so
    `--rollouts.bot-char MARTH` keeps working); any non-default whitelist
    wins. len==1 reproduces the fixed-character behavior exactly."""
    wl = [c.upper() for c in char_whitelist]
    if wl == ["FOX"]:
        return [bot_char.upper()]
    return wl


@dataclasses.dataclass
class EnvSpec:
    """Per-env assignment, fixed for the run."""

    # "cpu" | "teacher" | "reference" | "import" | "self" | "snapshot"
    # (snapshot = league env: opponent drawn per match)
    kind: str
    student_port: int  # 1 or 2
    opponent_char: str
    cpu_level: int = 9
    # "import" envs: which fixed brain this env is pinned to ("import:NAME")
    member: str = ""
    # pinned opponent character ("FOX" imports); None = redraw per game
    char_lock: tp.Optional[str] = None


@dataclasses.dataclass
class RolloutConfig:
    num_envs: int = 8
    unroll_length: int = 240  # 4s, slippi-ai's RL rollout length
    batch_steps: int = 1  # frames per inference flush; measured best on this rig (see docs)
    # overlap the per-frame opponent grid forwards (league + imports) on
    # their own CUDA streams while the student/groups run on the default
    # stream; the calls are independent and their small latency-bound
    # kernels coexist on the GPU. Serial on CPU and for the first frames
    # of a boot (compile/graph capture must be single-threaded).
    parallel_serving: bool = True
    bot_char: str = "FOX"
    stage: str = "FINAL_DESTINATION"
    games_per_dolphin: int = 20
    # Dolphin dual-core emulation (CPU + GPU threads). Off packs a big
    # headless fleet onto the cores better (one thread per Dolphin).
    dolphin_dual_core: bool = False  # measured: -4 ms/frame at 176 Dolphins
    # Opponent pool partition (see rl/pool.py). Defaults replicate the
    # simple all-teacher setup; production: everything not cpu/teacher/
    # reference/self is a LEAGUE env (kind "snapshot").
    cpu_envs: int = 0
    # -1 = teacher absorbs every env not assigned to cpu/ref/import/self/
    # league — a FOOTGUN with import_dedicated_envs (set teacher_envs 0
    # explicitly in league configs; train_rl asserts if league envs hit 0)
    teacher_envs: int = -1
    # The league grid (rl/agent.LeagueAgent): league_slices weight slices,
    # each serving league_envs / league_slices cells. A slice is a weight
    # cache entry — at most league_slices DISTINCT league members are
    # resident at once (107 MB each); envs draw their opponent per match
    # and sit wherever that member is loaded (rollouts._Grid). 0 = no
    # league envs.
    league_slices: int = 0
    # Stacked league weights dtype: "float16" halves VRAM per slice
    # (107 -> 54 MB), fp16 autocast on the grid forward. CUDA only.
    league_weights_dtype: str = "float16"
    # Student (and Phillip) rollout inference precision: "fp16" = fp16
    # autocast on the networks (sampling math stays fp32); gated by
    # scripts/precision_probe.py.
    rollout_precision: str = "fp16"
    # LEGACY (league_phillip mode only; ignored otherwise): Phillip's
    # league-agent capacity — max envs fighting him at once. 0 = 3 slices'
    # worth. v9 serves phillip via dedicated ref_envs instead.
    phillip_capacity: int = 0
    main12_prob: float = 0.6
    snapshot_interval: int = 500  # learner steps between student snapshots
    snapshot_keep: int = 30  # ghost archive cap; <=0 = never prune
    # Host-RAM cap on RESIDENT ghost weights when the archive is immortal
    # (snapshot_keep <= 0): ~107MB each, so an unbounded cache grows
    # ~8.6GB by step 40k. Evicted ghosts reload from disk on demand (the
    # league's warm() prefetch hides the latency).
    ghost_cache: int = 40
    # Dedicated envs PER import member (static: pinned brain, no league
    # draw, per-game char redraw unless the import is char-locked).
    # 0 = legacy behavior: imports are league members drawn by PFSP.
    import_dedicated_envs: int = 0
    partition_seed: int = 0
    headless: bool = True  # False: rendered window at normal speed (watch mode)
    log_tag: str = ""  # namespaces /tmp/smashbot-env-*.log between runs
    # Redraw the opponent character at each Dolphin recycle.
    redraw_chars: bool = True
    # Boot each Dolphin's replacement in the background during its final
    # game (recycle hot-swap). OFF by default: ~5% gain, riskier teardown.
    double_buffer: bool = False
    # Double-buffer the LEARNER: run learner.step(batch k) on its own CUDA
    # stream in a background thread while the worker collects batch k+1.
    # Rollouts become one gradient-update stale (PPO's importance ratios
    # absorb it; watch rl/actor_kl_mean); the student serves a dedicated
    # weight copy published at step boundaries so optimizer.step can never
    # tear a forward. CUDA only; serial on CPU.
    learner_overlap: bool = False
    # (historical: ref_shard_size tuned the retired in-worker TF bridge;
    # kept so old commands don't break. The worker ignores it now.)
    ref_shard_size: int = 16
    # Watchdog: max seconds the barrier waits on one env's payload before
    # crashing loudly (a supervisor/--runtime.restore auto turns that crash
    # into a ~20min self-heal instead of a silent overnight hang).
    env_timeout: float = 300.0
    # Seconds between successive envs' FIRST Dolphin boot (env i sleeps
    # i * boot_stagger). 250+ simultaneous cold boots exceed the port
    # connect timeout under contention (BOOT FAILURE 3/3 -> run dies);
    # ~0.15 spreads the storm over ~40s. Recycles are naturally staggered
    # by game lengths and don't use this.
    boot_stagger: float = 0.15
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
    # the student — loaded exactly like a ghost), optionally with a
    # per-import character lock "NAME=PATH@CHAR" (default lock: FOX —
    # the original imports are trained-fox opponents). An env fighting an
    # import pins the locked character for that match instead of
    # redrawing; "@ANY" imports stay unlocked and redraw their character
    # per game exactly like snapshots (for 12-char generalist imports).
    # Requires pfsp=True and league_slices > 0.
    league_imports: list[str] = dataclasses.field(default_factory=list)

    def import_members(self) -> dict[str, tuple[str, str]]:
        """Parsed league_imports: {NAME: (path, char_lock)}. Bad entries
        fail loudly (a silently dropped import would serve nothing and skew
        the per-match draw)."""
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

            if char == "ANY":
                char = None  # unlocked: redraw per game, like a snapshot
            else:
                assert char in MAIN_12, (
                    f"league import {name!r}: char lock {char!r} not in "
                    f"the policy-opponent roster {MAIN_12} (or ANY)"
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
        if self.learner_overlap:
            assert self.teacher_envs == 0, (
                "learner_overlap runs the learner concurrently with the "
                "worker; teacher_envs>0 would serve the SAME live teacher "
                "module from both threads (compiled-cudagraph state is not "
                "thread-safe) — fold the teacher into the league instead"
            )
        if self.league_teacher:
            assert self.teacher_envs == 0, (
                f"league_teacher folds the teacher into the PFSP league — "
                f"set teacher_envs=0 (got {self.teacher_envs}) and move "
                f"those envs into the league"
            )
            members.append("teacher")
        if self.league_cpu:
            assert self.cpu_envs == 0, (
                f"league_cpu folds the lvl-9 CPU into the PFSP league — "
                f"set cpu_envs=0 (got {self.cpu_envs}) and move those envs "
                f"into the league"
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
        if imports and self.import_dedicated_envs <= 0:
            # dedicated imports (v9) are static envs, NOT league members —
            # they never enter the draw and need no PFSP
            assert self.league_slices > 0, (
                f"league_imports serve through the league grid — set "
                f"league_slices > 0 (got {self.league_slices})"
            )
            members += [f"import:{name}" for name in imports]
        if members:
            assert self.pfsp, (
                "league_teacher/league_cpu require pfsp=True: league members "
                "earn/lose serving time through the payoff table, which the "
                "recency sampler never consults"
            )
        return members


@dataclasses.dataclass
class PPOConfig:
    num_epochs: int = 1
    epsilon: float = 1e-2  # log-space clip: ratio confined to [e^-eps, e^eps]
    beta: float = 0.0  # weight of KL(actor || policy)
    max_mean_actor_kl: float = 1e-4  # revert the update above this
    # Anomaly armor: |log ratio| beyond this is data corruption, not policy
    # drift (one update moves aKL ~1e-5; e^10 is impossible drift). Clamped
    # for the surrogate; occurrences logged + first few dumped for forensics.
    log_rho_clamp: float = 10.0


@dataclasses.dataclass
class RLConfig:
    learning_rate: float = 1e-4
    policy_gradient_weight: float = 1.0
    kl_teacher_weight: float = 1e-1
    reverse_kl_teacher_weight: float = 0.0
    entropy_weight: float = 0.0
    reward_halflife: float = 4.0  # seconds
    max_grad_norm: float = 1.0  # 0 = no clipping
    # Learner numeric precision: "fp32" (exact current behavior — no autocast
    # objects, no scaler) or "fp16" (cuda-only production path; cpu falls back
    # to fp32 with a loud warning). fp16 = torch.autocast(float16) around the
    # POLICY forward regions only (policy unroll, frozen-teacher unroll,
    # imitation unroll) + one GradScaler on the policy optimizer. The VALUE
    # net stays entirely fp32 — its fixed-pass forward/backward/step never
    # enter autocast (weakest fp16 arm in the probe, small compute share;
    # measured recipe: scripts/precision_probe.py fp16s arm, receipts in
    # /home/kage/drive2/ShineBot/probes/batch-0013549.pt.fidelity.json).
    precision: str = "fp32"
    # PPO policy pass in this many row chunks with gradient accumulation:
    # identical gradient and update, ~1/k the live activation memory
    # (rows x 240 unrolls), ~20 ms/step overhead at k=2 (measured)
    micro_batches: int = 1
    # fp16 loss-scale doubling interval, in learner steps (torch default
    # 2000 assumes a far higher step rate; see Learner.__init__)
    grad_scaler_growth_interval: int = 500
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    # --- opponent advantage imitation (docs/idea-opponent-learning.md) ---
    # Harvested opponent rows trained per step ON TOP of the full PPO batch
    # (nothing substituted out): -1 = every eligible row, N > 0 = a uniform
    # sample of N, 0 = fully dormant. Rows accumulate in chunks no larger
    # than the PPO micro-batch, so this costs learner time, not VRAM.
    imitation_rows: int = 0
    # MARWIL/AWR weighting: w = clip(exp(A_norm / beta), max=w_cap).
    imitation_beta: float = 1.0
    imitation_w_cap: float = 20.0
    # Loss coefficient: lambda_t * L_opp added to the policy loss; 0 = the
    # actor-side term is entirely absent (critic still trains on harvested
    # states when slots > 0). Decays linearly from imitation_lambda to
    # imitation_lambda * imitation_lambda_final_frac across runtime.steps.
    imitation_lambda: float = 0.0
    imitation_lambda_final_frac: float = 0.2

    @property
    def discount(self) -> float:
        return 0.5 ** (1 / (self.reward_halflife * 60))
