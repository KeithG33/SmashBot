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

    kind: str  # "cpu" | "teacher" | "reference" | "self" | "snapshot"
    group: int  # snapshot slot index (0 = freshest); -1 otherwise
    student_port: int  # 1 or 2
    opponent_char: str
    cpu_level: int = 9


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
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    # --- opponent advantage imitation (docs/idea-opponent-learning.md) ---
    # Memory-neutral substitution: up to imitation_slots harvested opponent
    # trajectories per step REPLACE randomly-chosen PPO trajectories (never
    # self-play seats; teacher/cpu first, then snapshot) so the learner batch
    # never exceeds num_envs trajectories. 0 = fully dormant.
    imitation_slots: int = 0
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
