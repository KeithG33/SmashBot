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
    # Teacher-KL leash decay: linear from kl_teacher_weight (progress 0)
    # to this value (progress 1). Negative = disabled (constant leash,
    # the historical behavior). ONE schedule, no resume special-casing:
    # a mid-run change extrapolates the line backward so the current
    # step sits on it (v10 @17k: start 0.1206522 -> 0.08 now -> 0.025
    # at 40k).
    kl_teacher_weight_final: float = -1.0
    reverse_kl_teacher_weight: float = 0.0
    entropy_weight: float = 0.0
    reward_halflife: float = 4.0  # seconds
    max_grad_norm: float = 1.0  # 0 = no clipping
    # AutoClip for the policy: clip to this percentile of the run's own
    # (unscaled) gradient norms instead of max_grad_norm; 0 = off
    autoclip_percentile: float = 0.0
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
