"""RL learner configuration, and the 12 characters the bots play."""

from __future__ import annotations

import dataclasses

MAIN_12 = [
    "FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "CPTFALCON",
    "PEACH", "YOSHI", "POPO", "LUIGI", "PIKACHU", "SAMUS",
]


@dataclasses.dataclass
class PPOConfig:
    num_epochs: int = 1
    epsilon: float = 1e-2  # log-space clip: ratio confined to [e^-eps, e^eps]
    beta: float = 0.0  # weight of KL(actor || policy)
    max_mean_actor_kl: float = 1e-4  # revert the update above this
    # |log ratio| beyond this is corrupt data, not drift (an update moves the
    # actor KL ~1e-5): clamped in the surrogate, logged, the first few dumped
    log_rho_clamp: float = 10.0


@dataclasses.dataclass
class RLConfig:
    learning_rate: float = 1e-4
    policy_gradient_weight: float = 1.0
    kl_teacher_weight: float = 1e-1
    # the teacher-KL leash decays from kl_teacher_weight (progress 0) to this
    # (progress 1) along kl_teacher_decay; negative = constant. There is no
    # resume special case: to change it mid-run, extrapolate the start back so
    # the current step's weight doesn't jump.
    kl_teacher_weight_final: float = -1.0
    reverse_kl_teacher_weight: float = 0.0
    # the reverse leash's decay, as kl_teacher_weight_final's; negative = constant
    reverse_kl_teacher_weight_final: float = -1.0
    # both leashes' decay path: "linear", or "exponential" (geometric, a
    # constant halving rate; needs positive endpoints)
    kl_teacher_decay: str = "linear"
    entropy_weight: float = 0.0
    reward_halflife: float = 4.0  # seconds
    max_grad_norm: float = 1.0  # 0 = no clipping
    # AutoClip for the policy: clip to this percentile of the run's own
    # (unscaled) gradient norms instead of max_grad_norm; 0 = off
    autoclip_percentile: float = 0.0
    # "fp32", or "fp16" (CUDA only; CPU warns and runs fp32): fp16 autocast
    # around the policy forwards (policy, teacher, imitation) and a GradScaler
    # on the policy optimizer. The value net stays fp32: it was the weakest
    # fp16 arm in scripts/precision_probe.py, for a small share of the compute.
    precision: str = "fp32"
    # the PPO policy pass in this many row chunks with gradient accumulation:
    # the same update in ~1/k the activation memory, ~20 ms/step slower at k=2
    micro_batches: int = 1
    # fp16 loss-scale doubling interval in learner steps (torch's 2000 assumes
    # a far higher step rate)
    grad_scaler_growth_interval: int = 500
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    # --- opponent advantage imitation (docs/idea-opponent-learning.md) ---
    # harvested opponent rows trained per step on top of the PPO batch: -1 =
    # every eligible row, N > 0 = a uniform sample of N, 0 = off. They run in
    # chunks no larger than the PPO micro-batch: learner time, not VRAM.
    imitation_rows: int = 0
    # MARWIL/AWR weighting: w = clip(exp(A_norm / beta), max=w_cap)
    imitation_beta: float = 1.0
    imitation_w_cap: float = 20.0
    # lambda_t * L_opp joins the policy loss, decaying linearly to
    # imitation_lambda * imitation_lambda_final_frac over runtime.schedule_steps;
    # 0 = no actor-side term (the critic still trains on harvested states)
    imitation_lambda: float = 0.0
    imitation_lambda_final_frac: float = 0.2

    @property
    def discount(self) -> float:
        return 0.5 ** (1 / (self.reward_halflife * 60))
