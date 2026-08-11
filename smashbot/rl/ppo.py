"""PPO learner: RL fine-tuning with a KL leash to the frozen imitation teacher.

Port of slippi-ai's rl/learner.py (vendored), redesigned for PyTorch and for
our rollout convention. Key differences from the vendor:

- Batch-first [B, T] everywhere, like the rest of smashbot.
- Trajectories store the SAMPLED action stream (action + logits at sample
  time), which is delay-aligned exactly like BC training frames — so the
  learner pairs policy outputs, teacher outputs, and advantages positionally,
  with no delayed-action queue reconstruction at learn time. The rollout
  collector owns delay bookkeeping (it already must, to drive the game).
- One epoch path (gradient accumulation over trajectory minibatches, one
  optimizer step per epoch) instead of the vendor's three compiled variants.

Loss (per vendor semantics):
    - policy_gradient_weight * clipped_surrogate(ratio, advantages)
    + ppo.beta * KL(actor || policy)          # smooth leash to rollout policy
    + kl_teacher_weight * KL(policy || teacher)   # mode-seeking human anchor
    + reverse_kl_teacher_weight * KL(teacher || policy)
    - entropy_weight * entropy(policy)

Safety: after each step() the post-update actor-KL is measured; if its mean
exceeds ppo.max_mean_actor_kl the whole update is reverted (vendor behavior).
"""

from __future__ import annotations

import copy
import dataclasses
import typing as tp

import torch
import tree

from slippi_ai.types import Frames, StateAction

from smashbot.networks import RecurrentState, _mask_state
from smashbot.policy import Policy
from smashbot.value import ValueFunction


@dataclasses.dataclass
class PPOConfig:
    num_epochs: int = 1
    epsilon: float = 1e-2  # log-space clip: ratio confined to [e^-eps, e^eps]
    beta: float = 0.0  # weight of KL(actor || policy)
    max_mean_actor_kl: float = 1e-4  # revert the update above this
    # DClamp-PPO (arXiv:2511.02577): steeper loss slope (alpha > 1) against
    # ratios drifting in the STRICT WRONG direction (below 1-dclamp_beta for
    # A>0, above 1+dclamp_beta for A<0). 0 disables. NOTE: the paper tunes
    # dclamp_beta ~0.2-0.4 for eps=0.2 multi-epoch PPO; our log-space
    # eps=1e-2 regime needs it rescaled to the actual ratio spread (~0.02).
    dclamp_alpha: float = 0.0
    dclamp_beta: float = 0.02


@dataclasses.dataclass
class RLConfig:
    learning_rate: float = 1e-4
    policy_gradient_weight: float = 1.0
    kl_teacher_weight: float = 1e-1
    reverse_kl_teacher_weight: float = 0.0
    entropy_weight: float = 0.0
    reward_halflife: float = 4.0  # seconds
    max_grad_norm: float = 0.0  # 0 = no clipping
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)

    @property
    def discount(self) -> float:
        return 0.5 ** (1 / (self.reward_halflife * 60))


class ActionData(tp.NamedTuple):
    # prev-action stream: controller_state[t] = the action sampled at frame
    # t-1, i.e. exactly what the agent fed as its input at frame t. This makes
    # Frames(state, action) identical in meaning to BC training frames.
    controller_state: tp.Any  # controller struct, [B, T+1, ...]
    # logits[t] = the actor's logits AT frame t (which sampled action â_t).
    # Position t of a learner unroll predicts â_t, so these pair 1:1.
    logits: tp.Any  # controller struct, [B, T+1, ...]


class Trajectory(tp.NamedTuple):
    """One rollout chunk, batch-first, agent-stream convention (see ActionData:
    actions.controller_state is the agent's *input* stream; actions.logits are
    sample-time logits). rewards[t] pairs the t -> t+1 transition, shifted by
    the rollout collector exactly as delay_lib.slice_delayed_frames shifts BC
    rewards. All [B, T+1] tensors overlap chunks by one frame.
    """

    states: tp.Any  # encoded Game struct, [B, T+1, ...]
    name: torch.Tensor  # [B, T+1]
    actions: ActionData  # [B, T+1]
    rewards: torch.Tensor  # [B, T]
    is_resetting: torch.Tensor  # [B, T+1]
    initial_state: RecurrentState  # policy recurrent state at chunk start


class LearnerState(tp.NamedTuple):
    """Learner-side recurrent states, carried across sequential chunks."""

    teacher: RecurrentState
    value: RecurrentState


def clipped_surrogate(
    log_rhos: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
    dclamp_alpha: float = 0.0,
    dclamp_beta: float = 0.02,
) -> torch.Tensor:
    """PPO objective (to maximize), clipped in log space: min(r*A, clip(r)*A).

    With dclamp_alpha > 1, adds DClamp-PPO's third min-term (arXiv:2511.02577):
    f(w) = alpha*w - (alpha-1)*(1 -+ dclamp_beta), a slope-alpha line active
    only where the ratio has drifted in the strict wrong direction, pulling
    it back toward 1 harder than PPO's slope-1 default."""
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.exp(torch.clamp(log_rhos, -epsilon, epsilon))
    objs = torch.minimum(rhos * advantages, clipped_rhos * advantages)
    if dclamp_alpha > 1.0:
        a, b = dclamp_alpha, dclamp_beta
        intercept = torch.where(
            advantages > 0,
            -(a - 1) * (1 - b),
            -(a - 1) * (1 + b),
        )
        objs = torch.minimum(objs, (a * rhos + intercept) * advantages)
    return objs


class _StructOps:
    """Sum-over-components distribution ops on controller logit structs."""

    def __init__(self, controller_embedding):
        self._embed = controller_embedding

    def _sum(self, struct) -> torch.Tensor:
        return sum(self._embed.flatten(struct))

    def log_prob(self, logits, actions) -> torch.Tensor:
        distances = self._embed.map(
            lambda e, t, a: e.distance(t, a), logits, actions
        )
        return -self._sum(distances)

    def kl(self, p_logits, q_logits) -> torch.Tensor:
        kls = self._embed.map(
            lambda e, p, q: e.logits_kl(p, q), p_logits, q_logits
        )
        return self._sum(kls)

    def entropy(self, logits) -> torch.Tensor:
        return self._sum(self._embed.map(lambda e, t: e.logits_entropy(t), logits))


class _Fixed(tp.NamedTuple):
    """Per-trajectory quantities that do not change across PPO epochs."""

    frames: Frames
    initial_policy_state: RecurrentState
    advantages: torch.Tensor  # [B, T], detached
    teacher_logits: tp.Any  # controller struct, [B, T]
    actor_logits: tp.Any  # controller struct, [B, T]
    actor_log_probs: torch.Tensor  # [B, T]


class Learner:
    """PPO + KL-to-teacher. The teacher is frozen; policy and value train."""

    def __init__(
        self,
        config: RLConfig,
        policy: Policy,
        teacher: Policy,
        value_function: ValueFunction,
    ):
        assert not policy.train_value_head, "RL uses the separate value network"
        self.config = config
        self.policy = policy
        self.teacher = teacher
        self.value_function = value_function

        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_function.parameters(), lr=config.learning_rate
        )
        self._ops = _StructOps(policy.controller_head.controller_embedding)

    def initial_state(self, batch_size: int, device=None) -> LearnerState:
        return LearnerState(
            teacher=self.teacher.initial_state(batch_size, device),
            value=self.value_function.initial_state(batch_size, device),
        )

    def _frames(self, traj: Trajectory) -> Frames:
        return Frames(
            state_action=StateAction(
                state=traj.states,
                action=traj.actions.controller_state,
                name=traj.name,
            ),
            is_resetting=traj.is_resetting,
            reward=traj.rewards,
        )

    def _fixed_pass(
        self, traj: Trajectory, state: LearnerState
    ) -> tuple[_Fixed, LearnerState, dict]:
        """Everything reusable across epochs: teacher logits, advantages (with
        a value-net update), and the actor's own log-probs — plus carried
        recurrent states."""
        frames = self._frames(traj)
        batch_size = traj.rewards.shape[0]

        # Actors reset mid-rollout invisibly to the learner; mask the policy's
        # carried state back to zeros wherever a chunk starts fresh.
        initial_policy_state = _mask_state(
            traj.is_resetting[:, 0],
            self.policy.initial_state(batch_size, traj.rewards.device),
            traj.initial_state,
        )

        with torch.no_grad():
            teacher_out = self.teacher.unroll(
                frames, state.teacher, discount=self.config.discount
            )

        value_out = self.value_function.outputs(
            frames, state.value, discount=self.config.discount
        )
        self.value_optimizer.zero_grad(set_to_none=True)
        value_out.loss.backward()
        self.value_optimizer.step()

        # Unroll position t (t = 0..T-1) predicts the action sampled at frame
        # t: its actor logits are logits[t], and the sampled action itself is
        # the NEXT entry of the prev-action stream, controller_state[t+1].
        actor_logits = tree.map_structure(lambda t: t[:, :-1], traj.actions.logits)
        actor_actions = tree.map_structure(
            lambda t: t[:, 1:], traj.actions.controller_state
        )
        with torch.no_grad():
            actor_log_probs = self._ops.log_prob(actor_logits, actor_actions)

        fixed = _Fixed(
            frames=frames,
            initial_policy_state=initial_policy_state,
            advantages=value_out.advantages,
            teacher_logits=teacher_out.logits,
            actor_logits=actor_logits,
            actor_log_probs=actor_log_probs,
        )
        # Detach carried recurrent states: the next chunk's backward must not
        # reach into this chunk's (already-freed) graph.
        detach = lambda t: t.detach() if isinstance(t, torch.Tensor) else t
        new_state = LearnerState(
            teacher=tree.map_structure(detach, teacher_out.final_state),
            value=tree.map_structure(detach, value_out.final_state),
        )
        return fixed, new_state, value_out.metrics

    def _policy_loss(self, fixed: _Fixed) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        out = self.policy.unroll(
            fixed.frames, fixed.initial_policy_state, discount=cfg.discount
        )

        log_rhos = out.log_probs - fixed.actor_log_probs
        surrogate = clipped_surrogate(
            log_rhos, fixed.advantages, cfg.ppo.epsilon,
            dclamp_alpha=cfg.ppo.dclamp_alpha, dclamp_beta=cfg.ppo.dclamp_beta,
        )

        # Forward KL to the teacher (expectation under the student's states):
        # mode-seeking — refine human play, free to drop human mistakes.
        teacher_kl = self._ops.kl(out.logits, fixed.teacher_logits)
        reverse_teacher_kl = self._ops.kl(fixed.teacher_logits, out.logits)
        actor_kl = self._ops.kl(fixed.actor_logits, out.logits)
        entropy = self._ops.entropy(out.logits)

        loss = (
            -cfg.policy_gradient_weight * surrogate
            + cfg.ppo.beta * actor_kl
            + cfg.kl_teacher_weight * teacher_kl
            + cfg.reverse_kl_teacher_weight * reverse_teacher_kl
            - cfg.entropy_weight * entropy
        ).mean()

        metrics = {
            "loss": loss.item(),
            "surrogate": surrogate.mean().item(),
            "teacher_kl": teacher_kl.mean().item(),
            "actor_kl_mean": actor_kl.mean().item(),
            "actor_kl_max": actor_kl.max().item(),
            "entropy": entropy.mean().item(),
            "ratio_mean": log_rhos.exp().mean().item(),
        }
        return loss, metrics

    def step(
        self, trajectories: tp.Sequence[Trajectory], state: LearnerState
    ) -> tuple[LearnerState, dict]:
        """One PPO update over a batch of trajectory chunks (minibatches).

        Runs the fixed passes (teacher + value update + advantages) once, then
        ppo.num_epochs gradient passes over all chunks, then a no-grad pass to
        measure post-update actor KL — reverting the update if it moved the
        policy beyond ppo.max_mean_actor_kl.
        """
        cfg = self.config

        fixed_list: list[_Fixed] = []
        value_metrics: list[dict] = []
        for traj in trajectories:
            fixed, state, vm = self._fixed_pass(traj, state)
            fixed_list.append(fixed)
            value_metrics.append(vm)

        snapshot = copy.deepcopy(self.policy.state_dict())

        epoch_metrics: list[dict] = []
        for _ in range(cfg.ppo.num_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            batch_metrics = []
            for fixed in fixed_list:
                loss, metrics = self._policy_loss(fixed)
                (loss / len(fixed_list)).backward()
                batch_metrics.append(metrics)
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), cfg.max_grad_norm
                )
            self.policy_optimizer.step()
            epoch_metrics.append(_mean_dicts(batch_metrics))

        # Post-update measurement (and trust-region backstop).
        with torch.no_grad():
            post = _mean_dicts([self._policy_loss(f)[1] for f in fixed_list])
        reverted = post["actor_kl_mean"] > cfg.ppo.max_mean_actor_kl
        if reverted:
            self.policy.load_state_dict(snapshot)

        metrics = {
            "epochs": epoch_metrics,
            "post_update": post,
            "value": _mean_dicts(value_metrics),
            "reverted": reverted,
        }
        return state, metrics


def _mean_dicts(dicts: tp.Sequence[dict]) -> dict:
    out = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        if key == "actor_kl_max":
            out[key] = max(vals)
        else:
            out[key] = sum(vals) / len(vals)
    return out
