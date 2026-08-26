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

import contextlib
import copy
import math
import random
import typing as tp

import torch
import tree

from slippi_ai.types import Frames, StateAction

from smashbot.networks import RecurrentState, _mask_state
from smashbot.rl.config import PPOConfig, RLConfig  # noqa: F401  (re-export)
from smashbot.policy import Policy
from smashbot.value import ValueFunction


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
    # Learner routing tag: "ppo" (on-policy student data, including both
    # seats of self-play envs) or "imitation" (harvested opponent seat, e.g.
    # Phillip's — states/actions/rewards from HIS seat, initial_state None).
    kind: str = "ppo"


def slice_trajectory_rows(traj: Trajectory, rows: tp.Sequence[int]) -> Trajectory:
    """Row (env-dim) subset of a Trajectory; initial_state may be None."""
    sel = torch.as_tensor(list(rows), dtype=torch.int64)

    def take(t):
        if isinstance(t, torch.Tensor):
            return t.index_select(0, sel.to(t.device))
        return t

    return Trajectory(
        states=tree.map_structure(take, traj.states),
        name=take(traj.name),
        actions=tree.map_structure(take, traj.actions),
        rewards=take(traj.rewards),
        is_resetting=take(traj.is_resetting),
        initial_state=(
            None if traj.initial_state is None
            else tree.map_structure(take, traj.initial_state)
        ),
        kind=traj.kind,
    )


class LearnerState(tp.NamedTuple):
    """Learner-side recurrent states, carried across sequential chunks."""

    teacher: RecurrentState
    value: RecurrentState


def clipped_surrogate(
    log_rhos: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """PPO objective (to maximize), clipped in log space: min(r*A, clip(r)*A)."""
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.exp(torch.clamp(log_rhos, -epsilon, epsilon))
    return torch.minimum(rhos * advantages, clipped_rhos * advantages)


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
    valid: torch.Tensor  # [B, T] float; 0 where position t's target is the
    # reset-substituted neutral at t+1 (a fictional action the actor never
    # sampled — the AR head's teacher-forcing chains diverge there and the
    # position carries no legitimate learning signal)


def imitation_weights(
    advantages: torch.Tensor,  # [B, T], will be detached
    valid: torch.Tensor,  # [B, T] float mask
    beta: float,
    w_cap: float,
) -> torch.Tensor:
    """MARWIL/AWR weighting for opponent-advantage imitation.

    A is detached, normalized over the VALID positions of this imitation
    minibatch, then w = clip(exp(A_norm / beta), max=w_cap). Returns [B, T]
    detached weights (unmasked; the loss applies `valid` itself)."""
    adv = advantages.detach()
    n = valid.sum().clamp(min=1.0)
    mean = (adv * valid).sum() / n
    var = (torch.square(adv - mean) * valid).sum() / n
    a_norm = (adv - mean) / (var.sqrt() + 1e-8)
    return torch.exp(a_norm / beta).clamp(max=w_cap)


class _ImitFixed(tp.NamedTuple):
    """Per-imitation-trajectory quantities fixed across PPO epochs."""

    frames: Frames
    weights: torch.Tensor  # [B, T], detached
    valid: torch.Tensor  # [B, T] float
    rows: int


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

        assert config.precision in ("fp32", "fp16"), config.precision
        self._device_type = next(policy.parameters()).device.type
        precision = config.precision
        if precision == "fp16" and self._device_type != "cuda":
            print(
                f"WARNING: precision=fp16 requested but the learner lives on "
                f"{self._device_type} — falling back to fp32 (fp16 autocast "
                f"is only the production path on cuda)",
                flush=True,
            )
            precision = "fp32"
        self.precision = precision
        self._amp_enabled = precision == "fp16"
        # One scaler, policy optimizer only (the value path never scales).
        # growth_interval is in SCALER STEPS: torch's 2000 default assumes
        # ~10 steps/s (3 min), but we run ~126 steps/HOUR, making a doubling
        # take ~16h — so a sparse fault rate can hold the scale down forever
        # (v5 ratcheted 65536->1024 and never recovered). Probe upward more
        # often; the cost of an over-high probe is one skipped update.
        self.grad_scaler = (
            torch.amp.GradScaler(
                self._device_type, init_scale=2.0 ** 16,
                growth_interval=config.grad_scaler_growth_interval,
            )
            if self._amp_enabled
            else None
        )

        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            value_function.parameters(), lr=config.learning_rate
        )
        self._ops = _StructOps(policy.controller_head.controller_embedding)
        # Imitation row-cap sampling RNG; seeded for reproducibility,
        # reseedable in tests.
        self._imit_rng = random.Random(0)

    def _autocast(self):
        """fp16-mode autocast for POLICY forward regions; a plain null
        context in fp32 mode (byte-identical legacy behavior: no autocast
        object is ever constructed)."""
        if not self._amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast(self._device_type, dtype=torch.float16)

    def _backward(self, loss: torch.Tensor) -> None:
        """Policy-loss backward: scaled through the GradScaler in fp16 mode
        (gradients underflow fp16 without it — the probe's unscaled fp16 arm
        lost 2/3 of the policy grad norm, 0.0206 vs fp32's 0.0597; the
        scaled arm recovered it, 0.0601), plain backward in fp32."""
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

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

        # Frozen-teacher forward follows the policy autocast (fp16 mode):
        # its logits only feed KL terms whose log_softmax autocast pins to
        # fp32 — the probe measured this exact path.
        with self._autocast(), torch.no_grad():
            teacher_out = self.teacher.unroll(
                frames, state.teacher, discount=self.config.discount
            )

        # VALUE island: everything from here through the value optimizer
        # step stays entirely fp32 — deliberately OUTSIDE any autocast scope
        # (fp16's weakest probe arm; small compute share). No scaler either:
        # fp32 gradients don't underflow.
        value_out = self.value_function.outputs(
            frames, state.value, discount=self.config.discount
        )
        self.value_optimizer.zero_grad(set_to_none=True)
        value_out.loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value_function.parameters(), float("inf")
        )
        if not torch.isfinite(value_grad_norm):
            print(f"NONFINITE VALUE GRAD NORM ({value_grad_norm}): "
                  "skipping value update", flush=True)
            self.value_optimizer.zero_grad(set_to_none=True)
        else:
            self.value_optimizer.step()

        # Unroll position t (t = 0..T-1) predicts the action sampled at frame
        # t: its actor logits are logits[t], and the sampled action itself is
        # the NEXT entry of the prev-action stream, controller_state[t+1].
        actor_logits = tree.map_structure(lambda t: t[:, :-1], traj.actions.logits)
        actor_actions = tree.map_structure(
            lambda t: t[:, 1:], traj.actions.controller_state
        )
        # Stored fp16 logits (sample-time + teacher) feed the KLs as
        # constants; sanitize rare inf/NaN so the loss stays finite.
        san = lambda t: torch.nan_to_num(
            t, nan=0.0, posinf=self.LOGIT_CLAMP, neginf=-self.LOGIT_CLAMP
        )
        actor_logits = tree.map_structure(san, actor_logits)
        teacher_logits = tree.map_structure(san, teacher_out.logits)
        with self._autocast(), torch.no_grad():
            actor_log_probs = self._ops.log_prob(actor_logits, actor_actions)

        fixed = _Fixed(
            frames=frames,
            initial_policy_state=initial_policy_state,
            advantages=value_out.advantages,
            teacher_logits=teacher_logits,
            actor_logits=actor_logits,
            actor_log_probs=actor_log_probs,
            valid=(~traj.is_resetting[:, 1:]).float(),
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
        # The whole loss runs under the policy autocast (fp16 mode), exactly
        # like the measured probe arm: the unroll's matmuls go fp16 while
        # every sensitive quantity (log-probs, KLs, entropies) is fp32
        # because autocast pins their log_softmax/bce ops to fp32. NOTE the
        # KL math must stay INSIDE the autocast for that pinning — outside
        # it, fp16 logits would flow through log_softmax at fp16.
        with self._autocast():
            return self._policy_loss_inner(fixed)

    LOGIT_CLAMP = 500.0  # real |logit| max ~125; kills only fp16 blowups

    def _policy_loss_inner(self, fixed: _Fixed) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        out = self.policy.unroll(
            fixed.frames, fixed.initial_policy_state, discount=cfg.discount
        )
        # Probe: pre-clamp max |logit| split by validity (NaN counted as
        # inf so it can't hide from max()).
        with torch.no_grad():
            vb = fixed.valid.bool()
            v_maxs, m_maxs = [], []
            for t in tree.flatten(out.logits):
                a = torch.nan_to_num(
                    t.detach().abs(), nan=float("inf"), posinf=float("inf")
                )
                v = vb.reshape(vb.shape + (1,) * (a.dim() - vb.dim())).expand_as(a)
                z = torch.zeros((), dtype=a.dtype, device=a.device)
                v_maxs.append(torch.where(v, a, z).amax())
                m_maxs.append(torch.where(v, z, a).amax())
            logit_absmax_valid = torch.stack(v_maxs).max().item()
            logit_absmax_masked = torch.stack(m_maxs).max().item()
            # advantage-path probe: the remaining suspect for rare
            # NONFINITE LOSS skips with finite logits
            adv = fixed.advantages.detach()
            adv_absmax = torch.nan_to_num(
                adv.abs(), nan=float("inf"), posinf=float("inf")
            ).max().item()
            adv_nonfinite = int((~torch.isfinite(adv)).sum().item())
        # Clamp defuses inf logits at masked positions (0-cotangent x
        # inf-jacobian NaNs backward); inert for real logits.
        out = out._replace(logits=tree.map_structure(
            lambda t: t.clamp(-self.LOGIT_CLAMP, self.LOGIT_CLAMP), out.logits
        ))

        valid = fixed.valid
        vbool = valid.bool()
        n_valid = valid.sum().clamp(min=1.0)
        log_rhos = out.log_probs - fixed.actor_log_probs
        # Masked positions are the reset-substituted targets the actor never
        # sampled; their AR teacher-forcing chains diverge, so |log_rho|
        # there is unbounded. exp() overflows above ~88.7 and the resulting
        # inf poisons the batch through the mask: inf*0 = NaN (loss), or
        # min()'s 0-cotangent x inf-jacobian = NaN (grads, finite loss).
        # where() — not a multiply — so an inf is DISCARDED, not multiplied.
        # Measure the masked tail BEFORE discarding it: this is the
        # quantity that has to exceed ~88.7 for the failure above, so
        # logging it is what confirms the diagnosis in production.
        with torch.no_grad():
            _m = torch.nan_to_num(
                log_rhos.detach().abs(), nan=float("inf"), posinf=float("inf")
            )
            z = torch.zeros((), dtype=_m.dtype, device=_m.device)
            log_rho_masked_absmax = torch.where(vbool, z, _m).max().item()
        log_rhos = torch.where(vbool, log_rhos, torch.zeros_like(log_rhos))
        # Detect on the post-where tensor but BEFORE clamping, and count
        # every position (the old code measured |log_rho|*valid, which is
        # identically 0 at masked positions and so could never see them).
        raw = log_rhos.detach()
        nonfinite = int((~torch.isfinite(raw)).sum().item())
        raw_abs_max = torch.nan_to_num(
            raw.abs(), nan=float("inf"), posinf=float("inf")
        ).max().item()
        anomalies = nonfinite + int(
            (torch.nan_to_num(raw.abs()) > cfg.ppo.log_rho_clamp).sum().item()
        )
        if anomalies:
            self._dump_anomaly(log_rhos, fixed)
        # Bound UNCONDITIONALLY: exp() must never see a value that can
        # overflow, whether or not the anomaly counter fired.
        log_rhos = torch.nan_to_num(
            log_rhos, nan=0.0,
            posinf=cfg.ppo.log_rho_clamp, neginf=-cfg.ppo.log_rho_clamp,
        )
        log_rhos = torch.clamp(
            log_rhos, -cfg.ppo.log_rho_clamp, cfg.ppo.log_rho_clamp
        )
        surrogate = clipped_surrogate(
            log_rhos, fixed.advantages, cfg.ppo.epsilon,
        )

        # Forward KL to the teacher (expectation under the student's states):
        # mode-seeking — refine human play, free to drop human mistakes.
        teacher_kl = self._ops.kl(out.logits, fixed.teacher_logits)
        reverse_teacher_kl = self._ops.kl(fixed.teacher_logits, out.logits)
        actor_kl = self._ops.kl(fixed.actor_logits, out.logits)
        entropy = self._ops.entropy(out.logits)

        # Only weighted terms enter the loss: a zero coefficient must DROP
        # the term, not multiply it (0.0 * inf = NaN would contaminate the
        # loss with a term that has no influence on it). actor_kl and
        # entropy are still computed above for metrics either way.
        per_pos = -cfg.policy_gradient_weight * surrogate
        for w, term in (
            (cfg.ppo.beta, actor_kl),
            (cfg.kl_teacher_weight, teacher_kl),
            (cfg.reverse_kl_teacher_weight, reverse_teacher_kl),
            (-cfg.entropy_weight, entropy),
        ):
            if w != 0.0:
                per_pos = per_pos + w * term
        loss = (per_pos * valid).sum() / n_valid
        # per-term nonfinite counts over VALID positions (one sync): a
        # nonfinite loss names the term it came from
        with torch.no_grad():
            terms = (surrogate, actor_kl, teacher_kl, reverse_teacher_kl,
                     entropy)
            bad = torch.stack([
                ((~torch.isfinite(t)) & valid.bool()).sum().float()
                for t in terms
            ]).tolist()

        vmean = lambda t: ((t * valid).sum() / n_valid).item()
        metrics = {
            "loss": loss.item(),
            "surrogate": vmean(surrogate),
            "teacher_kl": vmean(teacher_kl),
            "actor_kl_mean": vmean(actor_kl),
            "actor_kl_max": (actor_kl * valid).max().item(),
            "entropy": vmean(entropy),
            "ratio_mean": vmean(log_rhos.exp() * valid + (1 - valid)),
            "log_rho_abs_max": raw_abs_max,
            "log_rho_masked_absmax": log_rho_masked_absmax,
            "anomalous_samples": anomalies,
            "logit_absmax_valid": logit_absmax_valid,
            "logit_absmax_masked": logit_absmax_masked,
            "adv_absmax": adv_absmax,
            "adv_nonfinite": adv_nonfinite,
            "nf_surrogate": int(bad[0]),
            "nf_actor_kl": int(bad[1]),
            "nf_teacher_kl": int(bad[2]),
            "nf_reverse_kl": int(bad[3]),
            "nf_entropy": int(bad[4]),
        }
        return loss, metrics

    _anomaly_dumps = 0

    def _dump_anomaly(self, log_rhos: torch.Tensor, fixed: _Fixed) -> None:
        """Forensics for corrupted samples: where in the batch/time, near
        resets?, magnitudes. First 3 occurrences save full tensors."""
        bad = (log_rhos.detach().abs() > self.config.ppo.log_rho_clamp)
        idx = bad.nonzero()[:8].tolist()
        near_reset = fixed.frames.is_resetting.any(dim=1)
        print(f"ANOMALY: {bad.sum().item()} samples |log_rho|>"
              f"{self.config.ppo.log_rho_clamp} at (env,t)={idx}; "
              f"env-has-reset={[bool(near_reset[e]) for e, _ in idx]}")
        if Learner._anomaly_dumps < 3:
            Learner._anomaly_dumps += 1
            import time as _time

            path = f"/tmp/smashbot-anomaly-{int(_time.time())}.pt"
            torch.save(
                {"log_rhos": log_rhos.detach().cpu(),
                 "actor_log_probs": fixed.actor_log_probs.cpu(),
                 "advantages": fixed.advantages.cpu(),
                 "is_resetting": fixed.frames.is_resetting.cpu()},
                path,
            )
            print(f"ANOMALY: dumped {path}")

    # ------------------------------------------------ opponent imitation

    def lambda_at(self, progress: float) -> float:
        """Imitation coefficient at run fraction `progress` in [0, 1]:
        linear decay from imitation_lambda to
        imitation_lambda * imitation_lambda_final_frac."""
        cfg = self.config
        progress = min(max(progress, 0.0), 1.0)
        return cfg.imitation_lambda * (
            1.0 - (1.0 - cfg.imitation_lambda_final_frac) * progress
        )

    def _imitation_fixed(self, traj: Trajectory) -> tp.Optional[_ImitFixed]:
        """Fixed pass for one harvested opponent trajectory: critic update on
        its states (targets = discounted returns G_t along the opponent's
        seat), and detached MARWIL weights w from A = G - V. Returns None
        (trajectory dropped) on nonfinite inputs — anomaly armor."""
        frames = self._frames(traj)
        finite = all(
            bool(torch.isfinite(leaf).all())
            for leaf in tree.flatten(frames)
            if isinstance(leaf, torch.Tensor) and torch.is_floating_point(leaf)
        )
        if not finite:
            print("NONFINITE IMITATION INPUT: dropping trajectory", flush=True)
            return None
        batch_size = traj.rewards.shape[0]
        device = traj.rewards.device
        # Our policy/critic never ran over the opponent's stream during the
        # rollout, so there is no carried recurrent state: start from zeros.
        value_out = self.value_function.outputs(
            frames, self.value_function.initial_state(batch_size, device),
            discount=self.config.discount,
        )
        # The critic trains on these states with G_t targets (same guard as
        # the on-policy value update).
        self.value_optimizer.zero_grad(set_to_none=True)
        value_out.loss.backward()
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value_function.parameters(), float("inf")
        )
        if not torch.isfinite(value_grad_norm):
            print(f"NONFINITE IMITATION VALUE GRAD NORM ({value_grad_norm}): "
                  "skipping value update", flush=True)
            self.value_optimizer.zero_grad(set_to_none=True)
        else:
            self.value_optimizer.step()

        valid = (~traj.is_resetting[:, 1:]).float()
        weights = imitation_weights(
            value_out.advantages, valid,
            self.config.imitation_beta, self.config.imitation_w_cap,
        )
        if not torch.isfinite(weights).all():
            print("NONFINITE IMITATION WEIGHTS: dropping trajectory",
                  flush=True)
            return None
        return _ImitFixed(
            frames=frames, weights=weights, valid=valid, rows=batch_size
        )

    def _imitation_chunk_loss(
        self, imf: _ImitFixed, total_valid: float
    ) -> torch.Tensor:
        """One imitation chunk's share of L_opp = -(w * log pi(a_opp|s))
        averaged over EVERY valid harvested position this step: the chunk's
        masked sum over the step-wide denominator, so accumulating all
        chunks reproduces the full-batch mean exactly. Same teacher-forced
        unroll path (and, in fp16 mode, the same autocast + scaled
        backward) as PPO."""
        batch_size = imf.valid.shape[0]
        with self._autocast():
            out = self.policy.unroll(
                imf.frames,
                self.policy.initial_state(batch_size, imf.valid.device),
                discount=self.config.discount,
            )
            # masked-position NaN-grad armor (see _policy_loss_inner)
            logp = out.log_probs.clamp(-1e4, 0.0)
            return -(imf.weights * logp * imf.valid).sum() / total_valid

    @staticmethod
    def _imit_chunks(imf: _ImitFixed, chunk_rows: int) -> list:
        """Row-range VIEWS of an imitation fixed pass, each <= chunk_rows
        (the PPO chunk size) so no imitation chunk can raise the learner's
        activation peak above what the PPO chunks already set."""
        n = imf.rows
        out = []
        for lo in range(0, n, chunk_rows):
            hi = min(lo + chunk_rows, n)
            take = lambda t: t[lo:hi] if isinstance(t, torch.Tensor) else t
            out.append(_ImitFixed(
                frames=tree.map_structure(take, imf.frames),
                weights=imf.weights[lo:hi], valid=imf.valid[lo:hi],
                rows=hi - lo,
            ))
        return out

    @classmethod
    def _row_chunks(cls, fixed: _Fixed, k: int) -> list:
        """Contiguous row-range VIEWS, not copies (index_select here would
        hold the whole fixed pass on the card twice)."""
        n = fixed.valid.shape[0]
        bounds = [round(j * n / k) for j in range(k + 1)]

        def rng(lo, hi):
            take = lambda t: t[lo:hi] if isinstance(t, torch.Tensor) else t
            return _Fixed(*(tree.map_structure(take, field) for field in fixed))

        return [
            rng(bounds[j], bounds[j + 1])
            for j in range(k) if bounds[j + 1] > bounds[j]
        ]

    @staticmethod
    def _slice_fixed(fixed: _Fixed, rows: tp.Sequence[int]) -> _Fixed:
        """Row (env-dim) subset of a fixed pass (copy)."""
        sel = torch.as_tensor(list(rows), dtype=torch.int64)

        def take(t):
            if isinstance(t, torch.Tensor):
                return t.index_select(0, sel.to(t.device))
            return t

        return _Fixed(*(tree.map_structure(take, field) for field in fixed))

    def _plan_imitation(
        self, imit_trajs: list[Trajectory]
    ) -> tuple[list[_ImitFixed], dict]:
        """All harvested rows this step (or a uniform sample of
        imitation_rows of them when capped): per harvest group, the critic
        update + MARWIL weights. Nothing is substituted out of the PPO
        batch — micro-batching makes the extra rows a time cost, not a
        memory one (see step)."""
        cfg = self.config
        pool = [
            (ti, r) for ti, t in enumerate(imit_trajs)
            for r in range(t.rewards.shape[0])
        ]
        cap = cfg.imitation_rows
        chosen = (
            self._imit_rng.sample(pool, cap) if 0 < cap < len(pool) else pool
        )
        imit_fixed: list[_ImitFixed] = []
        for ti, traj in enumerate(imit_trajs):
            rows = sorted(r for t, r in chosen if t == ti)
            if not rows:
                continue
            if len(rows) < traj.rewards.shape[0]:
                traj = slice_trajectory_rows(traj, rows)
            imf = self._imitation_fixed(traj)
            if imf is not None:
                imit_fixed.append(imf)
        if not imit_fixed:
            return [], {}
        n = sum(imf.valid.sum().clamp(min=1.0) for imf in imit_fixed)
        w_mean = sum(
            (imf.weights * imf.valid).sum() for imf in imit_fixed
        ) / n
        w_max = max(
            (imf.weights * imf.valid).max().item() for imf in imit_fixed
        )
        stats = {
            "traj_count": sum(imf.rows for imf in imit_fixed),
            "w_mean": w_mean.item(),
            "w_max": w_max,
        }
        return imit_fixed, stats

    def step(
        self,
        trajectories: tp.Sequence[Trajectory],
        state: LearnerState,
        progress: float = 0.0,
    ) -> tuple[LearnerState, dict]:
        """One PPO update over a batch of trajectory chunks (minibatches).

        Runs the fixed passes (teacher + value update + advantages) once, then
        ppo.num_epochs gradient passes over all chunks, then a no-grad pass to
        measure post-update actor KL — reverting the update if it moved the
        policy beyond ppo.max_mean_actor_kl.

        Trajectories tagged kind="imitation" feed the opponent-advantage-
        imitation term: EVERY harvested row trains (or a uniform sample of
        imitation_rows of them), on top of the full PPO batch — nothing is
        substituted out. Both terms accumulate over row chunks no larger
        than the PPO micro-batch, so the activation peak is the PPO chunk's
        and extra rows only cost learner time. Ignored while
        imitation_rows == 0. `progress` (run fraction) drives lambda decay.
        """
        cfg = self.config
        ppo_trajs = [
            t for t in trajectories if getattr(t, "kind", "ppo") != "imitation"
        ]
        imit_trajs = [
            t for t in trajectories if getattr(t, "kind", "ppo") == "imitation"
        ]

        fixed_list: list[_Fixed] = []
        value_metrics: list[dict] = []
        for traj in ppo_trajs:
            fixed, state, vm = self._fixed_pass(traj, state)
            fixed_list.append(fixed)
            value_metrics.append(vm)

        imit_fixed: list[_ImitFixed] = []
        imit_stats: dict = {}
        if cfg.imitation_rows != 0 and imit_trajs and fixed_list:
            imit_fixed, imit_stats = self._plan_imitation(imit_trajs)
        lambda_t = self.lambda_at(progress)

        check_fixed = fixed_list  # post-update KL check: full rows, no grad
        train_fixed = fixed_list
        if cfg.micro_batches > 1:
            train_fixed = [
                c for f in fixed_list for c in self._row_chunks(f, cfg.micro_batches)
            ]
        # exact accumulation: each chunk's mean-over-valid loss weighted by its
        # share of all valid positions reproduces the full-batch mean
        total_valid = sum(float(f.valid.sum()) for f in train_fixed) or 1.0
        # imitation chunks are capped at the PPO chunk size (activation peak
        # unchanged) and share one step-wide denominator (exact mean)
        chunk_rows = max(f.valid.shape[0] for f in train_fixed) if train_fixed else 1
        imit_chunks = [c for imf in imit_fixed for c in self._imit_chunks(imf, chunk_rows)]
        total_imit_valid = sum(float(c.valid.sum()) for c in imit_chunks) or 1.0

        snapshot = copy.deepcopy(self.policy.state_dict())

        epoch_metrics: list[dict] = []
        imit_loss_val = 0.0
        for _ in range(cfg.ppo.num_epochs):
            self.policy_optimizer.zero_grad(set_to_none=True)
            any_backward = False
            batch_metrics = []
            for fixed in train_fixed:
                loss, metrics = self._policy_loss(fixed)
                if not torch.isfinite(loss):
                    print(
                        "NONFINITE LOSS: skipping minibatch (pre-clamp "
                        f"|logit| valid {metrics['logit_absmax_valid']:.1f} "
                        f"masked {metrics['logit_absmax_masked']:.1f} "
                        f"|adv| {metrics['adv_absmax']:.2f} "
                        f"adv_nf {metrics['adv_nonfinite']} | terms nf: "
                        f"surr {metrics['nf_surrogate']} "
                        f"akl {metrics['nf_actor_kl']} "
                        f"tkl {metrics['nf_teacher_kl']} "
                        f"rkl {metrics['nf_reverse_kl']} "
                        f"ent {metrics['nf_entropy']})",
                        flush=True,
                    )
                    batch_metrics.append(metrics)
                    # free the skipped chunk's graph before the next forward
                    # (else the live activation footprint doubles)
                    del loss
                    continue
                self._backward(loss * (float(fixed.valid.sum()) / total_valid))
                any_backward = True
                batch_metrics.append(metrics)
            # Stage tag for the grad guard: PPO and imitation backwards
            # accumulate into the SAME grads, so a NaN born in either
            # prints the same message. Snapshot finiteness between them
            # (one sync) to attribute the next event.
            ppo_grad_nonfinite = None
            if imit_chunks and lambda_t > 0.0 and any_backward:
                flags = [
                    (~torch.isfinite(p_.grad)).any()
                    for p_ in self.policy.parameters() if p_.grad is not None
                ]
                ppo_grad_nonfinite = bool(
                    torch.stack(flags).any().item()
                ) if flags else False
            if imit_chunks and lambda_t > 0.0:
                imit_losses = []
                for chunk in imit_chunks:
                    iloss = self._imitation_chunk_loss(chunk, total_imit_valid)
                    if not torch.isfinite(iloss):
                        print("NONFINITE IMITATION LOSS: skipping minibatch",
                              flush=True)
                        del iloss  # same graph release as above
                        continue
                    self._backward(lambda_t * iloss)
                    any_backward = True
                    imit_losses.append(iloss.item())
                if imit_losses:
                    imit_loss_val = sum(imit_losses)  # = step-wide mean
            use_scaler = self.grad_scaler is not None and any_backward
            if use_scaler:
                # Divide the loss scale back out BEFORE clipping/guarding so
                # (a) clip_grad_norm_ operates on true magnitudes and (b)
                # the nonfinite guard below reads honest numbers.
                self.grad_scaler.unscale_(self.policy_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"),
            )
            if not torch.isfinite(grad_norm):
                # A finite loss can still yield nonfinite gradients;
                # clip_grad_norm_ does not sanitize NaN. Skip the update.
                # inf vs nan discriminates the cause: fp16 OVERFLOW at this
                # scale produces inf (halving is the right response), bad
                # math produces nan (halving is useless — scale-independent).
                n_inf = n_nan = 0
                first = ""
                for nm, p_ in self.policy.named_parameters():
                    if p_.grad is None:
                        continue
                    gi = int(torch.isinf(p_.grad).sum().item())
                    gn = int(torch.isnan(p_.grad).sum().item())
                    if (gi or gn) and not first:
                        first = nm
                    n_inf += gi
                    n_nan += gn
                stage = (
                    "?" if ppo_grad_nonfinite is None
                    else ("ppo" if ppo_grad_nonfinite else "imitation")
                )
                print(f"NONFINITE GRAD NORM ({grad_norm}): skipping update "
                      f"(inf {n_inf} nan {n_nan} first={first} "
                      f"stage={stage})", flush=True)
                self.policy_optimizer.zero_grad(set_to_none=True)
                if use_scaler:
                    # unscale_ already recorded found_inf, so update() halves
                    # the scale — the right response whether the cause was
                    # fp16 overflow at this scale or genuinely bad math.
                    self.grad_scaler.update()
            elif use_scaler:
                # Two layers of skip, same semantics: our guard above catches
                # every nonfinite gradient FIRST (any inf/NaN element makes
                # the global norm nonfinite), and scaler.step's own internal
                # found_inf skip backstops it. Either way weights only move
                # on finite, unscaled, clipped gradients.
                self.grad_scaler.step(self.policy_optimizer)
                self.grad_scaler.update()
            else:
                self.policy_optimizer.step()
            epoch_metrics.append(_mean_dicts(batch_metrics))

        # Post-update measurement (and trust-region backstop).
        with torch.no_grad():
            post = _mean_dicts([self._policy_loss(f)[1] for f in check_fixed])
        # FAIL CLOSED: NaN compares False, so a contaminated measurement
        # would silently keep an update the trust region never cleared.
        post_kl = post["actor_kl_mean"]
        reverted = (
            not math.isfinite(post_kl) or post_kl > cfg.ppo.max_mean_actor_kl
        )
        if reverted and not math.isfinite(post_kl):
            print(f"NONFINITE POST-UPDATE actor_kl ({post_kl}): reverting",
                  flush=True)
        if reverted:
            self.policy.load_state_dict(snapshot)

        metrics = {
            "epochs": epoch_metrics,
            "post_update": post,
            "value": _mean_dicts(value_metrics),
            "reverted": reverted,
        }
        if imit_stats:
            metrics["imitation"] = dict(
                imit_stats, loss=imit_loss_val, **{"lambda": lambda_t}
            )
        return state, metrics


def _mean_dicts(dicts: tp.Sequence[dict]) -> dict:
    out = {}
    for key in dicts[0]:
        vals = [d[key] for d in dicts]
        if key in ("actor_kl_max", "log_rho_abs_max", "log_rho_masked_absmax",
                   "logit_absmax_valid", "logit_absmax_masked",
                   "adv_absmax"):
            out[key] = max(vals)
        elif key in ("anomalous_samples", "adv_nonfinite", "nf_surrogate",
                     "nf_actor_kl", "nf_teacher_kl", "nf_reverse_kl",
                     "nf_entropy"):
            out[key] = sum(vals)
        else:
            out[key] = sum(vals) / len(vals)
    return out
