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
from smashbot.training import GradClipper
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
    # seats of self-play envs) or "imitation" (a harvested opponent seat as
    # a replay, rollouts.HarvestAssembler: what it pressed, aligned to the
    # student's delay; no logits, initial_state None).
    kind: str = "ppo"
    # imitation only, [B, T]: positions whose target was pressed in their own game
    valid: tp.Optional[torch.Tensor] = None


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
        valid=None if traj.valid is None else take(traj.valid),
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
    reset0: torch.Tensor  # [B] bool: the chunk starts a fresh game (the
    # policy's initial state is masked to zeros per micro-batch chunk, not
    # for the whole batch up front — two full-batch state copies otherwise)


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
        # growth_interval is in scaler steps: torch's 2000 default assumes
        # ~10 steps/s; at our step rate a doubling would take ~16h and any
        # sparse fault rate would pin the scale down forever.
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
        self.policy_clipper = GradClipper(
            policy.parameters(), config.max_grad_norm, config.autoclip_percentile
        )
        self.value_optimizer = torch.optim.Adam(
            value_function.parameters(), lr=config.learning_rate
        )
        self._ops = _StructOps(policy.controller_head.controller_embedding)
        # Imitation row-cap sampling RNG; seeded for reproducibility,
        # reseedable in tests.
        self._imit_rng = random.Random(0)
        # Current teacher-KL leash weight; refreshed from the decay
        # schedule at each step() (constant when decay is disabled).
        # Initialized here so direct _policy_loss calls (tests) work.
        self._kl_teacher_w = config.kl_teacher_weight
        # Persistent trust-region snapshot buffers (see step): tensor
        # storages allocated once and copied into per step.
        self._snap_buffers: dict = {}

    def set_learning_rate(self, lr: float) -> set:
        """Point both optimizers at lr, leaving Adam's moments and step
        counts as they are; returns the rates they had."""
        groups = [g for opt in (self.policy_optimizer, self.value_optimizer) for g in opt.param_groups]
        before = {g["lr"] for g in groups}
        for group in groups:
            group["lr"] = lr
        return before

    def _snap_into(self, key: str, src):
        """Deep-copy `src` (a state dict) to CPU, REUSING the tensor
        storages from this key's previous snapshot wherever shapes/dtypes
        still match — the content is identical to a fresh _to_cpu, but
        ~1.3GB/step of host tensor churn becomes zero. Dict/list shells
        and non-tensor leaves (param_groups scalars) are rebuilt fresh
        each step (tiny). Structure changes (e.g. Adam state populating
        after the first optimizer step) fall back to fresh allocation for
        the changed leaves only. Aliasing: policy.load_state_dict copies
        out of the buffer, but Optimizer.load_state_dict keeps matching
        tensors by REFERENCE — the revert path therefore surrenders the
        opt buffer after restoring from it. Nothing else retains a
        snapshot across steps."""

        def into(dst, s):
            if isinstance(s, torch.Tensor):
                s = s.detach()
                if (isinstance(dst, torch.Tensor) and dst.shape == s.shape
                        and dst.dtype == s.dtype):
                    dst.copy_(s)
                    return dst
                return s.to("cpu", copy=True)
            if isinstance(s, dict):
                d = dst if isinstance(dst, dict) else {}
                return {k: into(d.get(k), v) for k, v in s.items()}
            if isinstance(s, (list, tuple)):
                d = (
                    list(dst)
                    if isinstance(dst, (list, tuple)) and len(dst) == len(s)
                    else [None] * len(s)
                )
                return type(s)(into(a, b) for a, b in zip(d, s))
            return copy.deepcopy(s)

        out = into(self._snap_buffers.get(key), src)
        self._snap_buffers[key] = out
        return out

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

    @staticmethod
    def _rows_take(struct, lo, hi, n):
        """Row-range VIEWS of a per-row struct: batch is dim 0 everywhere
        EXCEPT torch RNN states ([layers, B, H]) — the same disambiguation
        _mask_state/_row_chunks use."""
        def take(t):
            if not isinstance(t, torch.Tensor):
                return t
            if t.dim() >= 1 and t.shape[0] == n:
                return t[lo:hi]
            if t.dim() >= 2 and t.shape[1] == n:
                return t[:, lo:hi]
            return t
        return tree.map_structure(take, struct)

    @staticmethod
    def _rows_cat(structs, ref, n):
        """Stitch per-chunk structs back to full rows: each leaf concats
        along the batch dim of its FULL-BATCH reference leaf (chunk shapes
        alone are ambiguous — a 1-row chunk of a [layers, B, H] RNN state
        looks batch-first). Leaves whose reference has no batch dim must
        agree across chunks and pass through. Single chunk passes through
        untouched (keeps views)."""
        if len(structs) == 1:
            return structs[0]

        def cat(rf, *leaves):
            if not isinstance(leaves[0], torch.Tensor):
                return leaves[0]
            if rf.dim() >= 1 and rf.shape[0] == n:
                return torch.cat(leaves, dim=0)
            if rf.dim() >= 2 and rf.shape[1] == n:
                return torch.cat(leaves, dim=1)
            for lf in leaves[1:]:
                assert torch.equal(lf, leaves[0]), (
                    "batchless state leaf differs across row chunks"
                )
            return leaves[0]
        return tree.map_structure(cat, ref, *structs)

    def _fixed_pass(
        self, traj: Trajectory, state: LearnerState
    ) -> tuple[_Fixed, LearnerState, dict]:
        """Everything reusable across epochs: teacher logits, advantages (with
        a value-net update), and the actor's own log-probs — plus carried
        recurrent states.

        The expensive forwards (teacher unroll, value fwd+bwd, log-probs)
        run in row chunks of ceil(B / micro_batches): the unchunked fp32
        value backward over the full batch was the learner's largest
        constant VRAM block (micro_batches never touched it). Chunking is
        exact — rows are independent through every core, value-loss chunks
        accumulate weighted by their ROW share (the loss is a plain mean
        and T is constant across row chunks), the value optimizer steps
        ONCE per trajectory after all chunks, and per-row outputs/carried
        states are stitched back in row order. Merged metrics: counts sum,
        absmax takes max, means are row-weighted (uev approximately — its
        variance denominator is per-chunk; diagnostics only)."""
        frames = self._frames(traj)
        batch_size = traj.rewards.shape[0]

        # Actors reset mid-rollout invisibly to the learner; the policy's
        # carried state is masked back to zeros wherever a chunk starts
        # fresh — per micro-batch chunk, in _policy_loss_inner
        initial_policy_state = traj.initial_state

        budget = -(-batch_size // max(1, self.config.micro_batches))
        bounds = list(range(0, batch_size, budget)) + [batch_size]
        chunk_rows = [
            bounds[j + 1] - bounds[j] for j in range(len(bounds) - 1)
        ]
        self.value_optimizer.zero_grad(set_to_none=True)
        t_logits, t_states = [], []
        advantages, v_states, v_metrics = [], [], []
        a_logits, a_logps = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            ctraj = self._rows_take(traj, lo, hi, batch_size)
            cframes = self._frames(ctraj)
            # Frozen-teacher forward follows the policy autocast (fp16
            # mode): its logits only feed KL terms whose log_softmax
            # autocast pins to fp32 — the probe measured this exact path.
            with self._autocast(), torch.no_grad():
                teacher_out = self.teacher.unroll(
                    cframes,
                    self._rows_take(state.teacher, lo, hi, batch_size),
                )
            # VALUE island: everything from here through the value
            # optimizer step stays entirely fp32 — deliberately OUTSIDE
            # any autocast scope (fp16's weakest probe arm; small compute
            # share). No scaler either: fp32 gradients don't underflow.
            value_out = self.value_function.outputs(
                cframes,
                self._rows_take(state.value, lo, hi, batch_size),
                discount=self.config.discount,
            )
            (value_out.loss * ((hi - lo) / batch_size)).backward()

            # Unroll position t (t = 0..T-1) predicts the action sampled
            # at frame t: its actor logits are logits[t], and the sampled
            # action itself is the NEXT entry of the prev-action stream,
            # controller_state[t+1]. Stored fp16 logits (sample-time +
            # teacher) feed the KLs as constants; sanitize rare inf/NaN so
            # the loss stays finite.
            san = lambda t: torch.nan_to_num(
                t, nan=0.0, posinf=self.LOGIT_CLAMP, neginf=-self.LOGIT_CLAMP
            )
            ca_logits = tree.map_structure(
                lambda t: san(t[:, :-1]), ctraj.actions.logits
            )
            ca_actions = tree.map_structure(
                lambda t: t[:, 1:], ctraj.actions.controller_state
            )
            with self._autocast(), torch.no_grad():
                a_logps.append(self._ops.log_prob(ca_logits, ca_actions))
            a_logits.append(ca_logits)
            t_logits.append(tree.map_structure(san, teacher_out.logits))
            t_states.append(teacher_out.final_state)
            advantages.append(value_out.advantages)
            # detach HERE, not after the loop: a graph-attached final_state
            # would keep parts of this chunk's backward graph alive across
            # the remaining chunks — the exact pinning this pass exists to
            # prevent
            v_states.append(tree.map_structure(
                lambda t: t.detach() if isinstance(t, torch.Tensor) else t,
                value_out.final_state,
            ))
            v_metrics.append(value_out.metrics)

        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value_function.parameters(), float("inf")
        )
        if not torch.isfinite(value_grad_norm):
            print(f"NONFINITE VALUE GRAD NORM ({value_grad_norm}): "
                  "skipping value update", flush=True)
            self.value_optimizer.zero_grad(set_to_none=True)
        else:
            self.value_optimizer.step()

        metrics = dict(v_metrics[0])
        if len(v_metrics) > 1:
            for k in metrics:
                vals = [m[k] for m in v_metrics]
                if k.endswith("_nonfinite"):
                    metrics[k] = sum(vals)
                elif k.endswith("_absmax"):
                    metrics[k] = max(vals)
                else:  # loss / uev / return_mean / reward_mean
                    metrics[k] = sum(
                        v * r for v, r in zip(vals, chunk_rows)
                    ) / batch_size

        # logits/log-probs/advantages are [B, T, ...] slices of batch-first
        # trees: plain dim-0 concat. Only the recurrent STATES need the
        # reference-based batch-dim disambiguation.
        cat0 = (
            (lambda seq: seq[0]) if len(chunk_rows) == 1
            else (lambda seq: tree.map_structure(
                lambda *xs: torch.cat(xs, dim=0), *seq
            ))
        )
        fixed = _Fixed(
            frames=frames,
            initial_policy_state=initial_policy_state,
            advantages=cat0(advantages),
            teacher_logits=cat0(t_logits),
            actor_logits=cat0(a_logits),
            actor_log_probs=cat0(a_logps),
            valid=(~traj.is_resetting[:, 1:]).float(),
            reset0=traj.is_resetting[:, 0],
        )
        # Detach carried recurrent states: the next chunk's backward must not
        # reach into this chunk's (already-freed) graph.
        detach = lambda t: t.detach() if isinstance(t, torch.Tensor) else t
        new_state = LearnerState(
            teacher=tree.map_structure(
                detach, self._rows_cat(t_states, state.teacher, batch_size)
            ),
            value=tree.map_structure(
                detach, self._rows_cat(v_states, state.value, batch_size)
            ),
        )
        return fixed, new_state, metrics

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
        rows = fixed.valid.shape[0]
        init = _mask_state(
            fixed.reset0,
            self.policy.initial_state(rows, fixed.valid.device),
            fixed.initial_policy_state,
        )
        out = self.policy.unroll(fixed.frames, init)
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
            # advantage stream: the one loss input with no clamp of its own
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
        # Masked positions have unbounded |log_rho| (diverged AR chains);
        # exp() overflows past ~88.7 and inf reaches the batch through the
        # mask (inf*0 or min()'s 0-cotangent x inf-jacobian). where(), not
        # a multiply: an inf must be discarded, 0*inf is still NaN.
        # Measure the masked tail before discarding it (the quantity that
        # must exceed ~88.7 for the failure above).
        with torch.no_grad():
            _m = torch.nan_to_num(
                log_rhos.detach().abs(), nan=float("inf"), posinf=float("inf")
            )
            z = torch.zeros((), dtype=_m.dtype, device=_m.device)
            log_rho_masked_absmax = torch.where(vbool, z, _m).max().item()
        log_rhos = torch.where(vbool, log_rhos, torch.zeros_like(log_rhos))
        # Detect on every position, before clamping.
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
        # Bound unconditionally: exp() must never see an overflowable value.
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

        # A zero coefficient must DROP its term, not multiply it:
        # 0.0 * inf = NaN. (All terms still computed above for metrics.)
        per_pos = -cfg.policy_gradient_weight * surrogate
        for w, term in (
            (cfg.ppo.beta, actor_kl),
            (self._kl_teacher_w, teacher_kl),
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

    def kl_teacher_weight_at(self, progress: float) -> float:
        """Teacher-KL leash coefficient at run fraction `progress`: linear
        from kl_teacher_weight to kl_teacher_weight_final; constant (the
        historical behavior) while the final is negative."""
        cfg = self.config
        if cfg.kl_teacher_weight_final < 0:
            return cfg.kl_teacher_weight
        p = min(max(progress, 0.0), 1.0)
        return (
            cfg.kl_teacher_weight
            + (cfg.kl_teacher_weight_final - cfg.kl_teacher_weight) * p
        )

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

    def _imitation_fixed(
        self, traj: Trajectory, row_budget: int = 0,
    ) -> tp.Optional[_ImitFixed]:
        """Fixed pass for one harvested opponent trajectory: critic update on
        its states (targets = discounted returns G_t along the opponent's
        seat), and detached MARWIL weights w from A = G - V. Returns None
        (trajectory dropped) on nonfinite inputs — anomaly armor.

        The critic forward+backward runs in row chunks of `row_budget`
        (<=0: whole trajectory): harvest volume varies per step, and an
        unchunked fp32 backward over a max-harvest step is what set the
        learner's VRAM high-water mark (micro_batches never touched this
        pass). Chunking is exact: rows are independent through the value
        net, chunk losses accumulate weighted by their ROW share (the
        loss is a plain mean over all positions), the optimizer steps ONCE per
        trajectory after all chunks, and the MARWIL normalization runs on
        the CONCATENATED advantages — identical weights to the unchunked
        pass."""
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
        valid = traj.valid.float()
        step_rows = (
            batch_size if row_budget <= 0 else min(row_budget, batch_size)
        )
        # Our policy/critic never ran over the opponent's stream during the
        # rollout, so there is no carried recurrent state: start from zeros.
        self.value_optimizer.zero_grad(set_to_none=True)
        adv_chunks = []
        for lo in range(0, batch_size, step_rows):
            hi = min(lo + step_rows, batch_size)
            cf = tree.map_structure(
                lambda t: t[lo:hi] if isinstance(t, torch.Tensor) else t,
                frames,
            )
            value_out = self.value_function.outputs(
                cf, self.value_function.initial_state(hi - lo, device),
                discount=self.config.discount, detail=False,
            )
            # chunk share of the full-trajectory mean loss (a plain .mean()
            # over ALL positions — see value.py — so the share is the ROW
            # fraction): accumulating these reproduces the unchunked
            # gradient exactly
            share = (hi - lo) / batch_size
            (value_out.loss * share).backward()
            adv_chunks.append(value_out.advantages)
        # The critic trains on these states with G_t targets (same guard as
        # the on-policy value update).
        value_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.value_function.parameters(), float("inf")
        )
        if not torch.isfinite(value_grad_norm):
            print(f"NONFINITE IMITATION VALUE GRAD NORM ({value_grad_norm}): "
                  "skipping value update", flush=True)
            self.value_optimizer.zero_grad(set_to_none=True)
        else:
            self.value_optimizer.step()

        weights = imitation_weights(
            torch.cat(adv_chunks, dim=0), valid,
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

        def take(t, lo, hi):
            if not isinstance(t, torch.Tensor):
                return t
            # Batch is dim 0 everywhere EXCEPT torch RNN states, which are
            # [layers, B, H] — same disambiguation _mask_state uses. Slicing
            # dim 0 there would hand every chunk the full state and give the
            # last chunks an empty one.
            if t.dim() >= 1 and t.shape[0] == n:
                return t[lo:hi]
            if t.dim() >= 2 and t.shape[1] == n:
                return t[:, lo:hi]
            return t

        def rng(lo, hi):
            return _Fixed(*(
                tree.map_structure(lambda t: take(t, lo, hi), field)
                for field in fixed
            ))

        return [
            rng(bounds[j], bounds[j + 1])
            for j in range(k) if bounds[j + 1] > bounds[j]
        ]


    def _plan_imitation(
        self, imit_trajs: list[Trajectory], row_budget: int = 0,
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
            imf = self._imitation_fixed(traj, row_budget)
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
            # the imitation critic pass chunks to the PPO micro-batch's row
            # size, so a max-harvest step can never raise the VRAM peak
            # above what the PPO batch already sets
            budget = -(-max(
                f.valid.shape[0] for f in fixed_list
            ) // max(1, cfg.micro_batches))
            imit_fixed, imit_stats = self._plan_imitation(imit_trajs, budget)
        lambda_t = self.lambda_at(progress)
        self._kl_teacher_w = self.kl_teacher_weight_at(progress)

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

        # Trust-region snapshot: weights AND optimizer slots (weights
        # alone leave Adam's m/v carrying the rejected update). Copied into
        # PERSISTENT reusable buffers — allocating fresh host tensors every
        # step churns the allocator's heap unboundedly.
        snapshot = self._snap_into("policy", self.policy.state_dict())
        opt_snapshot = self._snap_into(
            "opt", self.policy_optimizer.state_dict()
        )

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
            # PPO and imitation backwards accumulate into the same grads;
            # snapshot finiteness between them so the guard can attribute.
            ppo_had_backward = any_backward
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
            grad_norm = self.policy_clipper.measure()
            if not math.isfinite(grad_norm):
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
                # the snapshot decides; with one backward there is no
                # ambiguity
                if ppo_grad_nonfinite is not None:
                    stage = "ppo" if ppo_grad_nonfinite else "imitation"
                else:
                    stage = "ppo" if ppo_had_backward else "imitation"
                print(f"NONFINITE GRAD NORM ({grad_norm}): skipping update "
                      f"(inf {n_inf} nan {n_nan} first={first} "
                      f"stage={stage})", flush=True)
                self.policy_optimizer.zero_grad(set_to_none=True)
                for m in batch_metrics:
                    m.update(grad_norm=grad_norm, clip_norm=self.policy_clipper.threshold)
                if use_scaler:
                    # unscale_ already recorded found_inf, so update() halves
                    # the scale — the right response whether the cause was
                    # fp16 overflow at this scale or genuinely bad math.
                    self.grad_scaler.update()
            else:
                clip = self.policy_clipper.clip(grad_norm)
                for m in batch_metrics:
                    m.update(clip)
                if use_scaler:
                    # Two layers of skip, same semantics: our guard above
                    # catches every nonfinite gradient FIRST (any inf/NaN
                    # element makes the global norm nonfinite), and
                    # scaler.step's own internal found_inf skip backstops
                    # it. Either way weights only move on finite, unscaled,
                    # clipped gradients.
                    self.grad_scaler.step(self.policy_optimizer)
                    self.grad_scaler.update()
                else:
                    self.policy_optimizer.step()
            epoch_metrics.append(_mean_dicts(batch_metrics))

        # Post-update measurement (and trust-region backstop).
        with torch.no_grad():
            post = _mean_dicts([self._policy_loss(f)[1] for f in check_fixed])
        # FAIL CLOSED: NaN > x is False — a contaminated measurement must
        # revert, not silently keep the update.
        post_kl = post["actor_kl_mean"]
        reverted = (
            not math.isfinite(post_kl) or post_kl > cfg.ppo.max_mean_actor_kl
        )
        if reverted and not math.isfinite(post_kl):
            print(f"NONFINITE POST-UPDATE actor_kl ({post_kl}): reverting",
                  flush=True)
        if reverted:
            self.policy.load_state_dict(snapshot)
            self.policy_optimizer.load_state_dict(opt_snapshot)
            # Optimizer.load_state_dict does NOT copy tensors whose
            # dtype+device already match — the live Adam state would then
            # ALIAS the reused buffer. Surrender it so the next step
            # fresh-allocates and a later revert restores the correct state.
            self._snap_buffers.pop("opt", None)

        metrics = {
            "epochs": epoch_metrics,
            "post_update": post,
            "value": _mean_dicts(value_metrics),
            "reverted": reverted,
        }
        # surface the (possibly decaying) leash weight beside teacher_kl
        post["kl_teacher_w"] = float(self._kl_teacher_w)
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
                   "adv_absmax", "reward_absmax", "value_absmax",
                   "target_absmax"):
            out[key] = max(vals)
        elif key in ("anomalous_samples", "adv_nonfinite", "nf_surrogate",
                     "nf_actor_kl", "nf_teacher_kl", "nf_reverse_kl",
                     "nf_entropy", "reward_nonfinite", "value_nonfinite",
                     "target_nonfinite"):
            out[key] = sum(vals)
        else:
            out[key] = sum(vals) / len(vals)
    return out
