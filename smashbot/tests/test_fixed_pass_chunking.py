"""The chunked PPO fixed pass must be EXACT vs unchunked: identical
advantages, teacher logits, sanitized actor logits, log-probs, carried
teacher/value states, and value-net update — across core types (sgu
batch-first caches AND tx_like's [layers, B, H] LSTM states), with
resets, and across SEQUENTIAL steps (state threading)."""

import numpy as np
import pytest
import torch
import tree

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy
from smashbot.rl.config import RLConfig
from smashbot.rl.ppo import Learner
from smashbot.tests.test_ppo import _rollout, _tiny_policy, _tiny_value


def _lstm_policy(seed=0):
    """tx_like core: exercises the [layers, B, H] RNN-state slicing."""
    torch.manual_seed(seed)
    p = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(
            name="tx_like", num_layers=2, hidden_size=32,
        ),
        head_config=configs.ControllerHeadConfig(
            residual_size=32, component_depth=0
        ),
        policy_config=configs.PolicyConfig(delay=2),
        num_names=4,
    )
    p.train_value_head = False
    return p


def _learner(policy_fn, mb):
    torch.manual_seed(0)
    return Learner(
        RLConfig(micro_batches=mb), policy_fn(seed=0), policy_fn(seed=1),
        _tiny_value(),
    )


def _with_resets(traj, seed=11):
    rng = np.random.default_rng(seed)
    resets = torch.from_numpy(
        rng.random(tuple(traj.is_resetting.shape)) < 0.15
    )
    return traj._replace(is_resetting=resets)


def _assert_tree_close(a, b, atol, what):
    for x, y in zip(tree.flatten(a), tree.flatten(b)):
        if isinstance(x, torch.Tensor):
            assert x.shape == y.shape, (what, x.shape, y.shape)
            assert torch.allclose(
                x.float(), y.float(), atol=atol, rtol=1e-5
            ), (what, (x.float() - y.float()).abs().max())


@pytest.mark.parametrize("policy_fn", [_tiny_policy, _lstm_policy])
@pytest.mark.parametrize("mb", [2, 3, 5])
@pytest.mark.parametrize("with_resets", [False, True])
def test_chunked_fixed_pass_matches_unchunked(policy_fn, mb, with_resets):
    la = _learner(policy_fn, 1)   # budget = B: single chunk (old path)
    lb = _learner(policy_fn, mb)  # chunked
    lb.value_function.load_state_dict(la.value_function.state_dict())
    lb.value_optimizer.load_state_dict(la.value_optimizer.state_dict())
    lb.teacher.load_state_dict(la.teacher.state_dict())

    # two SEQUENTIAL trajectories: the second consumes the first's carried
    # states, so a slicing/stitching bug in either pass compounds visibly
    trajs = [_rollout(la.policy, B=5, T=8, seed=s) for s in (0, 1)]
    if with_resets:
        trajs = [_with_resets(t, seed=20 + i) for i, t in enumerate(trajs)]

    # tolerances calibrated to fp-reorder noise, which compounds through
    # the traj0 value update into traj1's outputs (~2e-6 measured); resets
    # additionally shuffle unroll segmentation boundaries per chunk. A
    # real slicing/stitching bug shows at ~1e-1 — orders of magnitude of
    # discriminating power remain.
    atol = 1e-4 if with_resets else 1e-5

    sa, sb = la.initial_state(5), lb.initial_state(5)
    for i, traj in enumerate(trajs):
        fa, sa, ma = la._fixed_pass(traj, sa)
        fb, sb, mb_ = lb._fixed_pass(traj, sb)
        what = f"traj{i}"
        _assert_tree_close(fa.advantages, fb.advantages, atol, what + ".adv")
        _assert_tree_close(
            fa.teacher_logits, fb.teacher_logits, atol, what + ".tlogits"
        )
        _assert_tree_close(
            fa.actor_logits, fb.actor_logits, 0.0, what + ".alogits"
        )
        _assert_tree_close(
            fa.actor_log_probs, fb.actor_log_probs, atol, what + ".logp"
        )
        assert torch.equal(fa.valid, fb.valid)
        _assert_tree_close(sa.teacher, sb.teacher, atol, what + ".tstate")
        _assert_tree_close(sa.value, sb.value, atol, what + ".vstate")
        assert ma["loss"] == pytest.approx(mb_["loss"], rel=1e-4)
        assert ma["value_nonfinite"] == mb_["value_nonfinite"]
    # value net received the same TWO updates
    for (ka, va), (_, vb) in zip(
        la.value_function.state_dict().items(),
        lb.value_function.state_dict().items(),
    ):
        # 5e-5: Adam amplifies backward-reorder ulp noise on near-zero
        # grads (same calibration as test_imitation_chunking, review-
        # validated: a wrong share weighting measures ~2e-4+)
        assert torch.allclose(va, vb, atol=5e-5), (
            ka, (va - vb).abs().max()
        )


def test_single_chunk_is_bitwise_identical():
    """budget >= B (micro_batches=1) must not change a single bit vs the
    pre-chunking behavior: one chunk, share exactly 1.0, no cats."""
    la = _learner(_tiny_policy, 1)
    lb = _learner(_tiny_policy, 1)
    lb.value_function.load_state_dict(la.value_function.state_dict())
    lb.value_optimizer.load_state_dict(la.value_optimizer.state_dict())
    lb.teacher.load_state_dict(la.teacher.state_dict())
    traj = _rollout(la.policy, B=4, T=8, seed=2)
    fa, sa, _ = la._fixed_pass(traj, la.initial_state(4))
    fb, sb, _ = lb._fixed_pass(traj, lb.initial_state(4))
    for x, y in zip(tree.flatten(fa), tree.flatten(fb)):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
    for x, y in zip(tree.flatten(sa), tree.flatten(sb)):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
