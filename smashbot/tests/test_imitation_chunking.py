"""The chunked imitation critic pass (_imitation_fixed row_budget) must be
EXACT: identical MARWIL weights, identical value-net update, identical
_ImitFixed contents vs the unchunked pass — chunking bounds VRAM, never
changes math."""

import copy

import pytest
import torch
import tree

from smashbot.rl.config import RLConfig
from smashbot.rl.ppo import Learner
from smashbot.tests.test_ppo import _rollout, _tiny_policy, _tiny_value


def _make_learner(seed=0, **cfg_kwargs):
    torch.manual_seed(seed)
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    value = _tiny_value()
    return Learner(RLConfig(**cfg_kwargs), policy, teacher, value)


def _imit_traj(policy, B, seed):
    traj = _rollout(policy, B=B, T=8, seed=seed)
    return traj._replace(kind="imitation", valid=~traj.is_resetting[:, 1:])


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_chunked_imitation_fixed_is_exact(budget):
    """weights, valid, frames, AND the post-update value net all match the
    unchunked pass bitwise-or-eps for every chunking granularity (B=5:
    budgets cover uneven tails and single-row chunks)."""
    la = _make_learner(imitation_rows=-1)
    lb = _make_learner(imitation_rows=-1)
    lb.value_function.load_state_dict(la.value_function.state_dict())
    lb.value_optimizer.load_state_dict(la.value_optimizer.state_dict())

    traj = _imit_traj(la.policy, B=5, seed=3)
    imf_a = la._imitation_fixed(copy.deepcopy(traj), 0)       # unchunked
    imf_b = lb._imitation_fixed(copy.deepcopy(traj), budget)  # chunked

    assert imf_a is not None and imf_b is not None
    assert torch.allclose(imf_a.weights, imf_b.weights, atol=1e-6), (
        (imf_a.weights - imf_b.weights).abs().max()
    )
    assert torch.equal(imf_a.valid, imf_b.valid)
    for x, y in zip(tree.flatten(imf_a.frames), tree.flatten(imf_b.frames)):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
    # the critic UPDATE is the same update (chunk-share accumulation is the
    # full-batch gradient; optimizer stepped once in both)
    for (ka, va), (kb, vb) in zip(
        la.value_function.state_dict().items(),
        lb.value_function.state_dict().items(),
    ):
        assert ka == kb
        assert torch.allclose(va, vb, atol=1e-6), (ka, (va - vb).abs().max())


def test_step_budget_derivation_matches_micro_batches():
    """step() hands _plan_imitation ceil(ppo_rows / micro_batches)."""
    learner = _make_learner(imitation_rows=-1, micro_batches=2)
    seen = {}
    orig = learner._plan_imitation

    def spy(trajs, row_budget=0):
        seen["budget"] = row_budget
        return orig(trajs, row_budget)

    learner._plan_imitation = spy
    ppo = _rollout(learner.policy, B=5, T=8, seed=0)
    imit = _imit_traj(learner.policy, B=4, seed=7)
    state = learner.initial_state(5)
    learner.step([ppo, imit], state)
    assert seen["budget"] == 3  # ceil(5 / 2)
