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


def test_burn_in_warms_policy_and_critic_like_one_unbroken_unroll():
    """A chunk warmed on its prefix scores exactly as if both networks had run
    straight through prefix + chunk: the policy's log-probs (from the state
    _imit_chunks warms once per step) and the critic's advantages behind the
    MARWIL weights (warmed before its value pass). Resets inside the prefix
    are honoured."""
    from smashbot.rl.ppo import Prefix, imitation_weights

    P, T, B = 7, 8, 4
    learner = _make_learner(imitation_rows=-1)
    with torch.no_grad():   # zero-initialised output projections would ignore history
        for param in [*learner.policy.parameters(), *learner.value_function.parameters()]:
            param.add_(0.3 * torch.randn_like(param))
    full = _rollout(learner.policy, B=B, T=P + T, seed=5)
    resets = full.is_resetting.clone()
    resets[1, 3] = True
    full = full._replace(is_resetting=resets)
    at = lambda lo, hi: (lambda t: t[:, lo:hi])
    chunk = full._replace(
        states=tree.map_structure(at(P, None), full.states), name=full.name[:, P:],
        actions=full.actions._replace(
            controller_state=tree.map_structure(at(P, None), full.actions.controller_state)),
        rewards=full.rewards[:, P:], is_resetting=full.is_resetting[:, P:],
        kind="imitation", valid=~full.is_resetting[:, P + 1:],
        prefix=Prefix(
            state_action=learner._frames(full).state_action._replace(
                state=tree.map_structure(at(0, P), full.states), name=full.name[:, :P],
                action=tree.map_structure(at(0, P), full.actions.controller_state)),
            is_resetting=full.is_resetting[:, :P]))

    critic = copy.deepcopy(learner.value_function)
    with torch.no_grad():
        straight = critic.outputs(learner._frames(full), critic.initial_state(B),
                                  discount=learner.config.discount, detail=False)
        expected_policy = learner.policy.unroll(learner._frames(full),
                                                learner.policy.initial_state(B))
    valid = chunk.valid.float()
    expected_weights = imitation_weights(straight.advantages[:, P:], valid,
                                         learner.config.imitation_beta,
                                         learner.config.imitation_w_cap)

    imf = learner._imitation_fixed(chunk, 3)
    assert torch.allclose(imf.weights, expected_weights, atol=1e-5)
    (part,) = learner._imit_chunks(imf, B)
    with torch.no_grad():
        warmed = learner.policy.unroll(part.frames, part.initial_state)
    assert torch.allclose(warmed.log_probs, expected_policy.log_probs[:, P:], atol=1e-5)
    with torch.no_grad():
        cold = learner.policy.unroll(part.frames, learner.policy.initial_state(B))
    assert not torch.allclose(cold.log_probs, expected_policy.log_probs[:, P:], atol=1e-5)
