"""Persistent trust-region snapshot buffers (_snap_into): identical
content to a fresh _to_cpu every step, tensor storages actually reused,
and the revert path restores the CURRENT step's start (a stale-buffer bug
would restore an older step)."""

import torch
import tree

from smashbot.rl.config import RLConfig, PPOConfig
from smashbot.rl.ppo import Learner, _to_cpu
from smashbot.tests.test_ppo import _rollout, _tiny_policy, _tiny_value


def _learner(**kw):
    torch.manual_seed(0)
    return Learner(
        RLConfig(**kw), _tiny_policy(seed=0), _tiny_policy(seed=1),
        _tiny_value(),
    )


def _assert_same(a, b):
    assert type(a) is type(b) or (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    )
    if isinstance(a, torch.Tensor):
        assert a.device.type == "cpu"
        assert torch.equal(a, b.to("cpu")), (a - b.to("cpu")).abs().max()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    else:
        assert a == b


def test_snap_into_matches_fresh_copy_and_reuses_storage():
    learner = _learner()
    sd = learner.policy.state_dict()
    snap1 = learner._snap_into("policy", sd)
    _assert_same(snap1, _to_cpu(sd))
    # mutate weights, snapshot again: content tracks, storages reused
    with torch.no_grad():
        for p in learner.policy.parameters():
            p.add_(0.5)
    sd2 = learner.policy.state_dict()
    snap2 = learner._snap_into("policy", sd2)
    _assert_same(snap2, _to_cpu(sd2))
    ptrs1 = {
        k: v.data_ptr() for k, v in snap1.items()
        if isinstance(v, torch.Tensor)
    }
    reused = sum(
        1 for k, v in snap2.items()
        if isinstance(v, torch.Tensor) and v.data_ptr() == ptrs1.get(k)
    )
    assert reused == len(ptrs1), f"only {reused}/{len(ptrs1)} reused"


def test_snap_into_handles_optimizer_state_growth():
    """Adam state is empty before the first optimizer.step: the buffer
    must rebuild the changed leaves and match a fresh copy afterwards."""
    learner = _learner()
    s_empty = learner._snap_into("opt", learner.policy_optimizer.state_dict())
    _assert_same(s_empty, _to_cpu(learner.policy_optimizer.state_dict()))
    # populate Adam moments with one real update
    loss = sum(p.sum() for p in learner.policy.parameters())
    loss.backward()
    learner.policy_optimizer.step()
    s_full = learner._snap_into("opt", learner.policy_optimizer.state_dict())
    _assert_same(s_full, _to_cpu(learner.policy_optimizer.state_dict()))


def test_revert_restores_current_step_not_stale_buffer():
    """Two steps; the second is forced to revert (kl ceiling 0). The
    policy must come back to the SECOND step's starting weights — a
    stale-buffer bug would resurrect the first step's."""
    learner = _learner(
        ppo=PPOConfig(max_mean_actor_kl=1e9, num_epochs=1),
        imitation_rows=0,
    )
    traj = _rollout(learner.policy, B=4, T=8, seed=0)
    state = learner.initial_state(4)
    state, m1 = learner.step([traj], state, progress=0.0)
    assert not m1["reverted"]
    start2 = _to_cpu(learner.policy.state_dict())  # step-2 starting weights
    learner.config.ppo.max_mean_actor_kl = 0.0  # force revert
    traj2 = _rollout(learner.policy, B=4, T=8, seed=1)
    _, m2 = learner.step([traj2], state, progress=0.0)
    assert m2["reverted"]
    _assert_same(start2, _to_cpu(learner.policy.state_dict()))
