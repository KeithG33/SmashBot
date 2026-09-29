"""Teacher-KL leash decay schedule: exact at the endpoints and at v10's
mid-run splice point; disabled default is bitwise the historical constant;
step() actually applies the scheduled weights, the reverse leash its own."""

import math

import pytest
import torch

from smashbot.rl.config import RLConfig
from smashbot.rl.ppo import Learner
from smashbot.tests.test_ppo import _rollout, _tiny_policy, _tiny_value


def _learner(**kw):
    torch.manual_seed(0)
    return Learner(
        RLConfig(**kw), _tiny_policy(seed=0), _tiny_policy(seed=0),
        _tiny_value(),
    )


def test_schedule_exact_and_default_constant():
    # disabled (default -1): constant at kl_teacher_weight forever
    const = _learner(kl_teacher_weight=0.08)
    for p in (0.0, 0.425, 1.0, 7.0):
        assert const.kl_teacher_weight_at(p) == 0.08
    # v10 splice: start 0.1206522 -> 0.08 at progress 17000/40000 -> 0.025
    dec = _learner(
        kl_teacher_weight=0.1206522, kl_teacher_weight_final=0.025
    )
    assert dec.kl_teacher_weight_at(0.0) == pytest.approx(0.1206522)
    assert dec.kl_teacher_weight_at(17000 / 40000) == pytest.approx(
        0.08, abs=1e-7
    )
    assert dec.kl_teacher_weight_at(1.0) == pytest.approx(0.025)
    # clamped outside [0, 1]
    assert dec.kl_teacher_weight_at(1.7) == pytest.approx(0.025)
    assert dec.kl_teacher_weight_at(-0.2) == pytest.approx(0.1206522)


def test_step_applies_scheduled_weight():
    learner = _learner(
        kl_teacher_weight=0.1206522, kl_teacher_weight_final=0.025,
        imitation_rows=0,
    )
    traj = _rollout(learner.policy, B=3, T=8, seed=0)
    _, m = learner.step([traj], learner.initial_state(3), progress=0.425)
    assert learner._kl_teacher_w == pytest.approx(0.08, abs=1e-6)
    assert m["post_update"]["kl_teacher_w"] == pytest.approx(0.08, abs=1e-6)


def test_reverse_leash_decays_on_its_own_schedule():
    assert _learner().reverse_kl_teacher_weight_at(0.5) == 0.0   # default: off
    learner = _learner(
        kl_teacher_weight=0.025, kl_teacher_weight_final=0.0025,
        reverse_kl_teacher_weight=0.025, reverse_kl_teacher_weight_final=0.0025,
        imitation_rows=0,
    )
    assert learner.reverse_kl_teacher_weight_at(1.0) == pytest.approx(0.0025)
    traj = _rollout(learner.policy, B=3, T=8, seed=0)
    _, m = learner.step([traj], learner.initial_state(3), progress=0.5)
    assert learner._reverse_kl_teacher_w == pytest.approx(0.01375)
    assert m["post_update"]["reverse_kl_teacher_w"] == pytest.approx(0.01375)


def test_exponential_decay_is_geometric():
    learner = _learner(
        kl_teacher_weight=0.025, kl_teacher_weight_final=0.001,
        reverse_kl_teacher_weight=0.025, reverse_kl_teacher_weight_final=0.001,
        kl_teacher_decay="exponential",
    )
    for at in (learner.kl_teacher_weight_at, learner.reverse_kl_teacher_weight_at):
        assert at(0.0) == pytest.approx(0.025)
        assert at(0.5) == pytest.approx(0.005)   # the geometric mean
        assert at(1.0) == pytest.approx(0.001)
        assert at(1.7) == pytest.approx(0.001)
    halving = math.log(2) / math.log(25)
    for p in (0.0, 0.3, 0.6):
        assert learner.kl_teacher_weight_at(p + halving) == pytest.approx(learner.kl_teacher_weight_at(p) / 2)


def test_exponential_decay_rejects_zero_endpoints():
    with pytest.raises(ValueError):
        _learner(kl_teacher_weight=0.025, kl_teacher_weight_final=0.0, kl_teacher_decay="exponential")
    with pytest.raises(ValueError):
        _learner(kl_teacher_decay="geometric")
    _learner(kl_teacher_weight=0.025, kl_teacher_weight_final=-1.0, kl_teacher_decay="exponential")   # constant: fine
