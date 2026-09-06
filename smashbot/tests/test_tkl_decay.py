"""Teacher-KL leash decay schedule: exact at the endpoints and at v10's
mid-run splice point; disabled default is bitwise the historical constant;
step() actually applies the scheduled weight."""

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
