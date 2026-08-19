"""Unit tests for the learner-precision probe (scripts/precision_probe.py)
on a tiny synthetic setup: fp32-vs-fp32 self-comparison must be all-zero
deltas, the bf16 arm must run and produce finite numbers, reports must
JSON-round-trip, and the probe must never mutate weights. CUDA-only pieces
(memcurve proper) are skipped on cpu — and gated behind SMASHBOT_GPU_TESTS
so a routine test run never touches a GPU a live training run may own."""

import importlib.util
import json
import os
import pathlib

import pytest
import torch

from smashbot.rl.ppo import Learner, PPOConfig, RLConfig
from smashbot.tests.test_ppo import (
    _rollout,
    _tiny_policy,
    _tiny_value,
    copy_params,
)

_PROBE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts" / "precision_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "precision_probe", _PROBE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


@pytest.fixture()
def tiny():
    """Fresh-learner setup: teacher == policy init, so the ratio_mean == 1
    invariant and teacher_kl == 0 both hold on the rolled-out trajectory."""
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    value = _tiny_value()
    learner = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1.0)),
        policy, teacher, value,
    )
    probe.freeze_value_updates(learner)
    traj = _rollout(policy)
    return learner, traj


def test_fp32_self_comparison_is_all_zero(tiny):
    learner, traj = tiny
    a = probe.run_arm(learner, [traj], "fp32", "cpu")
    b = probe.run_arm(learner, [traj], "fp32", "cpu")
    cmp = probe.compare_arms(a, b, ratio_tol=1e-3)
    assert cmp["max_abs_dlogprob"] == 0.0
    assert cmp["max_abs_dratio"] == 0.0
    assert cmp["max_abs_dadvantage"] == 0.0
    assert cmp["d_akl_mean"] == 0.0
    assert cmp["d_loss"] == 0.0
    assert cmp["d_surrogate"] == 0.0
    assert cmp["d_teacher_kl"] == 0.0
    assert cmp["d_entropy"] == 0.0
    assert cmp["d_value_loss"] == 0.0
    assert cmp["d_policy_grad_norm"] == 0.0
    assert cmp["d_value_grad_norm"] == 0.0
    # fresh learner: the rollout/learner mismatch detector must pass
    assert cmp["ratio_invariant_ok"]
    assert cmp["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert cmp["akl_mean"] == pytest.approx(0.0, abs=1e-5)
    assert all(v == 0 for v in cmp["nonfinite"].values())
    assert cmp["dtypes"]["log_probs"] == "torch.float32"


def test_probe_never_mutates_weights(tiny):
    learner, traj = tiny
    pol_before = copy_params(learner.policy)
    val_before = copy_params(learner.value_function)
    probe.run_arm(learner, [traj], "fp32", "cpu")
    probe.run_arm(learner, [traj], "bf16", "cpu")
    for a, b in zip(pol_before, copy_params(learner.policy)):
        assert torch.equal(a, b)
    for a, b in zip(val_before, copy_params(learner.value_function)):
        assert torch.equal(a, b)
    # and no leftover grads
    assert all(p.grad is None for p in learner.policy.parameters())
    assert all(p.grad is None for p in learner.value_function.parameters())


def test_bf16_arm_runs_and_is_finite(tiny):
    learner, traj = tiny
    base = probe.run_arm(learner, [traj], "fp32", "cpu")
    arm = probe.run_arm(learner, [traj], "bf16", "cpu")
    cmp = probe.compare_arms(base, arm, ratio_tol=1e-3)
    for key, val in cmp.items():
        if isinstance(val, float):
            assert torch.isfinite(torch.tensor(val)), f"nonfinite {key}"
    assert all(v == 0 for v in cmp["nonfinite"].values())
    # bf16 must actually change the unroll dtype (the arm is real)...
    assert cmp["dtypes"]["unroll_logits_leaf"] == "torch.bfloat16"
    # ...while the sensitive path stays fp32 (autocast's fp32 ops +
    # value.py's fp32 island), per the experiments doc's design rules
    assert cmp["dtypes"]["log_probs"] == "torch.float32"
    assert cmp["dtypes"]["advantages"] == "torch.float32"
    assert cmp["dtypes"]["actor_log_probs"] == "torch.float32"
    # a tiny net in bf16 stays in the same numeric neighborhood
    assert cmp["max_abs_dlogprob"] < 1.0
    assert abs(cmp["ratio_mean"] - 1.0) < 0.2


def test_fidelity_report_json_round_trips(tiny):
    learner, traj = tiny
    arms = probe.run_fidelity(
        learner, [traj], "cpu", arms=("fp32", "bf16"), ratio_tol=1e-3
    )
    table = probe.format_table(arms)
    assert "fp32" in table and "bf16" in table
    assert "ratio_mean==1" in table
    blob = json.dumps(arms)
    restored = json.loads(blob)
    assert restored["fp32"]["max_abs_dlogprob"] == 0.0
    assert restored["bf16"]["ratio_mean"] == arms["bf16"]["ratio_mean"]


def test_synth_trajectory_matches_learner_recomputation(tiny):
    """The memcurve's synthetic batch must behave like real rollout data:
    sample-time logits come from the policy's own unroll, so a fresh
    learner's recomputation gives ratio == 1 and a finite loss (cpu stand-in
    for the cuda memcurve path)."""
    learner, _ = tiny
    traj = probe.synth_trajectory(learner.policy, rows=2, unroll=6,
                                  device="cpu")
    assert traj.rewards.shape == (2, 6)
    assert traj.is_resetting.shape == (2, 7)
    fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(2))
    with torch.no_grad():
        _, metrics = learner._policy_loss(fixed)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["anomalous_samples"] == 0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_full_learner_step_on_synth_batch(tiny):
    """memcurve drives learner.step on the synthetic batch; the cpu version
    of that exact call must run to completion with finite metrics."""
    learner, _ = tiny
    traj = probe.synth_trajectory(learner.policy, rows=2, unroll=6,
                                  device="cpu")
    _, metrics = learner.step([traj], learner.initial_state(2))
    assert torch.isfinite(torch.tensor(metrics["post_update"]["loss"]))


@pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SMASHBOT_GPU_TESTS")),
    reason="cuda-only; set SMASHBOT_GPU_TESTS=1 on an IDLE gpu "
           "(never beside a live training run)",
)
def test_memcurve_smoke_cuda(tiny):
    learner, _ = tiny
    learner.policy.to("cuda")
    learner.teacher.to("cuda")
    learner.value_function.to("cuda")
    results = probe.run_memcurve(learner, [2], unroll=6, arm="bf16")
    assert results and not results[0]["oom"]
    assert results[0]["peak_bytes"] > 0


def test_fp32_must_lead(tiny):
    learner, traj = tiny
    with pytest.raises(AssertionError):
        probe.run_fidelity(learner, [traj], "cpu", arms=("bf16", "fp32"))
