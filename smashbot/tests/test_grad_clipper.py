"""GradClipper: measuring leaves the gradients alone, clipping with the measured
norm is clip_grad_norm_ exactly, and AutoClip's threshold is np.percentile of
the whole history to the bit, across a resume too."""
import math

import numpy as np
import pytest
import torch

from smashbot.training import GradClipper, RunningPercentile


def _grads(lin, scale):
    lin.zero_grad()
    (lin(torch.randn(4, 8) * scale).pow(2).sum()).backward()


def _norm(lin):
    return math.sqrt(sum(p.grad.pow(2).sum().item() for p in lin.parameters()))


def _step(clipper):
    return clipper.clip(clipper.measure())


def test_measure_leaves_the_gradients_alone():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    _grads(lin, 1)
    lin.weight.grad[0, 0] = math.inf
    before = [p.grad.clone() for p in lin.parameters()]
    assert GradClipper(lin.parameters(), 1.0).measure() == math.inf
    for p, g in zip(lin.parameters(), before):
        assert torch.equal(p.grad, g)   # the finite gradients stay finite


def test_clipping_with_the_measured_norm_is_clip_grad_norm():
    torch.manual_seed(0)
    a, b = torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, 8) * 30
    for lin in (a, b):
        lin(x).pow(2).sum().backward()
    m = _step(GradClipper(a.parameters(), 1.0))
    reference = torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    assert m["grad_norm"] == reference.item()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa.grad, pb.grad)


def test_autoclip_tracks_the_percentile_and_bounds_a_spike():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    c = GradClipper(lin.parameters(), 0.0, 10.0)
    for i in range(30):
        _grads(lin, 30 if i == 20 else 1)
        m = _step(c)
        assert m["clip_norm"] == np.percentile(c.history, 10)
        assert _norm(lin) <= m["clip_norm"] * (1 + 1e-5)
        if i == 20:
            assert m["grad_norm"] > 20 * m["clip_norm"]
    assert len(c.history) == 30


def test_fixed_clip_and_exclusive_rules():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    _grads(lin, 30)
    m = _step(GradClipper(lin.parameters(), 1.0))
    assert m["clip_norm"] == 1.0 and _norm(lin) <= 1.0 + 1e-5 and m["grad_norm"] > 1.0
    try:
        GradClipper(lin.parameters(), 1.0, 10.0)
    except AssertionError:
        return
    raise AssertionError("two rules accepted")


def _norms(kind, n=2000):
    rng = np.random.default_rng(0)
    values = {
        "heavy-tailed": rng.lognormal(0, 1, n),
        "tied": np.round(rng.lognormal(0, 1, n), 1),
        "falling": np.linspace(5, 0.1, n) * rng.uniform(0.9, 1.1, n),
        "rising": np.linspace(0.1, 5, n) * rng.uniform(0.9, 1.1, n),
    }[kind]
    return [float(np.float32(v)) for v in values]   # norms arrive as float32 .item()s


@pytest.mark.parametrize("kind", ["heavy-tailed", "tied", "falling", "rising"])
@pytest.mark.parametrize("q", [10, 50, 90, 2.5, 17.5, 99.9, 0])   # 17.5/100 != 17.5*0.01
def test_the_running_percentile_is_numpys_to_the_bit(q, kind):
    norms = _norms(kind)
    running = RunningPercentile(q)
    for n, norm in enumerate(norms, 1):
        running.add(norm)
        assert running.value == float(np.percentile(norms[:n], q)), n


def test_a_resume_continues_the_threshold_to_the_bit():
    lin = torch.nn.Linear(2, 2)   # no gradients: clip only records and thresholds
    norms = _norms("heavy-tailed")
    whole = GradClipper(lin.parameters(), 0.0, 10.0)
    for norm in norms[:1000]:
        whole.clip(norm)
    built = GradClipper(lin.parameters(), 0.0, 10.0, history=whole.history)   # BC's resume
    restored = GradClipper(lin.parameters(), 0.0, 10.0)
    restored.restore_history(whole.history)                                    # RL's resume
    assert built.history is not whole.history
    for n, norm in enumerate(norms[1000:], 1001):
        expected = float(np.percentile(norms[:n], 10))
        assert [c.clip(norm)["clip_norm"] for c in (whole, built, restored)] == [expected] * 3, n
