"""GradClipper: measuring leaves the gradients alone, clipping with the measured
norm is clip_grad_norm_ exactly, and AutoClip's percentile threshold."""
import math

import numpy as np
import torch

from smashbot.training import GradClipper


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


def test_history_round_trip_gives_the_same_threshold():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    a = GradClipper(lin.parameters(), 0.0, 10.0)
    for _ in range(10):
        _grads(lin, 1)
        _step(a)
    b = GradClipper(lin.parameters(), 0.0, 10.0, history=a.history)
    assert b.threshold == a.threshold and b.history is not a.history
