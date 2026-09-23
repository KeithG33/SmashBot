"""GradClipper: fixed clip, AutoClip's percentile threshold, and the split
measure/clip the RL learner uses to keep non-finite steps out of the history."""
import math

import numpy as np
import torch

from smashbot.training import GradClipper


def _grads(lin, scale):
    lin.zero_grad()
    (lin(torch.randn(4, 8) * scale).pow(2).sum()).backward()


def _norm(lin):
    return math.sqrt(sum(p.grad.pow(2).sum().item() for p in lin.parameters()))


def test_autoclip_tracks_the_percentile_and_bounds_a_spike():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    c = GradClipper(lin.parameters(), 0.0, 10.0)
    for i in range(30):
        _grads(lin, 30 if i == 20 else 1)
        m = c()
        assert m["clip_norm"] == np.percentile(c.history, 10)
        assert _norm(lin) <= m["clip_norm"] * (1 + 1e-5)
        if i == 20:
            assert m["grad_norm"] > 20 * m["clip_norm"]
    assert len(c.history) == 30


def test_fixed_clip_and_exclusive_rules():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    _grads(lin, 30)
    m = GradClipper(lin.parameters(), 1.0)()
    assert m["clip_norm"] == 1.0 and _norm(lin) <= 1.0 + 1e-5 and m["grad_norm"] > 1.0
    try:
        GradClipper(lin.parameters(), 1.0, 10.0)
    except AssertionError:
        return
    raise AssertionError("two rules accepted")


def test_nonfinite_norm_never_enters_the_history():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    c = GradClipper(lin.parameters(), 0.0, 50.0)
    _grads(lin, 1)
    c()
    lin.weight.grad[0, 0] = float("nan")
    norm = c.measure()
    assert not math.isfinite(norm)
    m = c()   # the RL learner skips the step instead; __call__ must do the same
    assert len(c.history) == 1 and math.isnan(m["grad_norm"]) and m["clip_norm"] == c.threshold


def test_history_round_trip_gives_the_same_threshold():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 8)
    a = GradClipper(lin.parameters(), 0.0, 10.0)
    for _ in range(10):
        _grads(lin, 1); a()
    b = GradClipper(lin.parameters(), 0.0, 10.0, history=a.history)
    assert b.threshold == a.threshold and b.history is not a.history
