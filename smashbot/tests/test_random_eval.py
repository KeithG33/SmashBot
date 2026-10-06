"""The random-draw eval scores fresh random games at every eval, next to the
fixed set, and leaves training exactly as it would be without it: the same
weights, optimizer state, data position and RNG states."""

import dataclasses
import threading
import types

import numpy as np
import torch

from smashbot import train_bc
from smashbot.data import loader
from smashbot.tests.test_resume import _assert_same, _config, _latest


def test_random_eval_leaves_training_unchanged(tmp_path, capsys):
    base = _config(str(tmp_path), "base", 4)
    wide = dataclasses.replace(base, runtime=dataclasses.replace(
        base.runtime, tag="wide", wide_eval_groups=2, wide_eval_rows=3, wide_eval_batches=2))
    train_bc.main(base)
    train_bc.main(wide)
    out = capsys.readouterr().out
    wide_lines = [line for line in out.splitlines() if line.startswith("eval_wide @")]
    assert len(wide_lines) == 2 and all("nan" not in line for line in wide_lines)

    a, b = _latest(str(tmp_path), "base"), _latest(str(tmp_path), "wide")
    for key in ("policy", "value", "policy_opt", "value_opt", "train_data", "train_hidden", "value_hidden"):
        _assert_same(a[key], b[key], key)
    assert a["rng"]["python"] == b["rng"]["python"]
    assert all(np.array_equal(x, y) for x, y in zip(a["rng"]["numpy"], b["rng"]["numpy"]))
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])


def test_stop_waits_out_a_draw_in_progress(monkeypatch):
    """stop() mid-seating returns only once the draw thread has exited, and
    shuts the split down after it: a draw left running under a shut-down
    source hung interpreter exit (train_bc and score_wide, 2026-09-30)."""
    seating, release = threading.Event(), threading.Event()

    def slow_seat(*_):
        seating.set()
        release.wait()

    monkeypatch.setattr(loader, "seat_random", slow_seat)
    alive_at_shutdown = []
    split = types.SimpleNamespace(shutdown=lambda: alive_at_shutdown.append(stream._thread.is_alive()))
    stream = loader.RandomEvalStream(split, None, groups=2, batches=2, span=10, seed=0, num_workers=0)
    assert seating.wait(5)
    threading.Timer(6.0, release.set).start()   # outlasts the old 5 s join
    stream.stop()
    assert not stream._thread.is_alive() and alive_at_shutdown == [False]


def test_a_draw_depends_only_on_its_eval_step(monkeypatch):
    """Draw k is seeded by seed + k * interval, so a stream restarted at the
    next eval step draws exactly the games the first stream drew there."""
    picks = []
    monkeypatch.setattr(loader, "seat_random", lambda split, span, rng, workers: picks.append(int(rng.integers(1 << 30))))

    def draws(seed, n):
        picks.clear()
        stream = loader.RandomEvalStream(types.SimpleNamespace(shutdown=lambda: None), None, groups=1,
                                         batches=0, span=10, seed=seed, num_workers=0, interval=2500)
        for _ in range(n):
            next(stream)
        stream.stop()
        return picks[:n]

    from_start, after_restart = draws(7, 2), draws(7 + 2500, 1)
    assert from_start[0] != from_start[1] and from_start[1] == after_restart[0]
