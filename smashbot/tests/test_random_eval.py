"""The random-draw eval scores fresh random games at every eval, next to the
fixed set, and leaves training exactly as it would be without it: the same
weights, optimizer state, data position and RNG states."""

import dataclasses

import numpy as np
import torch

from smashbot import train_bc
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
