"""Stopping train_bc and resuming it is the same run.

One run of 2N steps must equal a run of N steps resumed for N more: same
weights, same optimizer moments, same recurrent states, same position in
both data streams. Reaching --runtime.steps IS a stop, so run B's first
half simply ends at N and its second half restores 'auto' with steps=2N.
The chunk is sized so rows exhaust their game after the resume point: the
next replay must come from the right place in the (mirror-doubled) cycle.
"""

import dataclasses

import torch
import tree
from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, saving, train_bc

N = 4
UNROLL = 1024  # toy game is 6402 frames: rows roll over at step 7


def _config(run_dir: str, tag: str, steps: int, autoclip: float = 0.0) -> train_bc.TrainConfig:
    return train_bc.TrainConfig(
        learner=configs.LearnerConfig(autoclip_percentile=autoclip),
        data=configs.DataConfig(
            dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET), mirror=True),
            batch_size=2, unroll_length=UNROLL, num_workers=0, prefetch=2, pin_memory=False,
        ),
        network=configs.NetworkConfig(name="tx_like", hidden_size=32, num_layers=1),
        value=configs.ValueConfig(hidden_size=32, num_layers=1),
        head=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        runtime=train_bc.RuntimeConfig(
            steps=steps, eval_interval=2, eval_batches=1, log_interval=1,
            checkpoint_interval=1000, tag=tag, run_dir=run_dir,
            wandb_mode="disabled", device="cpu", seed=0,
        ),
    )


def _latest(run_dir: str, tag: str) -> dict:
    return saving.load_checkpoint(f"{run_dir}/{tag}/latest.pt")["state"]


def _assert_same(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_resume_equals_uninterrupted(tmp_path):
    run_dir = str(tmp_path)
    train_bc.main(_config(run_dir, "whole", 2 * N))
    train_bc.main(_config(run_dir, "halves", N))
    cfg = _config(run_dir, "halves", 2 * N)
    cfg.runtime.restore = "auto"
    train_bc.main(cfg)

    whole, halves = _latest(run_dir, "whole"), _latest(run_dir, "halves")
    assert whole["step"] == halves["step"] == 2 * N
    assert whole["train_data"]["rows"] == halves["train_data"]["rows"]
    for key in ("policy", "value", "policy_opt", "value_opt", "train_data", "test_data",
                "train_hidden", "value_hidden", "eval_hidden", "eval_value_hidden"):
        _assert_same(whole[key], halves[key], key)


def test_autoclip_history_resumes(tmp_path):
    """AutoClip's threshold is a percentile of every step so far, so the
    history must be restored or the resumed run clips differently."""
    run_dir = str(tmp_path)
    train_bc.main(_config(run_dir, "whole", 2 * N, autoclip=50.0))
    train_bc.main(_config(run_dir, "halves", N, autoclip=50.0))
    cfg = _config(run_dir, "halves", 2 * N, autoclip=50.0)
    cfg.runtime.restore = "auto"
    train_bc.main(cfg)

    whole, halves = _latest(run_dir, "whole"), _latest(run_dir, "halves")
    assert len(whole["clip_history"]["policy"]) == 2 * N
    assert "value" not in whole["clip_history"], "AutoClip is policy-only"
    for key in ("policy", "value", "policy_opt", "value_opt", "clip_history"):
        _assert_same(whole[key], halves[key], key)


def test_resume_refuses_a_different_experiment(tmp_path):
    run_dir = str(tmp_path)
    train_bc.main(_config(run_dir, "run", 1))
    cfg = _config(run_dir, "run", 2)
    cfg.runtime.restore = "auto"
    cfg.learner.learning_rate = 3e-4
    try:
        train_bc.main(cfg)
    except ValueError as e:
        assert "learner.learning_rate" in str(e)
    else:
        raise AssertionError("resumed with a changed learning rate")


def test_a_setting_the_checkpoint_predates_is_held_to_its_default():
    import copy
    defaults = dataclasses.asdict(train_bc.TrainConfig())
    current = copy.deepcopy(defaults)
    saved = copy.deepcopy(defaults)
    del saved["network"]["window"]                     # an old checkpoint without this key
    train_bc._check_config(saved, current, defaults)
    current["network"]["window"] = defaults["network"]["window"] + 1
    try:
        train_bc._check_config(saved, current, defaults)
    except ValueError as e:
        assert "network.window" in str(e)
    else:
        raise AssertionError("a new setting was switched on over an old checkpoint")
