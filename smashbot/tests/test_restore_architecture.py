"""Restoring a checkpoint under a different architecture setting must fail
loudly: some settings change behavior but no parameter shape, so the weights
would load without complaint."""

import dataclasses

import pytest
from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, train_bc


def _config(run_dir: str, steps: int, reward_halflife: float) -> train_bc.TrainConfig:
    return train_bc.TrainConfig(
        data=configs.DataConfig(
            dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
            batch_size=2, unroll_length=16, num_workers=0, prefetch=2, pin_memory=False,
        ),
        network=configs.NetworkConfig(
            name="sgu", hidden_size=32, num_layers=1, window=8, attn_heads=2, attn_head_dim=8,
        ),
        value=configs.ValueConfig(hidden_size=32, num_layers=1, reward_halflife=reward_halflife),
        head=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        runtime=train_bc.RuntimeConfig(
            steps=steps, eval_interval=100, eval_batches=1, log_interval=1, checkpoint_interval=100,
            tag="run", run_dir=run_dir, wandb_mode="disabled", device="cpu", seed=0,
        ),
    )


def test_restore_under_a_changed_shape_free_setting_is_refused(tmp_path):
    train_bc.main(_config(str(tmp_path), 1, reward_halflife=4.0))
    changed = _config(str(tmp_path), 2, reward_halflife=2.0)
    changed.runtime.restore = "auto"
    with pytest.raises(ValueError, match="value.reward_halflife differs"):
        train_bc.main(changed)
    same = _config(str(tmp_path), 2, reward_halflife=4.0)
    same.runtime.restore = "auto"
    train_bc.main(same)


def test_checkpoints_that_predate_a_setting_take_its_default():
    current = train_bc.TrainConfig(network=configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1, window=8))
    saved = dataclasses.asdict(current)
    del saved["network"]["attn_heads"]
    defaults = dataclasses.asdict(train_bc.TrainConfig())
    train_bc._check_config(saved, dataclasses.asdict(current), defaults)
    current.network.attn_heads = 2
    with pytest.raises(ValueError, match="network.attn_heads"):
        train_bc._check_config(saved, dataclasses.asdict(current), defaults)
