"""Restoring a checkpoint under a different architecture setting must fail
loudly: attn_rope changes behavior but no parameter shape, so the weights
would load without complaint."""

import pytest
from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, train_bc


def _config(run_dir: str, steps: int, attn_rope: bool) -> train_bc.TrainConfig:
    return train_bc.TrainConfig(
        data=configs.DataConfig(
            dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
            batch_size=2, unroll_length=16, num_workers=0, prefetch=2, pin_memory=False,
        ),
        network=configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1, window=8, attn_rope=attn_rope),
        value=configs.ValueConfig(hidden_size=32, num_layers=1),
        head=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        runtime=train_bc.RuntimeConfig(
            steps=steps, eval_interval=100, eval_batches=1, log_interval=1, checkpoint_interval=100,
            tag="run", run_dir=run_dir, wandb_mode="disabled", device="cpu", seed=0,
        ),
    )


def test_restore_without_the_rope_flag_is_refused(tmp_path):
    train_bc.main(_config(str(tmp_path), 1, attn_rope=True))
    forgot = _config(str(tmp_path), 2, attn_rope=False)
    forgot.runtime.restore = "auto"
    with pytest.raises(ValueError, match="attn_rope=True"):
        train_bc.main(forgot)
    same = _config(str(tmp_path), 2, attn_rope=True)
    same.runtime.restore = "auto"
    train_bc.main(same)


def test_checkpoints_that_predate_a_setting_take_its_default():
    saved = {
        "network": {"name": "sgu", "hidden_size": 32, "num_layers": 1, "window": 8},
        "head": {}, "policy": {}, "value": {},
    }
    current = train_bc.TrainConfig(network=configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1, window=8))
    train_bc._check_architecture(saved, current)
    current.network.attn_rope = True
    with pytest.raises(ValueError):
        train_bc._check_architecture(saved, current)


def test_league_refuses_a_student_config_with_weightless_settings():
    import dataclasses
    from smashbot.rl.sim_league import SimLeague

    league = SimLeague.__new__(SimLeague)
    league._cfg = {"network": dataclasses.asdict(configs.NetworkConfig(name="sgu", attn_rope=True))}
    with pytest.raises(NotImplementedError, match="attn_rope"):
        league._make_skeleton()
