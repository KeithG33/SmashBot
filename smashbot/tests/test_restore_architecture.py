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


def test_compile_may_change_on_resume_but_precision_may_not():
    defaults = dataclasses.asdict(train_bc.TrainConfig())
    current = train_bc.TrainConfig()
    current.learner.compile = not current.learner.compile
    train_bc._check_config(defaults, dataclasses.asdict(current), defaults)
    current.learner.precision = "bf16" if current.learner.precision == "fp32" else "fp32"
    with pytest.raises(ValueError, match="learner.precision differs"):
        train_bc._check_config(defaults, dataclasses.asdict(current), defaults)


def test_a_removed_setting_in_a_saved_config_is_ignored():
    """gate_gelu / v_norm became unconditional; checkpoints that carry them
    must still build and resume."""
    saved = dataclasses.asdict(configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1, window=8))
    saved["gate_gelu"] = True
    assert configs.from_dict(configs.NetworkConfig, saved).hidden_size == 32
    cfg = train_bc.TrainConfig(network=configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1, window=8))
    whole = dataclasses.asdict(cfg)
    whole["network"]["gate_gelu"] = True
    train_bc._check_config(whole, dataclasses.asdict(cfg), dataclasses.asdict(train_bc.TrainConfig()))


def test_pre_paper_block_weights_are_refused_and_paper_block_weights_renamed():
    from smashbot import networks
    old_names = {"network.core.blocks.0.uv.weight": None}
    with pytest.raises(ValueError, match="paper block"):
        networks.check_loadable({"gate_gelu": False, "v_norm": True}, old_names)
    with pytest.raises(ValueError, match="paper block"):
        networks.check_loadable({}, old_names)
    networks.check_loadable({"gate_gelu": True, "v_norm": True}, old_names)
    assert networks.current_names(old_names) == {"network.core.blocks.0.uv.0.weight": None}
    networks.check_loadable({}, {"network.core.blocks.0.uv.0.weight": None})   # today's names need no flags


def test_optimizer_restores_past_a_removed_trailing_parameter():
    """Checkpoints from the built-in value head's era hold two more optimizer
    parameter ids than today's Policy has; they were registered last."""
    import torch
    from smashbot import saving
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 3)
    old_head = torch.nn.Linear(3, 1)
    old_opt = torch.optim.Adam(list(net.parameters()) + list(old_head.parameters()), lr=1e-3)
    (net(torch.randn(2, 4)).sum() + old_head(torch.randn(2, 3)).sum()).backward()
    old_opt.step()
    saved_opt = old_opt.state_dict()
    saved_module = {**{k: v for k, v in net.state_dict().items()},
                    **{"value_head." + k: v for k, v in old_head.state_dict().items()}}
    new_opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    saving.load_optimizer(new_opt, saved_opt, saved_module)
    assert len(new_opt.state_dict()["param_groups"][0]["params"]) == 2
    for i, p in enumerate(net.parameters()):
        assert torch.equal(new_opt.state[p]["exp_avg"], old_opt.state[list(old_opt.param_groups[0]["params"])[i]]["exp_avg"])
    assert saved_opt["param_groups"][0]["params"] == [0, 1, 2, 3], "the saved dict is not mutated"
    plain = torch.optim.Adam(net.parameters(), lr=1e-3)
    saving.load_optimizer(plain, plain.state_dict(), net.state_dict())   # nothing to drop
