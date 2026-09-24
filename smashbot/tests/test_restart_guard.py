"""A relaunch cannot silently start over an existing run: a fresh start into a
run directory that holds checkpoints is refused without touching anything,
"auto" and explicit restores need a checkpoint, and on an RL restore the
configured learning rate wins while Adam's state survives."""
import copy
import os

import pytest
import torch
from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, train_bc
from smashbot.rl import train_rl, train_sim
from smashbot.rl.config import RLConfig
from smashbot.rl.ppo import Learner
from smashbot.tests.test_ppo import _tiny_policy, _tiny_value
from smashbot.training import resolve_restore


def _config(run_dir, restore=""):
    return train_bc.TrainConfig(
        data=configs.DataConfig(
            dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
            batch_size=2, unroll_length=32, num_workers=0, prefetch=2, pin_memory=False,
        ),
        network=configs.NetworkConfig(name="tx_like", hidden_size=32, num_layers=1),
        value=configs.ValueConfig(hidden_size=32, num_layers=1),
        head=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        runtime=train_bc.RuntimeConfig(
            steps=2, eval_interval=1000, eval_batches=1, log_interval=1,
            checkpoint_interval=1000, tag="run", run_dir=str(run_dir), restore=restore,
            wandb_mode="disabled", device="cpu", seed=0,
        ),
    )


def _snapshot(run_dir):
    return {f: (os.stat(os.path.join(run_dir, f)).st_mtime_ns, open(os.path.join(run_dir, f), "rb").read())
            for f in sorted(os.listdir(run_dir))}


def test_a_fresh_start_into_an_existing_run_is_refused_and_changes_nothing(tmp_path):
    train_bc.main(_config(tmp_path))
    run_dir = tmp_path / "run"
    before = _snapshot(run_dir)
    with pytest.raises(FileExistsError, match="already holds latest.pt"):
        train_bc.main(_config(tmp_path))
    assert _snapshot(run_dir) == before
    train_bc.main(_config(tmp_path, restore="auto"))   # resuming it is still fine


@pytest.mark.parametrize("restore", ["auto", "/nowhere/latest.pt"])
def test_a_restore_needs_a_checkpoint(tmp_path, restore):
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        train_bc.main(_config(tmp_path, restore=restore))
    assert not (tmp_path / "run").exists()


def test_best_pt_alone_also_marks_an_existing_run(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already holds best.pt"):
        resolve_restore(str(tmp_path), "")
    assert resolve_restore(str(tmp_path / "new"), "") == ""


def _rl_config(run_dir, restore=""):
    cfg = train_rl.Config(ckpt="bc.pt")
    cfg.runtime.run_dir, cfg.runtime.tag, cfg.runtime.restore = str(run_dir), "run", restore
    cfg.runtime.wandb_mode = "disabled"
    return cfg


def test_the_rl_trainer_refuses_before_loading_anything(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "latest.pt").write_bytes(b"")
    with pytest.raises(FileExistsError, match="already holds latest.pt"):
        train_sim.run(_rl_config(tmp_path))
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):   # no silent fresh start
        train_sim.run(_rl_config(tmp_path / "elsewhere", restore="auto"))
    assert os.listdir(tmp_path) == ["run"] and os.listdir(tmp_path / "run") == ["latest.pt"]


def _learner(lr):
    torch.manual_seed(0)
    return Learner(RLConfig(learning_rate=lr), _tiny_policy(), _tiny_policy(), _tiny_value())


def test_the_configured_learning_rate_wins_and_adam_state_survives():
    old = _learner(3e-5)
    for opt, net in ((old.policy_optimizer, old.policy), (old.value_optimizer, old.value_function)):
        opt.zero_grad()
        sum(p.sum() for p in net.parameters()).backward()
        opt.step()
    saved = [copy.deepcopy(opt.state_dict()) for opt in (old.policy_optimizer, old.value_optimizer)]
    new = _learner(1e-4)
    new.policy_optimizer.load_state_dict(saved[0])
    new.value_optimizer.load_state_dict(saved[1])   # the checkpoint's 3e-5 comes back with it
    assert new.set_learning_rate(1e-5) == {3e-5}
    for opt, before in zip((new.policy_optimizer, new.value_optimizer), saved):
        assert all(group["lr"] == 1e-5 for group in opt.param_groups)
        state = opt.state_dict()["state"]
        for i, slots in before["state"].items():
            for name, value in slots.items():
                assert torch.equal(state[i][name], value), (i, name)
