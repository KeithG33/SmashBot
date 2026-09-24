"""BC stops on a non-finite update instead of training on it, and never writes
a checkpoint holding a NaN or inf: a bad gradient, loss, recurrent state or
eval injected at step 3 or 4 raises, and latest.pt stays the step-2 one."""
import math
import os

import pytest
import torch
import tree
from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, saving, train_bc
from smashbot.policy import Policy
from smashbot.value import ValueFunction


def _config(run_dir, steps=5, eval_interval=1000):
    return train_bc.TrainConfig(
        data=configs.DataConfig(
            dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
            batch_size=2, unroll_length=32, num_workers=0, prefetch=2, pin_memory=False,
        ),
        network=configs.NetworkConfig(name="tx_like", hidden_size=32, num_layers=1),
        value=configs.ValueConfig(hidden_size=32, num_layers=1),
        head=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        runtime=train_bc.RuntimeConfig(
            steps=steps, eval_interval=eval_interval, eval_batches=1, log_interval=1,
            checkpoint_interval=2, tag="run", run_dir=str(run_dir),
            wandb_mode="disabled", device="cpu", seed=0,
        ),
    )


def _poison(monkeypatch, cls, method, at_call, spoil, training=True):
    """On the at_call-th training (or eval) call of cls.method, hand its
    (loss, state, extras) to spoil."""
    original, calls = getattr(cls, method), [0]

    def wrapped(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if torch.is_grad_enabled() == training:
            calls[0] += 1
            if calls[0] == at_call:
                return spoil(self, out)
        return out

    monkeypatch.setattr(cls, method, wrapped)


def _inf_gradient(module, out):
    next(module.parameters()).register_hook(lambda g: g + math.inf)
    return out


def _nan_loss(module, out):
    loss, state, extras = out
    return loss * math.nan, state, extras


def _nan_state(module, out):
    loss, state, extras = out
    leaves = tree.flatten(state)
    poisoned = [t.clone() for t in leaves]
    next(t for t in poisoned if t.is_floating_point()).view(-1)[0] = math.nan
    return loss, tree.unflatten_as(state, poisoned), extras


def _latest_step(run_dir):
    return saving.load_checkpoint(os.path.join(run_dir, "run", "latest.pt"))["state"]["step"]


@pytest.mark.parametrize("cls,method,spoil,message", [
    (Policy, "imitation_loss", _inf_gradient, "non-finite policy update"),
    (ValueFunction, "loss", _inf_gradient, "non-finite value update"),
    (Policy, "imitation_loss", _nan_loss, "non-finite policy update"),
], ids=["policy gradient", "value gradient", "policy loss"])
def test_a_nonfinite_update_stops_before_the_optimizer_step(tmp_path, monkeypatch, cls, method, spoil, message):
    _poison(monkeypatch, cls, method, 3, spoil)
    with pytest.raises(FloatingPointError, match=message):
        train_bc.main(_config(tmp_path))
    assert _latest_step(tmp_path) == 2


def test_a_nonfinite_recurrent_state_is_never_written(tmp_path, monkeypatch):
    _poison(monkeypatch, Policy, "imitation_loss", 4, _nan_state)
    with pytest.raises(FloatingPointError, match="refusing to write latest.pt at step 4: non-finite train_hidden"):
        train_bc.main(_config(tmp_path))
    assert _latest_step(tmp_path) == 2


def test_a_nonfinite_eval_stops_training(tmp_path, monkeypatch):
    config = _config(tmp_path, eval_interval=3)
    warm_batches = -(-config.runtime.eval_burn_in // config.data.unroll_length)   # unscored
    _poison(monkeypatch, Policy, "imitation_loss", warm_batches + 1, _nan_loss, training=False)
    with pytest.raises(FloatingPointError, match="non-finite eval"):
        train_bc.main(config)
    assert _latest_step(tmp_path) == 2


def test_nonfinite_names_every_bad_state():
    good, bad = torch.zeros(3), torch.tensor([0.0, math.inf])
    states = {"weights": {"w": good}, "Adam": {"state": {0: {"exp_avg": bad}}}, "hidden": (good, [bad])}
    assert train_bc._nonfinite(states) == ["Adam", "hidden"]
    assert train_bc._nonfinite({"ints": torch.tensor([1, 2])}) == []
