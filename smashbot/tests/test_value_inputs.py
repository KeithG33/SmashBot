"""The value net takes the policy's inputs (embedding and tech mask) unless
its config says simple; configs saved before 2026-10-02 have no such key and
built simple value nets, which scripts/match_value_inputs.py converts."""
import dataclasses
import importlib.util
import pathlib

import torch

from smashbot import saving, train_bc
from smashbot.rl.train_rl import build_value_function
from smashbot.tests.test_resume import _latest
from smashbot.tests.test_warm_start import _runs
from smashbot.value import value_network_config

_path = pathlib.Path(__file__).parents[2] / "scripts" / "match_value_inputs.py"
_spec = importlib.util.spec_from_file_location("match_value_inputs", _path)
match_value_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match_value_inputs)

NETWORK = {"name": "sgu", "num_heads": 1, "window": 4, "embed": "enhanced", "embed_hidden_size": 16,
           "tech_mask_window": 4}
VALUE = {"name": "match", "hidden_size": 32, "num_layers": 1, "layout": ""}


def test_value_net_takes_the_policys_inputs_unless_saved_without_them():
    matched = value_network_config(NETWORK, {**VALUE, "inputs": "policy"})
    assert (matched.embed, matched.embed_hidden_size, matched.tech_mask_window) == ("enhanced", 16, 4)
    for value in ({**VALUE, "inputs": "simple"}, VALUE):   # explicit, and a config saved before the key
        simple = value_network_config(NETWORK, value)
        assert (simple.embed, simple.tech_mask_window) == ("simple", 0)


def test_a_simple_value_net_converts_and_its_run_resumes(tmp_path):
    io_config = _runs(tmp_path, value_inputs="simple")
    path = f"{tmp_path}/io/latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy_before = {k: v.clone() for k, v in ckpt["state"]["policy"].items()}
    report = match_value_inputs.match(ckpt)
    assert torch.isfinite(torch.tensor(report["value_input_layer_error"]))
    assert ckpt["config"]["value"]["inputs"] == "policy"
    assert any(k.startswith("network.enhanced.") for k in ckpt["state"]["value"])
    assert all(torch.equal(v, policy_before[k]) for k, v in ckpt["state"]["policy"].items())
    build_value_function(ckpt["config"], "cpu").load_state_dict(ckpt["state"]["value"])
    torch.save(ckpt, path)

    resumed = dataclasses.replace(
        io_config, value=dataclasses.replace(io_config.value, inputs="policy"),
        runtime=dataclasses.replace(io_config.runtime, steps=io_config.runtime.steps + 2, restore="auto"))
    train_bc.main(resumed)
    assert _latest(str(tmp_path), "io")["step"] == io_config.runtime.steps + 2
    assert saving.load_checkpoint(path)["config"]["value"]["inputs"] == "policy"
