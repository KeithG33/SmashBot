"""scripts/fix_enhanced_scale.py: a checkpoint trained from torch's N(0, 1)
tables comes back with tables at the init std, computing the same, and its
run resumes from it."""
import dataclasses
import importlib.util
import pathlib

import torch

from smashbot import train_bc
from smashbot.tests.test_resume import _latest
from smashbot.tests.test_warm_start import _runs

_path = pathlib.Path(__file__).parents[2] / "scripts" / "fix_enhanced_scale.py"
_spec = importlib.util.spec_from_file_location("fix_enhanced_scale", _path)
fix_enhanced_scale = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix_enhanced_scale)

TABLES = ("network.enhanced.embed_char.weight", "network.enhanced.embed_action.weight",
          "network.enhanced.embed_char_action.weight")


def test_fixed_checkpoint_computes_the_same_and_resumes(tmp_path):
    io_config = _runs(tmp_path)
    path = f"{tmp_path}/io/latest.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for name in TABLES:   # a torch-init-sized run
        ckpt["state"]["policy"][name].mul_(10)
    old = {k: v.clone() for k, v in ckpt["state"]["policy"].items()}

    report = fix_enhanced_scale.fix(ckpt)
    assert report["encoder_output_max_rel_diff"] < 1e-5
    new = ckpt["state"]["policy"]
    for name in TABLES[:2]:
        assert abs(new[name].std().item() * 16 ** 0.5 - 1) < 1e-4, name
    assert torch.allclose(new[TABLES[2]], old[TABLES[2]] * report["action_table_scale"])
    for k in old:
        if k not in TABLES and k != fix_enhanced_scale.ENCODER:
            assert torch.equal(new[k], old[k]), k
    torch.save(ckpt, path)

    resumed = dataclasses.replace(io_config, runtime=dataclasses.replace(
        io_config.runtime, steps=io_config.runtime.steps + 2, restore="auto"))
    train_bc.main(resumed)
    assert _latest(str(tmp_path), "io")["step"] == io_config.runtime.steps + 2
