"""warm_start: the new model has the core run's recurrent layers, the io
run's embeddings and head (but for the two fitted layers) and config, the
core run's value net with its previous-stick columns remapped exactly to the
joint buckets, and trains from there."""

import dataclasses

import numpy as np
import torch

from smashbot import configs, saving, train_bc, warm_start
from smashbot.rl.train_rl import build_value_function
from smashbot.tests.test_resume import _config, _latest

SGU = dict(name="sgu", num_layers=1, hidden_size=32, num_heads=1, window=4, attn_heads=1, attn_head_dim=8)


def _runs(tmp_path):
    base = _config(str(tmp_path), "core", 3)
    core = dataclasses.replace(base, network=configs.NetworkConfig(**SGU))
    io = dataclasses.replace(
        core, runtime=dataclasses.replace(core.runtime, tag="io"),
        policy=dataclasses.replace(core.policy, delay=core.policy.delay + 3),
        network=configs.NetworkConfig(**SGU, embed="enhanced", embed_hidden_size=16, tech_mask_window=4),
        head=dataclasses.replace(core.head, controller_type="balanced_v6_157"))
    train_bc.main(core)
    train_bc.main(io)
    return io


def test_bridge_keeps_each_part_where_it_belongs(tmp_path):
    io_config = _runs(tmp_path)
    out = f"{tmp_path}/warm.pt"
    report = warm_start.warm_start(f"{tmp_path}/core/latest.pt", f"{tmp_path}/io/latest.pt", out,
                                   n_batches=3, head_rows=2)
    assert all(np.isfinite(v) for v in report.values())
    warm, core, io = saving.load_checkpoint(out), _latest(str(tmp_path), "core"), _latest(str(tmp_path), "io")
    assert warm["config"] == saving.load_checkpoint(f"{tmp_path}/io/latest.pt")["config"]

    policy = warm["state"]["policy"]
    for k, v in policy.items():
        if k.startswith("network.core.") and not k.startswith("network.core.encoder."):
            assert torch.equal(v, core["policy"][k]), k
        elif not k.startswith(("network.core.encoder.", "controller_head.to_residual.")):
            assert torch.equal(v, io["policy"][k]), k

    grid_cfg = saving.load_checkpoint(f"{tmp_path}/core/latest.pt")["config"]
    src, dst = build_value_function(grid_cfg, "cpu"), build_value_function(warm["config"], "cpu")
    src.load_state_dict(core["value"])
    s_cols = warm_start.leaf_columns(src.network.embed_state_action)
    d_cols = warm_start.leaf_columns(dst.network.embed_state_action)
    bins = warm_start.stick_bins(dict(src.network.embed_state_action.embedding)["action"],
                                 dict(dst.network.embed_state_action.embedding)["action"])
    w_src, w_dst = core["value"]["network.core.encoder.weight"], warm["state"]["value"]["network.core.encoder.weight"]
    for stick, (xbin, ybin) in bins.items():
        want = w_src[:, s_cols[("action", stick, "x")]][:, xbin] + w_src[:, s_cols[("action", stick, "y")]][:, ybin]
        assert torch.equal(w_dst[:, d_cols[("action", stick)]], want)
    for path in (("action", "buttons", "A"), ("state", "p0", "percent"), ("name",)):
        assert torch.equal(w_dst[:, d_cols[path]], w_src[:, s_cols[path]]), path

    run = dataclasses.replace(io_config, runtime=dataclasses.replace(io_config.runtime, tag="warm", steps=2, init_from=out))
    train_bc.main(run)
    assert _latest(str(tmp_path), "warm")["step"] == 2
