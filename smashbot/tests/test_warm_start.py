"""warm_start: the new model has the core run's recurrent layers, the io
run's embeddings and head (but for the two fitted layers) and config, the
core run's value net (bridged onto the policy's enhanced inputs, or with its
previous-stick columns remapped exactly to the joint buckets when it stays
simple), and trains from there."""

import dataclasses

import numpy as np
import torch

from smashbot import configs, saving, train_bc, warm_start
from smashbot.data import loader
from smashbot.policy import build_policy_from_config
from smashbot.value import build_value_function
from smashbot.tests.test_resume import _config, _latest

SGU = dict(name="sgu", num_layers=1, hidden_size=32, num_heads=1, window=4, attn_heads=1, attn_head_dim=8)


def _runs(tmp_path, value_inputs="policy"):
    base = _config(str(tmp_path), "core", 3)
    core = dataclasses.replace(base, network=configs.NetworkConfig(**SGU))
    io = dataclasses.replace(
        core, runtime=dataclasses.replace(core.runtime, tag="io"),
        policy=dataclasses.replace(core.policy, delay=core.policy.delay + 3),
        network=configs.NetworkConfig(**SGU, embed="enhanced", embed_hidden_size=16, tech_mask_window=4),
        head=dataclasses.replace(core.head, controller_type="balanced_v6_157"),
        value=dataclasses.replace(core.value, inputs=value_inputs))
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
    tables = ("network.enhanced.embed_char.weight", "network.enhanced.embed_action.weight")
    for k, v in policy.items():
        if k.startswith("network.core.") and not k.startswith("network.core.encoder."):
            assert torch.equal(v, core["policy"][k]), k
        elif k in tables:   # factored from core's one-hot columns, at the init std
            assert abs(v.std().item() * 16 ** 0.5 - 1) < 0.1, k
        elif k == "network.enhanced.embed_char_action.weight":
            assert not v.any()
        elif not k.startswith(("network.core.encoder.", "controller_head.to_residual.")):
            assert torch.equal(v, io["policy"][k]), k
    assert all(0 < report[f"{leaf}_energy_kept"] <= 1 for leaf in ("action", "character"))

    value = warm["state"]["value"]   # on the policy's enhanced inputs: bridged from core's simple one
    assert any(k.startswith("network.enhanced.") for k in value)
    for k, v in value.items():
        if (k.startswith("network.core.") and not k.startswith("network.core.encoder.")) or k.startswith("head."):
            assert torch.equal(v, core["value"][k]), k

    run = dataclasses.replace(io_config, runtime=dataclasses.replace(io_config.runtime, tag="warm", steps=2, init_from=out))
    train_bc.main(run)
    assert _latest(str(tmp_path), "warm")["step"] == 2


def test_rescaling_the_tables_leaves_the_bridged_model_unchanged(tmp_path):
    _runs(tmp_path)
    core_ckpt, io_ckpt = (saving.load_checkpoint(f"{tmp_path}/{t}/latest.pt") for t in ("core", "io"))
    batches = warm_start.fit_batches(core_ckpt, io_ckpt["config"]["policy"]["delay"], 2)
    core = build_policy_from_config(core_ckpt["config"])
    core.load_state_dict(core_ckpt["state"]["policy"])
    outs = []
    for rescale in (False, True):
        io, target = (build_policy_from_config(io_ckpt["config"]) for _ in range(2))
        for p in (io, target):
            p.load_state_dict(io_ckpt["state"]["policy"])
        if rescale:
            with torch.no_grad():
                target.network.enhanced.embed_action.weight.mul_(10)   # a torch-init-sized table
                target.network.enhanced.embed_char_action.weight.mul_(10)
            warm_start.rescale_enhanced(target)
        warm_start.bridge_policy(core, io, target, batches, head_rows=2)
        frames = loader.batch_to_frames(batches[0], target.network)
        with torch.no_grad():
            outs.append(target.network.core.encoder(target.network.embed_sa(frames.state_action)))
    assert torch.allclose(outs[0], outs[1], rtol=1e-3, atol=1e-4)


def test_factored_tables_carry_the_cores_one_hot_columns(tmp_path):
    _runs(tmp_path)
    core_ckpt, io_ckpt = (saving.load_checkpoint(f"{tmp_path}/{t}/latest.pt") for t in ("core", "io"))
    core, target = build_policy_from_config(core_ckpt["config"]), build_policy_from_config(io_ckpt["config"])
    core.load_state_dict(core_ckpt["state"]["policy"])
    kept = warm_start.factor_enhanced_tables(core, target)
    cols = warm_start.leaf_columns(core.network.embed_state_action)
    w = core.network.core.encoder.weight.double()
    chars = torch.cat([w[:, cols[p]] for p in cols if p[0] == "state" and p[-1] == "character"])   # 4 slots x 33
    table = target.network.enhanced.embed_char.weight.double()                                     # 33 x 16
    readout = torch.linalg.lstsq(table, chars.T).solution                                          # the bridge's job
    assert torch.allclose(table @ readout, chars.T, atol=1e-6) == (kept["character_energy_kept"] > 1 - 1e-9)


def test_simple_value_net_stick_columns_remap_exactly(tmp_path):
    _runs(tmp_path, value_inputs="simple")
    out = f"{tmp_path}/warm.pt"
    warm_start.warm_start(f"{tmp_path}/core/latest.pt", f"{tmp_path}/io/latest.pt", out, n_batches=3, head_rows=2)
    warm, core = saving.load_checkpoint(out), _latest(str(tmp_path), "core")
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
