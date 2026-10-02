"""Warm-start a BC model from two checkpoints: the recurrent core of `core`
(a long run) and the input embeddings and controller head of `io` (a run
already in the target setup: its embedding, stick encoding and delay; the
new model's config is io's).

The two were trained apart, so their hidden units don't line up. Two
least-squares fits on real frames from core's data bridge them:
  - the input layer, so io's embedded features reproduce core's own
    input-layer output, frame by frame;
  - the head's to_residual, so core's output (through that input layer)
    lands where io's head had its residual, on the delay-aligned stream the
    new model trains on (the first batch warms the recurrent state).
The value net is core's, its previous-stick input columns remapped to io's
stick encoding when that changed: each bucket's column is the grid's x and y
columns at the bucket's position.

    python -m smashbot.warm_start <core.pt> <io.pt> <out.pt>
"""
from __future__ import annotations

import copy
import dataclasses
import sys

import numpy as np
import torch
import tree
from slippi_ai.data import DatasetConfig

from smashbot import configs, embed as embed_lib, saving
from smashbot.data import loader
from smashbot.delay import slice_delayed_frames
from smashbot.policy import build_policy_from_config
from smashbot.rl.train_rl import build_value_function


def leaf_columns(embedding) -> dict[tuple, slice]:
    """Each leaf's columns in a StructEmbedding's output, by path."""
    cols, offset = {}, 0

    def walk(e, path):
        nonlocal offset
        if isinstance(e, embed_lib.StructEmbedding):
            for k, child in e.embedding:
                walk(child, path + (str(k),))
        else:
            cols[path] = slice(offset, offset + e.size)
            offset += e.size

    walk(embedding, ())
    return cols


def fit_batches(ckpt: dict, delay: int, n: int) -> list:
    """The n raw batches after a checkpoint in its own data stream, with
    delay + 1 extra frames."""
    cfg, state = ckpt["config"], ckpt["state"]
    data = configs.from_dict(configs.DataConfig, {
        **cfg["data"], "dataset": configs.from_dict(DatasetConfig, cfg["data"]["dataset"])})
    data = dataclasses.replace(data, num_workers=0)
    sources = loader.make_sources(data, delay + 1, name_map=state["name_map"], train_state=state.get("train_data"))
    return [next(sources.train)[0].batch for _ in range(n)]


def _lstsq(xtx: torch.Tensor, xty: torch.Tensor, yy: torch.Tensor):
    """The ridge-stabilized least-squares coefficients and relative error."""
    beta = torch.linalg.solve(xtx + 1e-6 * torch.diagonal(xtx).mean() * torch.eye(len(xtx), dtype=xtx.dtype), xty)
    residual = yy - 2 * (beta * xty).sum() + (beta * (xtx @ beta)).sum()
    return beta, (residual.clamp_min(0) / yy).sqrt().item()


def _with_ones(x: torch.Tensor) -> torch.Tensor:
    x = x.flatten(0, -2).double()
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)


@torch.no_grad()
def rescale_enhanced(policy) -> dict:
    """The enhanced embed's character table, and its action table with the
    joint table (their sum is one input block), scaled to the tables' init
    std; the bridge refits the input layer, so the bridged model computes the
    same. Runs before 435101b trained tables from torch's N(0, 1) init."""
    enhanced = policy.network.enhanced
    if enhanced is None:
        return {}
    std = enhanced.embed_action.weight.shape[1] ** -0.5
    scales = {}
    for name, tables in (("char", (enhanced.embed_char,)),
                         ("action", (enhanced.embed_action, enhanced.embed_char_action))):
        scales[f"{name}_table_scale"] = scale = std / tables[0].weight.std().item()
        for table in tables:
            table.weight.mul_(scale)
    return scales


@torch.no_grad()
def bridge_policy(core, io, target, batches, head_rows: int) -> dict:
    """target (io's config, io's weights on entry) gets core's recurrent
    layers and the two fitted layers; returns the fits' relative errors."""
    assert hasattr(core.network.core, "encoder"), "the bridge fits the core's input Linear"
    state = target.state_dict()
    for k, v in core.state_dict().items():
        if k.startswith("network.core.") and not k.startswith("network.core.encoder."):
            state[k] = v.clone()
    target.load_state_dict(state)

    xtx = xty = yy = 0
    for b in batches:
        fc, ft = loader.batch_to_frames(b, core.network), loader.batch_to_frames(b, target.network)
        y = core.network.core.encoder(core.network.embed_sa(fc.state_action)).flatten(0, -2).double()
        x = _with_ones(target.network.embed_sa(ft.state_action))
        xtx, xty, yy = xtx + x.T @ x, xty + x.T @ y, yy + (y * y).sum()
    beta, input_error = _lstsq(xtx, xty, yy)
    target.network.core.encoder.weight.copy_(beta[:-1].T.float())
    target.network.core.encoder.bias.copy_(beta[-1].float())

    xtx = xty = yy = 0
    rows = slice(0, head_rows)
    st, si = target.network.initial_state(head_rows), io.network.initial_state(head_rows)
    for i, b in enumerate(batches):
        d = slice_delayed_frames(loader.batch_to_frames(tree.map_structure(lambda x: x[rows], b), target.network),
                                 target.delay)
        inputs, reset = tree.map_structure(lambda t: t[:, :-1], d.state_action), d.is_resetting[:, :-1]
        ct, st = target.network.unroll(inputs, reset, st)
        ci, si = io.network.unroll(inputs, reset, si)
        if i:
            x, y = _with_ones(ct), io.controller_head.to_residual(ci).flatten(0, -2).double()
            xtx, xty, yy = xtx + x.T @ x, xty + x.T @ y, yy + (y * y).sum()
    beta, residual_error = _lstsq(xtx, xty, yy)
    target.controller_head.to_residual.weight.copy_(beta[:-1].T.float())
    target.controller_head.to_residual.bias.copy_(beta[-1].float())
    return {"input_layer_error": input_error, "to_residual_error": residual_error}


def stick_bins(grid_controller, joint_controller) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per stick the joint encoding changed, each bucket's grid (x, y) bins
    at its position."""
    grid, joint = dict(grid_controller.embedding), dict(joint_controller.embedding)
    bins = {}
    for stick in ("main_stick", "c_stick"):
        if isinstance(joint[stick], embed_lib.JointStickEmbedding) and isinstance(grid[stick], embed_lib.StructEmbedding):
            axes = dict(grid[stick].embedding)
            xy = joint[stick].decode(np.arange(joint[stick].size))
            bins[stick] = (axes["x"].from_state(np.asarray(xy.x)).astype(np.int64),
                           axes["y"].from_state(np.asarray(xy.y)).astype(np.int64))
    return bins


def remap_value(src, dst) -> dict:
    """dst's state dict: src's weights, the previous stick action's input
    columns remapped to dst's stick encoding."""
    s_state, d_state = src.state_dict(), dst.state_dict()
    s_cols = leaf_columns(src.network.embed_state_action)
    d_cols = leaf_columns(dst.network.embed_state_action)
    w_src = s_state["network.core.encoder.weight"]
    w_dst = torch.zeros_like(d_state["network.core.encoder.weight"])
    for path, cols in d_cols.items():
        if path in s_cols:
            w_dst[:, cols] = w_src[:, s_cols[path]]
    bins = stick_bins(dict(src.network.embed_state_action.embedding)["action"],
                      dict(dst.network.embed_state_action.embedding)["action"])
    for stick, (xbin, ybin) in bins.items():
        x, y = s_cols[("action", stick, "x")], s_cols[("action", stick, "y")]
        w_dst[:, d_cols[("action", stick)]] = w_src[:, x][:, xbin] + w_src[:, y][:, ybin]
    return {**{k: v.clone() for k, v in s_state.items()}, "network.core.encoder.weight": w_dst}


def warm_start(core_path: str, io_path: str, out_path: str, n_batches: int = 5, head_rows: int = 128) -> dict:
    core_ckpt, io_ckpt = saving.load_checkpoint(core_path), saving.load_checkpoint(io_path)
    if core_ckpt["state"]["name_map"] != io_ckpt["state"]["name_map"]:
        raise SystemExit("the two runs' name maps differ: the name input would change meaning")
    config = io_ckpt["config"]
    core, io, target = (build_policy_from_config(c) for c in (core_ckpt["config"], config, config))
    core.load_state_dict(core_ckpt["state"]["policy"])
    io.load_state_dict(io_ckpt["state"]["policy"])
    target.load_state_dict(io_ckpt["state"]["policy"])
    for p in (core, io, target):
        p.eval()
    batches = fit_batches(core_ckpt, target.delay, n_batches)
    scales = rescale_enhanced(target)
    report = {**bridge_policy(core, io, target, batches, head_rows), **scales}

    src_value, value = build_value_function(core_ckpt["config"], "cpu"), build_value_function(config, "cpu")
    src_value.load_state_dict(core_ckpt["state"]["value"])
    value.load_state_dict(remap_value(src_value, value))

    torch.save({"config": config, "best_eval_loss": float("inf"), "version": io_ckpt["version"],
                "state": {"policy": target.state_dict(), "value": value.state_dict(),
                          "name_map": io_ckpt["state"]["name_map"], "step": 0,
                          "warm_start": {"core": core_path, "core_step": core_ckpt["state"]["step"],
                                         "io": io_path, "io_step": io_ckpt["state"]["step"], **report}}},
               out_path)
    return report


if __name__ == "__main__":
    print(warm_start(*sys.argv[1:4]))
