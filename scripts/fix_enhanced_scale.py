"""Rescale a BC checkpoint's enhanced-embed character and action tables to
their init std (what 435101b gives new runs) without changing what the model
computes: every encoder input column reading a table takes the inverse scale,
and Adam's moments follow both parameters, so the run resumes as from an
equivalent model whose tables now train at upstream's relative rate.

    python scripts/fix_enhanced_scale.py <ckpt.pt> <out.pt>
"""
import sys

import torch

from smashbot.embed import tree_map_to_torch
from smashbot.policy import build_policy_from_config
from smashbot.warm_start import fit_batches, rescale_enhanced
from smashbot.data import loader

TABLES = {"char_table_scale": ("network.enhanced.embed_char.weight",),
          "action_table_scale": ("network.enhanced.embed_action.weight",
                                 "network.enhanced.embed_char_action.weight")}
ENCODER = "network.core.encoder.weight"


@torch.no_grad()
def encoder_input_scale(policy, rescale) -> torch.Tensor:
    """Per input column of the core's encoder, how much rescale(policy) scales
    it: the enhanced embed's output on a dummy frame, before over after."""
    enhanced = policy.network.enhanced
    dummy = tree_map_to_torch(policy.network.embed_state_action.dummy((1,)))
    before = enhanced(dummy).double()[0]
    report = rescale(policy)
    after = enhanced(dummy).double()[0]
    changed = (after - before).abs() > 1e-9 * before.abs().max()
    ratio = torch.where(changed, after / before, torch.ones_like(before))
    known = torch.zeros_like(changed)
    for s in report.values():
        known |= changed & ((ratio - s).abs() < 1e-6 * s)
    assert torch.equal(known, changed), "a column scaled by neither table's factor"
    return ratio.float(), report


@torch.no_grad()
def fix(ckpt: dict) -> dict:
    policy = build_policy_from_config(ckpt["config"])
    policy.load_state_dict(ckpt["state"]["policy"])
    policy.eval()
    batch = fit_batches(ckpt, policy.delay, 1)[0]
    frames = loader.batch_to_frames(batch, policy.network)
    encoded = lambda: policy.network.core.encoder(policy.network.embed_sa(frames.state_action))
    want = encoded()

    ratio, report = encoder_input_scale(policy, rescale_enhanced)
    policy.network.core.encoder.weight.div_(ratio)
    got = encoded()
    report["encoder_output_max_rel_diff"] = ((got - want).abs().max() / want.abs().max()).item()
    assert report["encoder_output_max_rel_diff"] < 1e-5, report

    names = [n for n, _ in policy.named_parameters()]
    opt = ckpt["state"]["policy_opt"]
    ids = opt["param_groups"][0]["params"]
    assert len(opt["param_groups"]) == 1 and len(ids) == len(names)
    state = {name: opt["state"][i] for name, i in zip(names, ids) if i in opt["state"]}
    for key, tables in TABLES.items():
        s = report[key]
        for name in tables:            # p' = s p, so g' = g / s
            if name in state:
                state[name]["exp_avg"].div_(s)
                state[name]["exp_avg_sq"].div_(s * s)
    state[ENCODER]["exp_avg"].mul_(ratio)   # w' = w / r, so g' = r g
    state[ENCODER]["exp_avg_sq"].mul_(ratio * ratio)

    ckpt["state"]["policy"] = policy.state_dict()
    ckpt["state"]["enhanced_rescale"] = report
    return report


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print(fix(ckpt))
    torch.save(ckpt, out)
