"""Give a BC checkpoint's value net the policy's inputs (its embedding and
tech mask), which every value net silently lacked before 2026-10-02: the
value core and head carry over, the tables are factored from the value net's
own one-hot columns, and the input layer is fitted on real frames
(warm_start.bridge_value). The value optimizer starts fresh and the carried
value states gain the tech mask's counters; the policy is untouched.

    python scripts/match_value_inputs.py <ckpt.pt> <out.pt>
"""
import copy
import sys

import torch
import tree

from smashbot.policy import build_policy_from_config
from smashbot.rl.train_rl import build_value_function
from smashbot.warm_start import bridge_value, fit_batches


def carried(value, old):
    """A carried value state saved without the tech mask, in value's layout
    (the mask's counters start at zero)."""
    if not value.network.tech_mask_window:
        return old
    zero = torch.zeros(tree.flatten(old)[0].shape[0], dtype=torch.long)
    return {"core": old, "tech_mask": (zero, zero.clone())}


def match(ckpt: dict) -> dict:
    cfg = ckpt["config"]
    assert cfg["value"].get("inputs", "simple") == "simple", "the value net already has the policy's inputs"
    new_cfg = copy.deepcopy(cfg)
    new_cfg["value"]["inputs"] = "policy"
    src, dst = build_value_function(cfg, "cpu"), build_value_function(new_cfg, "cpu")
    src.load_state_dict(ckpt["state"]["value"])
    policy = build_policy_from_config(cfg)
    policy.load_state_dict(ckpt["state"]["policy"])
    for module in (src, dst, policy):
        module.eval()
    report = bridge_value(src, dst, policy, fit_batches(ckpt, policy.delay, 5))

    state = ckpt["state"]
    state["value"] = dst.state_dict()
    state["value_opt"] = torch.optim.Adam(dst.parameters(), lr=cfg["learner"]["learning_rate"]).state_dict()
    for key in ("value_hidden", "eval_value_hidden"):
        if state.get(key) is not None:
            state[key] = carried(dst, state[key])
    ckpt["config"] = new_cfg
    return report


if __name__ == "__main__":
    src, out = sys.argv[1:3]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    print(match(ckpt))
    torch.save(ckpt, out)
