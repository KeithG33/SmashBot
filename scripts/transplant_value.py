"""Give a BC checkpoint another BC run's value net. BC trains the critic apart
from the policy, so any value net trained on the same data, name map,
controller encoding and delay fits; the donor's optimizer and hidden states
are not carried over.

    python scripts/transplant_value.py <policy.pt> <value donor.pt> <out.pt>
"""
import sys

import torch

from smashbot import saving
from smashbot.rl.train_rl import build_value_function

SHARED = (("data", "max_names"), ("head", "axis_spacing"), ("head", "shoulder_spacing"),
          ("head", "controller_type"), ("network", "embed"), ("policy", "delay"))


def transplant(policy_path: str, donor_path: str, out_path: str) -> None:
    target, donor = saving.load_checkpoint(policy_path), saving.load_checkpoint(donor_path)
    for section, key in SHARED:
        mine, theirs = target["config"][section].get(key), donor["config"][section].get(key)
        if mine != theirs:
            raise SystemExit(f"{section}.{key}: {mine!r} here, {theirs!r} in the donor")
    for name, a, b in (("characters", target["config"]["data"]["dataset"]["allowed_characters"],
                        donor["config"]["data"]["dataset"]["allowed_characters"]),
                       ("name_map", target["state"]["name_map"], donor["state"]["name_map"])):
        if a != b:
            raise SystemExit(f"{name}: {a!r} here, {b!r} in the donor")

    value = dict(donor["config"]["value"])
    if value["name"] == "match":
        value["name"] = donor["config"]["network"]["name"]
    config = {**target["config"], "value": value}
    state = {k: v for k, v in target["state"].items()
             if k not in ("value_opt", "value_hidden", "eval_value_hidden")}
    state["value"] = donor["state"]["value"]
    build_value_function(config, "cpu").load_state_dict(state["value"])   # strict: every weight fits

    torch.save({"config": config, "state": state, "best_eval_loss": target["best_eval_loss"],
                "version": target["version"]}, out_path)
    print(f"{out_path}: policy of {policy_path} (step {target['state']['step']}), "
          f"value {value} of {donor_path} (step {donor['state']['step']})")


if __name__ == "__main__":
    transplant(*sys.argv[1:])
