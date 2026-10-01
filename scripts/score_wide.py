"""Score BC checkpoints the way train_bc's eval_wide scores a run, on one
shared draw of random test games: the same seed for every checkpoint, so each
sees the same games whatever its stick encoding.

    python scripts/score_wide.py [--seed N] [--groups 4] a/latest.pt b/latest.pt
"""
import argparse
import contextlib

import torch
from slippi_ai import data as data_lib
from slippi_ai.data import DatasetConfig

from smashbot import configs, saving
from smashbot.data import loader
from smashbot.networks import use_chunk_start_resets, use_packed_encoder
from smashbot.policy import StickScorer, build_policy_from_config
from smashbot.rl.train_rl import build_value_function
from smashbot.train_bc import RuntimeConfig, score


def score_checkpoint(path: str, seed: int, groups: int, device: str) -> dict:
    ckpt = saving.load_checkpoint(path)
    cfg, state = ckpt["config"], ckpt["state"]
    policy = build_policy_from_config(cfg).to(device)
    policy.load_state_dict(state["policy"])
    value_fn = build_value_function(cfg, device)
    value_fn.load_state_dict(state["value"])
    for net in (policy, value_fn):
        use_chunk_start_resets(net)
        use_packed_encoder(net)

    data = configs.from_dict(configs.DataConfig, {
        **cfg["data"], "dataset": configs.from_dict(DatasetConfig, cfg["data"]["dataset"])})
    delay = cfg["policy"]["delay"]
    _, test_replays = data_lib.train_test_split(data.dataset)
    rt = RuntimeConfig()
    warm_batches = -(-rt.eval_burn_in // data.unroll_length)
    stream = loader.random_eval_stream(
        test_replays, data, delay + 1, state["name_map"], policy.network,
        groups=groups, rows=rt.wide_eval_rows, batches=warm_batches + rt.wide_eval_batches, seed=seed)
    try:
        draw = next(stream)
    finally:
        stream.stop()

    if cfg["learner"]["precision"] == "bf16":
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext
    discount = 0.5 ** (1 / (cfg["value"]["reward_halflife"] * 60))
    return {"step": state["step"], **score(
        policy, value_fn, StickScorer(policy.controller_head, device), draw, rt.wide_eval_rows,
        delay, warm_batches, discount, autocast, device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for path in args.checkpoints:
        s = score_checkpoint(path, args.seed, args.groups, args.device)
        print(f"{path} @ {s['step']}: policy_loss {s['policy_loss']:.4f}, value_uev {s['value_uev']:.4f}, value_loss {s['value_loss']:.5f}"
              + "".join(f", {k} {v:.4f}" for k, v in s["sticks"].items()), flush=True)


if __name__ == "__main__":
    main()
