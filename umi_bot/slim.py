"""A checkpoint with only what serving reads: config, policy weights, name
map and step. Optimizers, the value net and the training state are most of a
BC checkpoint (1.9 GB -> 0.13 GB for QK).

  python -m umi_bot.slim models/qk-scratch-1057011.pt build/qk-1057011.pt
"""
import sys

import torch


def main(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    state = {k: ckpt["state"][k] for k in ("policy", "name_map", "step") if k in ckpt["state"]}
    torch.save({**ckpt, "state": state}, dst)


if __name__ == "__main__":
    main(*sys.argv[1:])
