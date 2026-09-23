"""Versioned checkpointing.

Checkpoint = torch.save of {"config", "state", "best_eval_loss", "version"},
mirroring slippi-ai's shape. `upgrade_checkpoint` maps old versions forward so
early checkpoints stay loadable as the config schema evolves.
"""

import dataclasses
import os
import typing as tp

import torch

VERSION = 1


def _upgraders() -> dict[int, tp.Callable[[dict], dict]]:
    # e.g. {1: _v1_to_v2}
    return {}


def upgrade_checkpoint(ckpt: dict) -> dict:
    version = ckpt.get("version", 0)
    while version < VERSION:
        upgrader = _upgraders().get(version)
        if upgrader is None:
            raise ValueError(f"No upgrader from checkpoint version {version}")
        ckpt = upgrader(ckpt)
        version = ckpt["version"]
    return ckpt


def save_checkpoint(
    path: str,
    config: tp.Any,  # dataclass
    state: dict,  # module/optimizer state_dicts + step
    best_eval_loss: float,
) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "config": dataclasses.asdict(config),
            "state": state,
            "best_eval_loss": best_eval_loss,
            "version": VERSION,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: never leave a torn checkpoint


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return upgrade_checkpoint(ckpt)


def load_optimizer(optimizer, saved_optimizer: dict, saved_module_state: dict, prefix: str = "value_head.") -> None:
    """Restore an optimizer whose module may have lost parameters since the
    checkpoint: the built-in value head's two tensors registered last, so a
    checkpoint that still carries them has two trailing parameter ids to drop."""
    dropped = sum(k.startswith(prefix) for k in saved_module_state)
    if dropped:
        saved_optimizer = {
            "state": dict(saved_optimizer["state"]),
            "param_groups": [dict(g) for g in saved_optimizer["param_groups"]],
        }
        group = saved_optimizer["param_groups"][-1]
        stale, group["params"] = group["params"][-dropped:], group["params"][:-dropped]
        for i in stale:
            saved_optimizer["state"].pop(i, None)
    optimizer.load_state_dict(saved_optimizer)

