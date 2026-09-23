"""RL fine-tuning entry point: PPO + KL-to-teacher over melee-sim-light
rollouts (rl/train_sim.py drives the loop).

Usage:
  python -m smashbot.rl.train_rl --ckpt /path/to/teacher.pt \
      --sim.num-envs 448 --runtime.steps 100000

The checkpoint provides everything: policy init, frozen teacher, critic init,
and the config/name_map (RL checkpoints stay play.py-compatible).
"""

from __future__ import annotations

import dataclasses
import os
import time

import tyro

from smashbot.rl.config import RLConfig
from smashbot.rl.train_sim import SimRolloutConfig


@dataclasses.dataclass
class RuntimeConfig:
    tag: str = "rl-dev"
    steps: int = 1000
    trajectories_per_step: int = 1
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    checkpoint_interval: int = 50
    log_interval: int = 1
    wandb_mode: str = "online"
    # wandb run id override (default: the tag). Needed when a tag's id was
    # deleted on the server — deleted ids are tombstoned and unreusable.
    wandb_id: str = ""
    name: str = "Master Player"
    compile: bool = True  # compile sample_n (the batched flush)
    restore: str = ""  # RL checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
    device: str = "cpu"  # rollouts are CPU-bound; learner device


@dataclasses.dataclass
class Config:
    ckpt: str = "/home/kage/drive2/ShineBot/models/mega-best-epoch1.8.pt"
    learner: RLConfig = dataclasses.field(default_factory=RLConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    backend: str = "sim"  # melee-sim-light rollouts; options under --sim
    sim: SimRolloutConfig = dataclasses.field(default_factory=SimRolloutConfig)


def build_value_function(cfg: dict, device: str):
    from smashbot import configs, embed as embed_lib
    from smashbot.networks import build_embed_network
    from smashbot.value import ValueFunction

    value_name = cfg["value"].get("name", "match")
    if value_name == "match":
        value_name = cfg["network"]["name"]
    net_cfg = configs.NetworkConfig(
        name=value_name,
        hidden_size=cfg["value"]["hidden_size"],
        num_layers=cfg["value"]["num_layers"],
        num_heads=cfg["network"]["num_heads"],
        window=cfg["value"].get("window", 0) or cfg["network"]["window"],
        layout=cfg["value"].get("layout", ""),
    )
    return ValueFunction(
        build_embed_network(
            embed_config=embed_lib.EmbedConfig(),
            controller_embedding=embed_lib.ControllerConfig(
                axis_spacing=cfg["head"]["axis_spacing"],
                shoulder_spacing=cfg["head"]["shoulder_spacing"],
            ).make_embedding(),
            num_names=cfg["data"]["max_names"],
            network_config=net_cfg,
        )
    ).to(device)


def _save_rl_checkpoint(
    path: str, config: dict, policy, value_fn, name_map, step: int, teacher: str
) -> None:
    """Same schema as BC checkpoints (config already a dict), so play.py and
    the eval harness load RL checkpoints unchanged."""
    import os

    import torch

    from smashbot import saving

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "config": config,
            "state": {
                "policy": policy.state_dict(),
                "value": value_fn.state_dict(),
                "policy_opt": _save_rl_checkpoint.policy_opt.state_dict(),
                "value_opt": _save_rl_checkpoint.value_opt.state_dict(),
                "name_map": name_map,
                "step": step,
                "teacher_ckpt": teacher,
                "trackers": _save_rl_checkpoint.tracker_states(),
                "clip_history": {"policy": _save_rl_checkpoint.clip_history()},
            },
            "best_eval_loss": None,
            "version": saving.VERSION,
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    """Sim-backend RL training (see rl/train_sim.py). The Dolphin RL fleet
    was removed after the sim backend validated; Dolphin remains for
    watch/play/eval (smashbot/eval/)."""
    args = tyro.cli(Config)
    assert args.backend == "sim", (
        f"backend {args.backend!r} removed; only 'sim' trains now"
    )
    from smashbot.rl import train_sim
    return train_sim.run(args)


if __name__ == "__main__":
    main()
