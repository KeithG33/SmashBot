"""RL fine-tuning entry point: PPO + KL-to-teacher over melee-sim-light
rollouts (rl/train_sim.py drives the loop).

Usage:
  python -m smashbot.rl.train_rl --ckpt /path/to/teacher.pt \
      --runtime.device cuda --runtime.steps 100000

The checkpoint provides everything: policy init, frozen teacher, critic init,
and the config/name_map (RL checkpoints stay play.py-compatible).
"""

from __future__ import annotations

import dataclasses
import os

import tyro

from smashbot.rl.ppo import RLConfig
from smashbot.rl.train_sim import SimRolloutConfig


@dataclasses.dataclass
class RuntimeConfig:
    tag: str = "rl-dev"
    steps: int = 1000
    # step at which the leash and imitation schedules reach their final values
    # and stay there (0 = steps), so steps can be extended without rescaling them
    schedule_steps: int = 0
    trajectories_per_step: int = 1
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    checkpoint_interval: int = 50
    log_interval: int = 1
    wandb_mode: str = os.environ.get("WANDB_MODE", "online")   # $WANDB_MODE sets the default
    # wandb run id override (default: the tag). Needed when a tag's id was
    # deleted on the server — deleted ids are tombstoned and unreusable.
    wandb_id: str = ""
    name: str = "Master Player"
    compile: bool = True  # compile the serving copy's sample and the learner cores
    restore: str = ""  # RL checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
    device: str = "cuda"  # the sim backend trains on CUDA only


@dataclasses.dataclass
class Config:
    ckpt: str  # BC checkpoint: the student's initial weights and the frozen teacher
    learner: RLConfig = dataclasses.field(default_factory=RLConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    sim: SimRolloutConfig = dataclasses.field(default_factory=SimRolloutConfig)


def _save_rl_checkpoint(
    path: str, config: dict, policy, value_fn, name_map, step: int, teacher: str
) -> None:
    """Same schema as BC checkpoints (config already a dict), so play.py and
    the eval harness load RL checkpoints unchanged."""
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
                "grad_scaler": _save_rl_checkpoint.grad_scaler(),
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
    args = tyro.cli(Config)
    from smashbot.rl import train_sim
    return train_sim.run(args)


if __name__ == "__main__":
    main()
