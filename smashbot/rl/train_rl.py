"""RL fine-tuning entry point: PPO + KL-to-teacher over live Dolphin rollouts.

Usage:
  python -m smashbot.rl.train_rl --ckpt /path/to/mega-best.pt \
      --rollouts.num-envs 8 --runtime.steps 10000

The checkpoint provides everything: policy init, frozen teacher, critic init,
and the config/name_map (RL checkpoints stay play.py-compatible).
"""

from __future__ import annotations

import copy
import dataclasses
import time

import torch
import tyro

from smashbot import configs, embed as embed_lib, saving
from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.networks import build_embed_network
from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.rl.ppo import Learner, RLConfig
from smashbot.rl.rollouts import DolphinRolloutWorker, RolloutConfig
from smashbot.rl.teacher_watch import TeacherWatcher
from smashbot.value import ValueFunction


@dataclasses.dataclass
class RuntimeConfig:
    tag: str = "rl-dev"
    steps: int = 1000
    trajectories_per_step: int = 1
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    checkpoint_interval: int = 50
    log_interval: int = 1
    wandb_mode: str = "online"
    name: str = "Master Player"
    compile: bool = True  # compile sample_n (the batched flush)
    # Hot-swappable teacher: poll this path (default: the --ckpt file) every
    # teacher_check_interval learner steps; on change, safely reload the
    # frozen teacher in place (see rl/teacher_watch.py).
    teacher_watch: str = ""
    teacher_check_interval: int = 20
    device: str = "cpu"  # rollouts are CPU-bound; learner device


@dataclasses.dataclass
class Config:
    ckpt: str = "/home/kage/drive2/ShineBot/models/mega-best-epoch1.8.pt"
    learner: RLConfig = dataclasses.field(default_factory=RLConfig)
    rollouts: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)


def build_value_function(cfg: dict, device: str) -> ValueFunction:
    value_name = cfg["value"].get("name", "match")
    if value_name == "match":
        value_name = cfg["network"]["name"]
    net_cfg = configs.NetworkConfig(
        name=value_name,
        hidden_size=cfg["value"]["hidden_size"],
        num_layers=cfg["value"]["num_layers"],
        num_heads=cfg["network"]["num_heads"],
        window=cfg["value"].get("window", 0) or cfg["network"]["window"],
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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "config": config,
            "state": {
                "policy": policy.state_dict(),
                "value": value_fn.state_dict(),
                "name_map": name_map,
                "step": step,
                "teacher_ckpt": teacher,
            },
            "best_eval_loss": None,
            "version": saving.VERSION,
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    args = tyro.cli(Config)
    device = args.runtime.device

    policy, name_map, step = load_policy(args.ckpt, device)
    policy.train_value_head = False
    teacher, _, _ = load_policy(args.ckpt, device)
    teacher.train_value_head = False

    ckpt = saving.load_checkpoint(args.ckpt)
    value_fn = build_value_function(ckpt["config"], device)
    value_fn.load_state_dict(ckpt["state"]["value"])
    name_code = resolve_name_code(name_map, args.runtime.name)
    print(f"teacher/init: {args.ckpt} (BC step {step}); conditioning code {name_code}")

    learner = Learner(args.learner, policy, teacher, value_fn)

    if args.runtime.compile:
        mode = "reduce-overhead" if device == "cuda" else "default"
        policy.sample = torch.compile(policy.sample, mode=mode)
        teacher.sample = torch.compile(teacher.sample, mode=mode)
        policy.sample_n = torch.compile(policy.sample_n, mode=mode)
        teacher.sample_n = torch.compile(teacher.sample_n, mode=mode)
    student_agent = BatchedPolicyAgent(
        policy, args.rollouts.num_envs, name_code=name_code, device=device,
        batch_steps=args.rollouts.batch_steps,
    )
    opponent_agent = None
    if args.rollouts.opponent == "teacher":
        opponent_agent = BatchedPolicyAgent(
            teacher, args.rollouts.num_envs, name_code=name_code, device=device,
            batch_steps=args.rollouts.batch_steps,
        )
    worker = DolphinRolloutWorker(args.rollouts, student_agent, opponent_agent)

    import wandb

    wandb.init(
        project="shinebot", id=args.runtime.tag, name=args.runtime.tag,
        mode=args.runtime.wandb_mode, config=dataclasses.asdict(args),
    )

    run_dir = f"{args.runtime.run_dir}/{args.runtime.tag}"
    state = learner.initial_state(args.rollouts.num_envs, device)
    watcher = TeacherWatcher(args.runtime.teacher_watch or args.ckpt)
    teacher_swaps = 0
    t0 = time.time()
    try:
        for i in range(args.runtime.steps):
            if i > 0 and i % args.runtime.teacher_check_interval == 0:
                new_teacher = watcher.poll()
                if new_teacher is not None:
                    teacher.load_state_dict(new_teacher)  # in-place copy
                    state = state._replace(
                        teacher=teacher.initial_state(
                            args.rollouts.num_envs, device
                        )
                    )
                    teacher_swaps += 1
                    print(f"[{i}] TEACHER SWAPPED (#{teacher_swaps})")
            trajectories = worker.collect(args.runtime.trajectories_per_step)
            state, metrics = learner.step(trajectories, state)

            if i % args.runtime.log_interval == 0:
                frames = (
                    (i + 1) * args.runtime.trajectories_per_step
                    * args.rollouts.num_envs * args.rollouts.unroll_length
                )
                log = {
                    "rl/" + k: v
                    for k, v in metrics["post_update"].items()
                }
                log.update({
                    "rl/value_" + k: v for k, v in metrics["value"].items()
                })
                log["rl/reverted"] = float(metrics["reverted"])
                log["rl/teacher_swaps"] = teacher_swaps
                log["rl/frames_per_sec"] = frames / (time.time() - t0)
                wandb.log(log, step=i)
                print(f"[{i}] {log}")

            if (i + 1) % args.runtime.checkpoint_interval == 0:
                _save_rl_checkpoint(
                    f"{run_dir}/latest.pt", ckpt["config"], policy, value_fn,
                    name_map, i, args.ckpt,
                )
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
