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
    teacher_check_interval: int = 100  # ~20 min at 64 envs (one step ~15s)
    restore: str = ""  # RL checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
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
                "policy_opt": _save_rl_checkpoint.policy_opt.state_dict(),
                "value_opt": _save_rl_checkpoint.value_opt.state_dict(),
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

    start_step = 0
    if args.runtime.restore:
        import os as os_lib

        rpath = args.runtime.restore
        if rpath == "auto":
            rpath = f"{args.runtime.run_dir}/{args.runtime.tag}/latest.pt"
            if not os_lib.path.exists(rpath):
                rpath = ""  # supervisor-friendly: no checkpoint = fresh start
                print("restore auto: no checkpoint yet, starting fresh")
        if rpath:
            rl_ckpt = saving.load_checkpoint(rpath)
            policy.load_state_dict(rl_ckpt["state"]["policy"])
            value_fn.load_state_dict(rl_ckpt["state"]["value"])
            if "policy_opt" in rl_ckpt["state"]:
                learner.policy_optimizer.load_state_dict(
                    rl_ckpt["state"]["policy_opt"]
                )
                learner.value_optimizer.load_state_dict(
                    rl_ckpt["state"]["value_opt"]
                )
            start_step = rl_ckpt["state"]["step"] + 1
            print(f"restored RL run from {rpath} at step {start_step}")
    _save_rl_checkpoint.policy_opt = learner.policy_optimizer
    _save_rl_checkpoint.value_opt = learner.value_optimizer

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

    from smashbot.rl.pool import SnapshotPool, make_partition

    rcfg = args.rollouts
    if not rcfg.log_tag:
        rcfg.log_tag = args.runtime.tag
    specs = make_partition(
        rcfg.num_envs, rcfg.cpu_envs, rcfg.teacher_envs,
        rcfg.snapshot_slots, rcfg.main12_prob, rcfg.partition_seed,
        ref_envs=rcfg.ref_envs,
    )
    opponents = {}
    slot_policies = []
    counts = {}
    for spec in specs:
        key = "teacher" if spec.kind == "teacher" else (
            ("slot", spec.group) if spec.kind == "snapshot" else (
                "reference" if spec.kind == "reference" else None
            )
        )
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    if "teacher" in counts:
        opponents["teacher"] = BatchedPolicyAgent(
            teacher, counts["teacher"], name_code=name_code, device=device,
            batch_steps=rcfg.batch_steps,
        )
    if "reference" in counts:
        # the ported medium-v2 (see scripts/port_ref_model.py): verified
        # 6.3e-13 vs TF at fp64. Delay 21 and its own name_map ride along
        # in the checkpoint; condition on ITS "Master Player" code.
        ref_policy, ref_names, _ = load_policy(rcfg.ref_ckpt, device)
        ref_policy.train_value_head = False
        ref_policy.requires_grad_(False)
        ref_policy.eval()
        if args.runtime.compile:
            # "default" mode: no CUDA-graph private pools. The learner peak
            # runs the GPU at its edge (2x 76MB-OOM at 22.4GB); opponent
            # policies trade a few ms/tick for ~1GB of headroom.
            ref_policy.sample = torch.compile(ref_policy.sample, mode="default")
        ref_code = resolve_name_code(ref_names, "Master Player")
        opponents["reference"] = BatchedPolicyAgent(
            ref_policy, counts["reference"], name_code=ref_code,
            device=device, batch_steps=rcfg.batch_steps,
        )
        print(f"reference: {rcfg.ref_ckpt} (delay {ref_policy.delay}, "
              f"name code {ref_code})")
    for key, n in counts.items():
        if key in ("teacher", "reference"):
            continue
        slot_policy, _, _ = load_policy(args.ckpt, device)  # init = teacher
        slot_policy.train_value_head = False
        slot_policy.requires_grad_(False)
        slot_policy.eval()
        if args.runtime.compile:
            slot_policy.sample = torch.compile(
                slot_policy.sample, mode="default"  # headroom > launch overhead
            )
        slot_policies.append((key[1], slot_policy))
        opponents[key] = BatchedPolicyAgent(
            slot_policy, n, name_code=name_code, device=device,
            batch_steps=rcfg.batch_steps,
        )
    snapshot_pool = SnapshotPool(
        f"{args.runtime.run_dir}/{args.runtime.tag}/snapshots",
        slots=rcfg.snapshot_slots,
    )
    worker = DolphinRolloutWorker(
        args.rollouts, student_agent, opponents=opponents, specs=specs
    )

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
        for i in range(start_step, args.runtime.steps):
            if (
                rcfg.snapshot_slots
                and i > 0
                and i % rcfg.snapshot_interval == 0
            ):
                snapshot_pool.save(policy, i)
                import random as _random

                assigns = snapshot_pool.assignments(_random.Random(i))
                for slot, slot_policy in slot_policies:
                    if slot < len(assigns):
                        slot_policy.load_state_dict(
                            torch.load(assigns[slot], map_location=device)
                        )
                print(f"[{i}] snapshot saved; slots refreshed")

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
                # frames THIS BOOT only: after a restore, i includes the
                # restored steps but t0 is boot time — crediting them made
                # fps read ~80k until the ghost frames washed out.
                frames = (
                    (i + 1 - start_step) * args.runtime.trajectories_per_step
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
                for kind, tracker in worker.trackers.items():
                    for k, v in tracker.stats().items():
                        log[f"rl/{kind}/{k}"] = v
                log["rl/frames_per_sec"] = frames / (time.time() - t0)
                wandb.log(log, step=i)
                games = sum(
                    log.get(f"rl/{k}/games_played", 0)
                    for k in ("cpu", "teacher", "snapshot", "reference")
                )
                ref_bit = (
                    f"R:{log.get('rl/reference/win_rate_recent', 0.5):.0%} "
                    if worker.ref_idx else ""
                )
                print(
                    f"[{i:4d}/{args.runtime.steps}] "
                    f"T:{log.get('rl/teacher/win_rate_recent', 0.5):.0%}"
                    f"/{log.get('rl/teacher/avg_stock_diff', 0):+.1f} "
                    f"S:{log.get('rl/snapshot/win_rate_recent', 0.5):.0%} "
                    f"C:{log.get('rl/cpu/win_rate_recent', 0.5):.0%} "
                    f"{ref_bit}"
                    f"({games:.0f}g) | "
                    f"kill@{log.get('rl/teacher/avg_percent_at_kill', 0):.0f}% "
                    f"die@{log.get('rl/teacher/avg_percent_at_death', 0):.0f}% | "
                    f"tKL {log['rl/teacher_kl']:.4f} "
                    f"aKL {log['rl/actor_kl_mean']:.5f} "
                    f"{'REVERTED ' if log['rl/reverted'] else ''}| "
                    f"{log['rl/frames_per_sec']:.0f} fps",
                    flush=True,
                )

            if (i + 1) % args.runtime.checkpoint_interval == 0:
                _save_rl_checkpoint(
                    f"{run_dir}/latest.pt", ckpt["config"], policy, value_fn,
                    name_map, i, args.ckpt,
                )
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
