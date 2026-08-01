"""Imitation learning (behavior cloning) training loop.

The production harness: train/eval split, separate policy and value networks
with separate optimizers, periodic eval on held-out games (key metric:
eval/policy_loss), best-eval + latest checkpoints with resume, wandb logging,
and a marginal-distribution baseline computed at startup.

Usage (from repo root):
  .venv/bin/python -m shinebot.train_bc --tag debug-fox-v0
  .venv/bin/python -m shinebot.train_bc --tag debug-fox-v0 --restore auto  # resume
"""

import dataclasses
import math
import os
import time
import typing as tp

import numpy as np
import torch
import tree
import tyro

from shinebot import configs, embed as embed_lib, saving
from shinebot.data import loader
from shinebot.delay import slice_delayed_frames
from shinebot.networks import build_embed_network
from shinebot.policy import build_policy
from shinebot.value import ValueFunction


@dataclasses.dataclass
class RuntimeConfig:
    steps: int = 20000
    eval_interval: int = 500
    eval_batches: int = 8
    log_interval: int = 50
    checkpoint_interval: int = 1000
    tag: str = "debug"
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    wandb_mode: str = "online"  # online | offline | disabled
    restore: str = ""  # checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_baseline: bool = True


@dataclasses.dataclass
class TrainConfig:
    data: configs.DataConfig = dataclasses.field(default_factory=configs.DataConfig)
    policy: configs.PolicyConfig = dataclasses.field(default_factory=configs.PolicyConfig)
    network: configs.NetworkConfig = dataclasses.field(default_factory=configs.NetworkConfig)
    head: configs.ControllerHeadConfig = dataclasses.field(
        default_factory=configs.ControllerHeadConfig
    )
    value: configs.ValueConfig = dataclasses.field(default_factory=configs.ValueConfig)
    learner: configs.LearnerConfig = dataclasses.field(
        default_factory=configs.LearnerConfig
    )
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)

    def __post_init__(self):
        if self.data.dataset.data_dir is None:
            root = "/home/kage/drive2/ShineBot/data/debug-fox/Root"
            self.data.dataset.data_dir = f"{root}/Parsed"
            self.data.dataset.meta_path = f"{root}/meta.json"
            self.data.dataset.allowed_characters = "fox"
            self.data.dataset.allowed_opponents = "all"


def marginal_baseline(source, num_batches: int = 10) -> float:
    """NLL of eval targets under component-wise marginal frequencies.

    The 'dumbest possible model' — predicting overall action frequencies with
    no state input. Eval loss must beat this or the model learned nothing.
    """
    counts: dict[str, np.ndarray] = {}

    def accumulate(batch):
        actions = batch.game.p0.controller
        flat = {
            **{f"b_{f}": getattr(actions.buttons, f) for f in actions.buttons._fields},
            "main_x": actions.main_stick.x, "main_y": actions.main_stick.y,
            "c_x": actions.c_stick.x, "c_y": actions.c_stick.y,
            "shoulder": actions.shoulder,
        }
        for k, v in flat.items():
            v = np.asarray(v)
            if v.dtype == np.float32:  # sticks/shoulder: discretize like training
                n = 16 if k != "shoulder" else 4
                v = (v * n + 0.5).astype(np.int64)
            else:
                v = v.astype(np.int64)
            hist = np.bincount(v.reshape(-1), minlength=18)
            counts[k] = counts.get(k, 0) + hist

    nll_sum, n_frames = 0.0, 0
    batches = []
    for _ in range(num_batches):
        batch_with_meta, _ = next(source)
        batches.append(batch_with_meta.batch)
        accumulate(batch_with_meta.batch)

    probs = {k: (c + 1e-9) / (c + 1e-9).sum() for k, c in counts.items()}
    for batch in batches:
        actions = batch.game.p0.controller
        flat = {
            **{f"b_{f}": getattr(actions.buttons, f) for f in actions.buttons._fields},
            "main_x": actions.main_stick.x, "main_y": actions.main_stick.y,
            "c_x": actions.c_stick.x, "c_y": actions.c_stick.y,
            "shoulder": actions.shoulder,
        }
        total = 0.0
        for k, v in flat.items():
            v = np.asarray(v)
            if v.dtype == np.float32:
                n = 16 if k != "shoulder" else 4
                v = (v * n + 0.5).astype(np.int64)
            else:
                v = v.astype(np.int64)
            total += -np.log(probs[k][v.reshape(-1)]).sum()
        nll_sum += total
        n_frames += v.size
    return nll_sum / n_frames


def main(config: TrainConfig) -> None:
    rt = config.runtime
    run_dir = os.path.join(rt.run_dir, rt.tag)
    os.makedirs(run_dir, exist_ok=True)
    device = rt.device
    discount = 0.5 ** (1 / (config.value.reward_halflife * 60))

    sources = loader.make_sources(config.data, extra_frames=config.policy.delay + 1)
    print(f"name_map: {sources.name_map}")

    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=config.head.axis_spacing,
            shoulder_spacing=config.head.shoulder_spacing,
        ),
        network_config=config.network,
        head_config=config.head,
        policy_config=config.policy,
        num_names=config.data.max_names,
    ).to(device)
    policy.train_value_head = False  # separate value network (production config)

    value_net_config = configs.NetworkConfig(
        hidden_size=config.value.hidden_size, num_layers=config.value.num_layers
    )
    value_fn = ValueFunction(
        build_embed_network(
            embed_config=embed_lib.EmbedConfig(),
            controller_embedding=embed_lib.ControllerConfig(
                axis_spacing=config.head.axis_spacing,
                shoulder_spacing=config.head.shoulder_spacing,
            ).make_embedding(),
            num_names=config.data.max_names,
            network_config=value_net_config,
        )
    ).to(device)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=config.learner.learning_rate)
    value_opt = torch.optim.Adam(value_fn.parameters(), lr=config.learner.learning_rate)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params/1e6:.1f}M params | value: "
          f"{sum(p.numel() for p in value_fn.parameters())/1e6:.1f}M params | "
          f"delay={config.policy.delay} | device={device}")

    step = 0
    best_eval_loss = math.inf
    if rt.restore:
        path = os.path.join(run_dir, "latest.pt") if rt.restore == "auto" else rt.restore
        ckpt = saving.load_checkpoint(path)
        policy.load_state_dict(ckpt["state"]["policy"])
        value_fn.load_state_dict(ckpt["state"]["value"])
        policy_opt.load_state_dict(ckpt["state"]["policy_opt"])
        value_opt.load_state_dict(ckpt["state"]["value_opt"])
        step = ckpt["state"]["step"]
        best_eval_loss = ckpt["best_eval_loss"]
        print(f"restored from {path} at step {step} (best eval {best_eval_loss:.4f})")

    import wandb

    wandb.init(
        project="shinebot",
        group="imitation",
        name=rt.tag,
        mode=rt.wandb_mode,
        config=dataclasses.asdict(config),
        resume="allow",
        id=rt.tag,
    )

    baseline = None
    if rt.compute_baseline:
        baseline = marginal_baseline(sources.test)
        print(f"marginal-distribution baseline (eval NLL): {baseline:.4f}")
        wandb.summary["baseline_nll"] = baseline

    B = config.data.batch_size
    train_hidden = policy.initial_state(B, device)
    value_hidden = value_fn.initial_state(B, device)
    eval_hidden = policy.initial_state(B, device)
    eval_value_hidden = value_fn.initial_state(B, device)

    train_stream = loader.TorchBatchStream(
        sources.train, config.data, encode_network=policy.network
    )
    eval_stream = loader.TorchBatchStream(
        sources.test, config.data, encode_network=policy.network
    )

    def to_device(frames):
        return tree.map_structure(lambda t: t.to(device, non_blocking=True), frames)

    def detach(state):
        return tree.map_structure(lambda t: t.detach(), state)

    def save(path_name: str):
        saving.save_checkpoint(
            os.path.join(run_dir, path_name),
            config,
            {
                "policy": policy.state_dict(),
                "value": value_fn.state_dict(),
                "policy_opt": policy_opt.state_dict(),
                "value_opt": value_opt.state_dict(),
                "step": step,
                "name_map": sources.name_map,
            },
            best_eval_loss,
        )

    def run_eval() -> dict:
        nonlocal eval_hidden, eval_value_hidden
        policy.eval()
        losses, value_metrics_acc = [], []
        with torch.no_grad():
            for _ in range(rt.eval_batches):
                frames, _ = next(eval_stream)
                frames = to_device(frames)
                loss, eval_hidden, m = policy.imitation_loss(frames, eval_hidden)
                sliced = slice_delayed_frames(frames, config.policy.delay)
                _, eval_value_hidden, vm = value_fn.loss(
                    sliced, eval_value_hidden, discount
                )
                losses.append(m["policy_loss"])
                value_metrics_acc.append(vm)
        policy.train()
        return {
            "policy_loss": float(np.mean(losses)),
            "value_uev": float(np.mean([m["uev"] for m in value_metrics_acc])),
        }

    print(f"training to step {rt.steps}; eval every {rt.eval_interval} steps")
    t0 = time.perf_counter()
    start_step = step
    try:
        while step < rt.steps:
            step += 1
            frames, epoch = next(train_stream)
            frames = to_device(frames)

            policy_loss, train_hidden, metrics = policy.imitation_loss(
                frames, train_hidden
            )
            train_hidden = detach(train_hidden)
            policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_opt.step()

            sliced = slice_delayed_frames(frames, config.policy.delay)
            sliced = tree.map_structure(lambda t: t.detach(), sliced)
            value_loss, value_hidden, value_metrics = value_fn.loss(
                sliced, value_hidden, discount
            )
            value_hidden = detach(value_hidden)
            value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_opt.step()

            if step % rt.log_interval == 0:
                dt = time.perf_counter() - t0
                fps = (step - start_step) * B * config.data.unroll_length / dt
                wandb.log(
                    {
                        "train/policy_loss": metrics["policy_loss"],
                        "train/epoch": epoch,
                        "train/frames_per_sec": fps,
                        **{f"train/controller/{k}": v
                           for k, v in metrics["controller_flat"].items()},
                        "train/value/loss": value_metrics["loss"],
                        "train/value/uev": value_metrics["uev"],
                    },
                    step=step,
                )
                print(
                    f"step {step:6d}  epoch {epoch:6.1f}  "
                    f"train {metrics['policy_loss']:7.4f}  "
                    f"value_uev {value_metrics['uev']:5.2f}  "
                    f"{fps/1e3:.0f}k frames/s"
                )

            if step % rt.eval_interval == 0:
                eval_metrics = run_eval()
                is_best = eval_metrics["policy_loss"] < best_eval_loss
                if is_best:
                    best_eval_loss = eval_metrics["policy_loss"]
                    save("best.pt")
                wandb.log(
                    {
                        "eval/policy_loss": eval_metrics["policy_loss"],
                        "eval/value_uev": eval_metrics["value_uev"],
                        "eval/best_policy_loss": best_eval_loss,
                    },
                    step=step,
                )
                marker = " *best*" if is_best else ""
                base = f"  (baseline {baseline:.3f})" if baseline else ""
                print(
                    f"  eval {eval_metrics['policy_loss']:7.4f}{base}{marker}"
                )

            if step % rt.checkpoint_interval == 0:
                save("latest.pt")
    finally:
        save("latest.pt")
        train_stream.stop()
        eval_stream.stop()
        wandb.finish()
        print(f"done at step {step}; best eval {best_eval_loss:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
