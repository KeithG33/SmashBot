"""Imitation learning (behavior cloning) training loop.

The production harness: train/eval split, separate policy and value networks
with separate optimizers, periodic eval on held-out games (key metric:
eval/policy_loss), best-eval + latest checkpoints with resume, wandb logging,
and tqdm/wandb progress reporting.

Usage (from repo root):
  .venv/bin/python -m shinebot.train_bc --tag debug-fox-v0
  .venv/bin/python -m shinebot.train_bc --tag debug-fox-v0 --restore auto  # resume
"""

import contextlib
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
    seed: int = 0  # seeds model init; makes A/B runs attributable


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


def main(config: TrainConfig) -> None:
    rt = config.runtime
    run_dir = os.path.join(rt.run_dir, rt.tag)
    os.makedirs(run_dir, exist_ok=True)
    device = rt.device
    torch.manual_seed(rt.seed)
    np.random.seed(rt.seed)
    discount = 0.5 ** (1 / (config.value.reward_halflife * 60))

    # On resume, the checkpoint's name_map is authoritative: indices are
    # frequency-assigned, so recomputing on changed data would permute them.
    restored_name_map = None
    if rt.restore:
        restore_path = (
            os.path.join(run_dir, "latest.pt") if rt.restore == "auto" else rt.restore
        )
        restored_name_map = saving.load_checkpoint(restore_path)["state"].get("name_map")

    sources = loader.make_sources(
        config.data,
        extra_frames=config.policy.delay + 1,
        name_map=restored_name_map,
    )
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

    if config.learner.precision == "bf16" and device == "cuda":
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    policy_loss_fn = policy.imitation_loss
    value_loss_fn = value_fn.loss
    if config.learner.compile and device == "cuda":
        # Whole-loss compile: dynamo graph-breaks around the cuDNN LSTM (fine)
        # and fuses the embedding/head/return math around it.
        policy_loss_fn = torch.compile(policy_loss_fn)
        value_loss_fn = torch.compile(value_loss_fn)
        print("torch.compile enabled (first steps will be slow while compiling)")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params/1e6:.1f}M params | value: "
          f"{sum(p.numel() for p in value_fn.parameters())/1e6:.1f}M params | "
          f"delay={config.policy.delay} | device={device}")

    step = 0
    best_eval_loss = math.inf
    if rt.restore:
        path = restore_path
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
                with autocast():
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

    from tqdm import tqdm

    pbar = tqdm(
        total=rt.steps, initial=step, unit="step", dynamic_ncols=True,
        desc=rt.tag, smoothing=0.05,
    )
    t_window = time.perf_counter()
    step_window = step
    try:
        while step < rt.steps:
            step += 1
            frames, epoch = next(train_stream)
            frames = to_device(frames)

            with autocast():
                policy_loss, train_hidden, metrics = policy_loss_fn(
                    frames, train_hidden
                )
            train_hidden = detach(train_hidden)
            policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            if config.learner.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.learner.max_grad_norm
                )
            policy_opt.step()

            sliced = slice_delayed_frames(frames, config.policy.delay)
            sliced = tree.map_structure(lambda t: t.detach(), sliced)
            with autocast():
                value_loss, value_hidden, value_metrics = value_loss_fn(
                    sliced, value_hidden, discount
                )
            value_hidden = detach(value_hidden)
            value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            if config.learner.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    value_fn.parameters(), config.learner.max_grad_norm
                )
            value_opt.step()

            if step % rt.log_interval == 0:
                now = time.perf_counter()
                fps = (step - step_window) * B * config.data.unroll_length / (
                    now - t_window
                )
                t_window, step_window = now, step
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
                pbar.set_postfix(
                    train=f"{metrics['policy_loss']:.4f}",
                    best_eval=(
                        f"{best_eval_loss:.4f}" if best_eval_loss < math.inf else "-"
                    ),
                    epoch=f"{epoch:.1f}",
                    fps=f"{fps/1e3:.0f}k",
                    refresh=False,
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
                pbar.write(
                    f"step {step:6d}  eval {eval_metrics['policy_loss']:7.4f}{marker}"
                )

            if step % rt.checkpoint_interval == 0:
                save("latest.pt")
            pbar.update(1)
    finally:
        pbar.close()
        save("latest.pt")
        train_stream.stop()
        eval_stream.stop()
        wandb.finish()
        print(f"done at step {step}; best eval {best_eval_loss:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
