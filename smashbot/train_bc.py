"""Imitation learning (behavior cloning) training loop.

The production harness: train/eval split, separate policy and value networks
with separate optimizers, periodic eval on held-out games (key metric:
eval/policy_loss), best-eval + latest checkpoints with resume, wandb logging,
and tqdm/wandb progress reporting.

Usage (from repo root):
  .venv/bin/python -m smashbot.train_bc --runtime.tag exp-baseline
  .venv/bin/python -m smashbot.train_bc --runtime.tag exp-baseline --runtime.restore auto
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

from smashbot import configs, embed as embed_lib, saving
from smashbot.data import loader
from smashbot.delay import slice_delayed_frames
from smashbot.networks import build_embed_network
from smashbot.policy import build_policy
from smashbot.value import ValueFunction


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
            root = "/home/kage/drive2/ShineBot/data/full/Root"
            self.data.dataset.data_dir = f"{root}/Parsed"
            # default: the seeded 20k experiment subset; big runs use meta.json
            self.data.dataset.meta_path = f"{root}/meta-20k.json"


def _state_rows(state, lo, hi, batch):
    """Rows [lo, hi) of a recurrent state; leaves are [B, ...] or torch-RNN [layers, B, H]."""
    def take(t):
        if not isinstance(t, torch.Tensor):
            return t
        return t[lo:hi] if t.dim() >= 1 and t.shape[0] == batch else t[:, lo:hi]
    return tree.map_structure(take, state)


def _state_cat(parts, batch):
    def cat(*ts):
        if not isinstance(ts[0], torch.Tensor):
            return ts[0]
        dim = 0 if ts[0].dim() >= 1 and sum(t.shape[0] for t in ts) == batch else 1
        return torch.cat(ts, dim=dim)
    return tree.map_structure(cat, *parts)



def _check_architecture(saved: dict, config: "TrainConfig") -> None:
    """Weights load into any model of the same shapes, so settings that change
    behavior without changing shapes (attn_rope) are only caught by value.
    Keys a checkpoint predates take their defaults."""
    for section, cls in (
        ("network", configs.NetworkConfig),
        ("head", configs.ControllerHeadConfig),
        ("policy", configs.PolicyConfig),
        ("value", configs.ValueConfig),
    ):
        then, now = cls(**saved[section]), getattr(config, section)
        if then != now:
            raise ValueError(f"--{section}.* differs from the checkpoint:\n  saved   {then}\n  current {now}")


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
    start_replay = 0
    start_test_replay = 0
    restored_eval_state = (None, None)
    if rt.restore:
        restore_path = (
            os.path.join(run_dir, "latest.pt") if rt.restore == "auto" else rt.restore
        )
        _restored = saving.load_checkpoint(restore_path)
        _check_architecture(_restored["config"], config)
        _rs = _restored["state"]
        restored_name_map = _rs.get("name_map")
        start_replay = _rs.get("replay_counter", 0)
        start_test_replay = _rs.get("test_replay_counter", 0)
        restored_eval_state = (_rs.get("eval_hidden"), _rs.get("eval_value_hidden"))

    sources = loader.make_sources(
        config.data,
        extra_frames=config.policy.delay + 1,
        name_map=restored_name_map,
        start_replay=start_replay,
        start_test_replay=start_test_replay,
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

    value_name = config.value.name
    if value_name == "match":
        value_name = config.network.name
    value_net_config = configs.NetworkConfig(
        name=value_name,
        hidden_size=config.value.hidden_size,
        num_layers=config.value.num_layers,
        num_heads=config.network.num_heads,
        window=config.value.window or config.network.window,
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
    if restored_eval_state[0] is not None:   # evals carry state across calls; restore it too
        eval_hidden, eval_value_hidden = tree.map_structure(
            lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, restored_eval_state)

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
                "replay_counter": sources.train.replay_counter,
                "test_replay_counter": sources.test.replay_counter,
                "eval_hidden": tree.map_structure(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, eval_hidden),
                "eval_value_hidden": tree.map_structure(lambda t: t.cpu() if isinstance(t, torch.Tensor) else t, eval_value_hidden),
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
        print(f"eval @ {step}: policy_loss {sum(losses) / len(losses):.6f}", flush=True)
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

            k = config.learner.grad_accum
            bounds = [(i * B // k, (i + 1) * B // k) for i in range(k)]
            mean = lambda ms: tree.map_structure(lambda *xs: sum(xs) / len(xs), *ms)

            policy_opt.zero_grad(set_to_none=True)
            hids, ms = [], []
            for lo, hi in bounds:
                fr = frames if k == 1 else tree.map_structure(lambda t: t[lo:hi], frames)
                hid = train_hidden if k == 1 else _state_rows(train_hidden, lo, hi, B)
                with autocast():
                    policy_loss, hid, m = policy_loss_fn(fr, hid)
                (policy_loss / k).backward()
                hids.append(detach(hid)); ms.append(m)
            train_hidden = hids[0] if k == 1 else _state_cat(hids, B)
            metrics = mean(ms)
            if config.learner.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), config.learner.max_grad_norm
                )
            policy_opt.step()

            value_opt.zero_grad(set_to_none=True)
            hids, ms = [], []
            for lo, hi in bounds:
                fr = frames if k == 1 else tree.map_structure(lambda t: t[lo:hi], frames)
                sliced = slice_delayed_frames(fr, config.policy.delay)
                sliced = tree.map_structure(lambda t: t.detach(), sliced)
                hid = value_hidden if k == 1 else _state_rows(value_hidden, lo, hi, B)
                with autocast():
                    value_loss, hid, m = value_loss_fn(sliced, hid, discount)
                (value_loss / k).backward()
                hids.append(detach(hid)); ms.append(m)
            value_hidden = hids[0] if k == 1 else _state_cat(hids, B)
            value_metrics = mean(ms)
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
