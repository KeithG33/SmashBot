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
import random
import signal
import time
import typing as tp

import numpy as np
import torch
import tree
import tyro

from smashbot import configs, embed as embed_lib, saving
from smashbot.data import loader
from smashbot.delay import slice_delayed_frames
from smashbot.networks import build_embed_network, check_loadable
from smashbot.policy import build_policy
from smashbot.training import GradClipper, compile_cores
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


def _to(state, device):
    return tree.map_structure(
        lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, state)


def _get_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def _resume_state(ckpt: tp.Optional[dict]) -> dict:
    """The checkpoint's state, with pre-row-tracking checkpoints mapped onto
    the same keys (cycle position only; rows restart fresh)."""
    if ckpt is None:
        return {}
    state = dict(ckpt["state"])
    if "train_data" not in state:
        print("WARNING: checkpoint predates exact resume; data rows restart at "
              "fresh replays and the run is not a bit-identical continuation")
        state["train_data"] = {"consumed": state.get("replay_counter", 0), "rows": None}
        state["test_data"] = {"consumed": state.get("test_replay_counter", 0), "rows": None}
    return state


_RESUME_FREE = {"runtime", "data.num_workers", "data.prefetch", "data.pin_memory"}


def _check_config(saved: dict, current: dict, defaults: dict, prefix: str = "") -> None:
    """A resumed run must be the same experiment; the runtime block and
    host-only data options are the only fields free to change. A setting the
    checkpoint predates ran at its default, so that is what it is held to;
    a setting the code no longer has is ignored (see configs.from_dict)."""
    for key in sorted(set(saved) | set(current)):
        path = f"{prefix}{key}"
        if path in _RESUME_FREE or path.split(".")[0] in _RESUME_FREE:
            continue
        if key not in current and key not in defaults:
            continue   # a removed setting: its behaviour became unconditional
        a, b = saved.get(key, defaults.get(key)), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            _check_config(a, b, defaults.get(key) or {}, prefix=f"{path}.")
        elif a != b:
            raise ValueError(f"config.{path} differs from the checkpoint: {a!r} -> {b!r}")


class _StopRequest:
    """First SIGINT/SIGTERM finishes the current step and checkpoints;
    a second one interrupts immediately."""

    def __init__(self):
        self.requested = False
        self.previous = {
            sig: signal.signal(sig, self._handle) for sig in (signal.SIGINT, signal.SIGTERM)
        }

    def restore(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print(f"stop requested (signal {signum}); finishing the current step", flush=True)


def main(config: TrainConfig) -> None:
    rt = config.runtime
    run_dir = os.path.join(rt.run_dir, rt.tag)
    os.makedirs(run_dir, exist_ok=True)
    device = rt.device
    torch.manual_seed(rt.seed)
    np.random.seed(rt.seed)
    discount = 0.5 ** (1 / (config.value.reward_halflife * 60))

    ckpt = None
    if rt.restore:
        restore_path = (
            os.path.join(run_dir, "latest.pt") if rt.restore == "auto" else rt.restore
        )
        ckpt = saving.load_checkpoint(restore_path)
        config.data.dataset.validate()
        _check_config(ckpt["config"], dataclasses.asdict(config), dataclasses.asdict(TrainConfig()))
    resume = _resume_state(ckpt)

    sources = loader.make_sources(
        config.data,
        extra_frames=config.policy.delay + 1,
        name_map=resume.get("name_map"),
        train_state=resume.get("train_data"),
        test_state=resume.get("test_data"),
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

    value_name = config.value.name
    if value_name == "match":
        value_name = config.network.name
    value_net_config = configs.NetworkConfig(
        name=value_name,
        hidden_size=config.value.hidden_size,
        num_layers=config.value.num_layers,
        num_heads=config.network.num_heads,
        window=config.value.window or config.network.window,
        layout=config.value.layout,
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
    # AutoClip is for the policy only: the value net's step-to-step norms swing
    # 4x, so a percentile threshold throttles its typical step (uev +3%, measured)
    clip_policy = GradClipper(policy.parameters(), config.learner.max_grad_norm,
                              config.learner.autoclip_percentile,
                              (resume.get("clip_history") or {}).get("policy"))
    clip_value = GradClipper(value_fn.parameters(), config.learner.max_grad_norm)

    if config.learner.precision == "bf16" and device == "cuda":
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext

    policy_loss_fn = policy.imitation_loss
    value_loss_fn = value_fn.loss
    if config.learner.compile and device == "cuda":
        compile_cores(policy, value_fn)
        print("torch.compile enabled (first steps will be slow while compiling)")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy: {n_params/1e6:.1f}M params | value: "
          f"{sum(p.numel() for p in value_fn.parameters())/1e6:.1f}M params | "
          f"delay={config.policy.delay} | device={device}")

    step = 0
    best_eval_loss = math.inf
    if ckpt is not None:
        check_loadable(ckpt["config"]["network"], resume["policy"])
        check_loadable({}, resume["value"])   # value nets never had the flags: uv.weight means pre-paper
        policy.load_state_dict(resume["policy"])
        value_fn.load_state_dict(resume["value"])
        saving.load_optimizer(policy_opt, resume["policy_opt"], resume["policy"])
        value_opt.load_state_dict(resume["value_opt"])
        step = resume["step"]
        best_eval_loss = ckpt["best_eval_loss"]
        print(f"restored from {restore_path} at step {step} (best eval {best_eval_loss:.4f})")

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
    hidden = {
        "train_hidden": policy.initial_state(B, device),
        "value_hidden": value_fn.initial_state(B, device),
        "eval_hidden": policy.initial_state(B, device),
        "eval_value_hidden": value_fn.initial_state(B, device),
    }
    for key in hidden:
        if resume.get(key) is not None:
            hidden[key] = _to(resume[key], device)
    train_hidden, value_hidden = hidden["train_hidden"], hidden["value_hidden"]
    eval_hidden, eval_value_hidden = hidden["eval_hidden"], hidden["eval_value_hidden"]
    if resume.get("rng") is not None:
        _set_rng(resume["rng"])
    train_data_state = resume.get("train_data")
    test_data_state = resume.get("test_data")

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
                "train_data": train_data_state,
                "test_data": test_data_state,
                "train_hidden": _to(train_hidden, "cpu"),
                "value_hidden": _to(value_hidden, "cpu"),
                "eval_hidden": _to(eval_hidden, "cpu"),
                "eval_value_hidden": _to(eval_value_hidden, "cpu"),
                "rng": _get_rng(),
                "clip_history": {"policy": clip_policy.history},
            },
            best_eval_loss,
        )

    def run_eval() -> dict:
        nonlocal eval_hidden, eval_value_hidden, test_data_state
        policy.eval()
        losses, value_metrics_acc = [], []
        with torch.no_grad():
            for _ in range(rt.eval_batches):
                frames, _, test_data_state = next(eval_stream)
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
    stop = _StopRequest()
    at_boundary = True
    try:
        while step < rt.steps and not stop.requested:
            at_boundary = False
            step += 1
            frames, epoch, train_data_state = next(train_stream)
            frames = to_device(frames)

            policy_opt.zero_grad(set_to_none=True)
            with autocast():
                policy_loss, train_hidden, metrics = policy_loss_fn(frames, train_hidden)
            policy_loss.backward()
            train_hidden = detach(train_hidden)
            clip_metrics = clip_policy()
            policy_opt.step()

            value_opt.zero_grad(set_to_none=True)
            sliced = slice_delayed_frames(frames, config.policy.delay)
            with autocast():
                value_loss, value_hidden, value_metrics = value_loss_fn(sliced, value_hidden, discount)
            value_loss.backward()
            value_hidden = detach(value_hidden)
            value_clip_metrics = clip_value()
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
                        **{f"train/{k}": v for k, v in clip_metrics.items() if math.isfinite(v)},
                        **{f"train/value/{k}": v for k, v in value_clip_metrics.items() if math.isfinite(v)},
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
            at_boundary = True
    finally:
        stop.restore()
        pbar.close()
        if at_boundary:
            save("latest.pt")
        else:
            print(f"interrupted mid-step {step}; latest.pt left as it was")
        train_stream.stop()
        eval_stream.stop()
        wandb.finish()
        print(f"done at step {step}; best eval {best_eval_loss:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
