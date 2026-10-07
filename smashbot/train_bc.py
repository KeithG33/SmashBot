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
import functools
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
from smashbot.networks import (check_loadable, keep_sgu_activations,
                               use_chunk_start_resets, use_packed_encoder)
from smashbot.policy import ActionLoss, EnergyScore, StickScorer, build_policy, imitation_metrics, stick_metrics
from smashbot.training import GradClipper, compile_cores, resolve_restore
from smashbot.value import build_value_function


@dataclasses.dataclass
class RuntimeConfig:
    steps: int = 20000
    eval_interval: int = 500
    # the eval set (eval/*, which picks best.pt): eval_groups groups of
    # eval_rows random test games, drawn once from a constant seed so every
    # eval scores the same frames; each row runs eval_burn_in frames of its own
    # history unscored, so the current model scores from its own warm state,
    # then eval_batches scored batches
    eval_groups: int = 4
    eval_rows: int = 1024
    eval_batches: int = 16
    eval_burn_in: int = 256
    # eval_wide/*: every eval also scores this many groups of fresh random
    # test games, shaped like the eval set (0: off)
    wide_eval_groups: int = 0
    log_interval: int = 50
    checkpoint_interval: int = 1000
    tag: str = "debug"
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    # online | offline | disabled; $WANDB_MODE sets the default, since this
    # value goes to wandb.init and would otherwise override the variable
    wandb_mode: str = os.environ.get("WANDB_MODE", "online")
    restore: str = ""  # checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
    # a fresh run's starting policy and value weights, from a checkpoint built
    # for this model config (smashbot.warm_start's); a restore takes precedence
    init_from: str = ""
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
        for device, state in enumerate(rng["cuda"][:torch.cuda.device_count()]):   # the GPUs this machine has
            torch.cuda.set_rng_state(state, device)


class _FiniteWatch:
    """Each step's losses and gradient norms, checked without making the CPU
    wait for the GPU: fused Adam skips a non-finite step on the GPU
    (found_inf), and the host reads the flags a step later, or before a
    save, and stops BC, since the data cursor and the recurrent state have
    already moved past the batch."""

    def __init__(self, optimizers, device: str):
        self.optimizers = optimizers
        self.flags = torch.zeros(4, pin_memory=device == "cuda")   # policy loss, norm, value loss, norm
        self.done = torch.cuda.Event() if device == "cuda" else None
        self.step = None

    def update(self, step: int, policy_loss, policy_norm, value_loss, value_norm) -> None:
        self.check()
        values = torch.stack([policy_loss.detach().float(), policy_norm, value_loss.detach().float(), value_norm])
        found_inf = (~torch.isfinite(values).all()).float()
        for opt in self.optimizers:
            opt.found_inf = found_inf
        self.flags.copy_(values, non_blocking=True)
        if self.done is not None:
            self.done.record()
        self.step = step

    def check(self) -> None:
        if self.step is None:
            return
        if self.done is not None:
            self.done.synchronize()
        values = self.flags.tolist()
        for net, loss, norm in zip(("policy", "value"), values[::2], values[1::2]):
            if not (math.isfinite(loss) and math.isfinite(norm)):
                raise FloatingPointError(f"step {self.step}: non-finite {net} update (loss {loss}, grad norm {norm})")
        self.step = None


def _nonfinite(states: dict) -> list:
    """Names of the states holding a NaN or inf in a floating tensor."""
    return [name for name, state in states.items()
            if not all(bool(torch.isfinite(t).all()) for t in tree.flatten(state)
                       if isinstance(t, torch.Tensor) and t.is_floating_point())]


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
    return state


_RESUME_FREE = {"runtime", "data.num_workers", "data.prefetch", "data.pin_memory", "learner.compile"}


def _check_config(saved: dict, current: dict, defaults: dict, prefix: str = "") -> None:
    """A resumed run must be the same experiment; the runtime block, the
    host-only data options and compile (kernel fusion: bf16 rounding, not
    the math) are the only fields free to change. A setting the
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


def score(policy, value_fn, stick_scorer: StickScorer, groups: list, rows: int, delay: int,
          warm_batches: int, discount: float, autocast, device) -> dict:
    """Policy loss, value UEV and stick metrics over groups of `rows` rows:
    each group starts cold, runs warm_batches unscored, then scores the
    rest."""
    policy.eval()
    losses, value_metrics_acc, stick_batches = [], [], []
    with torch.no_grad():
        for group in groups:
            eval_hidden = policy.initial_state(rows, device)
            eval_value_hidden = value_fn.initial_state(rows, device)
            for i, (frames, exact) in enumerate(group):
                frames = tree.map_structure(lambda t: t.to(device, non_blocking=True), frames)
                with autocast():
                    sliced = slice_delayed_frames(frames, delay)
                    outputs = policy.unroll(sliced, eval_hidden, joint_sticks=True)
                    eval_hidden = outputs.final_state
                    _, eval_value_hidden, vm = value_fn.loss(sliced, eval_value_hidden, discount)
                if i >= warm_batches:
                    losses.append(-outputs.log_probs.mean())
                    value_metrics_acc.append(vm)
                    human = tree.map_structure(lambda t: t[:, 1:], sliced.state_action.action)
                    exact = {k: v.to(device)[:, delay + 1:] for k, v in exact.items()}
                    stick_batches.append(stick_scorer.score(outputs.sticks, human, exact))
    policy.train()
    return {"policy_loss": torch.stack(losses).float().mean().item(),
            "value_uev": float(np.mean([m["uev"] for m in value_metrics_acc])),
            "value_loss": float(np.mean([m["loss"] for m in value_metrics_acc])),
            "sticks": stick_metrics(stick_batches)}


def main(config: TrainConfig) -> None:
    dataset = config.data.dataset
    if dataset.data_dir is not None and dataset.meta_path is None:   # the dataset's own full index
        dataset.meta_path = os.path.join(os.path.dirname(dataset.data_dir.rstrip("/")), "meta.json")
    dataset.validate()
    rt = config.runtime
    eval_set_id = (f"{rt.eval_groups}x{rt.eval_rows} games, {rt.eval_burn_in}+{rt.eval_batches}x"
                   f"{config.data.unroll_length} frames, seed {config.data.dataset.seed}")
    run_dir = os.path.join(rt.run_dir, rt.tag)
    restore_path = resolve_restore(run_dir, rt.restore)
    os.makedirs(run_dir, exist_ok=True)
    device = rt.device
    torch.manual_seed(rt.seed)
    np.random.seed(rt.seed)
    discount = 0.5 ** (1 / (config.value.reward_halflife * 60))

    ckpt = None
    if restore_path:
        ckpt = saving.load_checkpoint(restore_path)
        _check_config(ckpt["config"], dataclasses.asdict(config), dataclasses.asdict(TrainConfig()))
    resume = _resume_state(ckpt)
    init = saving.load_checkpoint(rt.init_from) if rt.init_from and ckpt is None else None
    if init is not None:
        for section, cls in (("network", configs.NetworkConfig), ("head", configs.ControllerHeadConfig),
                             ("policy", configs.PolicyConfig), ("value", configs.ValueConfig)):
            built, saved = getattr(config, section), configs.from_dict(cls, init["config"][section])
            if built != saved:
                raise SystemExit(f"--runtime.init-from {rt.init_from} was built for {section} {saved}, "
                                 f"this run's is {built}")

    sources = loader.make_sources(
        config.data,
        extra_frames=config.policy.delay + 1,
        name_map=resume.get("name_map", init["state"]["name_map"] if init else None),
        train_state=resume.get("train_data"),
    )
    print(f"name_map: {sources.name_map}")

    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=config.head.axis_spacing,
            shoulder_spacing=config.head.shoulder_spacing,
            type=config.head.controller_type,
        ),
        network_config=config.network,
        head_config=config.head,
        policy_config=config.policy,
        num_names=config.data.max_names,
    ).to(device)

    value_fn = build_value_function(dataclasses.asdict(config), device)

    use_chunk_start_resets(policy)
    use_chunk_start_resets(value_fn)
    keep_sgu_activations(policy)
    keep_sgu_activations(value_fn)
    use_packed_encoder(policy)
    use_packed_encoder(value_fn)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=config.learner.learning_rate, fused=True)
    value_opt = torch.optim.Adam(value_fn.parameters(), lr=config.learner.learning_rate, fused=True)
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

    extra = [(weight, term(policy.controller_head, device)) for weight, term in (
        (config.learner.energy_score_weight, EnergyScore), (config.learner.action_loss_weight, ActionLoss)) if weight]
    policy_loss_fn = functools.partial(policy.imitation_loss, extra=extra)
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
        # loading keeps the saved optimizer's setup: a non-fused Adam's flags and
        # its step counts on the CPU, where fused Adam needs them on the device
        for opt in (policy_opt, value_opt):
            for group in opt.param_groups:
                group["fused"], group["foreach"] = True, None
            for param, state in opt.state.items():
                state["step"] = state["step"].to(device=param.device, dtype=torch.float32)
        step = resume["step"]
        best_eval_loss = ckpt["best_eval_loss"]
        if resume.get("eval_set") != eval_set_id:   # a best scored on other frames means nothing here
            print(f"eval set changed ({resume.get('eval_set')} -> {eval_set_id}): best eval starts over")
            best_eval_loss = math.inf
        print(f"restored from {restore_path} at step {step} (best eval {best_eval_loss:.4f})")
    elif init is not None:
        policy.load_state_dict(init["state"]["policy"])
        value_fn.load_state_dict(init["state"]["value"])
        print(f"initialized from {rt.init_from}")

    import wandb

    wandb.init(
        project="shinebot",
        group="imitation",
        name=rt.tag,
        mode=rt.wandb_mode,
        config=dataclasses.asdict(config),
        resume="allow" if restore_path else "never",   # a fresh run never lands on an old wandb run
        id=rt.tag,
    )

    B = config.data.batch_size
    hidden = {
        "train_hidden": policy.initial_state(B, device),
        "value_hidden": value_fn.initial_state(B, device),
    }
    for key in hidden:
        if resume.get(key) is not None:
            hidden[key] = _to(resume[key], device)
    train_hidden, value_hidden = hidden["train_hidden"], hidden["value_hidden"]
    if resume.get("rng") is not None:
        _set_rng(resume["rng"])
    train_data_state = resume.get("train_data")

    train_stream = loader.TorchBatchStream(
        sources.train, config.data, encode_network=policy.network
    )
    sources.test.shutdown()   # the evals draw their own rows from its replays
    warm_batches = -(-rt.eval_burn_in // config.data.unroll_length)

    def eval_stream(groups: int, seed: int, interval: int = 0):
        return loader.random_eval_stream(
            sources.test.replays, config.data, config.policy.delay + 1, sources.name_map, policy.network,
            groups=groups, rows=rt.eval_rows, batches=warm_batches + rt.eval_batches, seed=seed,
            interval=interval)

    eval_seed = config.data.dataset.seed * 1_000_003
    fixed = eval_stream(rt.eval_groups, eval_seed)
    eval_set = next(fixed)
    fixed.stop()
    # a wide draw is seeded by its eval's step (never the eval set's seed), so
    # it draws the same games whether or not the run restarted before it
    wide_stream = eval_stream(rt.wide_eval_groups, eval_seed + (step // rt.eval_interval + 1) * rt.eval_interval,
                              rt.eval_interval) if rt.wide_eval_groups else None

    def to_device(frames):
        return tree.map_structure(lambda t: t.to(device, non_blocking=True), frames)

    def detach(state):
        return tree.map_structure(lambda t: t.detach(), state)

    def save(path_name: str):
        watch.check()
        bad = _nonfinite({
            "policy weights": policy.state_dict(), "value weights": value_fn.state_dict(),
            "policy Adam state": policy_opt.state_dict(), "value Adam state": value_opt.state_dict(),
            "train_hidden": train_hidden, "value_hidden": value_hidden,
        })
        if bad:
            raise FloatingPointError(f"refusing to write {path_name} at step {step}: non-finite {', '.join(bad)}")
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
                "train_hidden": _to(train_hidden, "cpu"),
                "value_hidden": _to(value_hidden, "cpu"),
                "rng": _get_rng(),
                "clip_history": {"policy": clip_policy.history},
                "eval_set": eval_set_id,
            },
            best_eval_loss,
        )

    stick_scorer = StickScorer(policy.controller_head, device)

    def score_groups(groups: list, rows: int, name: str) -> dict:
        start = time.perf_counter()
        scores = score(policy, value_fn, stick_scorer, groups, rows, config.policy.delay, warm_batches,
                       discount, autocast, device)
        policy_loss, value_uev, sticks = scores["policy_loss"], scores["value_uev"], scores["sticks"]
        if not (math.isfinite(policy_loss) and math.isfinite(value_uev)):
            raise FloatingPointError(f"step {step}: non-finite {name} (policy loss {policy_loss}, value uev {value_uev})")
        seconds = time.perf_counter() - start
        print(f"{name} @ {step}: policy_loss {policy_loss:.6f}"
              + "".join(f", {k} {v:.4f}" for k, v in sticks.items()) + f" ({seconds:.1f}s)", flush=True)
        return {**scores, "seconds": seconds}

    def run_eval() -> dict:
        return score_groups(eval_set, rt.eval_rows, "eval")

    def run_wide_eval() -> dict:
        return score_groups(next(wide_stream), rt.eval_rows, "eval_wide")

    from tqdm import tqdm

    pbar = tqdm(
        total=rt.steps, initial=step, unit="step", dynamic_ncols=True,
        desc=rt.tag, smoothing=0.05,
    )
    t_window = time.perf_counter()
    step_window = step
    watch = _FiniteWatch((policy_opt, value_opt), device)
    stop = _StopRequest()
    at_boundary = True
    try:
        while step < rt.steps and not stop.requested:
            at_boundary = False
            step += 1
            log_step = step % rt.log_interval == 0
            frames, epoch, train_data_state = next(train_stream)
            frames = to_device(frames)

            policy_opt.zero_grad(set_to_none=True)
            with autocast():
                policy_loss, train_hidden, distances = policy_loss_fn(frames, train_hidden)
            policy_loss.backward()
            train_hidden = detach(train_hidden)

            value_opt.zero_grad(set_to_none=True)
            sliced = slice_delayed_frames(frames, config.policy.delay)
            with autocast():
                value_loss, value_hidden, value_metrics = value_loss_fn(
                    sliced, value_hidden, discount, detail=log_step)
            value_loss.backward()
            value_hidden = detach(value_hidden)

            # the nets share no parameters, so both steps can wait for one check
            policy_norm, value_norm = clip_policy.norm(), clip_value.norm()
            watch.update(step, policy_loss, policy_norm, value_loss, value_norm)
            clip_metrics = clip_policy.clip(policy_norm)
            policy_opt.step()
            value_clip_metrics = clip_value.clip(value_norm)
            value_opt.step()

            if log_step:
                # the plain negative log-likelihood, as in eval: the loss without the extra terms
                nll = sum(d.mean() for d in tree.flatten(distances)) if extra else policy_loss
                metrics = imitation_metrics(nll, distances)
                now = time.perf_counter()
                fps = (step - step_window) * B * config.data.unroll_length / (
                    now - t_window
                )
                t_window, step_window = now, step
                wandb.log(
                    {
                        "train/policy_loss": metrics["policy_loss"],
                        **{f"train/{term.name}": term.value.item() for _, term in extra},
                        "train/epoch": epoch,
                        "train/frames_per_sec": fps,
                        **{f"train/controller/{k}": v
                           for k, v in metrics["controller_flat"].items()},
                        "train/value/loss": value_metrics["loss"],
                        "train/value/uev": value_metrics["uev"],
                        **{f"train/{k}": float(v) for k, v in clip_metrics.items() if math.isfinite(v)},
                        **{f"train/value/{k}": float(v) for k, v in value_clip_metrics.items() if math.isfinite(v)},
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
                wide = run_wide_eval() if wide_stream else None
                wandb.log(
                    {
                        "eval/policy_loss": eval_metrics["policy_loss"],
                        "eval/value_uev": eval_metrics["value_uev"],
                        "eval/value_loss": eval_metrics["value_loss"],
                        "eval/best_policy_loss": best_eval_loss,
                        **{f"eval/{k}": v for k, v in eval_metrics["sticks"].items()},
                        **({"eval_wide/policy_loss": wide["policy_loss"],
                            "eval_wide/value_uev": wide["value_uev"],
                            "eval_wide/value_loss": wide["value_loss"],
                            **{f"eval_wide/{k}": v for k, v in wide["sticks"].items()}} if wide else {}),
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
        try:
            if at_boundary:
                save("latest.pt")
            else:
                print(f"interrupted mid-step {step}; latest.pt left as it was")
        finally:
            train_stream.stop()
            if wide_stream:
                wide_stream.stop()
            wandb.finish()
        print(f"done at step {step}; best eval {best_eval_loss:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
