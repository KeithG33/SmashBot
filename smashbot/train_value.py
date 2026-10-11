"""Trains a value net alone on the BC data, as train_bc trains it beside the
policy: the same data stream (the same dataset seed gives the same batches),
loss, return discount and eval_wide, so its value_loss and value_uev compare
directly with a BC run's at the same step. No policy is built or trained, so
a value architecture is compared in a fraction of a BC run's time.

  .venv/bin/python -m smashbot.train_value $(cat <a BC run's args file>) \\
      --runtime.tag <its own tag> --value.num-layers 1 --value.layout l
"""
import contextlib
import dataclasses
import math
import os
import time

import numpy as np
import torch
import tree
import tyro

from smashbot import saving
from smashbot.data import loader
from smashbot.delay import slice_delayed_frames
from smashbot.networks import keep_sgu_activations, use_bf16_cells, use_chunk_start_resets, use_packed_encoder
from smashbot.train_bc import TrainConfig, _get_rng, _nonfinite, _resume_state, _set_rng, _StopRequest, _to
from smashbot.training import GradClipper, compile_cores, resolve_restore
from smashbot.value import build_value_function


def score(value_fn, groups: list, rows: int, delay: int, warm_batches: int, discount: float,
          autocast, device) -> dict:
    """train_bc.score's value half: each group starts cold, runs warm_batches
    unscored, then scores the rest."""
    metrics = []
    with torch.no_grad():
        for group in groups:
            hidden = value_fn.initial_state(rows, device)
            for i, (frames, _) in enumerate(group):
                frames = tree.map_structure(lambda t: t.to(device, non_blocking=True), frames)
                with autocast():
                    _, hidden, m = value_fn.loss(slice_delayed_frames(frames, delay), hidden, discount)
                if i >= warm_batches:
                    metrics.append(m)
    return {"value_uev": float(np.mean([m["uev"] for m in metrics])),
            "value_loss": float(np.mean([m["loss"] for m in metrics]))}


def main(config: TrainConfig) -> None:
    dataset = config.data.dataset
    if dataset.data_dir is not None and dataset.meta_path is None:
        dataset.meta_path = os.path.join(os.path.dirname(dataset.data_dir.rstrip("/")), "meta.json")
    dataset.validate()
    rt = config.runtime
    eval_id = (f"random {rt.eval_groups}x{rt.eval_rows} games, {rt.eval_burn_in}+{rt.eval_batches}x"
               f"{config.data.unroll_length} frames")
    run_dir = os.path.join(rt.run_dir, rt.tag)
    restore_path = resolve_restore(run_dir, rt.restore)
    os.makedirs(run_dir, exist_ok=True)
    device = rt.device
    torch.manual_seed(rt.seed)
    np.random.seed(rt.seed)
    discount = 0.5 ** (1 / (config.value.reward_halflife * 60))

    ckpt = saving.load_checkpoint(restore_path) if restore_path else None
    resume = _resume_state(ckpt)
    sources = loader.make_sources(config.data, extra_frames=config.policy.delay + 1,
                                  name_map=resume.get("name_map"), train_state=resume.get("train_data"),
                                  train=not config.data.producer_process)

    value_fn = build_value_function(dataclasses.asdict(config), device)
    use_chunk_start_resets(value_fn)
    keep_sgu_activations(value_fn)
    use_packed_encoder(value_fn)
    if config.learner.bf16_cells:
        if config.learner.precision != "bf16":
            raise ValueError("--learner.bf16-cells needs --learner.precision bf16")
        use_bf16_cells(value_fn)
    value_opt = torch.optim.Adam(value_fn.parameters(), lr=config.learner.learning_rate, fused=device == "cuda")
    clip = GradClipper(value_fn.parameters(), config.learner.max_grad_norm)
    if config.learner.precision == "bf16" and device == "cuda":
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext
    if config.learner.compile and device == "cuda":
        compile_cores(None, value_fn)
    print(f"value: {sum(p.numel() for p in value_fn.parameters()) / 1e6:.2f}M params | device={device}")

    step, best = 0, math.inf
    if ckpt is not None:
        value_fn.load_state_dict(resume["value"])
        value_opt.load_state_dict(resume["value_opt"])
        step, best = resume["step"], ckpt["best_eval_loss"]
        if resume.get("eval_set") != eval_id:
            best = math.inf
        print(f"restored from {restore_path} at step {step} (best eval value loss {best:.5f})")

    import wandb
    wandb.init(project="shinebot", group="value", name=rt.tag, mode=rt.wandb_mode,
               config=dataclasses.asdict(config), resume="allow" if restore_path else "never", id=rt.tag)

    B = config.data.batch_size
    hidden = _to(resume["value_hidden"], device) if resume.get("value_hidden") is not None else \
        value_fn.initial_state(B, device)
    if resume.get("rng") is not None:
        _set_rng(resume["rng"])
    train_data_state = resume.get("train_data")
    train_stream = loader.make_train_stream(sources, config.data, config.policy.delay + 1, train_data_state,
                                            value_fn.network)
    sources.test.shutdown()
    warm_batches = -(-rt.eval_burn_in // config.data.unroll_length)
    eval_stream = loader.random_eval_stream(
        sources.test.replays, config.data, config.policy.delay + 1, sources.name_map, value_fn.network,
        groups=rt.eval_groups, rows=rt.eval_rows, batches=warm_batches + rt.eval_batches,
        seed=config.data.dataset.seed * 1_000_003 + (step // rt.eval_interval + 1) * rt.eval_interval,
        interval=rt.eval_interval)

    def save(path_name: str):
        bad = _nonfinite({"value weights": value_fn.state_dict(), "value Adam state": value_opt.state_dict(),
                          "value_hidden": hidden})
        if bad:
            raise FloatingPointError(f"refusing to write {path_name} at step {step}: non-finite {', '.join(bad)}")
        saving.save_checkpoint(os.path.join(run_dir, path_name), config, {
            "value": value_fn.state_dict(), "value_opt": value_opt.state_dict(), "step": step,
            "name_map": sources.name_map, "train_data": train_data_state,
            "value_hidden": _to(hidden, "cpu"), "rng": _get_rng(), "eval_set": eval_id,
        }, best)

    t_window, step_window = time.perf_counter(), step
    stop = _StopRequest()
    at_boundary = True
    try:
        while step < rt.steps and not stop.requested:
            at_boundary = False
            step += 1
            log_step = step % rt.log_interval == 0
            frames, epoch, train_data_state = next(train_stream)
            frames = tree.map_structure(lambda t: t.to(device, non_blocking=True), frames)
            value_opt.zero_grad(set_to_none=True)
            with autocast():
                loss, hidden, metrics = value_fn.loss(slice_delayed_frames(frames, config.policy.delay),
                                                      hidden, discount, detail=log_step)
            loss.backward()
            hidden = tree.map_structure(lambda t: t.detach(), hidden)
            clip.clip(clip.norm())   # a non-finite run stops at its next save or eval
            value_opt.step()

            if log_step:
                now = time.perf_counter()
                fps = (step - step_window) * B * config.data.unroll_length / (now - t_window)
                t_window, step_window = now, step
                wandb.log({"train/value/loss": metrics["loss"], "train/value/uev": metrics["uev"],
                           "train/epoch": epoch, "train/frames_per_sec": fps}, step=step)
                print(f"step {step}: value loss {metrics['loss']:.5f} uev {metrics['uev']:.4f} "
                      f"| {fps / 1e3:.0f}k fps", flush=True)
            if step % rt.eval_interval == 0:
                ev = score(value_fn, next(eval_stream), rt.eval_rows, config.policy.delay, warm_batches,
                           discount, autocast, device)
                if not math.isfinite(ev["value_loss"]):
                    raise FloatingPointError(f"step {step}: non-finite eval value loss")
                if ev["value_loss"] < best:
                    best = ev["value_loss"]
                    save("best.pt")
                wandb.log({"eval_wide/value_loss": ev["value_loss"], "eval_wide/value_uev": ev["value_uev"],
                           "eval_wide/best_value_loss": best}, step=step)
                print(f"eval_wide @ {step}: value_loss {ev['value_loss']:.5f} uev {ev['value_uev']:.4f}",
                      flush=True)
            if step % rt.checkpoint_interval == 0:
                save("latest.pt")
            at_boundary = True
    finally:
        stop.restore()
        try:
            if at_boundary:
                save("latest.pt")
        finally:
            train_stream.stop()
            eval_stream.stop()
            wandb.finish()
        print(f"done at step {step}; best eval value loss {best:.5f}")


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
