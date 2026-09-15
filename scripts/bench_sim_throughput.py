"""Sim rollout throughput vs batch size and compile mode.

Measures student env-frames/sec through the real SimRolloutWorker path
(encode -> sample -> decode -> write -> step), under the compile mode the
training actually uses. Answers the batch-size choice with the compiled
number, not eager.

  python scripts/bench_sim_throughput.py --ckpt <v10.pt> \
      --batches 512,768,1024 --modes default,reduce-overhead --frames 300
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import melee_sim as msl
from smashbot.eval.game import load_policy
from smashbot.rl.sim_rollout import SimRolloutWorker


def compile_sample(policy, mode):
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 128
    policy.sample = torch.compile(policy.sample, mode=mode)


def bench(ckpt, batch, mode, frames, data_dir, device, selfplay):
    policy, name_map, _ = load_policy(ckpt, device)
    policy.eval()
    if mode != "none":
        compile_sample(policy, mode)
    opp = policy if selfplay else None
    w = SimRolloutWorker(
        policy, batch, unroll_length=80, data_dir=data_dir,
        stage=msl.Stage.FINAL_DESTINATION, chars=(msl.Character.FOX, msl.Character.MARTH),
        name_code=0, device=device, opponent_policy=opp, opp_name_code=0,
    )
    # warmup (also triggers compile/cudagraph capture)
    for _ in range(2):
        w.collect(30)
    if device == "cuda":
        torch.cuda.synchronize()
        peak0 = torch.cuda.max_memory_allocated()
    t0 = time.perf_counter()
    # collect in chunks, discarding trajectories each chunk so the benchmark
    # doesn't accumulate emitted chunks on-GPU (training consumes+frees each).
    done = 0
    chunk = 90
    while done < frames:
        n = min(chunk, frames - done)
        _ = w.collect(n)   # discarded -> bounded memory (one chunk at a time)
        done += n
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    w.close()
    efps = batch * frames / dt
    return efps, dt, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", default="512,768,1024")
    ap.add_argument("--modes", default="default,reduce-overhead")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-dir", default=os.environ.get("MSL_DATA_DIR"))
    ap.add_argument("--selfplay", action="store_true",
                    help="student vs student (2 forwards/frame) instead of student-only")
    args = ap.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    modes = args.modes.split(",")
    load = "self-play (2x fwd)" if args.selfplay else "student-only"
    print(f"# sim throughput | {load} | frames={args.frames} | device={args.device}")
    print(f"{'batch':>6} {'mode':>16} {'env-frames/s':>14} {'sec':>7} {'peakGB':>7}")
    for mode in modes:
        for b in batches:
            efps, dt, peak = bench(args.ckpt, b, mode, args.frames, args.data_dir,
                                   args.device, args.selfplay)
            print(f"{b:>6} {mode:>16} {efps:>14,.0f} {dt:>7.2f} {peak:>7.2f}", flush=True)


if __name__ == "__main__":
    main()
