"""Capture one REAL trajectory batch for offline learner-precision probes.

Tier-1 companion of docs/experiments-queue.md "Learner precision": the
fidelity probe (scripts/precision_probe.py) needs one batch of genuine
rollout trajectories — sample-time logits and all — so the fp32/bf16/fp16
learner passes can be diffed on identical data. This script boots a small
CPU-only worker (mirroring scripts/battery.py's construction: student vs
teacher+phillip yardsticks, half the envs each), collects trajectories until
one full batch exists, and torch.saves it with enough metadata to rebuild
the exact learner later.

Designed to run BESIDE the live training run: device is always cpu, the env
count is small (default 8 Dolphins), OMP threads are capped, and nothing
touches the GPU.

Usage:
  # from the live run's latest checkpoint (a full RL checkpoint):
  .venv/bin/python scripts/capture_batch.py \
      --ckpt /home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt

  # from a bare snapshot (battery's --snapshot/--config-from pattern):
  .venv/bin/python scripts/capture_batch.py \
      --snapshot .../snapshots/snapshot-0012345.pt \
      --config-from /home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt

  # construction check only (no Dolphins):
  .venv/bin/python scripts/capture_batch.py --ckpt ... --dry-run

NOTE: constructs a DolphinRolloutWorker -> spawn context re-imports
__main__, so everything Dolphin-touching lives under the __main__ guard.
"""

from __future__ import annotations

import os

# Modest CPU footprint next to a live training run; must be set before torch
# initializes its thread pools (same choice as battery.py).
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import dataclasses
import sys
import time

import torch

# battery.py lives beside this script; make the import location-independent
# (spawned env processes re-import __main__ with the same sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery  # noqa: E402

from smashbot.rl.rollouts import RolloutConfig  # noqa: E402

DEFAULT_OUT_DIR = "/home/kage/drive2/ShineBot/probes"
# One fixed 4-char slate (battery phase A): deterministic matchups; the
# probe cares about numerics, not roster coverage.
CAPTURE_SLATE = battery.PHASES[0][1]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", default="",
                     help="full RL checkpoint (config+state)")
    src.add_argument("--snapshot", default="",
                     help="bare policy state_dict; needs --config-from")
    ap.add_argument("--config-from", default=battery.DEFAULT_CONFIG_FROM,
                    help="full checkpoint whose config/name_map builds the "
                         "policy for --snapshot mode")
    ap.add_argument("--envs", type=int, default=8,
                    help="Dolphins to boot (half teacher, half phillip); "
                         "keep small beside a live run")
    ap.add_argument("--trajectories", type=int, default=1,
                    help="Trajectory chunks to collect (1 = one full "
                         "[envs, T+1] learner batch)")
    ap.add_argument("--unroll-length", type=int, default=240,
                    help="frames per chunk (T); production rollout length")
    ap.add_argument("--out", default="",
                    help=f"output path (default: "
                         f"{DEFAULT_OUT_DIR}/batch-<step>.pt)")
    ap.add_argument("--env-timeout", type=float, default=300.0)
    ap.add_argument("--yardstick-teacher", default=battery.YARDSTICK_TEACHER)
    ap.add_argument("--yardstick-phillip", default=battery.YARDSTICK_PHILLIP)
    ap.add_argument("--dry-run", action="store_true",
                    help="load policies, build specs/config, print the "
                         "plan, then exit WITHOUT booting Dolphins")
    return ap.parse_args(argv)


def build_rollout_config(args) -> RolloutConfig:
    """Mirror battery.py's worker config: fixed matchups, proven recycle
    path, no snapshot slots — the batch should look like ordinary on-policy
    student data against the standard yardsticks."""
    half = args.envs // 2
    return RolloutConfig(
        num_envs=args.envs,
        unroll_length=args.unroll_length,
        cpu_envs=0,
        teacher_envs=half,
        ref_envs=half,
        snapshot_slots=0,
        games_per_dolphin=200,   # high: recycles never interfere
        redraw_chars=False,      # fixed matchups
        double_buffer=False,     # proven env path
        log_tag="capture",       # env logs: /tmp/smashbot-env-capture-*.log
        env_timeout=args.env_timeout,
        ref_ckpt=args.yardstick_phillip,
    )


def main() -> None:
    args = parse_args()
    device = "cpu"  # always: the live training run owns the GPU

    policies, codes, rl_step, label = battery._load_policies(args, device)
    specs = battery.build_specs(args.envs, CAPTURE_SLATE)
    rcfg = build_rollout_config(args)

    out_path = args.out or os.path.join(
        DEFAULT_OUT_DIR, f"batch-{rl_step:07d}.pt"
    )
    print(f"capture: student {label} (rl step {rl_step}), device {device}",
          flush=True)
    print(f"capture: {args.envs} envs ({'/'.join(CAPTURE_SLATE)}), "
          f"{args.trajectories} chunk(s) of T={args.unroll_length} -> "
          f"{out_path}", flush=True)

    if args.dry_run:
        print("dry run: stopping before Dolphin boot")
        return

    from smashbot.rl.rollouts import DolphinRolloutWorker

    student_agent, opponents = battery._make_agents(
        policies, codes, args.envs, device
    )
    worker = DolphinRolloutWorker(
        rcfg, student_agent, opponents=opponents, specs=specs
    )
    t0 = time.monotonic()
    try:
        trajectories = worker.collect(args.trajectories)
    finally:
        worker.stop()
    elapsed = time.monotonic() - t0
    print(f"collected {len(trajectories)} trajectory chunk(s) in "
          f"{elapsed:.0f}s", flush=True)

    payload = {
        "trajectories": trajectories,
        "meta": {
            "student_ckpt": label,
            "student_mode": "snapshot" if args.snapshot else "ckpt",
            "config_from": args.config_from if args.snapshot else "",
            "rl_step": rl_step,
            "rollout_config": dataclasses.asdict(rcfg),
            "specs": [dataclasses.asdict(s) for s in specs],
            "slate": list(CAPTURE_SLATE),
            "yardstick_teacher": args.yardstick_teacher,
            "yardstick_phillip": args.yardstick_phillip,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": __import__("datetime").datetime.now().isoformat(
                timespec="seconds"
            ),
            "torch_version": torch.__version__,
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)
    rows = trajectories[0].rewards.shape[0]
    print(f"wrote {out_path} "
          f"({rows} rows x T={args.unroll_length}, step {rl_step})",
          flush=True)


if __name__ == "__main__":
    main()
