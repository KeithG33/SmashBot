"""Battery: fixed-yardstick evaluation of a student checkpoint.

Plays a FIXED, REPRODUCIBLE set of games against two frozen yardsticks and
writes a JSON report to scripts/battery_results/step-<STEP>.json:

  (a) yardstick-teacher-887500.pt  -- the frozen BC teacher ("teacher")
  (b) medium-v2-torch.pt           -- ported Phillip ("phillip"; served via
                                      the worker's "reference" env kind)

This is the project's honest strength measure: the training ticker's rolling
windows are noisy and matchup-drifting, while the battery pins the opponent,
the character slate, the seats, and the env layout, so numbers are comparable
across checkpoints.

Reproducibility choices (do not change casually -- they ARE the yardstick):
  - redraw_chars=False, double_buffer=False: fixed matchups, proven env path.
  - BATTERY_SLATE: a hardcoded 8-character rotation over the main-12 spread;
    env i of each yardstick group plays slate[i % 8], student_port alternating
    1/2. Same slate every run.
  - games_per_dolphin high (200) so Dolphin recycles never interfere.

Usage:
  # full RL checkpoint (has config + name_map + step in state):
  .venv/bin/python scripts/battery.py --ckpt runs/rl-pool-v3/latest.pt

  # bare policy state_dict from the snapshot pool (policy-only; the config
  # to BUILD the policy comes from --config-from, a full checkpoint):
  .venv/bin/python scripts/battery.py \
      --snapshot runs/rl-pool-v3/snapshots/snapshot-0001250.pt \
      --config-from runs/rl-pool-v3/latest.pt

NOTE: constructs a DolphinRolloutWorker -> spawn context re-imports __main__,
so everything Dolphin-touching lives under the __main__ guard / main().
"""

from __future__ import annotations

import os

# Modest CPU footprint next to a live training run; must be set before torch
# initializes its thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import dataclasses
import json
import time

import torch

from smashbot.rl.pool import EnvSpec
from smashbot.rl.rollouts import GameTracker, RolloutConfig

# Fixed 8-character rotation per yardstick: a spread of the main 12. Env i of
# each yardstick group gets BATTERY_SLATE[i % 8]; at the default 4 envs per
# yardstick the first 4 chars serve, at 16 envs the full slate serves. Same
# slate every battery run = comparable numbers across checkpoints.
BATTERY_SLATE = [
    "FOX", "FALCO", "MARTH", "SHEIK",
    "PEACH", "CPTFALCON", "JIGGLYPUFF", "SAMUS",
]

YARDSTICK_TEACHER = "/home/kage/drive2/ShineBot/models/yardstick-teacher-887500.pt"
YARDSTICK_PHILLIP = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
DEFAULT_CONFIG_FROM = "/home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_results")

# JSON report keys per yardstick block (schema pinned by test_battery.py).
YARDSTICK_KEYS = (
    "games", "wins", "losses", "draws", "win_rate",
    "avg_stock_diff", "avg_percent_at_kill", "avg_percent_at_death",
)


def build_specs(num_envs: int) -> list[EnvSpec]:
    """Deterministic env layout: first half teacher, second half reference
    (phillip), chars from BATTERY_SLATE in rotation, seats alternating 1/2
    within each group. Pure function of num_envs -- no RNG anywhere."""
    if num_envs < 2 or num_envs % 2 != 0:
        raise ValueError(f"num_envs must be even and >= 2, got {num_envs}")
    half = num_envs // 2
    specs = []
    for kind in ("teacher", "reference"):
        for i in range(half):
            specs.append(
                EnvSpec(
                    kind=kind,
                    group=-1,
                    student_port=1 + (i % 2),
                    opponent_char=BATTERY_SLATE[i % len(BATTERY_SLATE)],
                )
            )
    return specs


def result_filename(step: int) -> str:
    """step-<STEP>.json, zero-padded to match snapshot-<STEP>.pt naming so
    battery_watch.sh can pair snapshots with results by string surgery."""
    return f"step-{step:07d}.json"


def step_from_snapshot_path(path: str) -> int:
    """snapshot-0001250.pt -> 1250 (mirrors SnapshotPool._step_of)."""
    return int(os.path.basename(path).split("-")[1].split(".")[0])


def yardstick_block(tracker: GameTracker) -> dict:
    """Serialize one GameTracker into the per-yardstick report block.
    win_rate is over DECIDED games (draws excluded), matching the tracker's
    own convention; None when nothing was decided."""
    stats = tracker.stats()
    decided = tracker.wins + tracker.losses
    return {
        "games": stats["games_played"],
        "wins": tracker.wins,
        "losses": tracker.losses,
        "draws": tracker.draws,
        "win_rate": (tracker.wins / decided) if decided else None,
        "avg_stock_diff": stats["avg_stock_diff"],
        "avg_percent_at_kill": stats["avg_percent_at_kill"],
        "avg_percent_at_death": stats["avg_percent_at_death"],
    }


def make_report(
    student_ckpt: str,
    student_rl_step: int,
    stamp: str,
    trackers: dict,
    config_echo: dict,
) -> dict:
    """Assemble the JSON-serializable battery report. `trackers` maps env
    kind ("teacher" / "reference") to its GameTracker."""
    return {
        "student_ckpt": student_ckpt,
        "student_rl_step": student_rl_step,
        "timestamp": stamp,
        "char_slate": list(BATTERY_SLATE),
        "config": config_echo,
        "results": {
            "teacher": yardstick_block(trackers["teacher"]),
            "phillip": yardstick_block(trackers["reference"]),
        },
    }


def summary_line(report: dict) -> str:
    def pct(block):
        wr = block["win_rate"]
        return "--%" if wr is None else f"{wr:.0%}"

    t, p = report["results"]["teacher"], report["results"]["phillip"]
    return (
        f"BATTERY step {report['student_rl_step']}: "
        f"vs-teacher {pct(t)} ({t['games']}g) | "
        f"vs-phillip {pct(p)} ({p['games']}g)"
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", default="", help="full RL checkpoint (config+state)")
    src.add_argument("--snapshot", default="",
                     help="bare policy state_dict (snapshot pool); needs --config-from")
    ap.add_argument("--config-from", default=DEFAULT_CONFIG_FROM,
                    help="full checkpoint whose config/name_map builds the policy "
                         "for --snapshot mode")
    ap.add_argument("--envs", type=int, default=8, help="total envs (half per yardstick)")
    ap.add_argument("--games-per-side", type=int, default=24,
                    help="finished games required per yardstick")
    ap.add_argument("--max-minutes", type=float, default=90.0,
                    help="wall-clock budget; partial results are still reported")
    ap.add_argument("--stamp", default="", help="timestamp string for the report "
                    "(default: now, ISO format)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--yardstick-teacher", default=YARDSTICK_TEACHER)
    ap.add_argument("--yardstick-phillip", default=YARDSTICK_PHILLIP)
    ap.add_argument("--games-per-dolphin", type=int, default=200,
                    help="kept high so recycles never interfere with a battery")
    ap.add_argument("--env-timeout", type=float, default=300.0)
    return ap.parse_args(argv)


def _warm(policy, n: int, device: str) -> None:
    """One eager dummy forward at serving batch size: surfaces CUDA OOM at
    build time (where the cpu fallback can catch it) instead of mid-battery."""
    import numpy as np
    import tree

    from slippi_ai.types import StateAction

    def to_t(x):
        x = np.asarray(x)
        if x.dtype.kind in "iu":
            x = x.astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    dummy = tree.map_structure(to_t, policy.network.embed_state_action.dummy((n,)))
    sa = StateAction(state=dummy.state, action=dummy.action, name=dummy.name)
    h = policy.initial_state(n, device)
    with torch.no_grad():
        policy.sample(sa, h)


def _load_student(args, device: str):
    """Returns (policy, name_map, rl_step, label). Full checkpoints carry
    their own config and step; bare snapshots borrow config/name_map from
    --config-from and take their step from the filename."""
    from smashbot.eval.game import load_policy

    if args.snapshot:
        policy, name_map, _ = load_policy(args.config_from, device)
        state = torch.load(args.snapshot, map_location=device, weights_only=True)
        policy.load_state_dict(state)
        policy.eval()
        return policy, name_map, step_from_snapshot_path(args.snapshot), args.snapshot
    policy, name_map, step = load_policy(args.ckpt, device)
    return policy, name_map, int(step), args.ckpt


def _build_agents(args, device: str):
    """Load student + both yardsticks on `device`, eager only (torch.compile
    is deliberately NOT used: the live training run owns the GPU headroom).
    Raises torch OOM upward so main() can retry on cpu."""
    from smashbot.eval.game import load_policy, resolve_name_code
    from smashbot.rl.agent import BatchedPolicyAgent

    half = args.envs // 2

    student, student_names, rl_step, label = _load_student(args, device)
    teacher, teacher_names, _ = load_policy(args.yardstick_teacher, device)
    phillip, phillip_names, _ = load_policy(args.yardstick_phillip, device)
    for p in (student, teacher, phillip):
        p.train_value_head = False
        p.requires_grad_(False)
        p.eval()
        _warm(p, args.envs, device)

    student_agent = BatchedPolicyAgent(
        student, args.envs,
        name_code=resolve_name_code(student_names, "Master Player"), device=device,
    )
    opponents = {
        "teacher": BatchedPolicyAgent(
            teacher, half,
            name_code=resolve_name_code(teacher_names, "Master Player"),
            device=device,
        ),
        # phillip conditions on ITS OWN name_map's "Master Player" code; its
        # delay (21) rides in the checkpoint and BatchedPolicyAgent applies it.
        "reference": BatchedPolicyAgent(
            phillip, half,
            name_code=resolve_name_code(phillip_names, "Master Player"),
            device=device,
        ),
    }
    return student_agent, opponents, rl_step, label


def main() -> None:
    from smashbot.rl.rollouts import DolphinRolloutWorker

    args = parse_args()
    stamp = args.stamp or __import__("datetime").datetime.now().isoformat(
        timespec="seconds"
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        student_agent, opponents, rl_step, label = _build_agents(args, device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        print("CUDA OOM while building policies; falling back to cpu", flush=True)
        torch.cuda.empty_cache()
        device = "cpu"
        student_agent, opponents, rl_step, label = _build_agents(args, device)
    print(f"battery: student {label} (rl step {rl_step}), device {device}", flush=True)

    half = args.envs // 2
    specs = build_specs(args.envs)
    rcfg = RolloutConfig(
        num_envs=args.envs,
        cpu_envs=0,
        teacher_envs=half,
        ref_envs=half,
        snapshot_slots=0,
        games_per_dolphin=args.games_per_dolphin,
        redraw_chars=False,     # fixed matchups: the yardstick must not drift
        double_buffer=False,    # proven recycle path only
        log_tag="battery",      # env logs: /tmp/smashbot-env-battery-*.log
        env_timeout=args.env_timeout,
        ref_ckpt=args.yardstick_phillip,
    )
    worker = DolphinRolloutWorker(rcfg, student_agent, opponents=opponents, specs=specs)

    target = args.games_per_side
    deadline = time.monotonic() + args.max_minutes * 60
    t0 = time.monotonic()
    last_print = 0.0
    collects = 0
    try:
        while True:
            worker.collect(1)
            collects += 1
            done = {
                k: worker.trackers[k].stats()["games_played"]
                for k in ("teacher", "reference")
            }
            now = time.monotonic()
            if now - last_print > 30:
                frames = collects * rcfg.unroll_length
                print(
                    f"[battery {now - t0:5.0f}s] teacher {done['teacher']}/{target} "
                    f"phillip {done['reference']}/{target} games | "
                    f"{frames} frames/env | {frames * args.envs / (now - t0):.0f} fps",
                    flush=True,
                )
                last_print = now
            if done["teacher"] >= target and done["reference"] >= target:
                break
            if now > deadline:
                print("[battery] --max-minutes reached; writing partial results",
                      flush=True)
                break
    finally:
        worker.stop()

    config_echo = dataclasses.asdict(rcfg)
    config_echo.update(
        envs=args.envs,
        games_per_side=args.games_per_side,
        max_minutes=args.max_minutes,
        device=device,
        yardstick_teacher=args.yardstick_teacher,
        yardstick_phillip=args.yardstick_phillip,
        student_mode="snapshot" if args.snapshot else "ckpt",
        config_from=args.config_from if args.snapshot else "",
        elapsed_seconds=round(time.monotonic() - t0, 1),
    )
    report = make_report(label, rl_step, stamp, worker.trackers, config_echo)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, result_filename(rl_step))
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out_path}", flush=True)
    print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
