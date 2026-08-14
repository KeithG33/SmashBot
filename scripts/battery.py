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

Full main-12 coverage via SEQUENTIAL PHASES (the project principle behind
make_partition's stratified(): guaranteed roster coverage, never random
holes). One battery = 3 phases x --envs envs; each phase boots a fresh
DolphinRolloutWorker whose 4-character slate covers a third of the main 12
per yardstick, so the union across phases is exactly MAIN_12 -- while never
exceeding --envs Dolphins at once. Everything is deterministic: no RNG in
spec construction, phase order fixed, seats alternating.

Reproducibility choices (do not change casually -- they ARE the yardstick):
  - redraw_chars=False, double_buffer=False: fixed matchups, proven env path.
  - PHASES: hardcoded 4-char slates; env i of each yardstick group plays
    slate[i % 4], student_port alternating 1/2. Same slates every run.
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

# Three fixed 4-character phase slates whose union is exactly the main 12.
# Within a phase, env i of each yardstick group gets slate[i % 4]; at the
# default 4 envs per yardstick every slate character serves every phase.
# Same phases every battery run = comparable numbers across checkpoints.
PHASES = [
    ("A", ["FOX", "FALCO", "MARTH", "SHEIK"]),
    ("B", ["PEACH", "CPTFALCON", "JIGGLYPUFF", "SAMUS"]),
    ("C", ["YOSHI", "POPO", "LUIGI", "PIKACHU"]),
]
# All 12, in phase order (the report's char_slate field).
BATTERY_SLATE = [c for _, slate in PHASES for c in slate]

YARDSTICK_TEACHER = "/home/kage/drive2/ShineBot/models/yardstick-teacher-887500.pt"
YARDSTICK_PHILLIP = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
DEFAULT_CONFIG_FROM = "/home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery_results")

# JSON report keys per yardstick block (schema pinned by test_battery.py).
YARDSTICK_KEYS = (
    "games", "wins", "losses", "draws", "win_rate",
    "avg_stock_diff", "avg_percent_at_kill", "avg_percent_at_death",
)


def build_specs(num_envs: int, slate: list[str]) -> list[EnvSpec]:
    """Deterministic env layout for one phase: first half teacher, second
    half reference (phillip), chars from `slate` in rotation, seats
    alternating 1/2 within each group. Pure function -- no RNG anywhere."""
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
                    opponent_char=slate[i % len(slate)],
                )
            )
    return specs


def phase_game_targets(games_per_side: int) -> list[int]:
    """Split the per-yardstick game budget across the 3 phases, remainder to
    the earliest phases; sums exactly to games_per_side."""
    n = len(PHASES)
    return [games_per_side // n + (1 if i < games_per_side % n else 0)
            for i in range(n)]


def per_char_counts(specs: list[EnvSpec], games_per_env: list[int]) -> dict:
    """Map per-env finished-game counts onto {"teacher"|"phillip": {char:
    games}} using the phase's spec layout (each char is served by a fixed
    env per yardstick, so env game counts ARE per-char counts)."""
    label = {"teacher": "teacher", "reference": "phillip"}
    out: dict = {"teacher": {}, "phillip": {}}
    for spec, games in zip(specs, games_per_env):
        d = out[label[spec.kind]]
        d[spec.opponent_char] = d.get(spec.opponent_char, 0) + games
    return out


def merge_char_counts(counts: list[dict]) -> dict:
    """Sum per-char game counts across phases."""
    out: dict = {"teacher": {}, "phillip": {}}
    for c in counts:
        for side in out:
            for char, games in c.get(side, {}).items():
                out[side][char] = out[side].get(char, 0) + games
    return out


class _CountingConn:
    """Transparent Pipe proxy that counts finished games per env by watching
    for final_stocks in the payload stream. Pure observation -- every call is
    forwarded unchanged, so the worker behaves identically. (The worker's own
    GameTrackers aggregate per KIND; per-character coverage needs per-ENV
    counts, and wrapping the pipe is the least invasive way to get them
    without touching smashbot/rl/.)"""

    def __init__(self, conn):
        self._conn = conn
        self.games = 0

    def poll(self, *args, **kwargs):
        return self._conn.poll(*args, **kwargs)

    def recv(self):
        payload = self._conn.recv()
        if isinstance(payload, dict) and payload.get("final_stocks") is not None:
            self.games += 1
        return payload

    def send(self, *args, **kwargs):
        return self._conn.send(*args, **kwargs)


def result_filename(step: int) -> str:
    """step-<STEP>.json, zero-padded to match snapshot-<STEP>.pt naming so
    battery_all.sh can pair snapshots with results by string surgery."""
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


def merge_trackers(trackers: list[GameTracker]) -> GameTracker:
    """Pool per-phase GameTrackers into one aggregate tracker. Safe at
    battery scale: the tracker windows (100/200) exceed any battery's game
    and event counts, so nothing is silently dropped."""
    merged = GameTracker()
    for t in trackers:
        merged.wins += t.wins
        merged.losses += t.losses
        merged.draws += t.draws
        merged.diffs.extend(t.diffs)
        merged.kill_percents.extend(t.kill_percents)
        merged.death_percents.extend(t.death_percents)
    return merged


def make_report(
    student_ckpt: str,
    student_rl_step: int,
    stamp: str,
    phases: list[dict],
    config_echo: dict,
) -> dict:
    """Assemble the JSON-serializable battery report from per-phase entries:
    {"phase": str, "slate": [4 chars], "elapsed_seconds": float,
     "trackers": {"teacher": GameTracker, "reference": GameTracker}}.
    Top-level results are the cross-phase totals; the phases section keeps
    the per-slate detail."""
    merged = {
        kind: merge_trackers([p["trackers"][kind] for p in phases])
        for kind in ("teacher", "reference")
    }
    merged_chars = merge_char_counts([p.get("per_char", {}) for p in phases])
    return {
        "student_ckpt": student_ckpt,
        "student_rl_step": student_rl_step,
        "timestamp": stamp,
        "char_slate": list(BATTERY_SLATE),
        "config": config_echo,
        "results": {
            "teacher": yardstick_block(merged["teacher"]),
            "phillip": yardstick_block(merged["reference"]),
        },
        # games per character per yardstick, summed over phases: the roster-
        # coverage receipt (every main-12 char should be >= 1 in a full run)
        "per_char": merged_chars,
        "phases": [
            {
                "phase": p["phase"],
                "slate": list(p["slate"]),
                "elapsed_seconds": p["elapsed_seconds"],
                "teacher": yardstick_block(p["trackers"]["teacher"]),
                "phillip": yardstick_block(p["trackers"]["reference"]),
                "per_char": p.get("per_char", {}),
            }
            for p in phases
        ],
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
    ap.add_argument("--envs", type=int, default=8,
                    help="envs per phase (half per yardstick); never more Dolphins "
                         "than this at once")
    ap.add_argument("--games-per-side", type=int, default=24,
                    help="finished games required per yardstick, total across phases")
    ap.add_argument("--max-minutes", type=float, default=90.0,
                    help="wall-clock budget, split evenly across the 3 phases; "
                         "partial results are still reported")
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


def _load_policies(args, device: str):
    """Load student + both yardsticks on `device`, eager only (torch.compile
    is deliberately NOT used: the live training run owns the GPU headroom).
    Raises torch OOM upward so main() can retry on cpu. Loaded ONCE; each
    phase wraps them in fresh BatchedPolicyAgents (fresh recurrent state)."""
    from smashbot.eval.game import load_policy, resolve_name_code

    student, student_names, rl_step, label = _load_student(args, device)
    teacher, teacher_names, _ = load_policy(args.yardstick_teacher, device)
    phillip, phillip_names, _ = load_policy(args.yardstick_phillip, device)
    for p in (student, teacher, phillip):
        p.train_value_head = False
        p.requires_grad_(False)
        p.eval()
        _warm(p, args.envs, device)

    codes = {
        "student": resolve_name_code(student_names, "Master Player"),
        "teacher": resolve_name_code(teacher_names, "Master Player"),
        # phillip conditions on ITS OWN name_map's "Master Player" code; its
        # delay (21) rides in the checkpoint and BatchedPolicyAgent applies it.
        "phillip": resolve_name_code(phillip_names, "Master Player"),
    }
    policies = {"student": student, "teacher": teacher, "phillip": phillip}
    return policies, codes, rl_step, label


def _make_agents(policies: dict, codes: dict, num_envs: int, device: str):
    """Fresh per-phase agents: recurrent state, delay queues, and prev-action
    buffers all start clean for the phase's new env processes."""
    from smashbot.rl.agent import BatchedPolicyAgent

    half = num_envs // 2
    student_agent = BatchedPolicyAgent(
        policies["student"], num_envs, name_code=codes["student"], device=device
    )
    opponents = {
        "teacher": BatchedPolicyAgent(
            policies["teacher"], half, name_code=codes["teacher"], device=device
        ),
        "reference": BatchedPolicyAgent(
            policies["phillip"], half, name_code=codes["phillip"], device=device
        ),
    }
    return student_agent, opponents


def _run_phase(
    phase_name: str,
    slate: list[str],
    target: int,
    budget_seconds: float,
    policies: dict,
    codes: dict,
    rcfg: RolloutConfig,
    device: str,
) -> dict:
    """One phase: fresh worker over this slate's specs, collect until both
    yardsticks reach `target` finished games or the budget lapses, then stop
    the worker cleanly. Returns the phase entry for make_report."""
    from smashbot.rl.rollouts import DolphinRolloutWorker

    specs = build_specs(rcfg.num_envs, slate)
    student_agent, opponents = _make_agents(policies, codes, rcfg.num_envs, device)
    worker = DolphinRolloutWorker(
        rcfg, student_agent, opponents=opponents, specs=specs
    )
    # Boot the envs now so the payload pipes exist, then wrap each in a
    # counting proxy: per-env finished-game counts give the per-character
    # coverage receipt (see _CountingConn). _conns is worker-internal, but
    # observation-only wrapping beats forking the rollout code.
    worker._ensure_started()
    counters = [_CountingConn(c) for c in worker._conns]
    worker._conns = counters
    print(f"[battery] phase {phase_name} ({'/'.join(slate)}): "
          f"{target} games per yardstick, {budget_seconds / 60:.0f} min budget",
          flush=True)
    t0 = time.monotonic()
    deadline = t0 + budget_seconds
    last_print = 0.0
    collects = 0
    try:
        while target > 0:
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
                    f"[battery {phase_name} {now - t0:5.0f}s] "
                    f"teacher {done['teacher']}/{target} "
                    f"phillip {done['reference']}/{target} games | "
                    f"{frames} frames/env | "
                    f"{frames * rcfg.num_envs / (now - t0):.0f} fps",
                    flush=True,
                )
                last_print = now
            if done["teacher"] >= target and done["reference"] >= target:
                break
            if now > deadline:
                print(f"[battery] phase {phase_name} budget reached; "
                      f"keeping partial results", flush=True)
                break
    finally:
        worker.stop()
    return {
        "phase": phase_name,
        "slate": slate,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "trackers": {k: worker.trackers[k] for k in ("teacher", "reference")},
        "per_char": per_char_counts(specs, [c.games for c in counters]),
    }


def main() -> None:
    args = parse_args()
    stamp = args.stamp or __import__("datetime").datetime.now().isoformat(
        timespec="seconds"
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        policies, codes, rl_step, label = _load_policies(args, device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        print("CUDA OOM while building policies; falling back to cpu", flush=True)
        torch.cuda.empty_cache()
        device = "cpu"
        policies, codes, rl_step, label = _load_policies(args, device)
    print(f"battery: student {label} (rl step {rl_step}), device {device}", flush=True)

    half = args.envs // 2
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

    targets = phase_game_targets(args.games_per_side)
    phase_budget = args.max_minutes * 60 / len(PHASES)
    t0 = time.monotonic()
    phase_entries = []
    for (phase_name, slate), target in zip(PHASES, targets):
        phase_entries.append(
            _run_phase(
                phase_name, slate, target, phase_budget,
                policies, codes, rcfg, device,
            )
        )

    config_echo = dataclasses.asdict(rcfg)
    config_echo.update(
        envs=args.envs,
        games_per_side=args.games_per_side,
        phase_game_targets=targets,
        max_minutes=args.max_minutes,
        device=device,
        yardstick_teacher=args.yardstick_teacher,
        yardstick_phillip=args.yardstick_phillip,
        student_mode="snapshot" if args.snapshot else "ckpt",
        config_from=args.config_from if args.snapshot else "",
        elapsed_seconds=round(time.monotonic() - t0, 1),
    )
    report = make_report(label, rl_step, stamp, phase_entries, config_echo)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, result_filename(rl_step))
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out_path}", flush=True)
    print(summary_line(report), flush=True)


if __name__ == "__main__":
    main()
