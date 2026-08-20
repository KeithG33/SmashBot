"""Sharded tournament driver: one PROCESS per pair (own GIL, own Dolphins).

The in-process wave scheduler in tournament.py runs pair-workers as threads;
at 64 concurrent pairs the per-frame python of 128 seats serializes on one
GIL and per-env fps collapses (live-caught: 128 dolphins, load 7, 0 games
in 18 min). This wrapper runs each pair as its own tournament.py subprocess
(2 contestants, --envs = games-per-pair, cpu inference) in bounded batches,
then merges the per-pair JSONs into one standings report via the tool's own
compute_standings.

Usage mirrors tournament.py:
  .venv/bin/python scripts/tournament_sharded.py \
      --contestants NAME=PATH ... [final=...:full] \
      --config-from CKPT --games-per-pair 2 --concurrency 40
"""

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "bin", "python")
TOOL = os.path.join(REPO, "scripts", "tournament.py")
OUT_DIR = os.path.join(REPO, "scripts", "tournament_results")


def _load_tool():
    spec = importlib.util.spec_from_file_location("_tournament_mod", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tournament_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contestants", nargs="+", required=True)
    ap.add_argument("--config-from", required=True)
    ap.add_argument("--phillip", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="include medium-v2 (default OFF here; the fox-only "
                         "ghost bracket usually runs without it)")
    ap.add_argument("--games-per-pair", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=40,
                    help="max simultaneous pair processes (each runs "
                         "games-per-pair dolphins + 1 python)")
    ap.add_argument("--char-mode", default="fox", choices=("fox", "main12"))
    ap.add_argument("--pair-timeout", type=float, default=900,
                    help="seconds per pair process before it is killed "
                         "(partials from finished games are still merged)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    names = [c.split("=", 1)[0] for c in args.contestants]
    assert len(names) == len(set(names)), "duplicate contestant names"
    by_name = dict(c.split("=", 1) for c in args.contestants)
    if args.phillip:
        by_name["phillip"] = (
            "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt:full"
        )
        names.append("phillip")
    pairs = list(itertools.combinations(names, 2))
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    shard_dir = os.path.join(OUT_DIR, f"sharded-{stamp}")
    os.makedirs(shard_dir)
    print(f"sharded tournament: {len(names)} contestants, {len(pairs)} "
          f"pairs, {args.games_per_pair} games/pair, "
          f"concurrency {args.concurrency}", flush=True)

    running: dict = {}
    done = 0
    queue = list(enumerate(pairs))
    t0 = time.monotonic()
    while queue or running:
        while queue and len(running) < args.concurrency:
            i, (a, b) = queue.pop(0)
            out = os.path.join(shard_dir, f"pair-{i:04d}.json")
            cmd = [
                PY, TOOL,
                "--contestants", f"{a}={by_name[a]}", f"{b}={by_name[b]}",
                "--config-from", args.config_from,
                "--no-phillip",
                "--games-per-pair", str(args.games_per_pair),
                "--envs", str(args.games_per_pair),
                "--char-mode", args.char_mode,
                "--device", "cpu",
                "--max-minutes", str(args.pair_timeout / 60),
                "--out", out,
            ]
            env = {**os.environ,
                   # batch-2 cpu inference gains nothing from intra-op
                   # threads; N procs x 48-thread default pools thrashed
                   # the box to load 160+ (live-caught)
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            running[i] = (
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, env=env,
                ),
                time.monotonic(), a, b, out,
            )
        for i, (proc, started, a, b, out) in list(running.items()):
            rc = proc.poll()
            if rc is None and time.monotonic() - started > args.pair_timeout:
                proc.kill()
                rc = -9
            if rc is not None:
                del running[i]
                done += 1
                if done % 20 == 0 or not (queue or running):
                    el = time.monotonic() - t0
                    print(f"[{el:6.0f}s] {done}/{len(pairs)} pairs done "
                          f"({len(running)} running)", flush=True)
        time.sleep(2)

    tool = _load_tool()
    pair_results = []
    missing = []
    for i, (a, b) in enumerate(pairs):
        out = os.path.join(shard_dir, f"pair-{i:04d}.json")
        try:
            with open(out) as f:
                rep = json.load(f)
            pair_results.extend(rep["pairs"])
        except (OSError, json.JSONDecodeError, KeyError):
            missing.append((a, b))
            pair_results.append({
                "a": a, "b": b, "envs": 0, "games": 0, "wins_a": 0, "wins_b": 0,
                "draws": 0, "stock_diffs_a": [], "elapsed_seconds": 0.0,
                "error": "shard missing/unreadable",
            })
    merged = {
        "timestamp": stamp,
        "config": {
            "games_per_pair": args.games_per_pair,
            "char_mode": args.char_mode,
            "sharded": True,
            "concurrency": args.concurrency,
        },
        "pairs": pair_results,
        **tool.compute_standings(names, pair_results),
    }
    dest = args.out or os.path.join(OUT_DIR, f"sharded-{stamp}.json")
    with open(dest, "w") as f:
        json.dump(merged, f, indent=1)
    print(tool.format_table(merged), flush=True)
    if missing:
        print(f"MISSING PAIRS ({len(missing)}): {missing[:10]}", flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
