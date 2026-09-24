"""Sim eval battery: a student checkpoint against a slate of reference
opponents on melee-sim-light. Every opponent plays the same fixed games
(--games of them, stratified over the 12 characters, or the 144-pair grid),
each played to its end and counted once. The JSON report is rewritten after
every opponent, so a failure keeps the results before it.

Default slate: the five Phillip tiers, the fixed opponents RL trains and
measures against.

  .venv/bin/python scripts/battery.py --ckpt <student.pt> [--games 96]
      [--device cpu] [--opponents gm,master] [--out report.json]

CPU-friendly (runs alongside GPU training); GPU when free is ~10x faster.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from smashbot import paths
from smashbot.eval.sim_arena import MatchSet, full_grid, load_player, stratified

SLATE = {
    "medium":  paths.MODELS_DIR / "medium-v2-torch.pt",
    "plat":    paths.MODELS_DIR / "plat-torch.pt",
    "diamond": paths.MODELS_DIR / "diamond-torch.pt",
    "master":  paths.MODELS_DIR / "master-torch.pt",
    "gm":      paths.MODELS_DIR / "gm-torch.pt",
}


def _write(path: str, report: dict) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(report, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="student checkpoint (full, or bare weights with --config-from)")
    ap.add_argument("--config-from", default="",
                    help="full checkpoint whose config builds a bare --ckpt")
    ap.add_argument("--games", type=int, default=96,
                    help="games per opponent, stratified over the 12 characters")
    ap.add_argument("--grid", action="store_true",
                    help="the full 144-pair character grid instead, --grid-games "
                         "per pair, per-pair results in the report")
    ap.add_argument("--grid-games", type=int, default=1)
    ap.add_argument("--envs", type=int, default=48, help="games played at once")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--opponents", default="",
                    help=f"comma list from the slate {list(SLATE)} (default: all)")
    ap.add_argument("--data-dir", default=str(paths.MSL_DATA_DIR))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    names = [n.strip() for n in args.opponents.split(",") if n.strip()] or list(SLATE)
    unknown = [n for n in names if n not in SLATE]
    if unknown:
        ap.error(f"unknown opponents {unknown}; the slate is {list(SLATE)}")
    student, student_code, step = load_player(args.ckpt, args.device, args.config_from)
    opponents = {n: load_player(str(SLATE[n]), args.device) for n in names}   # a bad slate fails before any game
    pairs = full_grid(args.seed) if args.grid else stratified(args.games, args.seed)
    slate = pairs * args.grid_games if args.grid else pairs
    print(f"battery: {args.ckpt} (step {step}) vs {names} | {len(slate)} games each "
          f"on {args.device}", flush=True)

    report = {"ckpt": args.ckpt, "step": step, "seed": args.seed, "games": len(slate),
              "opponents": {}}
    for name, (opponent, opponent_code, _) in opponents.items():
        ms = MatchSet(student, opponent, slate, args.data_dir, args.envs, args.device,
                      student_name_code=student_code, opp_name_code=opponent_code)
        ms.run()
        ms.close()
        st = ms.stats()
        if args.grid:
            st["pairs"] = {f"{a}|{b}": [list(r) for r in ms.results[i::len(pairs)]]
                           for i, (a, b) in enumerate(pairs)}
        report["opponents"][name] = st
        print(f"  vs {name:8s} win {st['win_rate']:.3f} ({st['wins']}-{st['losses']}"
              f"-{st['draws']} of {st['games']}) | stockdiff {st['avg_stock_diff']:+.2f} | "
              f"kill@{st['avg_percent_at_kill']:.0f}% die@{st['avg_percent_at_death']:.0f}%",
              flush=True)
        if args.out:
            _write(args.out, report)
    if args.out:
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
