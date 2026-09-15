"""Sim eval battery: the student checkpoint vs a slate of reference
opponents on melee-sim-light. Deterministic pairs, GameTracker stats
(winrate, stock diff, kill/death percents), JSON report.

Default slate: the five phillip tiers + the three fox imports — the same
opponents training measures against, so battery numbers are directly
comparable to the run's rl/* tracker panels.

  .venv/bin/python scripts/battery.py --ckpt <student.pt> [--games 96]
      [--device cpu] [--opponents gm,master,imp9000] [--out report.json]

CPU-friendly (runs alongside GPU training); GPU when free is ~10x faster.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.eval.sim_arena import MatchSet, full_grid, stratified  # noqa: F401

MODELS = "/home/kage/drive2/ShineBot/models"
V10_SNAPS = "/home/kage/drive2/ShineBot/runs/rl-pool-v10/snapshots"
SLATE = {
    "medium":  f"{MODELS}/medium-v2-torch.pt",
    "plat":    f"{MODELS}/plat-torch.pt",
    "diamond": f"{MODELS}/diamond-torch.pt",
    "master":  f"{MODELS}/master-torch.pt",
    "gm":      f"{MODELS}/gm-torch.pt",
    "imp9000": f"{MODELS}/rl-v3-tournament1st-step0009000.pt",
    "imp10000": f"{MODELS}/rl-best-step0010000-phillip56.pt",
    "s9500":   "/home/kage/drive2/ShineBot/runs/rl-pool-v3/snapshots/snapshot-0009500.pt",
}
FOX_LOCKED = {"imp9000", "imp10000", "s9500"}   # fox-mclaude char locks


def load_opponent(name: str, path: str, device: str, config_from: str):
    """Full checkpoints load directly; bare state_dicts (fox imports /
    ghosts) build from the student checkpoint's config."""
    try:
        pol, nm, _ = load_policy(path, device)
        code = resolve_name_code(nm, "Master Player", verbose=False)
    except Exception:
        from smashbot.rl.sim_league import SimLeague
        lg = SimLeague(None, os.path.dirname(path), phillips={},
                       fox_imports={}, config_from=config_from,
                       device=device)
        pol = lg._make_skeleton()
        lg._load_into(pol, path)
        code = 1
    pol.eval()
    pol.requires_grad_(False)
    return pol, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="student checkpoint (full, or bare state_dict "
                         "with --config-from)")
    ap.add_argument("--config-from", default="",
                    help="full ckpt whose config builds a bare --ckpt")
    ap.add_argument("--games", type=int, default=96,
                    help="min decided games per opponent")
    ap.add_argument("--envs", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--opponents", default="",
                    help="comma list from the slate (default: all)")
    ap.add_argument("--grid", action="store_true",
                    help="full 144-pair character grid per opponent (the "
                         "v10-vs-gm baseline mode), per-pair results in the "
                         "report")
    ap.add_argument("--data-dir",
                    default=os.environ.get("MSL_DATA_DIR",
                                           "/home/kage/drive2/ShineBot/msl-data"))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    try:
        student, snm, step = load_policy(args.ckpt, args.device)
        sc = resolve_name_code(snm, "Master Player", verbose=False)
    except Exception:
        assert args.config_from, "bare student ckpt needs --config-from"
        from smashbot.rl.sim_league import SimLeague
        lg = SimLeague(None, os.path.dirname(args.ckpt), phillips={},
                       fox_imports={}, config_from=args.config_from,
                       device=args.device)
        student = lg._make_skeleton()
        lg._load_into(student, args.ckpt)
        sc, step = 1, None
    student.eval()
    names = [n.strip() for n in args.opponents.split(",") if n.strip()] \
        or list(SLATE)
    print(f"battery: {os.path.basename(args.ckpt)} (step {step}) vs "
          f"{names} | {args.games}+ games each on {args.device}", flush=True)

    report = {"ckpt": args.ckpt, "step": step, "seed": args.seed,
              "opponents": {}}
    for name in names:
        opp, oc = load_opponent(name, SLATE[name], args.device,
                                args.config_from or args.ckpt)
        if args.grid:
            pairs = full_grid(args.seed)
        elif name in FOX_LOCKED:
            pairs = stratified(args.envs, opponent_char="FOX", seed=args.seed)
        else:
            pairs = stratified(args.envs, seed=args.seed)
        ms = MatchSet(student, opp, pairs, args.data_dir, args.device,
                      student_name_code=sc, opp_name_code=oc)
        ms.run(min_games=args.games)
        st = ms.stats()
        if args.grid:   # per-pair first decisions (grid/baseline mode)
            st["pairs"] = {f"{a}|{b}": list(ms.first[i])
                           for i, (a, b) in enumerate(ms.pairs)
                           if i in ms.first}
        ms.close()
        del opp
        report["opponents"][name] = st
        print(f"  vs {name:9s} win {st['win_rate_recent']:.3f} "
              f"(ema {st['win_rate_ema']:.3f}) over {st['games']}g | "
              f"stockdiff {st['avg_stock_diff']:+.2f} | "
              f"kill@{st['avg_percent_at_kill']:.0f}% "
              f"die@{st['avg_percent_at_death']:.0f}%", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
