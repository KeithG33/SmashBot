"""Sim round-robin tournament between checkpoints: every pair plays a
deterministic slate to completion; standings by winrate then stock diff.

  .venv/bin/python scripts/tournament.py --ckpts a.pt b.pt c.pt \
      [--games 48] [--envs 24] [--device cpu] [--out standings.json]

Bare state_dicts (snapshots) are built from --config-from (default: the
first full checkpoint given). Runs on CPU alongside GPU training; the sim
is deterministic and pairs are seeded, so results replay exactly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import torch

from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.eval.sim_arena import MatchSet, stratified


def load_any(path: str, device: str, config_from: str):
    try:
        pol, nm, step = load_policy(path, device)
        code = resolve_name_code(nm, "Master Player", verbose=False)
    except Exception:
        from smashbot.rl.sim_league import SimLeague
        lg = SimLeague(os.path.dirname(path), phillips={},
                       fox_imports={}, config_from=config_from, device=device)
        pol = lg._make_skeleton()
        lg._load_into(pol, path)
        code, step = 1, None
    pol.eval()
    pol.requires_grad_(False)
    return pol, code, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=48,
                    help="min decided games per pairing")
    ap.add_argument("--envs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--config-from", default="",
                    help="full ckpt whose config builds bare state_dicts "
                         "(default: first --ckpts entry)")
    ap.add_argument("--data-dir",
                    default=os.environ.get("MSL_DATA_DIR",
                                           "/home/kage/drive2/ShineBot/msl-data"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg_from = args.config_from or args.ckpts[0]

    entries = {}
    for p in args.ckpts:
        pol, code, step = load_any(p, args.device, cfg_from)
        entries[os.path.basename(p)] = (pol, code)
        print(f"loaded {os.path.basename(p)}"
              + (f" (step {step})" if step is not None else ""), flush=True)

    results = {}      # (a, b) -> stats for a-as-student vs b
    for a, b in itertools.combinations(entries, 2):
        pa, ca = entries[a]
        pb, cb = entries[b]
        pairs = stratified(args.envs, seed=args.seed)
        ms = MatchSet(pa, pb, pairs, args.data_dir, args.device,
                      student_name_code=ca, opp_name_code=cb)
        ms.run(min_games=args.games)
        st = ms.stats()
        ms.close()
        results[f"{a}|{b}"] = st
        print(f"  {a} vs {b}: {st['win_rate_recent']:.3f} over "
              f"{st['games']}g (stockdiff {st['avg_stock_diff']:+.2f})",
              flush=True)

    # standings: mean winrate across pairings (a's wins vs b, 1-b's vs a)
    score = {k: [] for k in entries}
    for key, st in results.items():
        a, b = key.split("|")
        score[a].append(st["win_rate_recent"])
        score[b].append(1.0 - st["win_rate_recent"])
    print("\nstandings:")
    for k, xs in sorted(score.items(), key=lambda kv: -sum(kv[1]) / max(1, len(kv[1]))):
        print(f"  {k:40s} {sum(xs)/max(1,len(xs)):.3f}")
    if args.out:
        json.dump({"results": results,
                   "standings": {k: sum(v)/max(1,len(v)) for k, v in score.items()}},
                  open(args.out, "w"), indent=1)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
