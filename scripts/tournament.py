"""Sim round-robin tournament between checkpoints: every pairing plays the
same fixed games (--games of them, stratified over the 12 characters), each
played to its end and counted once. Standings: mean win rate over pairings.
Each entry meets all the entries before it at once, as slices of a stacked
grid (sim_arena.MatchSet), so a round-robin of N is N-1 runs.

  .venv/bin/python scripts/tournament.py --ckpts a.pt b.pt c.pt \
      [--games 48] [--envs 24] [--device cpu] [--out standings.json]

Entries are labelled by as many trailing path components as keep them
distinct, so two runs' best.pt stay two entries. Bare weights (league
snapshots) are built from --config-from (default: the first --ckpts entry,
which must then be a full checkpoint). The sim is deterministic and the
games are seeded, so results replay.
"""
from __future__ import annotations

import argparse
import collections
import json
import time

import torch

from smashbot import paths
from smashbot.eval.sim_arena import MatchSet, dittos, load_player, stratified, unique_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=48,
                    help="games per pairing, stratified over the 12 characters")
    ap.add_argument("--dittos", type=int, default=0,
                    help="instead: this many games of each character against itself")
    ap.add_argument("--envs", type=int, default=24, help="games played at once per pairing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--config-from", default="",
                    help="full checkpoint whose config builds bare weights "
                         "(default: the first --ckpts entry)")
    ap.add_argument("--data-dir", default=str(paths.MSL_DATA_DIR))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    labels = unique_labels(args.ckpts)
    players = {}
    for label, path in zip(labels, args.ckpts):
        policy, code, step = load_player(path, "cpu", args.config_from or args.ckpts[0])
        players[label] = (policy, code)
        print(f"loaded {label}" + (f" (step {step})" if step is not None else ""), flush=True)

    slate = dittos(args.dittos) if args.dittos else stratified(args.games, args.seed)
    results = {}
    win_rates = {label: [] for label in players}
    for j, b in enumerate(labels[1:], 1):
        earlier = labels[:j]
        start = time.perf_counter()
        ms = MatchSet([players[a] for a in earlier], players[b][0], slate, args.data_dir,
                      args.envs, args.device, opp_name_code=players[b][1])
        ms.run()
        ms.close()
        players[b][0].to("cpu")
        print(f"  {b} vs {j} earlier: {(time.perf_counter() - start) / 60:.1f} min", flush=True)
        for i, a in enumerate(earlier):
            st = ms.stats(i)
            results[f"{a}|{b}"] = st
            win_rates[a].append(st["wins"] / st["games"])
            win_rates[b].append(st["losses"] / st["games"])
            print(f"  {a} vs {b}: {st['wins']}-{st['losses']}-{st['draws']} of {st['games']} "
                  f"(stockdiff {st['avg_stock_diff']:+.2f})", flush=True)
            by_char = collections.defaultdict(lambda: [0, 0, 0])
            for (char, _), outcome in zip(ms.slate, ms.outcomes[i]):
                by_char[char][0 if outcome > 0 else 1 if outcome < 0 else 2] += 1
            print("    by " + a + "'s character: " + ", ".join(
                f"{c} {w}-{l}" + (f"-{d}" if d else "") for c, (w, l, d) in by_char.items()), flush=True)

    standings = {label: sum(v) / len(v) for label, v in win_rates.items()}
    print("\nstandings (mean win rate over pairings):")
    for label, rate in sorted(standings.items(), key=lambda kv: -kv[1]):
        print(f"  {label:48s} {rate:.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"paths": dict(zip(labels, args.ckpts)), "results": results,
                       "standings": standings}, f, indent=1)
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
