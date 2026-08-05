"""Headless eval batteries: N games of a checkpoint vs an opponent spec,
fanned across parallel Dolphin workers, aggregated into a JSON report.

Usage:
  python -m smashbot.eval.evaluate --ckpt best.pt --opponent cpu:9 --num-games 50
  python -m smashbot.eval.evaluate --ckpt new.pt --opponent ckpt:old.pt --parallel 8

Opponent specs: cpu:<level>[:<CHAR>]  |  ckpt:<path>[:<CHAR>]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from smashbot.eval import report as report_lib


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
    except OSError:
        return "unknown"


def _worker(
    worker_id: int,
    args_dict: dict,
    n_games: int,
    queue: mp.Queue,
) -> None:
    """One eval worker: own policy instance(s), own Dolphin, n_games games."""
    # imports here: torch/melee must load post-fork in the spawned process
    from smashbot.eval import game as game_lib
    from smashbot.eval.agent import DelayedAgent

    args = argparse.Namespace(**args_dict)
    opponent = game_lib.Opponent.parse(args.opponent)
    try:
        policy, name_map, _ = game_lib.load_policy(args.ckpt, args.device)
        if args.compile:
            game_lib.maybe_compile(policy, args.device, verbose=worker_id == 0)
        name_code = game_lib.resolve_name_code(name_map, args.name, verbose=False)

        import melee
        from slippi_ai import dolphin as dolphin_lib

        players = {
            1: dolphin_lib.AI(character=melee.Character[args.bot_char.upper()]),
            2: opponent.make_player(),
        }

        agents = {}
        agents[1] = DelayedAgent(
            policy, own_port=1, opponent_port=2, name_code=name_code,
            console_delay=0, temperature=args.temperature, device=args.device,
        )
        if opponent.kind == "ckpt":
            opp_policy, opp_map, _ = game_lib.load_policy(opponent.ckpt_path, args.device)
            if args.compile:
                game_lib.maybe_compile(opp_policy, args.device, verbose=False)
            agents[2] = DelayedAgent(
                opp_policy, own_port=2, opponent_port=1,
                name_code=game_lib.resolve_name_code(opp_map, args.name, verbose=False),
                console_delay=0, temperature=args.temperature, device=args.device,
            )

        done = 0
        while done < n_games:
            dolphin = game_lib.make_dolphin(players, headless=True, stage=args.stage)
            got_any = False
            try:
                batch = min(args.games_per_dolphin, n_games - done)
                for record in game_lib.run_games(
                    dolphin, agents, num_games=batch,
                    max_frames_per_game=args.max_game_minutes * 60 * 60,
                ):
                    got_any = True
                    queue.put(("record", worker_id, record.to_dict()))
                    done += 1
            finally:
                dolphin.stop()
            if not got_any:
                raise RuntimeError("dolphin produced no completed games")
        queue.put(("done", worker_id, None))
    except Exception as e:  # surface worker failures to the main process
        queue.put(("error", worker_id, f"{type(e).__name__}: {e}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", default="cpu:9",
                    help="cpu:<level>[:<CHAR>] | ckpt:<path>[:<CHAR>]")
    ap.add_argument("--num-games", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--bot-char", default="FOX")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--name", default="Master Player")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--games-per-dolphin", type=int, default=10,
                    help="restart Dolphin every N games (memory leaks)")
    ap.add_argument("--max-game-minutes", type=int, default=8)
    ap.add_argument("--out", default="",
                    help="report path (default runs/eval/<ts>_<tag>.json)")
    args = ap.parse_args()

    opponent = args.opponent
    from smashbot.eval.game import Opponent
    Opponent.parse(opponent)  # fail fast on bad specs
    if opponent.startswith("human"):
        ap.error("eval batteries need a scriptable opponent (cpu:/ckpt:)")

    n_workers = min(args.parallel, args.num_games)
    per_worker = [args.num_games // n_workers] * n_workers
    for i in range(args.num_games % n_workers):
        per_worker[i] += 1

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    args_dict = vars(args)
    # non-daemon: libmelee's slippstream spawns its own child process, which
    # daemonic workers are not allowed to do. Cleanup is explicit below.
    workers = [
        ctx.Process(target=_worker, args=(i, args_dict, n, queue))
        for i, n in enumerate(per_worker)
    ]
    t0 = time.time()
    for w in workers:
        w.start()
    print(f"{n_workers} workers x {per_worker} games | {args.ckpt} vs {opponent}")

    records: list[report_lib.GameRecord] = []
    active, errors = n_workers, []
    while active > 0:
        kind, wid, payload = queue.get()
        if kind == "record":
            records.append(report_lib.GameRecord(**payload))
            r = records[-1]
            print(f"[{len(records):3d}/{args.num_games}] worker {wid}: "
                  f"{r.winner or 'draw':4s} {r.bot_stocks}-{r.opp_stocks} "
                  f"({r.frames / 60:.0f}s)")
        elif kind == "done":
            active -= 1
        else:
            active -= 1
            errors.append(f"worker {wid}: {payload}")
            print(f"WORKER FAILED — {errors[-1]}")
    for w in workers:
        w.join(timeout=30)
        if w.is_alive():
            w.terminate()
            w.join(timeout=10)

    tag = Path(args.ckpt).parent.name or Path(args.ckpt).stem
    out = args.out or (
        f"runs/eval/{time.strftime('%Y%m%d_%H%M')}_{tag}_vs_"
        f"{opponent.replace(':', '').replace('/', '_')}.json"
    )
    meta = {
        "ckpt": args.ckpt,
        "opponent": opponent,
        "bot_char": args.bot_char,
        "stage": args.stage,
        "temperature": args.temperature,
        "name": args.name,
        "git_sha": _git_sha(),
        "wall_seconds": round(time.time() - t0, 1),
        "errors": errors,
    }
    report = report_lib.save_report(out, meta, records)
    print(f"\n{report_lib.format_summary(report)}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
