"""Ghost tournament: round-robin evaluation between frozen policies.

Post-training snapshot selection: play every unordered pair of "contestants"
(bare snapshot state_dicts sharing --config-from, full checkpoints, and
optionally the ported Phillip medium-v2 as a calibration anchor) against each
other, headless and maximally parallel, and rank them.

Scheduling: pairs are packed into WAVES sized to the --envs Dolphin budget.
Within a wave every pair gets its own small DolphinRolloutWorker (battery.py's
building blocks, one worker per pairing) running in its own thread, so up to
--envs Dolphins run at once while each worker keeps the proven
one-student + one-opponent-group layout. Seats alternate ports across each
pair's envs, so engine port bias cancels over the pair's games.

Devices: --device cpu is the safe default (inference is batched per
contestant, and each pair's batch is small). Post-run the machine is free, so
--device cuda is also fine and faster for big waves; there is no fallback
logic — pick explicitly.

Usage (post-v3 example: 6 snapshots + the final checkpoint + phillip):
  .venv/bin/python scripts/tournament.py \
      --contestants \
        s1250=runs/rl-pool-v3/snapshots/snapshot-0001250.pt \
        s2500=runs/rl-pool-v3/snapshots/snapshot-0002500.pt \
        final=runs/rl-pool-v3/latest.pt:full \
      --config-from runs/rl-pool-v3/latest.pt \
      --games-per-pair 8 --envs 64 --char-mode fox

NOTE: constructs DolphinRolloutWorkers -> the spawn context re-imports
__main__, so everything Dolphin-touching runs under the __main__ guard.
"""

from __future__ import annotations

import os

# Modest per-process torch threading: parallelism comes from the Dolphin
# fleet + per-pair threads, not intra-op threads. Must precede torch init.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import dataclasses
import json
import threading
import time

import torch

from smashbot.rl.pool import MAIN_12, EnvSpec
from smashbot.rl.rollouts import GameTracker, RolloutConfig

DEFAULT_CONFIG_FROM = "/home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt"
DEFAULT_PHILLIP = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
PHILLIP_NAME = "phillip"
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tournament_results"
)

STAGE_ALIASES = {
    "FD": "FINAL_DESTINATION",
    "BF": "BATTLEFIELD",
    "DL": "DREAMLAND",
    "YS": "YOSHIS_STORY",
    "FOD": "FOUNTAIN_OF_DREAMS",
    "PS": "POKEMON_STADIUM",
}


# ------------------------------------------------------------ pure plumbing


@dataclasses.dataclass(frozen=True)
class Contestant:
    """One tournament entrant.

    kind: "snapshot" = bare policy state_dict (architecture borrowed from
    --config-from); "full" = self-contained checkpoint (config + name_map in
    the file, battery's --ckpt path — also how Phillip loads)."""

    name: str
    path: str
    kind: str  # "snapshot" | "full"


def parse_contestant(entry: str) -> Contestant:
    """"NAME=PATH" (bare snapshot state_dict) or "NAME=PATH:full" (full
    checkpoint). Bad entries fail loudly."""
    name, eq, rest = entry.partition("=")
    if not (eq and name and rest):
        raise ValueError(
            f"bad contestant entry {entry!r}: want NAME=PATH or NAME=PATH:full"
        )
    if not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError(
            f"bad contestant name {name!r}: names key tables and JSON — "
            f"alphanumeric/-/_/. only"
        )
    kind = "snapshot"
    if rest.endswith(":full"):
        kind = "full"
        rest = rest[: -len(":full")]
    if not rest:
        raise ValueError(f"bad contestant entry {entry!r}: empty path")
    return Contestant(name=name, path=rest, kind=kind)


def resolve_contestants(
    entries: list[str], phillip: bool, phillip_ckpt: str
) -> list[Contestant]:
    """Parse all entries, append the Phillip calibration contestant when
    enabled, and reject duplicate names."""
    contestants = [parse_contestant(e) for e in entries]
    if phillip:
        contestants.append(
            Contestant(name=PHILLIP_NAME, path=phillip_ckpt, kind="full")
        )
    seen: set[str] = set()
    for c in contestants:
        if c.name in seen:
            raise ValueError(
                f"duplicate contestant name {c.name!r}"
                + (
                    f" (rename yours: {PHILLIP_NAME!r} is reserved for "
                    f"--phillip)"
                    if c.name == PHILLIP_NAME
                    else ""
                )
            )
        seen.add(c.name)
    if len(contestants) < 2:
        raise ValueError(
            f"need at least 2 contestants, got {len(contestants)}"
        )
    return contestants


def enumerate_pairs(names: list[str]) -> list[tuple[str, str]]:
    """All unordered pairs, in input order (deterministic)."""
    return [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]


@dataclasses.dataclass(frozen=True)
class PairPlan:
    a: str
    b: str
    num_envs: int


def envs_per_pair(games_per_pair: int, envs: int) -> int:
    """Dedicated envs for one pairing: enough to play games_per_pair mostly
    in parallel, even (seat balance: ports alternate across the pair's envs),
    never above the wave budget."""
    if envs < 2:
        raise ValueError(f"--envs must be >= 2, got {envs}")
    per = min(games_per_pair, envs)
    per -= per % 2
    return max(2, per)


def plan_waves(
    pairs: list[tuple[str, str]], games_per_pair: int, envs: int
) -> list[list[PairPlan]]:
    """Greedy first-fit packing of pairs into sequential waves such that the
    total Dolphin count of a wave never exceeds `envs`. Every pair appears in
    exactly one wave."""
    per = envs_per_pair(games_per_pair, envs)
    waves: list[list[PairPlan]] = []
    current: list[PairPlan] = []
    used = 0
    for a, b in pairs:
        if used + per > envs:
            waves.append(current)
            current, used = [], 0
        current.append(PairPlan(a=a, b=b, num_envs=per))
        used += per
    if current:
        waves.append(current)
    return waves


def build_pair_specs(num_envs: int, char_mode: str) -> list[EnvSpec]:
    """Env layout for one pairing: every env kind="teacher" (contestant B's
    agent serves the whole opponent group), seats alternating 1/2 so the
    pair's games are port-balanced. Characters: "fox" pins FOX both seats
    (peak-vs-peak between fox specialists); "main12" boots a rotation over
    the main 12 and the worker's per-game redraw takes it from there.
    Deterministic — no RNG in spec construction."""
    if num_envs < 2 or num_envs % 2 != 0:
        raise ValueError(f"pair env count must be even and >= 2, got {num_envs}")
    if char_mode not in ("fox", "main12"):
        raise ValueError(f"unknown char mode {char_mode!r}")
    return [
        EnvSpec(
            kind="teacher",
            group=-1,
            student_port=1 + (i % 2),
            opponent_char=(
                "FOX" if char_mode == "fox" else MAIN_12[i % len(MAIN_12)]
            ),
        )
        for i in range(num_envs)
    ]


def resolve_stage(stage: str) -> str:
    """FD/BF/... aliases or full melee.Stage names, case-insensitive."""
    s = stage.upper()
    return STAGE_ALIASES.get(s, s)


# ---------------------------------------------------------------- standings


def empty_pair_result(plan: PairPlan, error: str | None = None) -> dict:
    return {
        "a": plan.a,
        "b": plan.b,
        "envs": plan.num_envs,
        "games": 0,
        "wins_a": 0,
        "wins_b": 0,
        "draws": 0,
        "stock_diffs_a": [],
        "elapsed_seconds": 0.0,
        "error": error,
    }


def pair_result_from_tracker(
    plan: PairPlan,
    tracker: GameTracker,
    elapsed_seconds: float,
    error: str | None,
) -> dict:
    """Serialize one pair's GameTracker (student seat = contestant A, so
    tracker wins/diffs are from A's perspective)."""
    return {
        "a": plan.a,
        "b": plan.b,
        "envs": plan.num_envs,
        "games": tracker.wins + tracker.losses + tracker.draws,
        "wins_a": tracker.wins,
        "wins_b": tracker.losses,
        "draws": tracker.draws,
        "stock_diffs_a": [int(d) for d in tracker.diffs],
        "elapsed_seconds": round(elapsed_seconds, 1),
        "error": error,
    }


def compute_standings(names: list[str], pair_results: list[dict]) -> dict:
    """Aggregate pair results into per-contestant standings, a head-to-head
    matrix, and a ranking.

    win_rate is over DECIDED games (draws excluded, the repo convention);
    None when nothing was decided — zero-game contestants are reported
    honestly, never invented. Ranking: overall win_rate desc, vs-phillip
    win_rate as tiebreak (contestants without a decided phillip game — and
    phillip himself — tiebreak below any measured vs-phillip rate), then
    input order for full determinism."""
    per: dict[str, dict] = {
        n: {
            "games": 0, "wins": 0, "losses": 0, "draws": 0,
            "stock_diff_sum": 0, "vs_phillip_wins": 0, "vs_phillip_decided": 0,
        }
        for n in names
    }
    h2h: dict[str, dict[str, dict]] = {n: {} for n in names}
    for r in pair_results:
        a, b = r["a"], r["b"]
        for me, opp, wins, losses, sign in (
            (a, b, r["wins_a"], r["wins_b"], 1),
            (b, a, r["wins_b"], r["wins_a"], -1),
        ):
            s = per[me]
            s["games"] += r["games"]
            s["wins"] += wins
            s["losses"] += losses
            s["draws"] += r["draws"]
            s["stock_diff_sum"] += sign * sum(r["stock_diffs_a"])
            if opp == PHILLIP_NAME:
                s["vs_phillip_wins"] += wins
                s["vs_phillip_decided"] += wins + losses
            h2h[me][opp] = {
                "wins": wins, "losses": losses,
                "draws": r["draws"], "games": r["games"],
            }

    standings = []
    for n in names:
        s = per[n]
        decided = s["wins"] + s["losses"]
        vs_p = (
            s["vs_phillip_wins"] / s["vs_phillip_decided"]
            if s["vs_phillip_decided"]
            else None
        )
        standings.append(
            {
                "name": n,
                "games": s["games"],
                "wins": s["wins"],
                "losses": s["losses"],
                "draws": s["draws"],
                "win_rate": (s["wins"] / decided) if decided else None,
                "vs_phillip": None if n == PHILLIP_NAME else vs_p,
                "avg_stock_diff": (
                    s["stock_diff_sum"] / s["games"] if s["games"] else None
                ),
            }
        )

    order = {n: i for i, n in enumerate(names)}
    key = lambda st: (
        -(st["win_rate"] if st["win_rate"] is not None else -1.0),
        -(st["vs_phillip"] if st["vs_phillip"] is not None else -1.0),
        order[st["name"]],
    )
    ranking = [st["name"] for st in sorted(standings, key=key)]
    return {"standings": standings, "head_to_head": h2h, "ranking": ranking}


def make_report(
    contestants: list[Contestant],
    pair_results: list[dict],
    stamp: str,
    config_echo: dict,
) -> dict:
    names = [c.name for c in contestants]
    return {
        "timestamp": stamp,
        "contestants": [dataclasses.asdict(c) for c in contestants],
        "config": config_echo,
        "pairs": pair_results,
        **compute_standings(names, pair_results),
    }


def format_table(report: dict) -> str:
    """Pretty stdout tables: ranking + head-to-head matrix (rows show
    "wins-losses" from the ROW contestant's perspective)."""
    pct = lambda x: "  --" if x is None else f"{x:4.0%}"
    sd = lambda x: "   --" if x is None else f"{x:+5.2f}"
    by_name = {st["name"]: st for st in report["standings"]}
    names = report["ranking"]
    w = max(8, max(len(n) for n in names))
    lines = ["", f"{'#':>2} {'name':<{w}} {'games':>5} {'W-L-D':>9} "
                 f"{'win%':>4} {'vsPhil':>6} {'stockD':>6}"]
    for i, n in enumerate(names):
        st = by_name[n]
        wld = f"{st['wins']}-{st['losses']}-{st['draws']}"
        lines.append(
            f"{i + 1:>2} {n:<{w}} {st['games']:>5} {wld:>9} "
            f"{pct(st['win_rate'])} {pct(st['vs_phillip']):>6} "
            f"{sd(st['avg_stock_diff'])}"
        )
    lines.append("")
    cw = max(5, max(len(n) for n in names))
    lines.append("head-to-head (row's W-L vs column):")
    lines.append(" " * (w + 1) + " ".join(f"{n:>{cw}}" for n in names))
    h2h = report["head_to_head"]
    for a in names:
        cells = []
        for b in names:
            if a == b:
                cells.append(f"{'.':>{cw}}")
                continue
            cell = h2h.get(a, {}).get(b)
            cells.append(
                f"{'?':>{cw}}" if cell is None
                else f"{cell['wins']}-{cell['losses']}".rjust(cw)
            )
        lines.append(f"{a:<{w}} " + " ".join(cells))
    zero = [r for r in report["pairs"] if r["games"] == 0]
    if zero:
        lines.append("")
        lines.append("pairs with NO finished games (honest zeros):")
        for r in zero:
            lines.append(f"  {r['a']} vs {r['b']}: {r['error'] or 'no games'}")
    return "\n".join(lines)


def format_schedule(waves: list[list[PairPlan]], games_per_pair: int) -> str:
    lines = []
    for wi, wave in enumerate(waves):
        total = sum(p.num_envs for p in wave)
        lines.append(f"wave {wi + 1}/{len(waves)}: {total} dolphins")
        for p in wave:
            lines.append(
                f"  {p.a} vs {p.b}: {p.num_envs} envs, "
                f"{games_per_pair} games target"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--contestants", nargs="+", required=True, metavar="NAME=PATH[:full]",
        help="bare snapshot state_dicts (NAME=PATH, arch from --config-from) "
             "and/or full checkpoints (NAME=PATH:full)")
    ap.add_argument("--config-from", default=DEFAULT_CONFIG_FROM,
                    help="full checkpoint whose config/name_map builds the "
                         "policy for snapshot contestants")
    ap.add_argument("--phillip", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="include the ported Phillip medium-v2 as a "
                         "calibration contestant (--no-phillip to drop)")
    ap.add_argument("--phillip-ckpt", default=DEFAULT_PHILLIP)
    ap.add_argument("--games-per-pair", type=int, default=8,
                    help="finished games required per pairing")
    ap.add_argument("--envs", type=int, default=64,
                    help="max Dolphins at once (wave budget)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="inference device; post-run the GPU is free too "
                         "(--device cuda)")
    ap.add_argument("--char-mode", default="fox", choices=("fox", "main12"),
                    help="fox: BOTH seats FOX (peak-vs-peak between fox "
                         "specialists); main12: per-game character redraws")
    ap.add_argument("--stage", default="FD",
                    help="FD/BF/DL/YS/FOD/PS alias or full melee.Stage name")
    ap.add_argument("--max-minutes", type=float, default=90.0,
                    help="total wall-clock budget; partial results are still "
                         "reported")
    ap.add_argument("--out", default="",
                    help=f"JSON report path (default "
                         f"{RESULTS_DIR}/<timestamp>.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve contestants + print the schedule, then stop "
                         "before booting any Dolphin")
    ap.add_argument("--games-per-dolphin", type=int, default=200,
                    help="kept high so Dolphin recycles never interfere")
    ap.add_argument("--env-timeout", type=float, default=300.0)
    return ap.parse_args(argv)


# ------------------------------------------------- Dolphin-touching (main)


def _load_contestant_policies(
    contestants: list[Contestant], config_from: str, device: str
) -> dict[str, tuple[torch.nn.Module, int]]:
    """{name: (frozen policy, name_code)}. Snapshot contestants borrow the
    --config-from architecture/name_map (battery's --snapshot pattern); full
    contestants (and Phillip) carry their own."""
    from smashbot.eval.game import load_policy, resolve_name_code

    out: dict[str, tuple[torch.nn.Module, int]] = {}
    for c in contestants:
        if c.kind == "snapshot":
            policy, name_map, _ = load_policy(config_from, device)
            state = torch.load(c.path, map_location=device, weights_only=True)
            policy.load_state_dict(state)
        else:
            policy, name_map, _ = load_policy(c.path, device)
        policy.train_value_head = False
        policy.requires_grad_(False)
        policy.eval()
        out[c.name] = (policy, resolve_name_code(name_map, "Master Player"))
        print(f"loaded {c.name}: {c.path} ({c.kind})", flush=True)
    return out


def _run_pair(
    plan: PairPlan,
    games_target: int,
    deadline: float,
    policies: dict,
    args: argparse.Namespace,
    results: dict,
    progress: dict,
) -> None:
    """One pairing, one worker, one thread: contestant A on the student seat
    of every env, contestant B serving the whole "teacher" opponent group.
    Collect until the pair's game target or the shared deadline; any crash
    keeps the tracker's partial results (battery-style honesty)."""
    from smashbot.rl.agent import BatchedPolicyAgent
    from smashbot.rl.rollouts import DolphinRolloutWorker

    key = (plan.a, plan.b)
    n = plan.num_envs
    rcfg = RolloutConfig(
        num_envs=n,
        cpu_envs=0,
        teacher_envs=n,
        ref_envs=0,
        snapshot_slots=0,
        bot_char="FOX",
        stage=resolve_stage(args.stage),
        games_per_dolphin=args.games_per_dolphin,
        redraw_chars=(args.char_mode == "main12"),
        double_buffer=False,  # proven recycle path only
        log_tag=f"tourney-{plan.a}-vs-{plan.b}",
        env_timeout=args.env_timeout,
        char_whitelist=(list(MAIN_12) if args.char_mode == "main12"
                        else ["FOX"]),
    )
    pol_a, code_a = policies[plan.a]
    pol_b, code_b = policies[plan.b]
    # Fresh agents per pair: clean recurrent state / delay queues, and a
    # contestant appearing in several pairs of one wave shares its policy
    # MODULE (inference-only forwards are thread-safe) but never its state.
    student = BatchedPolicyAgent(pol_a, n, name_code=code_a, device=args.device)
    opponent = BatchedPolicyAgent(pol_b, n, name_code=code_b, device=args.device)
    specs = build_pair_specs(n, args.char_mode)
    worker = None
    error = None
    t0 = time.monotonic()
    try:
        worker = DolphinRolloutWorker(
            rcfg, student, opponents={"teacher": opponent}, specs=specs
        )
        while True:
            worker.collect(1)
            done = worker.trackers["teacher"].stats()["games_played"]
            progress[key] = done
            if done >= games_target:
                break
            if time.monotonic() > deadline:
                error = "budget reached; partial results kept"
                break
    except Exception as e:  # env death/timeout: keep partials, report why
        error = f"{type(e).__name__}: {e}"
    finally:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
    elapsed = time.monotonic() - t0
    tracker = worker.trackers["teacher"] if worker is not None else GameTracker()
    results[key] = pair_result_from_tracker(plan, tracker, elapsed, error)
    progress[key] = results[key]["games"]
    r = results[key]
    tag = f" [{error}]" if error else ""
    print(
        f"[pair {plan.a} vs {plan.b}] done: {r['wins_a']}-{r['wins_b']}-"
        f"{r['draws']} in {elapsed:.0f}s{tag}",
        flush=True,
    )


def _run_wave(
    wave_idx: int,
    num_waves: int,
    wave: list[PairPlan],
    games_target: int,
    deadline: float,
    policies: dict,
    args: argparse.Namespace,
) -> list[dict]:
    """Run every pairing of the wave concurrently (one worker thread each,
    total Dolphins <= --envs) and return their pair results."""
    results: dict = {}
    progress: dict = {(p.a, p.b): 0 for p in wave}
    threads = [
        threading.Thread(
            target=_run_pair,
            args=(p, games_target, deadline, policies, args, results, progress),
            name=f"pair-{p.a}-vs-{p.b}",
        )
        for p in wave
    ]
    total_envs = sum(p.num_envs for p in wave)
    print(
        f"[wave {wave_idx + 1}/{num_waves}] {len(wave)} pairs, "
        f"{total_envs} dolphins, target {games_target} games/pair",
        flush=True,
    )
    t0 = time.monotonic()
    for t in threads:
        t.start()
    last_print = time.monotonic()
    while any(t.is_alive() for t in threads):
        time.sleep(2)
        now = time.monotonic()
        if now - last_print > 30:
            done = sum(progress.values())
            print(
                f"[wave {wave_idx + 1} {now - t0:5.0f}s] "
                f"{done}/{games_target * len(wave)} games "
                f"({len(results)}/{len(wave)} pairs finished)",
                flush=True,
            )
            last_print = now
    for t in threads:
        t.join()
    return [
        results.get((p.a, p.b), empty_pair_result(p, "worker never reported"))
        for p in wave
    ]


def main() -> None:
    args = parse_args()
    stamp = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    contestants = resolve_contestants(
        args.contestants, args.phillip, args.phillip_ckpt
    )
    names = [c.name for c in contestants]
    pairs = enumerate_pairs(names)
    waves = plan_waves(pairs, args.games_per_pair, args.envs)
    total_games = args.games_per_pair * len(pairs)
    print(
        f"tournament: {len(contestants)} contestants, {len(pairs)} pairs, "
        f"{len(waves)} waves, {total_games} games target, "
        f"char-mode {args.char_mode}, stage {resolve_stage(args.stage)}, "
        f"device {args.device}",
        flush=True,
    )
    print(format_schedule(waves, args.games_per_pair), flush=True)

    policies = _load_contestant_policies(
        contestants, args.config_from, args.device
    )
    if args.dry_run:
        print("DRY RUN: contestants resolved and schedule printed; "
              "stopping before any Dolphin boots.", flush=True)
        return

    t0 = time.monotonic()
    deadline = t0 + args.max_minutes * 60
    pair_results: list[dict] = []
    for wi, wave in enumerate(waves):
        if time.monotonic() > deadline - 60:
            print(
                f"[wave {wi + 1}/{len(waves)}] skipped: budget exhausted",
                flush=True,
            )
            pair_results.extend(
                empty_pair_result(p, "skipped: budget exhausted") for p in wave
            )
            continue
        pair_results.extend(
            _run_wave(
                wi, len(waves), wave, args.games_per_pair, deadline,
                policies, args,
            )
        )

    config_echo = dict(
        contestant_entries=list(args.contestants),
        config_from=args.config_from,
        phillip=args.phillip,
        phillip_ckpt=args.phillip_ckpt if args.phillip else "",
        games_per_pair=args.games_per_pair,
        envs=args.envs,
        device=args.device,
        char_mode=args.char_mode,
        stage=resolve_stage(args.stage),
        max_minutes=args.max_minutes,
        games_per_dolphin=args.games_per_dolphin,
        env_timeout=args.env_timeout,
        num_pairs=len(pairs),
        num_waves=len(waves),
        elapsed_seconds=round(time.monotonic() - t0, 1),
    )
    report = make_report(contestants, pair_results, stamp, config_echo)

    out_path = args.out or os.path.join(
        RESULTS_DIR, stamp.replace(":", "-") + ".json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(format_table(report), flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
