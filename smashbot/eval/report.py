"""Eval battery statistics and reports (import-light: no torch/melee here)."""

from __future__ import annotations

import dataclasses
import json
import math
import time
import typing as tp
from pathlib import Path


@dataclasses.dataclass
class GameRecord:
    """One game, from the bot's perspective (bot on port 1)."""

    winner: str | None  # "bot" | "opp" | None (draw/timeout)
    bot_stocks: int
    opp_stocks: int
    bot_damage_dealt: float  # sum of opponent percent gains
    bot_damage_taken: float
    frames: int
    timeout: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate; (0, 1) when n == 0."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def aggregate(records: tp.Sequence[GameRecord]) -> dict:
    n = len(records)
    wins = sum(1 for r in records if r.winner == "bot")
    losses = sum(1 for r in records if r.winner == "opp")
    draws = n - wins - losses
    decided = wins + losses
    lo, hi = wilson_interval(wins, decided)
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
    return {
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / decided) if decided else 0.0,
        "win_rate_ci95": [lo, hi],
        "avg_stock_diff": mean([r.bot_stocks - r.opp_stocks for r in records]),
        "avg_damage_dealt": mean([r.bot_damage_dealt for r in records]),
        "avg_damage_taken": mean([r.bot_damage_taken for r in records]),
        "avg_game_seconds": mean([r.frames / 60.0 for r in records]),
        "timeouts": sum(1 for r in records if r.timeout),
    }


def save_report(
    path: str | Path,
    meta: dict,
    records: tp.Sequence[GameRecord],
) -> dict:
    """Writes the battery report JSON and returns it."""
    report = {
        "meta": {**meta, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "aggregate": aggregate(records),
        "games": [r.to_dict() for r in records],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def format_summary(report: dict) -> str:
    a = report["aggregate"]
    lo, hi = a["win_rate_ci95"]
    return (
        f"{a['wins']}W-{a['losses']}L-{a['draws']}D over {a['games']} games | "
        f"win rate {a['win_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%}) | "
        f"stock diff {a['avg_stock_diff']:+.2f} | "
        f"dmg {a['avg_damage_dealt']:.0f}/{a['avg_damage_taken']:.0f} per game | "
        f"avg length {a['avg_game_seconds']:.0f}s"
        + (f" | {a['timeouts']} timeouts" if a["timeouts"] else "")
    )
