"""Unit tests for the eval harness: stats math, opponent specs, report I/O.

No Dolphin required — the single-game engine itself is covered by the live
smoke battery (see docs/); these tests pin down everything computable.
"""

import json

import pytest

from smashbot.eval import report as report_lib
from smashbot.eval.game import Opponent
from smashbot.eval.report import GameRecord


def _rec(winner, bot=4, opp=0, frames=7200, timeout=False):
    return GameRecord(
        winner=winner, bot_stocks=bot, opp_stocks=opp,
        bot_damage_dealt=400.0, bot_damage_taken=250.0,
        frames=frames, timeout=timeout,
    )


def test_wilson_interval_bounds():
    assert report_lib.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = report_lib.wilson_interval(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.15
    lo, hi = report_lib.wilson_interval(50, 50)
    assert 0.85 < lo < 1.0 and hi == 1.0
    lo, hi = report_lib.wilson_interval(25, 50)
    assert lo < 0.5 < hi
    # more games -> tighter interval
    lo2, hi2 = report_lib.wilson_interval(250, 500)
    assert (hi2 - lo2) < (hi - lo)


def test_aggregate():
    records = [_rec("bot"), _rec("bot", bot=2, opp=0), _rec("opp", bot=0, opp=3),
               _rec(None, bot=1, opp=1, timeout=True)]
    a = report_lib.aggregate(records)
    assert a["games"] == 4
    assert (a["wins"], a["losses"], a["draws"]) == (2, 1, 1)
    assert a["win_rate"] == pytest.approx(2 / 3)  # draws excluded from rate
    assert a["timeouts"] == 1
    assert a["avg_stock_diff"] == pytest.approx((4 + 2 - 3 + 0) / 4)
    assert a["avg_game_seconds"] == pytest.approx(120.0)


def test_aggregate_empty():
    a = report_lib.aggregate([])
    assert a["games"] == 0 and a["win_rate"] == 0.0
    assert a["win_rate_ci95"] == [0.0, 1.0]


def test_opponent_spec_parsing():
    o = Opponent.parse("cpu:9")
    assert (o.kind, o.level, o.character) == ("cpu", 9, "MARTH")
    o = Opponent.parse("cpu:3:FALCO")
    assert (o.kind, o.level, o.character) == ("cpu", 3, "FALCO")
    o = Opponent.parse("ckpt:/path/to/best.pt")
    assert (o.kind, o.ckpt_path, o.character) == ("ckpt", "/path/to/best.pt", "FOX")
    o = Opponent.parse("ckpt:/path/best.pt:MARTH")
    assert (o.kind, o.ckpt_path, o.character) == ("ckpt", "/path/best.pt", "MARTH")
    assert Opponent.parse("human").kind == "human"
    with pytest.raises(ValueError):
        Opponent.parse("cpu:0")
    with pytest.raises(ValueError):
        Opponent.parse("ckpt:")
    with pytest.raises(ValueError):
        Opponent.parse("wombo:9")


def test_save_report_roundtrip(tmp_path):
    records = [_rec("bot"), _rec("opp", bot=0, opp=2)]
    out = tmp_path / "report.json"
    report = report_lib.save_report(out, {"ckpt": "x.pt", "opponent": "cpu:9"}, records)
    on_disk = json.loads(out.read_text())
    assert on_disk["aggregate"] == report["aggregate"]
    assert on_disk["meta"]["ckpt"] == "x.pt"
    assert len(on_disk["games"]) == 2
    # records reconstruct cleanly
    rebuilt = [GameRecord(**g) for g in on_disk["games"]]
    assert rebuilt[0].winner == "bot" and rebuilt[1].opp_stocks == 2
    summary = report_lib.format_summary(report)
    assert "1W-1L-0D" in summary
