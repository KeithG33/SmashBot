"""Unit tests for the battery's pure parts (scripts/battery.py): env-spec
construction determinism, character-slate coverage, seat alternation, and the
JSON report schema round-trip. No Dolphin, no checkpoints."""

import importlib.util
import json
import pathlib

import pytest

from smashbot.rl.pool import MAIN_12
from smashbot.rl.rollouts import GameTracker

_BATTERY_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "battery.py"
)


def _load_battery():
    spec = importlib.util.spec_from_file_location("battery", _BATTERY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


battery = _load_battery()


def test_specs_deterministic():
    # No RNG anywhere: identical layout every call, and across module reloads.
    assert battery.build_specs(8) == battery.build_specs(8)
    assert battery.build_specs(16) == battery.build_specs(16)
    assert battery.build_specs(8) == _load_battery().build_specs(8)


def test_specs_split_and_kinds():
    specs = battery.build_specs(8)
    assert len(specs) == 8
    kinds = [s.kind for s in specs]
    assert kinds.count("teacher") == 4 and kinds.count("reference") == 4


@pytest.mark.parametrize("num_envs", [2, 8, 16])
def test_specs_char_slate(num_envs):
    slate = battery.BATTERY_SLATE
    assert len(slate) == 8 and len(set(slate)) == 8
    assert set(slate) <= set(MAIN_12)  # policy opponents must be on-roster
    specs = battery.build_specs(num_envs)
    half = num_envs // 2
    for kind in ("teacher", "reference"):
        chars = [s.opponent_char for s in specs if s.kind == kind]
        # rotation through the fixed slate, same for both yardsticks
        assert chars == [slate[i % len(slate)] for i in range(half)]
    # at 16 envs each yardstick covers the full slate
    if num_envs >= 16:
        for kind in ("teacher", "reference"):
            assert set(s.opponent_char for s in specs if s.kind == kind) == set(slate)


def test_specs_port_alternation():
    specs = battery.build_specs(8)
    for kind in ("teacher", "reference"):
        ports = [s.student_port for s in specs if s.kind == kind]
        assert ports == [1, 2, 1, 2]


@pytest.mark.parametrize("bad", [0, 3, 7])
def test_specs_reject_odd_or_tiny(bad):
    with pytest.raises(ValueError):
        battery.build_specs(bad)


def _fake_trackers():
    teacher, reference = GameTracker(), GameTracker()
    for diff in (2, 1, -1, 0):  # 2 wins, 1 loss, 1 draw
        teacher.add_game((4, 4 - diff) if diff >= 0 else (4 + diff, 4))
    teacher.add_kill(80.0)
    teacher.add_death(120.0)
    for diff in (-2, -1, 1):  # 1 win, 2 losses
        reference.add_game((4, 4 - diff) if diff >= 0 else (4 + diff, 4))
    return {"teacher": teacher, "reference": reference}


def test_report_schema_roundtrip():
    trackers = _fake_trackers()
    report = battery.make_report(
        student_ckpt="/runs/rl-pool-v3/latest.pt",
        student_rl_step=1250,
        stamp="2026-08-13T12:00:00",
        trackers=trackers,
        config_echo={"envs": 8, "games_per_side": 24, "redraw_chars": False},
    )
    back = json.loads(json.dumps(report))  # JSON-serializable end to end
    assert back["student_rl_step"] == 1250
    assert back["char_slate"] == battery.BATTERY_SLATE
    assert back["config"]["redraw_chars"] is False
    for side in ("teacher", "phillip"):
        assert set(back["results"][side]) == set(battery.YARDSTICK_KEYS)
    t = back["results"]["teacher"]
    assert (t["games"], t["wins"], t["losses"], t["draws"]) == (4, 2, 1, 1)
    assert t["win_rate"] == pytest.approx(2 / 3)  # decided games only
    assert t["avg_percent_at_kill"] == pytest.approx(80.0)
    assert t["avg_percent_at_death"] == pytest.approx(120.0)
    p = back["results"]["phillip"]
    assert p["win_rate"] == pytest.approx(1 / 3)
    assert p["avg_stock_diff"] == pytest.approx((-2 - 1 + 1) / 3)


def test_report_win_rate_none_when_undecided():
    trackers = {"teacher": GameTracker(), "reference": GameTracker()}
    trackers["teacher"].add_game((2, 2))  # draw only
    report = battery.make_report("x", 1, "t", trackers, {})
    assert report["results"]["teacher"]["win_rate"] is None
    assert report["results"]["phillip"]["games"] == 0
    json.dumps(report)  # None must still serialize


def test_summary_line():
    report = battery.make_report("x", 1250, "t", _fake_trackers(), {})
    line = battery.summary_line(report)
    assert line == "BATTERY step 1250: vs-teacher 67% (4g) | vs-phillip 33% (3g)"


def test_result_filename_matches_snapshot_naming():
    # battery_watch.sh pairs snapshot-<STEP>.pt with step-<STEP>.json by
    # string surgery; the zero-padding must agree with SnapshotPool.save.
    assert battery.result_filename(1250) == "step-0001250.json"
    assert battery.step_from_snapshot_path(
        "/runs/rl-pool-v3/snapshots/snapshot-0001250.pt"
    ) == 1250
