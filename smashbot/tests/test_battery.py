"""Unit tests for the battery's pure parts (scripts/battery.py): phase-slate
coverage of the main 12, env-spec construction determinism, seat alternation,
game-budget phasing, and the JSON report schema round-trip. No Dolphin, no
checkpoints."""

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


def test_phases_cover_main12_exactly():
    # THE design requirement: the union of the phase slates is exactly the
    # main 12 (per yardstick -- both get every slate), with no overlap.
    assert len(battery.PHASES) == 3
    slates = [slate for _, slate in battery.PHASES]
    assert all(len(s) == 4 for s in slates)
    union = set().union(*map(set, slates))
    assert union == set(MAIN_12)
    assert sum(len(s) for s in slates) == 12  # disjoint
    assert battery.BATTERY_SLATE == [c for s in slates for c in s]


def test_served_chars_cover_main12_at_default_envs():
    # At --envs 8 (4 per yardstick), every phase serves its full 4-char
    # slate to BOTH yardsticks, so one battery covers all 12 per yardstick.
    served = {"teacher": set(), "reference": set()}
    for _, slate in battery.PHASES:
        for spec in battery.build_specs(8, slate):
            served[spec.kind].add(spec.opponent_char)
    assert served["teacher"] == set(MAIN_12)
    assert served["reference"] == set(MAIN_12)


def test_specs_deterministic():
    # No RNG anywhere: identical layout every call, and across module reloads.
    for _, slate in battery.PHASES:
        assert battery.build_specs(8, slate) == battery.build_specs(8, slate)
    slate_a = battery.PHASES[0][1]
    assert battery.build_specs(8, slate_a) == _load_battery().build_specs(8, slate_a)


def test_specs_split_kinds_and_rotation():
    slate = battery.PHASES[1][1]
    specs = battery.build_specs(8, slate)
    assert len(specs) == 8
    kinds = [s.kind for s in specs]
    assert kinds.count("teacher") == 4 and kinds.count("reference") == 4
    for kind in ("teacher", "reference"):
        chars = [s.opponent_char for s in specs if s.kind == kind]
        assert chars == [slate[i % len(slate)] for i in range(4)]
    # smaller phases serve a prefix of the slate; larger ones wrap around
    small = battery.build_specs(6, slate)
    assert [s.opponent_char for s in small if s.kind == "teacher"] == slate[:3]
    big = battery.build_specs(12, slate)
    assert [s.opponent_char for s in big if s.kind == "reference"] == (
        slate + slate[:2]
    )


def test_specs_port_alternation():
    specs = battery.build_specs(8, battery.PHASES[0][1])
    for kind in ("teacher", "reference"):
        ports = [s.student_port for s in specs if s.kind == kind]
        assert ports == [1, 2, 1, 2]


@pytest.mark.parametrize("bad", [0, 3, 7])
def test_specs_reject_odd_or_tiny(bad):
    with pytest.raises(ValueError):
        battery.build_specs(bad, battery.PHASES[0][1])


def test_phase_game_targets():
    assert battery.phase_game_targets(24) == [8, 8, 8]
    assert battery.phase_game_targets(6) == [2, 2, 2]
    assert battery.phase_game_targets(7) == [3, 2, 2]
    assert battery.phase_game_targets(1) == [1, 0, 0]
    for n in range(0, 30):
        assert sum(battery.phase_game_targets(n)) == n


def test_per_char_counts_maps_envs_to_chars():
    slate = battery.PHASES[0][1]
    specs = battery.build_specs(8, slate)
    counts = battery.per_char_counts(specs, [1, 2, 0, 3, 4, 0, 1, 2])
    assert counts["teacher"] == {"FOX": 1, "FALCO": 2, "MARTH": 0, "SHEIK": 3}
    assert counts["phillip"] == {"FOX": 4, "FALCO": 0, "MARTH": 1, "SHEIK": 2}


def test_merge_char_counts_covers_main12():
    per_phase = []
    for _, slate in battery.PHASES:
        specs = battery.build_specs(8, slate)
        per_phase.append(battery.per_char_counts(specs, [2] * 8))
    merged = battery.merge_char_counts(per_phase)
    for side in ("teacher", "phillip"):
        assert set(merged[side]) == set(MAIN_12)
        assert all(g == 2 for g in merged[side].values())
    # duplicate chars accumulate rather than overwrite
    twice = battery.merge_char_counts([per_phase[0], per_phase[0]])
    assert twice["teacher"]["FOX"] == 4


def _tracker(diffs, kills=(), deaths=()):
    t = GameTracker()
    for diff in diffs:
        t.add_game((4, 4 - diff) if diff >= 0 else (4 + diff, 4))
    for k in kills:
        t.add_kill(k)
    for d in deaths:
        t.add_death(d)
    return t


def _fake_phases():
    """3 phases: teacher totals 2W-1L-1D, phillip totals 1W-2L."""
    entries = []
    per_phase = [
        {"teacher": _tracker([2], kills=(80.0,), deaths=(120.0,)),
         "reference": _tracker([-2])},
        {"teacher": _tracker([1, -1]), "reference": _tracker([-1])},
        {"teacher": _tracker([0]), "reference": _tracker([1])},
    ]
    for (name, slate), trackers in zip(battery.PHASES, per_phase):
        specs = battery.build_specs(8, slate)
        entries.append({
            "phase": name,
            "slate": slate,
            "elapsed_seconds": 60.0,
            "trackers": trackers,
            "per_char": battery.per_char_counts(specs, [1] * 8),
        })
    return entries


def test_report_schema_roundtrip_and_aggregation():
    report = battery.make_report(
        student_ckpt="/runs/rl-pool-v3/latest.pt",
        student_rl_step=1250,
        stamp="2026-08-13T12:00:00",
        phases=_fake_phases(),
        config_echo={"envs": 8, "games_per_side": 24, "redraw_chars": False},
    )
    back = json.loads(json.dumps(report))  # JSON-serializable end to end
    assert back["student_rl_step"] == 1250
    assert back["char_slate"] == battery.BATTERY_SLATE
    assert back["config"]["redraw_chars"] is False
    for side in ("teacher", "phillip"):
        assert set(back["results"][side]) == set(battery.YARDSTICK_KEYS)
    # merged totals across phases
    t = back["results"]["teacher"]
    assert (t["games"], t["wins"], t["losses"], t["draws"]) == (4, 2, 1, 1)
    assert t["win_rate"] == pytest.approx(2 / 3)  # decided games only
    assert t["avg_stock_diff"] == pytest.approx((2 + 1 - 1 + 0) / 4)
    assert t["avg_percent_at_kill"] == pytest.approx(80.0)
    assert t["avg_percent_at_death"] == pytest.approx(120.0)
    p = back["results"]["phillip"]
    assert (p["wins"], p["losses"]) == (1, 2)
    assert p["win_rate"] == pytest.approx(1 / 3)
    assert p["avg_stock_diff"] == pytest.approx((-2 - 1 + 1) / 3)
    # per-phase detail preserved
    assert [ph["phase"] for ph in back["phases"]] == ["A", "B", "C"]
    assert [ph["slate"] for ph in back["phases"]] == [
        s for _, s in battery.PHASES
    ]
    assert back["phases"][0]["teacher"]["wins"] == 1
    assert back["phases"][1]["teacher"]["games"] == 2
    assert back["phases"][2]["phillip"]["win_rate"] == pytest.approx(1.0)
    # per-char coverage receipt: merged across phases = full main 12
    for side in ("teacher", "phillip"):
        assert set(back["per_char"][side]) == set(MAIN_12)
        assert all(g == 1 for g in back["per_char"][side].values())
    assert set(back["phases"][0]["per_char"]["teacher"]) == set(
        battery.PHASES[0][1]
    )


def test_report_win_rate_none_when_undecided():
    phases = []
    for name, slate in battery.PHASES:
        phases.append({
            "phase": name, "slate": slate, "elapsed_seconds": 0.0,
            "trackers": {"teacher": GameTracker(), "reference": GameTracker()},
        })
    phases[0]["trackers"]["teacher"].add_game((2, 2))  # draw only
    report = battery.make_report("x", 1, "t", phases, {})
    assert report["results"]["teacher"]["win_rate"] is None
    assert report["results"]["phillip"]["games"] == 0
    json.dumps(report)  # None must still serialize


def test_summary_line():
    report = battery.make_report("x", 1250, "t", _fake_phases(), {})
    line = battery.summary_line(report)
    assert line == "BATTERY step 1250: vs-teacher 67% (4g) | vs-phillip 33% (3g)"


def test_result_filename_matches_snapshot_naming():
    # battery_all.sh pairs snapshot-<STEP>.pt with step-<STEP>.json by
    # string surgery; the zero-padding must agree with SnapshotPool.save.
    assert battery.result_filename(1250) == "step-0001250.json"
    assert battery.step_from_snapshot_path(
        "/runs/rl-pool-v3/snapshots/snapshot-0001250.pt"
    ) == 1250
