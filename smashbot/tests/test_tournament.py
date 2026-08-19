"""Unit tests for the ghost tournament's pure parts (scripts/tournament.py):
contestant parsing (NAME=PATH forms, phillip inclusion), pair enumeration and
wave-scheduling math (all pairs covered, env budget respected), env-spec
construction, standings/ranking from synthetic results, JSON round-trip, and
the --dry-run path against tiny tmp checkpoints. No Dolphin anywhere."""

import dataclasses
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import torch

from smashbot import configs, embed as embed_lib, saving
from smashbot.policy import build_policy
from smashbot.rl.pool import MAIN_12
from smashbot.rl.rollouts import GameTracker

_TOURN_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tournament.py"
)


def _load_tournament():
    spec = importlib.util.spec_from_file_location("tournament", _TOURN_PATH)
    mod = importlib.util.module_from_spec(spec)
    # dataclass processing resolves string annotations via
    # sys.modules[cls.__module__], so the module must be registered first
    sys.modules["tournament"] = mod
    spec.loader.exec_module(mod)
    return mod


tourn = _load_tournament()


# ------------------------------------------------------- contestant parsing


def test_parse_contestant_forms():
    c = tourn.parse_contestant("s1250=/runs/snapshots/snapshot-0001250.pt")
    assert c.name == "s1250"
    assert c.path == "/runs/snapshots/snapshot-0001250.pt"
    assert c.kind == "snapshot"

    c = tourn.parse_contestant("final=/runs/latest.pt:full")
    assert (c.name, c.path, c.kind) == ("final", "/runs/latest.pt", "full")


@pytest.mark.parametrize(
    "bad",
    ["nopath", "=path.pt", "name=", "na me=x.pt", "a=:full"],
)
def test_parse_contestant_rejects(bad):
    with pytest.raises(ValueError):
        tourn.parse_contestant(bad)


def test_resolve_contestants_includes_phillip_by_default():
    cs = tourn.resolve_contestants(
        ["a=x.pt", "b=y.pt:full"], phillip=True, phillip_ckpt="/models/p.pt"
    )
    assert [c.name for c in cs] == ["a", "b", tourn.PHILLIP_NAME]
    assert cs[-1].kind == "full"
    assert cs[-1].path == "/models/p.pt"
    assert cs[1].kind == "full"

    cs = tourn.resolve_contestants(
        ["a=x.pt", "b=y.pt"], phillip=False, phillip_ckpt="/models/p.pt"
    )
    assert [c.name for c in cs] == ["a", "b"]


def test_resolve_contestants_rejects_collisions_and_singletons():
    with pytest.raises(ValueError, match="duplicate"):
        tourn.resolve_contestants(
            ["a=x.pt", "a=y.pt"], phillip=False, phillip_ckpt=""
        )
    # "phillip" is reserved while --phillip is on
    with pytest.raises(ValueError, match="duplicate"):
        tourn.resolve_contestants(
            ["phillip=x.pt", "b=y.pt"], phillip=True, phillip_ckpt="p.pt"
        )
    with pytest.raises(ValueError, match="at least 2"):
        tourn.resolve_contestants(["a=x.pt"], phillip=False, phillip_ckpt="")


# ------------------------------------------------ pairs and wave scheduling


def test_enumerate_pairs_all_unordered():
    names = ["a", "b", "c", "d"]
    pairs = tourn.enumerate_pairs(names)
    assert len(pairs) == 6  # 4 choose 2
    assert len(set(map(frozenset, pairs))) == 6  # all distinct, unordered
    assert all(a != b for a, b in pairs)
    assert pairs[0] == ("a", "b")  # deterministic input order


def test_envs_per_pair_even_and_bounded():
    assert tourn.envs_per_pair(8, 64) == 8
    assert tourn.envs_per_pair(7, 64) == 6  # rounded down to even
    assert tourn.envs_per_pair(1, 64) == 2  # floor of 2
    assert tourn.envs_per_pair(100, 16) == 16  # capped at the wave budget
    assert tourn.envs_per_pair(100, 15) == 14  # capped AND even
    with pytest.raises(ValueError):
        tourn.envs_per_pair(8, 1)


def test_plan_waves_covers_all_pairs_within_budget():
    # THE post-v3 shape: 8 contestants -> 28 pairs, 8 games/pair, 64 envs.
    names = [f"c{i}" for i in range(8)]
    pairs = tourn.enumerate_pairs(names)
    assert len(pairs) == 28
    waves = tourn.plan_waves(pairs, games_per_pair=8, envs=64)
    # 8 envs/pair -> 8 pairs/wave -> ceil(28/8) = 4 waves
    assert len(waves) == 4
    assert [len(w) for w in waves] == [8, 8, 8, 4]
    for wave in waves:
        assert sum(p.num_envs for p in wave) <= 64  # dolphin budget
        assert all(p.num_envs == 8 for p in wave)
    # every pair exactly once
    scheduled = [(p.a, p.b) for wave in waves for p in wave]
    assert scheduled == pairs


def test_plan_waves_budget_respected_odd_shapes():
    names = [f"c{i}" for i in range(5)]
    pairs = tourn.enumerate_pairs(names)  # 10 pairs
    waves = tourn.plan_waves(pairs, games_per_pair=6, envs=10)
    scheduled = [(p.a, p.b) for wave in waves for p in wave]
    assert scheduled == pairs
    for wave in waves:
        assert sum(p.num_envs for p in wave) <= 10


def test_build_pair_specs_fox_mode():
    specs = tourn.build_pair_specs(8, "fox")
    assert len(specs) == 8
    assert all(s.kind == "teacher" and s.group == -1 for s in specs)
    assert all(s.opponent_char == "FOX" for s in specs)
    # seat balance: half the envs on each port
    ports = [s.student_port for s in specs]
    assert ports.count(1) == ports.count(2) == 4
    # deterministic
    assert specs == tourn.build_pair_specs(8, "fox")


def test_build_pair_specs_main12_rotation():
    specs = tourn.build_pair_specs(12, "main12")
    assert [s.opponent_char for s in specs] == MAIN_12


def test_build_pair_specs_rejects_bad_input():
    with pytest.raises(ValueError):
        tourn.build_pair_specs(7, "fox")  # odd
    with pytest.raises(ValueError):
        tourn.build_pair_specs(8, "sheik-only")


def test_resolve_stage_aliases():
    assert tourn.resolve_stage("FD") == "FINAL_DESTINATION"
    assert tourn.resolve_stage("fd") == "FINAL_DESTINATION"
    assert tourn.resolve_stage("BATTLEFIELD") == "BATTLEFIELD"


# -------------------------------------------------------- standings/ranking


def _pair(a, b, wins_a, wins_b, draws=0, diffs=None):
    games = wins_a + wins_b + draws
    if diffs is None:
        diffs = [2] * wins_a + [-2] * wins_b + [0] * draws
    return {
        "a": a, "b": b, "envs": 8, "games": games,
        "wins_a": wins_a, "wins_b": wins_b, "draws": draws,
        "stock_diffs_a": diffs, "elapsed_seconds": 1.0, "error": None,
    }


def test_standings_ranking_and_tiebreak():
    names = ["s1", "s2", "phillip"]
    # s1 and s2 tie 4-4 head to head; s2 does better against phillip.
    results = [
        _pair("s1", "s2", 4, 4),
        _pair("s1", "phillip", 4, 4),
        _pair("s2", "phillip", 6, 2),
    ]
    out = tourn.compute_standings(names, results)
    by = {st["name"]: st for st in out["standings"]}
    assert by["s1"]["win_rate"] == pytest.approx(0.5)
    assert by["s2"]["win_rate"] == pytest.approx(10 / 16)
    assert by["s2"]["vs_phillip"] == pytest.approx(0.75)
    assert by["s1"]["vs_phillip"] == pytest.approx(0.5)
    assert by["phillip"]["vs_phillip"] is None
    # s2 leads on win rate; phillip (6 wins / 16) last
    assert out["ranking"] == ["s2", "s1", "phillip"]
    # head-to-head is symmetric and from each row's perspective
    assert out["head_to_head"]["s2"]["phillip"]["wins"] == 6
    assert out["head_to_head"]["phillip"]["s2"]["wins"] == 2


def test_standings_vs_phillip_breaks_win_rate_tie():
    names = ["s1", "s2", "phillip"]
    results = [
        _pair("s1", "s2", 5, 3),
        _pair("s1", "phillip", 3, 5),
        _pair("s2", "phillip", 5, 3),
    ]
    out = tourn.compute_standings(names, results)
    by = {st["name"]: st for st in out["standings"]}
    # both 8-8 overall, but s2's vs-phillip 5/8 beats s1's 3/8
    assert by["s1"]["win_rate"] == by["s2"]["win_rate"]
    assert out["ranking"][0] == "s2"


def test_standings_zero_games_reported_honestly():
    names = ["s1", "s2"]
    plan = tourn.PairPlan(a="s1", b="s2", num_envs=8)
    results = [tourn.empty_pair_result(plan, "env 3 died")]
    out = tourn.compute_standings(names, results)
    for st in out["standings"]:
        assert st["games"] == 0
        assert st["win_rate"] is None
        assert st["avg_stock_diff"] is None
    assert out["ranking"] == ["s1", "s2"]  # input order on total ties


def test_pair_result_from_tracker_perspective():
    plan = tourn.PairPlan(a="A", b="B", num_envs=4)
    t = GameTracker()
    t.add_game((3, 0))  # A wins by 3
    t.add_game((0, 2))  # B wins by 2
    t.add_game((1, 1))  # draw
    r = tourn.pair_result_from_tracker(plan, t, 12.34, None)
    assert (r["wins_a"], r["wins_b"], r["draws"], r["games"]) == (1, 1, 1, 3)
    assert r["stock_diffs_a"] == [3, -2, 0]
    assert r["error"] is None


# -------------------------------------------------------- report round-trip


def test_report_json_roundtrip(tmp_path):
    contestants = [
        tourn.Contestant("s1", "/runs/s1.pt", "snapshot"),
        tourn.Contestant("s2", "/runs/s2.pt", "snapshot"),
        tourn.Contestant("phillip", "/models/p.pt", "full"),
    ]
    results = [
        _pair("s1", "s2", 5, 3),
        _pair("s1", "phillip", 4, 4),
        tourn.empty_pair_result(
            tourn.PairPlan("s2", "phillip", 8), "budget reached"
        ),
    ]
    report = tourn.make_report(
        contestants, results, "2026-08-19T12:00:00", {"envs": 64}
    )
    path = tmp_path / "report.json"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    with open(path) as fh:
        assert json.load(fh) == report
    # and the table renders every contestant + the honest-zero note
    table = tourn.format_table(report)
    for name in ("s1", "s2", "phillip"):
        assert name in table
    assert "NO finished games" in table


# ------------------------------------------------------------ CLI defaults


def test_parse_args_defaults():
    args = tourn.parse_args(["--contestants", "a=x.pt", "b=y.pt"])
    assert args.games_per_pair == 8
    assert args.envs == 64
    assert args.device == "cpu"
    assert args.char_mode == "fox"
    assert args.stage == "FD"
    assert args.phillip is True
    assert args.dry_run is False
    args = tourn.parse_args(
        ["--contestants", "a=x.pt", "b=y.pt", "--no-phillip", "--dry-run",
         "--device", "cuda", "--char-mode", "main12"]
    )
    assert args.phillip is False
    assert args.dry_run is True
    assert args.device == "cuda"
    assert args.char_mode == "main12"


# ------------------------------------------------------ --dry-run end to end


_NET = dict(name="sgu", num_layers=1, hidden_size=64, num_heads=1, window=4)
_HEAD = dict(residual_size=32, component_depth=0)


def _tiny_policy(seed=0):
    torch.manual_seed(seed)
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(**_NET),
        head_config=configs.ControllerHeadConfig(**_HEAD),
        policy_config=configs.PolicyConfig(delay=2),
        num_names=4,
    )


def _write_full_ckpt(path, seed=0):
    """A checkpoint shaped exactly as eval.game.load_policy expects."""
    policy = _tiny_policy(seed)
    cfg = {
        "network": dataclasses.asdict(configs.NetworkConfig(**_NET)),
        "head": dataclasses.asdict(configs.ControllerHeadConfig(**_HEAD)),
        "policy": dataclasses.asdict(configs.PolicyConfig(delay=2)),
        "data": {"max_names": 4},
    }
    torch.save(
        {
            "config": cfg,
            "state": {
                "policy": policy.state_dict(),
                "name_map": {"Master Player": 1},
                "step": 7,
            },
            "best_eval_loss": 0.0,
            "version": saving.VERSION,
        },
        path,
    )


def test_dry_run_resolves_and_stops_before_dolphin(tmp_path):
    full = tmp_path / "latest.pt"
    _write_full_ckpt(full, seed=0)
    for i, seed in enumerate((1, 2)):
        torch.save(
            _tiny_policy(seed).state_dict(),
            tmp_path / f"snapshot-000{i}.pt",
        )
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable, str(_TOURN_PATH),
            "--contestants",
            f"s0={tmp_path / 'snapshot-0000.pt'}",
            f"s1={tmp_path / 'snapshot-0001.pt'}",
            f"final={full}:full",
            "--config-from", str(full),
            "--phillip-ckpt", str(full),  # tiny stand-in; loads via the same path
            "--games-per-pair", "2", "--envs", "4",
            "--dry-run", "--out", str(out),
        ],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "DRY RUN" in proc.stdout
    # 4 contestants (3 + phillip) -> 6 pairs, all resolved and scheduled
    assert "6 pairs" in proc.stdout
    for name in ("s0", "s1", "final", "phillip"):
        assert f"loaded {name}:" in proc.stdout
    assert "wave 1/" in proc.stdout
    # stopped before booting anything: no report written on a dry run
    assert not out.exists()
