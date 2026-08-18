"""Unit tests for scripts/watch_live.py agent construction: both CLI modes
(full checkpoint and snapshot/--config-from), per-policy delay and name_code,
async wrapper type, and spec/character resolution. No Dolphin, no real
checkpoints -- tiny policies saved to tmp_path."""

import dataclasses
import importlib.util
import pathlib

import pytest
import torch

from smashbot import configs, embed as embed_lib, saving
from smashbot.eval.agent import AsyncDelayedAgent
from smashbot.policy import build_policy
from smashbot.rl.pool import MAIN_12

_WATCH_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "watch_live.py"
)


def _load_watch():
    spec = importlib.util.spec_from_file_location("watch_live", _WATCH_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


watch = _load_watch()

_NET = dict(hidden_size=64, num_layers=2)
_HEAD = dict(residual_size=32, component_depth=1)


def _tiny_policy(delay: int):
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(**_NET),
        head_config=configs.ControllerHeadConfig(**_HEAD),
        policy_config=configs.PolicyConfig(delay=delay),
        num_names=4,
    )


def _write_full_ckpt(path, delay: int, name_map: dict, step=7):
    """A checkpoint shaped exactly as eval.game.load_policy expects."""
    policy = _tiny_policy(delay)
    cfg = {
        "network": dataclasses.asdict(configs.NetworkConfig(**_NET)),
        "head": dataclasses.asdict(configs.ControllerHeadConfig(**_HEAD)),
        "policy": dataclasses.asdict(configs.PolicyConfig(delay=delay)),
        "data": {"max_names": 4},
    }
    torch.save(
        {
            "config": cfg,
            "state": {
                "policy": policy.state_dict(),
                "name_map": name_map,
                "step": step,
            },
            "best_eval_loss": 0.0,
            "version": saving.VERSION,
        },
        path,
    )


def _stop(agents: dict) -> None:
    for agent in agents.values():
        agent._in_q.put(None)
        agent._thread.join(timeout=5)


def test_build_agents_full_ckpts(tmp_path):
    p1, p2 = tmp_path / "p1.pt", tmp_path / "p2.pt"
    _write_full_ckpt(p1, delay=3, name_map={"Master Player": 2, "Foe": 0})
    _write_full_ckpt(p2, delay=5, name_map={"Somebody Else": 1}, step=42)
    specs = {
        1: watch.resolve_spec(str(p1), "", ""),
        2: watch.resolve_spec(str(p2), "", ""),
    }
    agents, infos = watch.build_agents(specs)
    try:
        assert set(agents) == {1, 2}
        # the async wrapper is the whole point: inference off the frame loop
        assert all(type(a) is AsyncDelayedAgent for a in agents.values())
        assert agents[1]._ports == (1, 2)
        assert agents[2]._ports == (2, 1)
        # each agent keeps ITS checkpoint's delay...
        assert agents[1].delay == 3 and agents[2].delay == 5
        # ...and resolves the name in ITS OWN name_map (missing -> 0)
        assert int(agents[1]._name.item()) == 2
        assert int(agents[2]._name.item()) == 0
        assert infos[1].step == 7 and infos[2].step == 42
        assert infos[1].delay == 3 and infos[2].delay == 5
    finally:
        _stop(agents)


def test_build_agents_snapshot_mode(tmp_path):
    config_from = tmp_path / "latest.pt"
    _write_full_ckpt(config_from, delay=4, name_map={"Master Player": 3})
    snap_policy = _tiny_policy(delay=4)
    with torch.no_grad():
        for p in snap_policy.parameters():
            p.fill_(0.25)
    snapshot = tmp_path / "snapshot-0001250.pt"
    torch.save(snap_policy.state_dict(), snapshot)
    p2 = tmp_path / "p2.pt"
    _write_full_ckpt(p2, delay=6, name_map={"Master Player": 1})

    specs = {
        1: watch.resolve_spec("", str(snapshot), str(config_from)),
        2: watch.resolve_spec(str(p2), "", str(config_from)),
    }
    agents, infos = watch.build_agents(specs)
    try:
        assert all(type(a) is AsyncDelayedAgent for a in agents.values())
        # snapshot side: delay + name_map come from --config-from
        assert agents[1].delay == 4
        assert int(agents[1]._name.item()) == 3
        assert infos[1].step is None  # bare snapshots carry no step
        # ...but the WEIGHTS come from the snapshot file
        assert torch.all(next(agents[1].policy.parameters()) == 0.25)
        # full-ckpt side untouched by config-from
        assert agents[2].delay == 6
        assert int(agents[2]._name.item()) == 1
    finally:
        _stop(agents)


def test_resolve_spec_rules():
    # neither flag -> the seat's default applies
    s = watch.resolve_spec("", "", "cfg.pt", default_snapshot="snap.pt")
    assert (s.ckpt, s.snapshot) == ("", "snap.pt")
    s = watch.resolve_spec("", "", "", default_ckpt="full.pt")
    assert (s.ckpt, s.snapshot) == ("full.pt", "")
    # an explicit --pN overrides the seat's default snapshot
    s = watch.resolve_spec("mine.pt", "", "cfg.pt", default_snapshot="snap.pt")
    assert (s.ckpt, s.snapshot) == ("mine.pt", "")
    assert s.label == "mine.pt"
    with pytest.raises(ValueError):
        watch.resolve_spec("a.pt", "b.pt", "cfg.pt")
    with pytest.raises(ValueError):
        watch.resolve_spec("", "snap.pt", "")  # snapshot needs config-from


def test_cli_defaults_resolve():
    args = watch.parse_args([])
    specs = watch.resolve_specs(args)
    assert specs[1].snapshot == watch.DEFAULT_P1_SNAPSHOT
    assert specs[1].config_from == watch.DEFAULT_CONFIG_FROM
    assert specs[1].ckpt == ""
    assert specs[2].ckpt == watch.DEFAULT_P2
    assert specs[2].snapshot == ""
    assert args.games == 1
    assert args.compile and args.mute and args.save_replays
    assert args.p1_char == "FOX" and args.p2_char == "random"
    # explicit --p1 suppresses the default snapshot for that seat
    args = watch.parse_args(["--p1", "/tmp/other.pt"])
    specs = watch.resolve_specs(args)
    assert specs[1].ckpt == "/tmp/other.pt" and specs[1].snapshot == ""


def test_resolve_char():
    assert watch.resolve_char("marth") == "MARTH"
    assert watch.resolve_char("FOX") == "FOX"
    assert watch.resolve_char("random") in MAIN_12
    assert watch.resolve_char("") in MAIN_12
    rng = __import__("random").Random(0)
    assert watch.resolve_char("random", rng) in MAIN_12
    with pytest.raises(ValueError):
        watch.resolve_char("GOKU")
