"""MatchSet plays a fixed slate: every game once and to its end, each
counted once, and a game's kills and deaths on that game, its last stock
included."""
import sys

import pytest
import torch

from smashbot import paths

if paths.MELEE_SIM_DIR.is_dir():
    sys.path.insert(0, str(paths.MELEE_SIM_DIR))
msl = pytest.importorskip("melee_sim")
if not paths.MSL_DATA_DIR.is_dir():
    pytest.skip(f"no melee-sim-light data at {paths.MSL_DATA_DIR}", allow_module_level=True)

from smashbot.eval import sim_arena  # noqa: E402
from smashbot.tests.test_ppo import _tiny_policy  # noqa: E402


def test_every_game_of_the_slate_is_played_to_its_end_and_counted_once(monkeypatch):
    one_stock = msl.MatchConfig
    monkeypatch.setattr(msl, "MatchConfig", lambda **kw: one_stock(**{**kw, "stocks": 1}))
    monkeypatch.setattr(sim_arena, "MAX_GAME_FRAMES", 900)   # 15 s: some games end on the timer
    torch.manual_seed(1)
    ms = sim_arena.MatchSet(_tiny_policy(0), _tiny_policy(1), sim_arena.stratified(4, seed=3),
                            str(paths.MSL_DATA_DIR), envs=2)   # the last two games go to whichever env frees first
    ms.run()
    ms.close()
    assert None not in ms.results
    assert any(0 in r for r in ms.results) and (1, 1) in ms.results   # both ways a game ends
    for (s0, s1), deaths, kills in zip(ms.results, ms.death_percents, ms.kill_percents):
        assert (len(deaths), len(kills)) == (1 - s0, 1 - s1)   # a game's last stock is its own
    st = ms.stats()
    assert (st["games"], st["wins"] + st["losses"] + st["draws"]) == (4, 4)
    assert st["win_rate"] == st["wins"] / 4
