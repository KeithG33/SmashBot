"""The simulator's frames must encode as the replays BC trains on
(slippi_db.parse_peppi) and as Dolphin (libmelee) would: same ids, same units,
on both the structured path and the flat path the captured student reads."""

import collections
import sys

import melee
import numpy as np
import pytest

from smashbot import paths

if paths.MELEE_SIM_DIR.is_dir():
    sys.path.insert(0, str(paths.MELEE_SIM_DIR))
msl = pytest.importorskip("melee_sim")
if not paths.MSL_DATA_DIR.is_dir():
    pytest.skip(f"no melee-sim-light data at {paths.MSL_DATA_DIR}", allow_module_level=True)

from smashbot.rl import sim_env  # noqa: E402

# what libmelee reports in Dolphin for each sim stage (only Dream Land is spelled differently)
LIBMELEE_STAGE = {
    msl.Stage.FOUNTAIN_OF_DREAMS: melee.Stage.FOUNTAIN_OF_DREAMS,
    msl.Stage.POKEMON_STADIUM: melee.Stage.POKEMON_STADIUM,
    msl.Stage.YOSHIS_STORY: melee.Stage.YOSHIS_STORY,
    msl.Stage.DREAM_LAND_N64: melee.Stage.DREAMLAND,
    msl.Stage.BATTLEFIELD: melee.Stage.BATTLEFIELD,
    msl.Stage.FINAL_DESTINATION: melee.Stage.FINAL_DESTINATION,
}


def _env(stage, p0=msl.Character.FOX, p1=msl.Character.MARTH):
    env = msl.EnvBatch(batch_size=1, length=64, data_dir=str(paths.MSL_DATA_DIR))
    env.configure_match(stage=stage, players=[msl.PlayerConfig(p0), msl.PlayerConfig(p1)])
    env.reset_all()
    return env


def _first_frame(stage):
    env = _env(stage)
    obs = env.current_frame.copy()
    env.close()
    return obs


def _first_items(obs):
    return sim_env.ItemSlots(len(obs)).place(obs["items"], np.ones(len(obs), bool))


def test_every_sim_stage_has_a_libmelee_twin():
    assert set(LIBMELEE_STAGE) == set(msl.Stage)


@pytest.mark.parametrize("stage", list(msl.Stage), ids=lambda s: s.name)
def test_stage_encodes_as_the_replays_and_dolphin_do(stage):
    obs = _first_frame(stage)
    internal = LIBMELEE_STAGE[stage].value
    # the replay parser's conversion of the id the sim reports lands on the same stage
    assert melee.enums.to_internal_stage(int(obs["stage_id"][0])).value == internal
    items = _first_items(obs)
    assert int(sim_env.obs_to_game(obs, items).stage[0]) == internal
    flat = sim_env.FlatFrames("cpu")
    assert int(flat.view(flat.to_device(sim_env.encode_flats(obs, items))).stage[0]) == internal


def test_items_keep_their_slot_for_life_as_in_replays():
    """Fox and Falco mashing lasers: the sim lists live items packed, so a laser
    moves slots whenever an older one vanishes; placed as the parsers place
    them, each keeps one slot for its whole life."""
    env = _env(msl.Stage.FINAL_DESTINATION, msl.Character.FOX, msl.Character.FALCO)
    slots, new_game = sim_env.ItemSlots(1), np.ones(1, bool)
    listed_at, placed_at = collections.defaultdict(set), collections.defaultdict(set)
    for f in range(600):
        if env.t >= env.length:
            env.reset_cursor()
        obs = env.current_frame
        items = slots.place(obs["items"], new_game)
        listed = obs["items"][0][obs["items"][0]["exists"].astype(bool)]
        for k, spawn_id in enumerate(listed["spawn_id"].tolist()):
            listed_at[spawn_id].add(k)
        live = items[0]["exists"].astype(bool)
        for slot in np.flatnonzero(live):
            placed_at[int(items[0, slot]["spawn_id"])].add(int(slot))
        assert sorted(items[0][live]["spawn_id"]) == sorted(listed["spawn_id"])
        game = sim_env.obs_to_game(obs, items)
        for slot in range(15):
            item = getattr(game.items, f"item_{slot}")
            assert bool(item.exists[0]) == live[slot]
            assert item.x[0] == items[0, slot]["pos_x"]
        pressed = np.array([[.5, .5, .5, .5, 0] + [0] * 8], np.float32)
        pressed[:, 6] = float(f % 3 == 0)   # B: lasers
        sim_env.write_controller_rows(env, pressed, player=0)
        sim_env.write_controller_rows(env, pressed, player=1)
        is_resetting, _ = env.step_and_reset()
        new_game = np.asarray(is_resetting, bool)
    env.close()
    assert any(len(s) > 1 for s in listed_at.values()), "fixture never shifted an item: it tests nothing"
    assert all(len(s) == 1 for s in placed_at.values())
