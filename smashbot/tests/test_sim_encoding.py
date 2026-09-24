"""The simulator's frames must encode as the replays BC trains on
(slippi_db.parse_peppi) and as Dolphin (libmelee) would: same ids, same units,
on both the structured path and the flat path the captured student reads."""

import sys

import melee
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


def _first_frame(stage):
    env = msl.EnvBatch(batch_size=1, length=8, data_dir=str(paths.MSL_DATA_DIR))
    env.configure_match(stage=stage, players=[msl.PlayerConfig(msl.Character.FOX),
                                              msl.PlayerConfig(msl.Character.MARTH)])
    env.reset_all()
    obs = env.current_frame.copy()
    env.close()
    return obs


def test_every_sim_stage_has_a_libmelee_twin():
    assert set(LIBMELEE_STAGE) == set(msl.Stage)


@pytest.mark.parametrize("stage", list(msl.Stage), ids=lambda s: s.name)
def test_stage_encodes_as_the_replays_and_dolphin_do(stage):
    obs = _first_frame(stage)
    internal = LIBMELEE_STAGE[stage].value
    # the replay parser's conversion of the id the sim reports lands on the same stage
    assert melee.enums.to_internal_stage(int(obs["stage_id"][0])).value == internal
    assert int(sim_env.obs_to_game(obs).stage[0]) == internal
    flat = sim_env.FlatFrames("cpu")
    assert int(flat.view(flat.to_device(sim_env.encode_flats(obs))).stage[0]) == internal
