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

from slippi_db.parse_peppi import RANDALL_HLR, RANDALL_INTERVAL  # noqa: E402
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
    game = sim_env.obs_to_game(obs, items)
    assert int(game.stage[0]) == internal
    flat = sim_env.FlatFrames("cpu")
    assert int(flat.view(flat.to_device(sim_env.encode_flats(obs, items))).stage[0]) == internal
    if stage is not msl.Stage.YOSHIS_STORY:   # the parsers give Randall only on Yoshi's Story
        assert game.randall.x[0] == game.randall.y[0] == 0


def _laser_match(frames=600):
    """Fox and Falco on Final Destination mashing lasers: items spawn and vanish
    constantly, and staled hits leave fractional percent."""
    env = _env(msl.Stage.FINAL_DESTINATION, msl.Character.FOX, msl.Character.FALCO)
    slots, new_game = sim_env.ItemSlots(1), np.ones(1, bool)
    for f in range(frames):
        if env.t >= env.length:
            env.reset_cursor()
        obs = env.current_frame
        yield obs, slots.place(obs["items"], new_game)
        pressed = np.array([[.5, .5, .5, .5, 0] + [0] * 8], np.float32)
        pressed[:, 6] = float(f % 3 == 0)   # B: lasers
        sim_env.write_controller_rows(env, pressed, player=0)
        sim_env.write_controller_rows(env, pressed, player=1)
        is_resetting, _ = env.step_and_reset()
        new_game = np.asarray(is_resetting, bool)
    env.close()


def test_percent_is_whole_as_in_replays():
    fractional = 0
    for obs, items in _laser_match():
        game = sim_env.obs_to_game(obs, items)
        for player, slot in ((game.p0, 0), (game.p1, 1)):
            percent = obs["slots"][0, slot]["percent"]
            fractional += percent % 1 != 0
            assert player.percent[0] == np.floor(percent)
    assert fractional, "no staled hit left a fractional percent: the fixture tests nothing"


def test_randall_is_the_replay_parsers_cycle():
    """The sim reports the cloud's collision surface, a fraction of a unit
    off libmelee's position for the frame; the replays and Dolphin give
    libmelee's, so that is what we encode."""
    env = _env(msl.Stage.YOSHIS_STORY)
    differs = False
    for _ in range(400):
        if env.t >= env.length:
            env.reset_cursor()
        obs = env.current_frame
        frame = int(obs["frame_id"][0])
        game = sim_env.obs_to_game(obs, _first_items(obs))
        height, left, right = RANDALL_HLR[(frame + RANDALL_INTERVAL) % RANDALL_INTERVAL]
        assert (game.randall.x[0], game.randall.y[0]) == (np.float32((left + right) / 2), np.float32(height))
        dolphin_y, dolphin_left, dolphin_right = melee.randall_position(frame)
        assert game.randall.y[0] == np.float32(dolphin_y)
        assert game.randall.x[0] == np.float32((dolphin_left + dolphin_right) / 2)
        differs |= abs(float(obs["stage"]["randall"]["x"][0]) - (left + right) / 2) > 0.01
        env.step_and_reset()
    env.close()
    assert differs, "the sim's own Randall matched: the fixture tests nothing"


def test_items_keep_their_slot_for_life_as_in_replays():
    """The sim lists live items packed, so a laser moves slots whenever an older
    one vanishes; placed as the parsers place them, each keeps one slot for its
    whole life."""
    listed_at, placed_at = collections.defaultdict(set), collections.defaultdict(set)
    for obs, items in _laser_match():
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
    assert any(len(s) > 1 for s in listed_at.values()), "fixture never shifted an item: it tests nothing"
    assert all(len(s) == 1 for s in placed_at.values())


NANA = 11   # Nana's own character id, as the replays carry her (Popo is 10)


def test_ice_climbers_follower_is_observed_as_nana():
    """The sim observes Popo's follower beside him, as the replays carry her in
    Player.nana; a player with no follower keeps an all-zero nana."""
    env = _env(msl.Stage.FINAL_DESTINATION, p0=msl.Character.ICE_CLIMBERS, p1=msl.Character.FOX)
    for _ in range(120):
        env.step_and_reset()
    obs = env.current_frame.copy()
    env.close()
    popo, nana = obs["slots"][0, 0], obs["followers"][0, 0]
    assert (nana["present"], nana["char_id"]) == (1, NANA)
    assert abs(float(nana["pos_x"]) - float(popo["pos_x"])) < 20
    assert not obs["followers"][0, 1].tobytes().strip(b"\0")
    game = sim_env.obs_to_game(obs, _first_items(obs))
    assert bool(game.p0.nana.exists[0]) and game.p0.nana.character[0] == NANA
    assert game.p0.nana.x[0] == nana["pos_x"]
    assert not any(np.any(leaf) for leaf in game.p1.nana)


ICS_DITTO = paths.MELEE_SIM_DIR / "replays/validation/icies/ics-ditto-d18-2025-05_Game_20250518T163949.slpz"


def _lanes(post, present: np.ndarray) -> np.ndarray:
    """A fighter's Slippi post-frame rows as the sim's observation carries them:
    upstream's replay validator's expected lanes (tools/validation/native.c),
    which the sim matches bit-exact, zeroed where the fighter has no row."""
    import pyarrow.compute as pc
    from melee_sim import dtypes

    get = lambda a: pc.fill_null(a, 0).to_numpy(zero_copy_only=False)
    lanes = np.zeros(len(present), dtypes.gamestate_player_dtype())
    lanes["present"] = present
    lanes["char_id"] = get(post.character)
    lanes["pos_x"], lanes["pos_y"] = get(post.position.x), get(post.position.y)
    lanes["facing"] = get(post.direction) > 0
    lanes["on_ground"] = get(post.airborne) == 0
    lanes["action_id"] = get(post.state)
    lanes["jumps_left"] = get(post.jumps)
    lanes["percent"] = get(post.percent)
    lanes["shield_hp"] = get(post.shield)
    lanes["hurtbox_state"] = get(post.hurtbox_state)
    lanes["invulnerable"] = lanes["hurtbox_state"] != 0
    lanes[~present] = np.zeros(1, lanes.dtype)
    return lanes


def test_nana_encodes_as_the_replay_parser_does():
    """An Ice Climbers ditto, every frame: the player and Nana fields the sim
    observes for this game (its replay rows as the sim's lanes) embed exactly
    as parse_peppi's Game embeds for BC, Nana's absent frames included."""
    if not ICS_DITTO.is_file() or ICS_DITTO.stat().st_size < 1024:
        pytest.skip(f"{ICS_DITTO.name} not fetched (git lfs pull --include='replays/validation/icies/**')")
    import tree
    from slippi_ai import types
    from slippi_db import parse_peppi
    from smashbot import embed as embed_lib
    from tools.validation.slpz import replay_path_for_peppi

    with replay_path_for_peppi(ICS_DITTO) as slp:
        peppi = parse_peppi.read_slippi(str(slp))
    replay = types.array_to_nt(types.Game, parse_peppi.from_peppi(peppi))
    embed = embed_lib.make_player_embedding()
    absent = 0
    for port, player in zip(peppi.frames.ports, (replay.p0, replay.p1)):
        present = ~np.isnan(port.follower.post.position.x.to_numpy(zero_copy_only=False))
        sim = sim_env._player(_lanes(port.leader.post, np.ones(len(present), bool)),
                              _lanes(port.follower.post, present))
        for got, want in zip(tree.flatten(embed.from_state(sim)), tree.flatten(embed.from_state(player))):
            np.testing.assert_array_equal(got, want)
        absent += int((~present).sum())
    assert absent, "Nana never left: the absent frames are untested"


def test_nana_reward_is_the_value_targets_nana_term():
    """An Ice Climbers ditto, every frame: Nana's part of the RL reward, from
    what the sim observes (her replay rows as its lanes), equals her part of
    the value targets BC trains on (slippi_ai compute_rewards, nana_ratio).
    Percent is taken whole as the replays store it; the leader's units are
    the reward's own concern, not Nana's."""
    if not ICS_DITTO.is_file() or ICS_DITTO.stat().st_size < 1024:
        pytest.skip(f"{ICS_DITTO.name} not fetched (git lfs pull --include='replays/validation/icies/**')")
    import torch
    from slippi_ai import reward, types
    from slippi_db import parse_peppi
    from smashbot.rl.rollouts import Followers, compute_reward
    from tools.validation.slpz import replay_path_for_peppi

    with replay_path_for_peppi(ICS_DITTO) as slp:
        peppi = parse_peppi.read_slippi(str(slp))
    replay = types.array_to_nt(types.Game, parse_peppi.from_peppi(peppi))
    followers = []
    for port in peppi.frames.ports:
        present = ~np.isnan(port.follower.post.position.x.to_numpy(zero_copy_only=False))
        lanes = _lanes(port.follower.post, present)
        lanes["percent"] = np.floor(lanes["percent"])
        followers.append(lanes)
    obs = {"followers": np.stack(followers, axis=1)}          # [F, 2], frames as envs
    f = Followers(*map(torch.as_tensor, sim_env.follower_stats(obs)))
    prev, cur = (Followers(*(x[:-1] for x in f)), Followers(*(x[1:] for x in f)))
    still = torch.zeros(len(f.present) - 1, 2)
    no_reset = torch.zeros(len(still), dtype=torch.bool)
    got = compute_reward(still, still, still, still, no_reset,
                         prev_followers=prev, followers=cur).numpy()
    want = reward.compute_rewards(replay) - reward.compute_rewards(replay, nana_ratio=0)
    np.testing.assert_allclose(got, want, atol=1e-6)
    deaths = (cur.dying & ~prev.dying & cur.present).sum().item()
    assert deaths and np.count_nonzero(want) > deaths, "no Nana deaths or damage to compare"
