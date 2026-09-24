"""ItemSlots must place the sim's items exactly as the replay and Dolphin
parsers do: one slippi_db ItemAssigner per env, fresh at each game's first
frame, fed every frame (an empty frame frees slots), unused slots zeroed."""

import numpy as np

from slippi_db.parsing_utils import ItemAssigner
from smashbot.rl.sim_env import ItemSlots

DTYPE = np.dtype([("exists", "u1"), ("spawn_id", "<u4"), ("pos_x", "<f4")])
SLOTS = 15

# per frame: new-game flag per env, then each env's live items (spawn_id, x) in the sim's listing order
FRAMES = [
    ((1, 1, 1, 1), ([(1, .1)], [(10, 1.), (11, 1.1), (12, 1.2)], [], [])),         # simultaneous spawns
    ((0, 0, 0, 0), ([(1, .1), (2, .2)], [(11, 1.1), (12, 1.2)], [], [])),
    ((0, 0, 0, 0), ([(2, .2)], [(12, 1.2)], [], [(40, 4.)])),                      # 2 keeps its slot
    ((0, 1, 0, 0), ([(2, .2), (3, .3)], [(12, 5.)], [], [(40, 4.), (41, 4.1)])),  # env 1's new game reuses id 12
    ((0, 0, 0, 0), ([(3, .3), (2, .2)], [(12, 5.), (13, 5.1)], [], [(41, 4.1)])), # reordered listing
    ((0, 0, 1, 0), ([], [(13, 5.1)], [], [])),                                       # empty frames free slots
    ((0, 0, 0, 1), ([(2, .2)], [], [(7, 7.)], [(40, 9.)])),                          # 2 reappears; env 3 new game
    ((0, 0, 0, 0), ([(2, .2), (5, .5), (6, .6)], [], [(7, 7.)], [(40, 9.), (42, 9.2)])),
]


def _as_sim_lists(frame):
    items = np.zeros((len(frame), SLOTS), DTYPE)
    for e, live in enumerate(frame):
        for k, (spawn_id, x) in enumerate(live):
            items[e, k] = (1, spawn_id, x)
    return items


def _as_parsers_place():
    assigners = [None] * len(FRAMES[0][0])
    for new_game, frame in FRAMES:
        placed = np.zeros((len(frame), SLOTS), DTYPE)
        for e, live in enumerate(frame):
            if new_game[e]:
                assigners[e] = ItemAssigner()
            for slot, (spawn_id, x) in zip(assigners[e].assign([i for i, _ in live]), live):
                placed[e, slot] = (1, spawn_id, x)
        yield placed


def test_items_are_placed_as_the_replay_and_dolphin_parsers_place_them():
    slots = ItemSlots(len(FRAMES[0][0]))
    for t, ((new_game, frame), expected) in enumerate(zip(FRAMES, _as_parsers_place())):
        placed = slots.place(_as_sim_lists(frame), np.array(new_game, bool))
        np.testing.assert_array_equal(placed, expected, err_msg=f"frame {t}")


def test_an_item_keeps_its_slot_when_an_earlier_one_vanishes():
    slots = ItemSlots(1)
    slots.place(_as_sim_lists([[(1, .1), (2, .2)]]), np.ones(1, bool))
    placed = slots.place(_as_sim_lists([[(2, .2)]]), np.zeros(1, bool))   # the sim now lists 2 first
    assert placed[0, 1]["spawn_id"] == 2 and not placed[0, 0]["exists"]
