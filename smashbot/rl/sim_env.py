"""melee-sim-light backend: batched, deterministic Melee for RL rollouts.

Replaces the Dolphin fleet (env_process.py + DolphinRolloutWorker) with a
single-process batched sim. One EnvBatch steps N games per frame far faster
than the GPU can serve inference, so rollout collection becomes a synchronous
vectorized loop and the GPU is the bottleneck.

This module is the seam between the sim and our stack:
  * obs_to_game / encode_obs : MslObservation (numpy struct) -> our encoded
    Game struct, the exact input BatchedPolicyAgent.step expects.
  * write_controllers        : decoded controller struct -> MslInput.
Both use GALE01 ids directly (sim and slippi-ai share them), so it is field
renaming, not value translation.

char/action/stage ids are GALE01 in both the sim and slippi_ai.types, so no
mapping tables are needed.
"""
from __future__ import annotations

import numpy as np

from slippi_ai import types
from smashbot import embed as embed_lib

# Built once; pure structure (no torch state), safe to share.
_EMBED = embed_lib.EmbedConfig().make_game_embedding()
_DUMMY = _EMBED.dummy()
_RANDALL_T = type(_DUMMY.randall)
_FOD_T = type(_DUMMY.fod_platforms)
_ITEMS_T = type(_DUMMY.items)
_ITEM_T = type(_DUMMY.items.item_0)
_N_ITEMS = 15


def _empty_nana(n: int) -> types.Nana:
    z = np.zeros(n, np.float32)
    zi = np.zeros(n, np.int64)
    zb = np.zeros(n, np.bool_)
    return types.Nana(
        exists=zb, percent=z, facing=z, x=z, y=z, action=zi, invulnerable=zb,
        character=zi, jumps_left=zi, shield_strength=z, on_ground=zb,
    )


def _player(slot: np.ndarray) -> types.Player:
    """One player slot of an MslObservation batch -> slippi_ai Player.

    slot: structured array [N] for one viewpoint-relative slot. nana is left
    empty (the sim's per-player slots don't expose Ice Climbers' follower;
    Popo plays with nana.exists=0 -- a minor obs gap, tracked as follow-up).
    """
    return types.Player(
        percent=slot["percent"].astype(np.float32),
        facing=slot["facing"].astype(np.bool_),        # 0 left / 1 right -> bool right
        x=slot["pos_x"].astype(np.float32),
        y=slot["pos_y"].astype(np.float32),
        action=slot["action_id"].astype(np.int64),     # GALE01 action id
        invulnerable=slot["invulnerable"].astype(np.bool_),
        character=slot["char_id"].astype(np.int64),     # GALE01 character id
        jumps_left=slot["jumps_left"].astype(np.int64),
        shield_strength=slot["shield_hp"].astype(np.float32),
        on_ground=slot["on_ground"].astype(np.bool_),
        controller=None,                                # not encoded in the game state
        nana=_empty_nana(len(slot)),
    )


def _items(items: np.ndarray) -> object:
    """items: structured array [N, 15] -> our Items struct."""
    return _ITEMS_T(**{
        f"item_{k}": _ITEM_T(
            exists=items[:, k]["exists"].astype(np.bool_),
            type=items[:, k]["type"].astype(np.int64),
            state=items[:, k]["state"].astype(np.int64),
            x=items[:, k]["pos_x"].astype(np.float32),
            y=items[:, k]["pos_y"].astype(np.float32),
        )
        for k in range(_N_ITEMS)
    })


def obs_to_game(obs: np.ndarray, self_slot: int = 0, opp_slot: int = 1) -> types.Game:
    """MslObservation batch [N] -> our batched Game struct.

    self_slot / opp_slot pick which viewpoint-relative slots become p0 (us)
    and p1 (them). Swapping them yields the opponent's perspective for free
    (positions are world coords; only the slot roles change), which is how we
    build the player-1 input from the same frame.
    """
    slots = obs["slots"]  # [N, 4] structured
    stage = obs["stage"]
    return types.Game(
        p0=_player(slots[:, self_slot]),
        p1=_player(slots[:, opp_slot]),
        stage=obs["stage_id"].astype(np.int64),         # GALE01 stage id
        randall=_RANDALL_T(
            x=stage["randall"]["x"].astype(np.float32),
            y=stage["randall"]["y"].astype(np.float32),
        ),
        fod_platforms=_FOD_T(
            left=stage["fod_platforms"]["left"].astype(np.float32),
            right=stage["fod_platforms"]["right"].astype(np.float32),
        ),
        items=_items(obs["items"]),
    )


def encode_obs(obs: np.ndarray, self_slot: int = 0, opp_slot: int = 1):
    """MslObservation batch -> encoded Game struct (numpy leaves), the input
    BatchedPolicyAgent.step / infer expects (still numpy; the agent moves it
    to torch/device)."""
    return _EMBED.from_state(obs_to_game(obs, self_slot, opp_slot))


# --------------------------------------------------------------- action side

# A/B/X/Y/Z/L/R/D_UP -- our decoded controller and the sim's use the same set.
_BUTTONS = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")


def write_controllers(env, controllers, player: int) -> None:
    """Write BatchedPolicyAgent.execute()'s output -- a list of N decoded
    Controller structs (sticks in [0,1] centered at 0.5, buttons bool) -- into
    the sim's controller buffer for `player` at the current step row. Sticks
    pass through unchanged (both use the [0,1] convention); buttons map by name.
    """
    import melee_sim as msl

    N = env.batch_size
    ctrl = msl.neutral_controller((env.length, N))
    t = env.t
    ctrl.main_stick.x[t] = np.fromiter((c.main_stick.x for c in controllers), np.float32, N)
    ctrl.main_stick.y[t] = np.fromiter((c.main_stick.y for c in controllers), np.float32, N)
    ctrl.c_stick.x[t] = np.fromiter((c.c_stick.x for c in controllers), np.float32, N)
    ctrl.c_stick.y[t] = np.fromiter((c.c_stick.y for c in controllers), np.float32, N)
    ctrl.shoulder[t] = np.fromiter((c.shoulder for c in controllers), np.float32, N)
    b0 = controllers[0].buttons
    for name in _BUTTONS:
        if hasattr(b0, name) and hasattr(ctrl.buttons, name):
            getattr(ctrl.buttons, name)[t] = np.fromiter(
                (bool(getattr(c.buttons, name)) for c in controllers), bool, N)
    msl.write_controller(env.controller_action_view, ctrl, player=player)


# controller_rows column order = tree.flatten(Controller):
#   0 main_x  1 main_y  2 c_x  3 c_y  4 shoulder  5..12 buttons A B X Y Z L R D_UP
def write_controller_rows(env, rows: np.ndarray, player: int) -> None:
    """Vectorized twin of write_controllers for the flat [N, 13] row format
    (encode.controller_rows / BatchedPolicyAgent flat_controllers /
    LeagueAgent.execute). Writes ONLY the current step's row of the action
    ring via env.current_action_frame -- the previous version built a full
    [length, N] neutral controller and whole-buffer-assigned it through
    msl.write_controller, ~240x the numpy traffic, per player, per frame."""
    p = env.current_action_frame["players"][..., int(player)]
    p["main_stick_x"] = rows[:, 0]
    p["main_stick_y"] = rows[:, 1]
    p["c_stick_x"] = rows[:, 2]
    p["c_stick_y"] = rows[:, 3]
    p["shoulder"] = rows[:, 4]
    b = p["buttons"]
    for j, name in enumerate(_BUTTONS):
        b[name] = rows[:, 5 + j] > 0.5
