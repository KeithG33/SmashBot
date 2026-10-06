"""melee-sim-light backend: batched, deterministic Melee for RL rollouts.

One EnvBatch steps N games per frame far faster than the GPU can serve
inference, so rollout collection is a synchronous vectorized loop and the
GPU is the bottleneck.

This module is the seam between the sim and our stack:
  * obs_to_game / encode_obs : MslObservation (numpy struct) -> our encoded
    Game struct, the exact input BatchedPolicyAgent.step expects.
Character and action ids are the game's internal ids in both the sim and the
replays, so those fields are renamed, not translated. The stage is the
exception: the sim reports Slippi's external stage id, while the replays
(slippi_db.parse_peppi) and Dolphin (libmelee) carry the internal one.
"""
from __future__ import annotations

import melee
import numpy as np

from slippi_ai import reward as reward_lib
from slippi_ai import types
from slippi_db.parsing_utils import ItemAssigner
from smashbot import embed as embed_lib

# Built once; pure structure (no torch state), safe to share.
_EMBED = embed_lib.EmbedConfig().make_game_embedding()
_DUMMY = _EMBED.dummy()
_RANDALL_T = type(_DUMMY.randall)
_FOD_T = type(_DUMMY.fod_platforms)
_ITEMS_T = type(_DUMMY.items)
_ITEM_T = type(_DUMMY.items.item_0)
_N_ITEMS = 15
# external stage id -> internal: the replay parser's own conversion, as a lookup
_INTERNAL_STAGE = np.array(
    [melee.enums.to_internal_stage(i).value for i in range(256)], np.int64)
# Yoshi's Story's cloud as the replays carry it: libmelee's height and edges
# for the frame, on its 1200-frame cycle (slippi_db.parse_peppi)
_RANDALL_HLR = np.array([melee.stages.randall_position(f) for f in range(1200)])


def _fighter(slot: np.ndarray) -> dict:
    """One fighter record of an MslObservation batch (a player slot or its
    follower), [N] structured, in the replays' conventions (parse_peppi)."""
    return dict(
        percent=np.floor(slot["percent"]).astype(np.float32),   # replays store whole percent
        facing=slot["facing"].astype(np.bool_),        # 0 left / 1 right -> bool right
        x=slot["pos_x"].astype(np.float32),
        y=slot["pos_y"].astype(np.float32),
        action=slot["action_id"].astype(np.int64),     # GALE01 action id
        invulnerable=slot["invulnerable"].astype(np.bool_),
        character=slot["char_id"].astype(np.int64),     # GALE01 character id
        jumps_left=slot["jumps_left"].astype(np.int64),
        shield_strength=slot["shield_hp"].astype(np.float32),
        on_ground=slot["on_ground"].astype(np.bool_),
    )


def _player(slot: np.ndarray, follower: np.ndarray) -> types.Player:
    """A player slot and its follower (Ice Climbers' Nana; zeroed while
    absent, as the replays carry her) -> slippi_ai Player."""
    return types.Player(
        **_fighter(slot),
        controller=None,                                # not encoded in the game state
        nana=types.Nana(exists=follower["present"].astype(np.bool_), **_fighter(follower)),
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


class ItemSlots:
    """Keeps each item in one slot for its lifetime, as the replays do
    (slippi_db's ItemAssigner, one per game). The sim lists live items
    packed, so without this an item shifts slots whenever an earlier one
    vanishes."""

    def __init__(self, num_envs: int):
        self._assigners = [ItemAssigner() for _ in range(num_envs)]
        self._listed = np.full((num_envs, _N_ITEMS), -1, np.int64)   # last frame's listing
        self._slot = np.zeros((num_envs, _N_ITEMS), np.int64)        # slot of each listed item

    def place(self, items: np.ndarray, new_game: np.ndarray) -> np.ndarray:
        """items [N, 15] as the sim lists them; new_game [N] marks each game's
        first frame. Returns [N, 15] with every item in its slot, the rest zero."""
        live = items["exists"].astype(bool)
        listed = np.where(live, items["spawn_id"].astype(np.int64), -1)
        for e in np.flatnonzero(new_game):
            self._assigners[e] = ItemAssigner()
        # a listing changes only when an item spawns or vanishes (an emptied
        # listing frees its slots); unchanged, the assigner would return the
        # same slots without changing state
        for e in np.flatnonzero(new_game | (listed != self._listed).any(axis=1)):
            pos = np.flatnonzero(live[e])
            self._slot[e, pos] = self._assigners[e].assign(listed[e, pos].tolist())
        self._listed = listed
        env, pos = np.nonzero(live)
        placed = np.zeros_like(items)
        placed[env, self._slot[env, pos]] = items[env, pos]
        return placed


def obs_to_game(obs: np.ndarray, items: np.ndarray, self_slot: int = 0,
                opp_slot: int = 1) -> types.Game:
    """MslObservation batch [N] -> our batched Game struct; items are the
    frame's items in their slots (ItemSlots.place).

    self_slot / opp_slot pick which viewpoint-relative slots become p0 (us)
    and p1 (them). Swapping them yields the opponent's perspective for free
    (positions are world coords; only the slot roles change), which is how we
    build the player-1 input from the same frame.
    """
    slots, followers = obs["slots"], obs["followers"]  # [N, 4] structured
    stage = obs["stage"]
    stage_id = _INTERNAL_STAGE[obs["stage_id"]]
    on_yoshis = stage_id == melee.Stage.YOSHIS_STORY.value
    height, left, right = _RANDALL_HLR[(obs["frame_id"] + 1200) % 1200].T
    return types.Game(
        p0=_player(slots[:, self_slot], followers[:, self_slot]),
        p1=_player(slots[:, opp_slot], followers[:, opp_slot]),
        stage=stage_id,
        randall=_RANDALL_T(
            x=np.where(on_yoshis, (left + right) / 2, 0).astype(np.float32),
            y=np.where(on_yoshis, height, 0).astype(np.float32),
        ),
        fod_platforms=_FOD_T(
            left=stage["fod_platforms"]["left"].astype(np.float32),
            right=stage["fod_platforms"]["right"].astype(np.float32),
        ),
        items=_items(items),
    )


def encode_obs(obs: np.ndarray, items: np.ndarray, self_slot: int = 0, opp_slot: int = 1):
    """MslObservation batch -> encoded Game struct (numpy leaves), the input
    BatchedPolicyAgent.step / infer expects (still numpy; the agent moves it
    to torch/device)."""
    return _EMBED.from_state(obs_to_game(obs, items, self_slot, opp_slot))


# ---- flat encoding ----
# One encode + three host->GPU copies per frame; perspective swap is a
# column permutation, per-group views are 3 row gathers + struct views
# (encode.unflatten_typed_torch) instead of ~120 per-leaf launches each.
from smashbot import encode as _encode  # noqa: E402

_LAYOUT = _encode.layout_of(_DUMMY)
_SWAP_PERM_NP = _encode.swap_perm(_DUMMY, _LAYOUT)


def encode_flats(obs: np.ndarray, items: np.ndarray) -> tuple:
    """(bools[N,B], ints[N,I], floats[N,F]) numpy flats of the UNSWAPPED
    (player-0 view) encoded frame."""
    return _encode.flatten_typed_batched(
        _EMBED.from_state(obs_to_game(obs, items)), len(obs))


class FlatFrames:
    """Device-side frame views built from the flats: student view, swapped
    (opponent) view, and row-gathered sub-views — all views into three
    tensors."""

    _KINDS = ("bool", "int", "float")

    def __init__(self, device):
        import torch
        self.device = device
        self.perm = {
            k: (None if p is None else torch.from_numpy(p).to(device))
            for k, p in _SWAP_PERM_NP.items()
        }

    def to_device(self, flats_np: tuple) -> tuple:
        import torch
        return tuple(
            torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
            for a in flats_np
        )

    def swap(self, flats: tuple) -> tuple:
        return tuple(
            t if self.perm[k] is None else t.index_select(-1, self.perm[k])
            for k, t in zip(self._KINDS, flats)
        )

    def view(self, flats: tuple, rows=None, lead=None):
        if rows is not None:
            flats = tuple(t.index_select(0, rows) for t in flats)
        if lead is not None:
            flats = tuple(t.view(*lead, t.shape[-1]) for t in flats)
        return _encode.unflatten_typed_torch(_DUMMY, _LAYOUT, *flats)


# --------------------------------------------------------------- action side

# A/B/X/Y/Z/L/R/D_UP -- our decoded controller and the sim's use the same set.
_BUTTONS = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")


class ShardedEnvBatch:
    """EnvBatch's interface over several batches stepped in parallel, one
    thread each: melee-sim-light drives a batch from any one thread at a time,
    and a process's batches share its game data. current_frame is a copy of
    the shards' frames in env order."""

    def __init__(self, batch_size: int, shards: int, **kwargs):
        import concurrent.futures
        import melee_sim as msl
        sizes = [len(a) for a in np.array_split(np.arange(batch_size), shards)]
        self.bounds = np.cumsum([0] + sizes)
        self.shards = [msl.EnvBatch(batch_size=n, **kwargs) for n in sizes]
        self.batch_size, self.length = batch_size, self.shards[0].length
        self._pool = concurrent.futures.ThreadPoolExecutor(shards, thread_name_prefix="sim-shard")

    def _spans(self):
        return zip(self.shards, self.bounds[:-1], self.bounds[1:])

    @property
    def t(self) -> int:
        return self.shards[0].t

    @property
    def current_frame(self) -> np.ndarray:
        return np.concatenate([s.current_frame for s in self.shards])

    def configure_matches(self, configs, *, env_ids=None) -> None:
        if env_ids is None:
            for s, lo, hi in self._spans():
                s.configure_matches(configs[lo:hi])
            return
        ids = np.asarray(env_ids)
        for s, lo, hi in self._spans():
            mine = np.nonzero((ids >= lo) & (ids < hi))[0]
            if len(mine):
                s.configure_matches([configs[i] for i in mine], env_ids=ids[mine] - lo)

    def reset_all(self) -> None:
        for s in self.shards:
            s.reset_all()

    def reset_cursor(self) -> None:
        for s in self.shards:
            s.reset_cursor()

    def step_and_reset(self) -> tuple[np.ndarray, np.ndarray]:
        stepped = list(self._pool.map(lambda s: s.step_and_reset(), self.shards))
        return np.concatenate([r for r, _ in stepped]), np.concatenate([t for _, t in stepped])

    def close(self) -> None:
        self._pool.shutdown()
        for s in self.shards:
            s.close()


def write_controller_rows(env, rows: np.ndarray, player: int) -> None:
    """Flat [N, 13] controller rows (encode.controller_rows, as the agents
    execute them) -> the current step's row of the sim's action ring only."""
    if isinstance(env, ShardedEnvBatch):
        for s, lo, hi in env._spans():
            _write_rows(s, rows[lo:hi], player)
    else:
        _write_rows(env, rows, player)


def _write_rows(env, rows: np.ndarray, player: int) -> None:
    p = env.current_action_frame["players"][..., int(player)]
    p["main_stick_x"] = rows[:, 0]
    p["main_stick_y"] = rows[:, 1]
    p["c_stick_x"] = rows[:, 2]
    p["c_stick_y"] = rows[:, 3]
    p["shoulder"] = rows[:, 4]
    b = p["buttons"]
    for j, name in enumerate(_BUTTONS):
        b[name] = rows[:, 5 + j] > 0.5


def states_to_torch(encoded, device):
    """Encoded struct (numpy leaves) -> torch on device (int64/bool/float32,
    the learner's conventions); the worker uses FlatFrames instead."""
    import torch
    import tree

    def cvt(x):
        a = np.asarray(x)
        if a.dtype == np.bool_:
            return torch.as_tensor(a, device=device)
        if a.dtype.kind in "iu":
            return torch.as_tensor(a.astype(np.int64), device=device)
        return torch.as_tensor(a.astype(np.float32), device=device)
    return tree.map_structure(cvt, encoded)


def seat_stats(obs):
    """(stocks[N,2], percent[N,2]) with col0=self (slot0), col1=opp (slot1)."""
    s = obs["slots"]
    stocks = np.stack([s[:, 0]["stocks"], s[:, 1]["stocks"]], axis=1).astype(np.float32)
    percent = np.stack([s[:, 0]["percent"], s[:, 1]["percent"]], axis=1).astype(np.float32)
    return stocks, percent


def follower_stats(obs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(present, dying, percent), each [N, 2] in seat_stats' columns: the
    seats' followers (Nana), zeroed while absent."""
    f = obs["followers"][:, :2]
    return (f["present"].astype(np.bool_), reward_lib.is_dying(f["action_id"]),
            f["percent"].astype(np.float32))
