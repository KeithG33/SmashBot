"""Name maps: new runs embed Unknown players (the empty name) and names the map lacks as all zeros, give
every label in the data a slot of its own, and leave checkpoints' saved maps encoding as they always did."""

import collections
import copy
import dataclasses

import numpy as np
import pytest
import torch

from slippi_ai import data as data_lib
from slippi_ai import nametags
from slippi_ai.paths import TOY_DATASET

from smashbot import saving, train_bc
from smashbot.configs import DataConfig
from smashbot.data import loader
from smashbot.embed import OneHotEmbedding, OneHotPolicy

MAX_NAMES = 16
LABELS = ["GM Player", "Master Player", "Platinum Player", "Diamond Player"]
RANKED_MAP = {"Platinum Player": 0, "Master Player": 1, "Diamond Player": 2}   # the ranked BC checkpoints'
PORTED_MAP = {"Platinum Player": 0, "Master Player": 1, "Diamond Player": 2, "": 3, "Hax": 4, "XX#02": 4}

_Player = collections.namedtuple("_Player", "name")
_Replay = collections.namedtuple("_Replay", "main_player")


def _replays(counts: dict[str, int]) -> list[_Replay]:
    return [_Replay(_Player(name)) for name, n in counts.items() for _ in range(n)]


def _embedded(name_map: dict[str, int], names: list[str]) -> torch.Tensor:
    """The model's name columns for each name, encoded as the data pipeline encodes them."""
    encode = nametags.name_encoder(name_map)
    embedding = OneHotEmbedding("name", MAX_NAMES, dtype=np.int32, one_hot_policy=OneHotPolicy.EMPTY)
    return embedding(torch.tensor([encode(name) for name in names]))


TOURNAMENT = _replays({"": 50, "Master Player": 9, "GM Player": 4, "Platinum Player": 3, "Diamond Player": 2})


def test_unknown_embeds_as_zeros_and_labels_get_distinct_slots():
    name_map = loader.create_name_map(TOURNAMENT, MAX_NAMES, zero_unknown=True)
    rows = _embedded(name_map, ["", "Somebody Not In The Data", *LABELS])
    assert rows[:2].abs().sum() == 0
    assert (rows[2:].sum(-1) == 1).all()
    assert len({tuple(row.tolist()) for row in rows[2:]}) == len(LABELS)


def test_a_ranked_warm_start_keeps_its_slots_and_gm_gets_a_new_one():
    name_map = loader.create_name_map(TOURNAMENT, MAX_NAMES, zero_unknown=True, base=RANKED_MAP)
    assert {name: name_map[name] for name in RANKED_MAP} == RANKED_MAP
    assert name_map["GM Player"] not in RANKED_MAP.values()
    rows = _embedded(name_map, ["", *LABELS])
    assert rows[0].abs().sum() == 0
    assert len({tuple(row.tolist()) for row in rows[1:]}) == len(LABELS)


def test_a_ported_maps_unknown_slot_stays_reserved_on_a_warm_start():
    name_map = loader.create_name_map(TOURNAMENT, MAX_NAMES, zero_unknown=True, base=PORTED_MAP)
    assert name_map["GM Player"] not in PORTED_MAP.values()
    assert _embedded(name_map, [""]).abs().sum() == 0


def test_saved_maps_encode_as_before():
    encode = nametags.name_encoder(RANKED_MAP)
    assert [encode(name) for name in ["Platinum Player", "Master Player", "Diamond Player", "", "GM Player"]] \
        == [0, 1, 2, 3, 3]
    encode = nametags.name_encoder(PORTED_MAP)
    assert [encode(name) for name in ["", "XX#02", "Somebody"]] == [3, 4, 5]
    assert saving.resolve_name_code(RANKED_MAP, "GM Player", verbose=False) == 0
    assert saving.resolve_name_code(PORTED_MAP, "", verbose=False) == 3


def test_without_the_setting_maps_are_built_as_before():
    ranked = _replays({"Platinum Player": 5, "Master Player": 4, "Diamond Player": 3})
    assert loader.create_name_map(ranked, MAX_NAMES) == RANKED_MAP
    assert loader.create_name_map(TOURNAMENT, MAX_NAMES) == {
        "": 0, "Master Player": 1, "GM Player": 2, "Platinum Player": 3, "Diamond Player": 4}


def _toy(**kwargs) -> DataConfig:
    return DataConfig(dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)), batch_size=2,
                      unroll_length=16, num_workers=0, pin_memory=False, **kwargs)


@pytest.mark.parametrize("zero_unknown_names", [True, False])
def test_a_resumed_map_is_used_as_saved(zero_unknown_names):
    sources = loader.make_sources(_toy(zero_unknown_names=zero_unknown_names), 4, name_map=dict(RANKED_MAP))
    assert sources.name_map == RANKED_MAP


def test_a_new_run_builds_a_zero_unknown_map():
    name_map = loader.make_sources(_toy(), 4).name_map
    assert name_map[""] == MAX_NAMES
    assert all(code < MAX_NAMES for name, code in name_map.items() if name)


def test_a_checkpoint_from_before_the_setting_resumes():
    current = dataclasses.asdict(train_bc.TrainConfig())
    saved = copy.deepcopy(current)
    del saved["data"]["zero_unknown_names"]
    train_bc._check_config(saved, current, current)
