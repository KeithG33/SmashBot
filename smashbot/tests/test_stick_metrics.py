"""Eval's stick metrics and the joint stick encoding: a stick's
distribution over its buckets must agree with the head's own teacher-forced
distance, the joint tables must cover every position Melee reads, and the
scores must read positions the way the game does."""

import numpy as np
import pytest
import torch
import tree
from slippi_ai.types import Buttons, Controller, Stick

from smashbot import embed as embed_lib
from smashbot.heads import AutoRegressive
from smashbot.policy import StickScorer, bucket_reads

B, T = 6, 5


def _controller(stick_table=""):
    return embed_lib.get_controller_embedding(axis_spacing=16, stick_table=stick_table)


def _random_raw(rng):
    """Replay-like controllers: each stick a position the game reads."""
    points = np.array(sorted({tuple(p) for p in embed_lib.stick_read(
        np.stack(np.meshgrid(np.arange(-80, 81), np.arange(-80, 81), indexing="ij"), -1)).reshape(-1, 2)}))
    stick = lambda: Stick(*((points[rng.integers(len(points), size=(B, T))] + 80) / 160).astype(np.float32)
                          .transpose(2, 0, 1))
    return Controller(main_stick=stick(), c_stick=stick(), shoulder=rng.random((B, T), dtype=np.float32),
                      buttons=Buttons(*(rng.random((B, T)) < 0.5 for _ in Buttons._fields)))


def _grid(x, y):
    return (x + 80) // 10 * 17 + (y + 80) // 10


@pytest.mark.parametrize("stick_table", ["", "balanced_v2", "balanced_v6"])
def test_stick_distribution_matches_distance(stick_table):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    embed_controller = _controller(stick_table)
    head = AutoRegressive(embed_controller, input_size=16, residual_size=8, component_depth=1)
    for block in head.res_blocks:
        torch.nn.init.normal_(block.decoder.weight, std=0.5)
    inputs = torch.randn(B, T, 16)
    prev, human = (tree.map_structure(torch.from_numpy, embed_controller.from_state(_random_raw(rng)))
                   for _ in range(2))

    log_probs = head.stick_log_probs(inputs, prev, human)
    distance = head.distance(inputs, prev, human).distance
    scorer = StickScorer(head, "cpu")
    for name in ("main_stick", "c_stick"):
        log_p, d = log_probs[name], getattr(distance, name)
        torch.testing.assert_close(log_p.logsumexp(-1), torch.zeros(B, T))
        at_human = log_p.gather(-1, scorer._bucket(name, getattr(human, name))[..., None]).squeeze(-1)
        torch.testing.assert_close(at_human, -(d.x + d.y) if stick_table == "" else -d)


@pytest.mark.parametrize("table, main_size, main_neutral", [("balanced_v2", 237, 116), ("balanced_v6", 241, 118)])
def test_joint_tables_cover_every_position(table, main_size, main_neutral):
    for which, size, neutral in (("main_stick", main_size, main_neutral), ("c_stick", 75, 35)):
        stick = embed_lib.JointStickEmbedding(which, table, which)
        assert stick.size == size and stick.lookup.shape == (161, 161) and stick.lookup.max() == size - 1
        # every bucket decodes to one of its own positions
        np.testing.assert_array_equal(stick.from_state(stick.decode(np.arange(size))), np.arange(size))
        np.testing.assert_array_equal(stick.from_state(Stick(np.float32([0.5]), np.float32([0.5]))), [neutral])
        # any other position (a grid or custom_v1 press) lands where the game reads it
        raw = np.array([[80, 80], [-80, 10], [10, -20], [53, 53], [-57.4, 12.6]])
        read = embed_lib.stick_read(np.rint(raw))
        want = stick.from_state(Stick(*((read + 80) / 160).astype(np.float32).T))
        np.testing.assert_array_equal(stick.from_state(Stick(*((raw + 80) / 160).astype(np.float32).T)), want)


def test_grid_reads_as_melee():
    reads = bucket_reads(_controller().map(lambda e: e).main_stick)
    assert len(np.unique(reads, axis=0)) == 161
    assert len({tuple(reads[_grid(x, y)]) for x in range(-20, 21, 10) for y in range(-20, 21, 10)}) == 1
    # (80, +-10) clamps to (79, +-9), whose y is in the deadzone: (79, 0), not a full press
    assert tuple(reads[_grid(80, 10)]) == tuple(reads[_grid(80, -10)]) == (79, 0)
    assert tuple(reads[_grid(60, -10)]) == tuple(reads[_grid(60, 10)]) == (60, 0)


def _sure(bucket, size):
    return torch.log_softmax(100 * torch.nn.functional.one_hot(torch.tensor(bucket), size).float(), -1).view(1, 1, -1)


def test_scores():
    grid_head = AutoRegressive(_controller(), input_size=4)
    grid = StickScorer(grid_head, "cpu")
    human = Controller(main_stick=Stick(x=torch.tensor([[14]]), y=torch.tensor([[8]])),   # (60, 0)
                       c_stick=None, shoulder=None, buttons=None)
    exact = {"main_stick": torch.tensor([[[60, 0]]])}
    score = lambda scorer, bucket, size, human, exact: scorer.score(
        {"main_stick": _sure(bucket, size)}, human, exact)["main_stick"]
    torch.testing.assert_close(score(grid, _grid(60, 0), 289, human, exact), torch.tensor([1.0, 1.0, 0.0]))
    torch.testing.assert_close(score(grid, _grid(60, 10), 289, human, exact), torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(score(grid, _grid(70, 0), 289, human, exact), torch.tensor([0.0, 0.0, 10.0]))

    joint_head = AutoRegressive(_controller("balanced_v2"), input_size=4)
    joint = StickScorer(joint_head, "cpu")
    stick = joint.sticks["main_stick"]
    bucket = int(stick.from_state(Stick(np.float32([(57 + 80) / 160]), np.float32([(30 + 80) / 160])))[0])
    centre = tuple(int(v) for v in stick.positions[bucket])
    assert centre != (57, 30)   # a member of the bucket other than its decode point
    human = Controller(main_stick=torch.tensor([[bucket]]), c_stick=None, shoulder=None, buttons=None)
    for position, want_read in ((centre, 1.0), ((57, 30), 0.0)):
        got = score(joint, bucket, 237, human, {"main_stick": torch.tensor([[position]])})
        miss = float(np.hypot(centre[0] - position[0], centre[1] - position[1]))
        torch.testing.assert_close(got, torch.tensor([1.0, want_read, miss]))
