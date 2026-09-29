"""Eval's stick metrics: the joint p(x, y) must agree with the head's own
teacher-forced distance, and the read tables with Melee's clamp and deadzone."""

import numpy as np
import torch
import tree
from slippi_ai.types import Buttons, Controller, Stick

from smashbot import embed as embed_lib
from smashbot.heads import AutoRegressive
from smashbot.policy import _stick_tables, stick_scores

B, T = 6, 5


def _random_controller(embed_controller, rng):
    axis = lambda: rng.random((B, T), dtype=np.float32)
    raw = Controller(
        main_stick=Stick(x=axis(), y=axis()),
        c_stick=Stick(x=axis(), y=axis()),
        shoulder=axis(),
        buttons=Buttons(*(rng.random((B, T)) < 0.5 for _ in Buttons._fields)),
    )
    return tree.map_structure(torch.from_numpy, embed_controller.from_state(raw))


def _bucket(x, y):
    return (x + 80) // 10 * 17 + (y + 80) // 10


def test_joint_matches_distance():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    embed_controller = embed_lib.get_controller_embedding(axis_spacing=16)
    head = AutoRegressive(embed_controller, input_size=16, residual_size=8, component_depth=1)
    for block in head.res_blocks:
        torch.nn.init.normal_(block.decoder.weight, std=0.5)
    inputs = torch.randn(B, T, 16)
    prev, human = (_random_controller(embed_controller, rng) for _ in range(2))

    joint = head.stick_log_probs(inputs, prev, human)
    distance = head.distance(inputs, prev, human).distance
    for name in ("main_stick", "c_stick"):
        log_p, stick = joint[name], getattr(human, name)
        torch.testing.assert_close(log_p.logsumexp((-2, -1)), torch.zeros(B, T))
        at_human = log_p.flatten(-2).gather(-1, (stick.x.long() * 17 + stick.y.long())[..., None])
        expected = -(getattr(distance, name).x + getattr(distance, name).y)
        torch.testing.assert_close(at_human.squeeze(-1), expected)


def test_grid_reads_as_melee():
    outcome, distance = _stick_tables(17, torch.device("cpu"))
    assert len(outcome.unique()) == 161
    centre = {outcome[_bucket(x, y)].item() for x in range(-20, 21, 10) for y in range(-20, 21, 10)}
    assert len(centre) == 1
    assert outcome[_bucket(60, -10)] == outcome[_bucket(60, 0)] == outcome[_bucket(60, 10)]
    # (80, +-10) clamps to (79, +-9), whose y is in the deadzone: (79, 0), not a full press
    assert outcome[_bucket(80, 10)] == outcome[_bucket(80, -10)] != outcome[_bucket(80, 0)]
    assert distance[_bucket(80, 10), _bucket(80, 0)] == 1


def test_scores():
    human = Controller(main_stick=Stick(x=torch.tensor([[14]]), y=torch.tensor([[8]])),
                       c_stick=None, shoulder=None, buttons=None)   # (60, 0)
    sure = lambda x, y: torch.log_softmax(
        100 * torch.nn.functional.one_hot(torch.tensor(_bucket(x, y)), 289).float(), -1).view(1, 1, 17, 17)

    exact = stick_scores({"main_stick": sure(60, 0)}, human)["main_stick"]
    torch.testing.assert_close(exact, torch.tensor([1.0, 1.0, 0.0]))
    same_read = stick_scores({"main_stick": sure(60, 10)}, human)["main_stick"]
    torch.testing.assert_close(same_read, torch.tensor([0.0, 1.0, 0.0]))
    near = stick_scores({"main_stick": sure(70, 0)}, human)["main_stick"]
    torch.testing.assert_close(near, torch.tensor([0.0, 0.0, 10.0]))
