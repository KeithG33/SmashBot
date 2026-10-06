"""The joint sticks' action loss: any bucket in the human's game region
counts, so it never exceeds the bucket cross-entropy, is near zero when the
probability sits on another bucket of the right action, and BC trains with
it."""

import dataclasses

import torch
import torch.nn.functional as F
from slippi_ai.types import Controller

from smashbot import embed as embed_lib, train_bc
from smashbot.heads import AutoRegressive
from smashbot.policy import ActionLoss
from smashbot.tests.test_resume import _config, _latest

TABLE = "balanced_v6_157"


def _action_loss():
    head = AutoRegressive(embed_lib.get_controller_embedding(stick_table=TABLE), input_size=4)
    loss = ActionLoss(head, "cpu")
    loss.bucket_region = {"main_stick": loss.bucket_region["main_stick"]}
    return loss, head.embed_struct.main_stick


def _score(loss, logits, human):
    return loss(Controller(main_stick=logits, c_stick=None, shoulder=None, buttons=None),
                Controller(main_stick=human, c_stick=None, shoulder=None, buttons=None))


def test_never_above_cross_entropy():
    loss, stick = _action_loss()
    torch.manual_seed(0)
    logits, human = torch.randn(4, 6, stick.size), torch.randint(stick.size, (4, 6))
    ce = F.cross_entropy(logits.reshape(-1, stick.size), human.reshape(-1))
    assert _score(loss, logits, human) <= ce


def test_right_action_wrong_bucket_costs_nothing():
    loss, stick = _action_loss()
    region = loss.bucket_region["main_stick"]
    human = next(b for b in range(stick.size) if (region == region[b]).sum() > 1)
    other_same = next(b for b in range(stick.size) if b != human and region[b] == region[human])
    other_wrong = next(b for b in range(stick.size) if region[b] != region[human])
    sure = lambda b: torch.log_softmax(100 * F.one_hot(torch.tensor([[b]]), stick.size).float(), -1)
    target = torch.tensor([[human]])
    assert _score(loss, sure(other_same), target) < 1e-4
    assert _score(loss, sure(other_wrong), target) > 50
    assert F.cross_entropy(sure(other_same).reshape(1, -1), target.reshape(-1)) > 50


def test_bc_trains_with_it(tmp_path):
    cfg = _config(str(tmp_path), "action", 4)
    cfg = dataclasses.replace(cfg, head=dataclasses.replace(cfg.head, controller_type=TABLE),
                              learner=dataclasses.replace(cfg.learner, action_loss_weight=0.25, energy_score_weight=1.0))
    train_bc.main(cfg)
    assert _latest(str(tmp_path), "action")["step"] == 4
