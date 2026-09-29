"""The joint sticks' energy score: the Q = P @ D form matches the double sum,
near misses cost less than far ones, the true distribution scores best, and
BC trains with it."""

import dataclasses

import numpy as np
import torch
from slippi_ai.types import Controller

from smashbot import embed as embed_lib, train_bc
from smashbot.heads import AutoRegressive
from smashbot.policy import EnergyScore
from smashbot.tests.test_resume import _config, _latest

TABLE = "balanced_v6_157"


def _energy():
    head = AutoRegressive(embed_lib.get_controller_embedding(stick_table=TABLE), input_size=4)
    return EnergyScore(head, "cpu"), head.embed_struct.main_stick


def _main_only(energy):
    energy.distance = {"main_stick": energy.distance["main_stick"]}
    return energy


def _score(energy, logits, human):
    return energy(Controller(main_stick=logits, c_stick=None, shoulder=None, buttons=None),
                  Controller(main_stick=human, c_stick=None, shoulder=None, buttons=None))


def test_matches_the_double_sum():
    energy, stick = _energy()
    energy = _main_only(energy)
    torch.manual_seed(0)
    logits, human = torch.randn(3, 5, stick.size), torch.randint(stick.size, (3, 5))
    xy = torch.tensor(stick.positions, dtype=torch.float32) / 80
    d = torch.cdist(xy, xy)
    p = torch.softmax(logits, -1)
    want = (torch.einsum("btk,btk->bt", p, d[human]) - 0.5 * torch.einsum("btj,jk,btk->bt", p, d, p)).mean()
    torch.testing.assert_close(_score(energy, logits, human), want)


def test_near_misses_cost_less():
    energy, stick = _energy()
    energy = _main_only(energy)
    xy = stick.positions.astype(float)
    human = 0
    order = np.argsort(np.hypot(*(xy - xy[human]).T))
    sure = lambda b: torch.log(torch.nn.functional.one_hot(torch.tensor([[b]]), stick.size).float() + 1e-9)
    on, near, far = (_score(energy, sure(b), torch.tensor([[human]])) for b in (human, order[1], order[-1]))
    assert on < 1e-4 < near < far


def test_true_distribution_scores_best():
    """Proper: averaged over humans drawn from r, the score is lowest at p = r."""
    energy, stick = _energy()
    energy = _main_only(energy)
    rng = np.random.default_rng(0)
    r = torch.tensor(rng.dirichlet(np.ones(stick.size) * 0.3), dtype=torch.float32)
    humans = torch.arange(stick.size)

    def expected(p):
        logits = torch.log(p).expand(stick.size, stick.size).unsqueeze(1)   # every human, same p
        per_human = torch.stack([_score(energy, logits[h:h + 1], humans[h:h + 1, None]) for h in range(stick.size)])
        return (r * per_human).sum()

    best = expected(r)
    for mix in (torch.full_like(r, 1 / stick.size), torch.roll(r, 1)):
        assert best < expected(0.5 * r + 0.5 * mix)


def test_bc_trains_with_it(tmp_path):
    cfg = _config(str(tmp_path), "energy", 4)
    cfg = dataclasses.replace(cfg, head=dataclasses.replace(cfg.head, controller_type=TABLE),
                              learner=dataclasses.replace(cfg.learner, energy_score_weight=1.0))
    train_bc.main(cfg)
    assert _latest(str(tmp_path), "energy")["step"] == 4
