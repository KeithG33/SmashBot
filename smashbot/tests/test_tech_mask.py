"""The tech mask over a training unroll must match serving frame for frame:
the same masked opponent actions and the same carried filter state, whole or
split into chunks."""

import collections

import torch

from smashbot.networks import BACKWARD_TECH, FORWARD_TECH, NEUTRAL_TECH, StateActionNetwork

Player = collections.namedtuple("Player", "action")
State = collections.namedtuple("State", "p1")
StateAction = collections.namedtuple("StateAction", "state")


class _Filter:
    tech_mask_window = 4
    _mask_tech = StateActionNetwork._mask_tech
    _mask_tech_unroll = StateActionNetwork._mask_tech_unroll


def _sequences(B=8, T=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    choices = torch.tensor([NEUTRAL_TECH, FORWARD_TECH, BACKWARD_TECH, 0, 14, 57])
    runs = choices[torch.randint(len(choices), (B, T // 3), generator=g)]
    lengths = torch.randint(1, 7, (B, T // 3), generator=g)
    rows = [torch.repeat_interleave(r, n)[:T] for r, n in zip(runs, lengths)]
    action = torch.stack([torch.nn.functional.pad(r, (0, T - len(r)), value=14) for r in rows]).to(torch.int16)
    reset = torch.rand(B, T, generator=g) < 0.05
    prev = choices[torch.randint(len(choices), (B,), generator=g)]
    count = torch.randint(0, 6, (B,), generator=g)
    return action, reset, prev, count


def _serving(f, action, reset, prev, count):
    masked = []
    for t in range(action.shape[1]):
        prev, count = (torch.where(reset[:, t], 0, v) for v in (prev, count))
        sa, (prev, count) = f._mask_tech(StateAction(State(Player(action[:, t]))), prev, count)
        masked.append(sa.state.p1.action)
    return torch.stack(masked, dim=1), prev, count


def _unroll(f, action, reset, prev, count):
    sa, (prev, count) = f._mask_tech_unroll(StateAction(State(Player(action))), reset, prev, count)
    return sa.state.p1.action, prev, count


def test_unroll_matches_serving():
    f = _Filter()
    for seed in range(5):
        action, reset, prev, count = _sequences(seed=seed)
        want = _serving(f, action, reset, prev, count)
        got = _unroll(f, action, reset, prev, count)
        for w, g in zip(want, got):
            assert torch.equal(w.long(), g.long())
        assert (want[0] != action).any()   # the mask actually fired


def test_chunked_unroll_carries_state():
    f = _Filter()
    action, reset, prev, count = _sequences(seed=7)
    whole = _unroll(f, action, reset, prev, count)
    first = _unroll(f, action[:, :19], reset[:, :19], prev, count)
    second = _unroll(f, action[:, 19:], reset[:, 19:], first[1], first[2])
    assert torch.equal(torch.cat([first[0], second[0]], 1).long(), whole[0].long())
    assert torch.equal(second[1], whole[1]) and torch.equal(second[2], whole[2])
