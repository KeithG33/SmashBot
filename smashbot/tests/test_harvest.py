"""Harvested seats are replays: HarvestAssembler must produce exactly what BC's
slicing makes of the same gameplay recorded as a replay, match the old
record-based path for an opponent at the student's delay, and realign an
opponent at a longer delay (Phillip's 21 against 18) onto the student's."""
import collections
import typing as tp

import numpy as np
import torch
import tree
from slippi_ai.types import Frames, StateAction

from smashbot import embed as embed_lib, encode
from smashbot.delay import slice_delayed_frames
from smashbot.rl.agent import FrameRecord
from smashbot.rl.rollouts import ChunkAssembler, HarvestAssembler

EMBED = embed_lib.ControllerConfig().make_embedding()
NEUTRAL = np.array([0.5, 0.5, 0.5, 0.5, 0.0] + [0.0] * 8, dtype=np.float32)


class State(tp.NamedTuple):
    stage: torch.Tensor


def _game_starts(frames, starts):
    """[N, F] reset flags from each env's game-start frames (0 always starts)."""
    resets = np.zeros((len(starts), frames), dtype=bool)
    for n, s in enumerate(starts):
        resets[n, [0, *s]] = True
    return resets


def _encoded(rows):
    """Rows [..., 13] -> the student's encoded controller, as torch."""
    return tree.map_structure(
        lambda x: torch.from_numpy(np.ascontiguousarray(
            x.astype(np.int64) if x.dtype.kind in "iu" else x)),
        EMBED.from_state(encode.controller_from_rows(rows)))


def _equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(tree.flatten(a), tree.flatten(b)))


def _harvest(pressed, resets, rewards, T, D, tenure=None, burn_in=0):
    """Feed a timeline through the assembler in the worker's order: frame f,
    then the reward of f-1 -> f. pressed [N, F, 13], resets [N, F], rewards
    [N, F-1] (transition f -> f+1), tenure [N, F] (default: one per row)."""
    N, F = resets.shape
    tenure = np.zeros((N, F), dtype=np.int64) if tenure is None else tenure
    asm = HarvestAssembler(T, D, EMBED, name_code=3, burn_in=burn_in)
    chunks = []
    for f in range(F):
        asm.push_frame(State(stage=torch.arange(N) * 1000.0 + f), pressed[:, f],
                       torch.from_numpy(resets[:, f]), tenure[:, f])
        if f > 0:
            asm.push_reward(torch.from_numpy(rewards[:, f - 1]))
        if asm.ready():
            chunks.append(asm.emit())
    return chunks


def _expected_valid(resets, t0, T, D):
    """Position t is valid iff no game starts in frames t+1 .. t+D+1."""
    return torch.from_numpy(np.stack(
        [~resets[:, t0 + t + 1:t0 + t + D + 2].any(1) for t in range(T)], axis=1))


def _button_rows(N, F):
    """Each (env, frame) presses a distinct button pattern: code n*64 + f."""
    codes = np.arange(N)[:, None] * 64 + np.arange(F)[None]
    rows = np.tile(NEUTRAL, (N, F, 1))
    rows[..., 5:] = (codes[..., None] >> np.arange(8)) & 1
    return rows


def test_harvest_matches_bc_replay_alignment():
    """The same gameplay as a replay (replay_action[f+1] = pressed[f], reward
    f = transition f -> f+1), sliced by BC: identical states, resets and
    action stream everywhere, identical rewards at valid positions, zero
    reward elsewhere; observation t targets what was pressed at t+D."""
    T, D, N, F = 6, 3, 2, 44
    resets = _game_starts(F, [[13, 27], [20]])
    pressed = _button_rows(N, F)
    rewards = (np.arange(1, N + 1)[:, None] * 100.0 + np.arange(F - 1)[None]).astype(np.float32)
    chunks = _harvest(pressed, resets, rewards, T, D)
    assert len(chunks) == (F - D - 1) // T

    replay_action = np.concatenate([np.tile(NEUTRAL, (N, 1, 1)), pressed[:, :-1]], axis=1)
    saw_invalid = False
    for c, chunk in enumerate(chunks):
        t0 = c * T
        window = slice(t0, t0 + T + D + 1)
        bc = slice_delayed_frames(Frames(
            state_action=StateAction(
                state=State(stage=torch.arange(N)[:, None] * 1000.0 + torch.arange(F)[None, window]),
                action=_encoded(replay_action[:, window]),
                name=torch.zeros(N, T + D + 1, dtype=torch.int64)),
            is_resetting=torch.from_numpy(resets[:, window]),
            reward=torch.from_numpy(rewards[:, t0:t0 + T + D])), D)

        assert torch.equal(chunk.states.stage, bc.state_action.state.stage)
        assert torch.equal(chunk.is_resetting, bc.is_resetting)
        assert _equal(chunk.actions.controller_state, bc.state_action.action)
        assert _equal(tree.map_structure(lambda x: x[:, 1:], chunk.actions.controller_state),
                      _encoded(pressed[:, t0 + D:t0 + T + D]))
        valid = _expected_valid(resets, t0, T, D)
        assert torch.equal(chunk.valid, valid)
        assert torch.equal(chunk.rewards[valid], bc.reward[valid])
        assert not chunk.rewards[~valid].any()
        assert (chunk.name == 3).all() and chunk.kind == "imitation"
        saw_invalid |= bool((~valid).any())
    assert saw_invalid


def _opponent(resets, delay, seed):
    """An opponent seat as the agents run it: at a game start its queue
    refills with `delay` neutral rows and its previous action is neutral;
    each frame pops the row it presses, then queues its new decision.
    Returns (pressed rows, records' prev actions, decision rows)."""
    N, F = resets.shape
    rng = np.random.default_rng(seed)
    decisions = np.concatenate([rng.random((N, F, 5)), rng.random((N, F, 8)) > 0.5], -1)
    decisions = encode.controller_rows(EMBED.decode(EMBED.from_state(
        encode.controller_from_rows(decisions.astype(np.float32)))))
    pressed, prev = np.empty_like(decisions), np.empty_like(decisions)
    for n in range(N):
        queue = collections.deque()
        for f in range(F):
            if resets[n, f]:
                queue = collections.deque([NEUTRAL] * delay)
            prev[n, f] = NEUTRAL if resets[n, f] else decisions[n, f - 1]
            pressed[n, f] = queue.popleft()
            queue.append(decisions[n, f])
    return pressed, prev, decisions


def test_equal_delay_matches_old_record_path():
    """An opponent at the student's delay (the PFSP snapshots): the old path
    (its own records through ChunkAssembler) and the replay harvest agree on
    every valid position's states, previous action, target and reward."""
    T, D, N, F = 8, 4, 3, 70
    resets = _game_starts(F, [[19, 40], [33], [9, 50]])
    rewards = np.random.default_rng(1).normal(size=(N, F - 1)).astype(np.float32)
    pressed, prev, _ = _opponent(resets, D, seed=2)
    new = _harvest(pressed, resets, rewards, T, D)

    old_asm, old = ChunkAssembler(T, D), []
    for f in range(F):
        if f > 0:
            old_asm.push_reward(torch.from_numpy(rewards[:, f - 1]))
        old_asm.push_frame(
            FrameRecord(state=State(stage=torch.arange(N) * 1000.0 + f),
                        prev_action=_encoded(prev[:, f]), logits=torch.zeros(N),
                        name=torch.zeros(N, dtype=torch.int64)),
            torch.from_numpy(resets[:, f]), torch.zeros(1) if f % T == 0 else None)
        if old_asm.ready():
            old.append(old_asm.emit())

    for a, b in zip(old, new):
        assert torch.equal(a.states.stage, b.states.stage)
        assert torch.equal(a.is_resetting, b.is_resetting)
        v = b.valid
        for x, y in zip(tree.flatten(a.actions.controller_state),
                        tree.flatten(b.actions.controller_state)):
            assert torch.equal(x[:, :-1][v], y[:, :-1][v])   # previous actions
            assert torch.equal(x[:, 1:][v], y[:, 1:][v])     # targets
        assert torch.equal(a.rewards[v], b.rewards[v])
        assert not b.rewards[~v].any()


def test_longer_delay_opponent_is_realigned():
    """Phillip at 21 against a student at 18: observation t targets Phillip's
    decision from t-3 with its decision from t-4 as the previous action, and
    neutral where that decision would predate the game (what was pressed)."""
    T, D, lag, N, F = 8, 18, 3, 2, 90
    resets = _game_starts(F, [[37], [52]])
    pressed, _, decisions = _opponent(resets, D + lag, seed=3)
    chunks = _harvest(pressed, resets, np.zeros((N, F - 1), np.float32), T, D)

    def decided(n, f, obs):
        game_start = max(s for s in np.nonzero(resets[n])[0] if s <= obs)
        return decisions[n, f] if f >= game_start else NEUTRAL

    for c, chunk in enumerate(chunks):
        for n in range(N):
            for t in np.nonzero(chunk.valid[n].numpy())[0]:
                obs = c * T + t
                stream = tree.map_structure(lambda x: x[n, t:t + 2], chunk.actions.controller_state)
                assert _equal(stream, _encoded(np.stack([decided(n, obs - lag - 1, obs),
                                                         decided(n, obs - lag, obs)])))


def test_rows_that_change_occupant_are_left_out():
    """A seat whose occupant changed at any frame the chunk reads (states
    through T, lookahead through T+D) is dropped from that chunk only."""
    T, D, N, F = 5, 2, 3, 24
    resets = _game_starts(F, [[], [], []])
    tenure = np.zeros((N, F), dtype=np.int64)
    tenure[1, 6:] = 1           # reseated at 6: chunk 0 (frames 0..7) and 1 (5..12) span it
    tenure[2, :] = -1           # idle throughout
    chunks = _harvest(_button_rows(N, F), resets, np.zeros((N, F - 1), np.float32), T, D, tenure)
    assert [c.states.stage[:, 0].tolist() for c in chunks[:3]] == [
        [0.0], [5.0], [10.0, 1010.0]]

    asm = HarvestAssembler(T, D, EMBED, name_code=0)
    for f in range(T + D + 1):
        asm.push_frame(State(stage=torch.zeros(1)), NEUTRAL[None], torch.zeros(1, dtype=torch.bool),
                       np.full(1, -1))
        if f > 0:
            asm.push_reward(torch.zeros(1))
    assert asm.ready() and asm.emit() is None


def test_burn_in_prefix_is_the_rows_own_history():
    """Each chunk carries the frames before it (up to burn_in) with the same
    alignment as the chunk: prefix frame f has state f and pressed[f+D-1] as
    its previous action. A tenure change inside the prefix restarts it at
    the new tenure's first frame; a row with no history of its own starts
    its chunk cold."""
    T, D, H, N, F = 6, 3, 10, 3, 40
    resets = _game_starts(F, [[15], [], []])
    tenure = np.zeros((N, F), dtype=np.int64)
    tenure[1, 16:] = 1          # moved at 16: chunk 3 (18..) keeps 16, 17 as history
    tenure[2, :18] = -1         # seated at 18 (idle before): chunk 3 has no history
    tenure[2, 18:] = 2
    pressed = _button_rows(N, F)
    chunks = _harvest(pressed, resets, np.zeros((N, F - 1), np.float32), T, D, tenure, burn_in=H)
    assert chunks[0].prefix is None

    for c, chunk in enumerate(chunks[1:], start=1):
        t0 = c * T
        P = min(H, t0)
        rows = [n for n in range(N) if (tenure[n, t0:t0 + T + D + 1] == tenure[n, t0]).all()
                and tenure[n, t0] >= 0]
        pre = chunk.prefix
        assert torch.equal(pre.state_action.state.stage,
                           torch.tensor([[n * 1000.0 + f for f in range(t0 - P, t0)] for n in rows]))
        assert _equal(pre.state_action.action, _encoded(pressed[rows, t0 - P + D - 1:t0 + D - 1]))
        for i, n in enumerate(rows):
            own = tenure[n, t0 - P:t0] == tenure[n, t0]
            first = P - int(np.argmin(own[::-1])) if not own.all() else 0
            expected = resets[n, t0 - P:t0].copy()
            if 0 < first < P:
                expected[first] = True
            assert pre.is_resetting[i].tolist() == expected.tolist(), (c, n)
            assert bool(chunk.is_resetting[i, 0]) == bool(resets[n, t0] or first == P), (c, n)
    third = chunks[3]
    assert third.prefix.is_resetting[1].tolist() == [False] * 8 + [True, False]
    assert bool(third.is_resetting[2, 0])
