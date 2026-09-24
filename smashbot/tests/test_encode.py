"""smashbot.encode: the typed flat layout the sim worker ships frames in,
and the flat controller rows it sends to the sim."""

import numpy as np
import tree

from smashbot import embed as embed_lib
from smashbot import encode


def _random_raw(embedding, rng, shape=(), stress_policies=True):
    """A random raw struct in the embedding's input schema. ERROR-policy
    one-hots stay in range; CLAMP/EXTRA/EMPTY ones get out-of-range values
    so the policy branches are exercised."""
    def gen(e):
        if isinstance(e, embed_lib.MLPWrapper):
            return e._embed.map(gen)
        if isinstance(e, embed_lib.DiscreteEmbedding):
            return rng.random(shape, dtype=np.float32)
        if isinstance(e, embed_lib.OneHotEmbedding):
            lo, hi = 0, e.input_size
            if stress_policies and e.one_hot_policy is not embed_lib.OneHotPolicy.ERROR:
                lo, hi = -3, e.input_size + 3
            return rng.integers(lo, hi, size=shape, dtype=np.int64)
        if isinstance(e, embed_lib.BoolEmbedding):
            return rng.integers(0, 2, size=shape).astype(bool)
        if isinstance(e, embed_lib.FloatEmbedding):
            return (rng.standard_normal(shape) * 50).astype(np.float32)
        raise TypeError(type(e))
    return embedding.map(gen)


def test_typed_flat_round_trip_matches_worker_encode():
    """flatten_typed_batched -> unflatten_typed_torch equals the per-leaf
    stack+from_numpy path exactly (values, dtypes, shapes) for a batch of
    random frames."""
    import torch

    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(3)
    frames = [game.from_state(_random_raw(game, rng)) for _ in range(6)]
    # today's worker path
    batched = tree.map_structure(lambda *xs: np.stack(xs), *frames)
    ref = tree.map_structure(
        lambda x: torch.from_numpy(np.ascontiguousarray(
            x.astype(np.int64) if x.dtype.kind in "iu" else x)), batched)
    # flat path
    layout = encode.layout_of(game.dummy())
    b, i, f = (torch.from_numpy(x) for x in encode.flatten_typed_batched(batched, len(frames)))
    got = encode.unflatten_typed_torch(game.dummy(), layout, b, i, f)
    for x, y in zip(tree.flatten(got), tree.flatten(ref)):
        assert x.dtype == y.dtype and x.shape == y.shape
        assert torch.equal(x, y)


def test_split_rows_builder_matches_unflatten_as():
    """agent._split_rows (compiled constructor) == tree.unflatten_as per row
    on the controller struct, values and types."""
    from smashbot.rl.agent import _split_rows

    ctrl = embed_lib.ControllerConfig().make_embedding()
    rng = np.random.default_rng(5)
    raw = _random_raw(ctrl, rng, (6,))
    rows = _split_rows(raw, 6)
    for i in range(6):
        ref = tree.unflatten_as(raw, [leaf[i] for leaf in tree.flatten(raw)])
        assert type(rows[i]) is type(ref)
        for x, y in zip(tree.flatten(rows[i]), tree.flatten(ref)):
            assert np.array_equal(x, y) and np.asarray(x).dtype == np.asarray(y).dtype


def test_flat_controller_round_trip():
    """controller_rows -> controller_from_rows reproduces the struct exactly
    (stick floats bit-equal, buttons as bools), incl. the neutral controller;
    leaf order is the Controller tree order."""
    from smashbot.eval.agent import _neutral_controller

    ctrl = embed_lib.ControllerConfig().make_embedding()
    rng = np.random.default_rng(9)
    decoded = ctrl.decode(ctrl.from_state(_random_raw(ctrl, rng, (6,))))
    rows = encode.controller_rows(decoded)
    assert rows.shape == (6, len(tree.flatten(decoded))) and rows.dtype == np.float32
    got = encode.controller_from_rows(rows)
    assert type(got) is type(decoded)
    for x, y in zip(tree.flatten(got), tree.flatten(decoded)):
        assert np.asarray(x).dtype == np.asarray(y).dtype, (x, y)
        assert np.array_equal(x, y)
    neutral = _neutral_controller()
    row = encode.controller_rows(tree.map_structure(lambda x: np.asarray(x)[None], neutral))[0]
    got = encode.controller_from_rows(row)
    for x, y in zip(tree.flatten(got), tree.flatten(neutral)):
        assert np.array_equal(x, y)
