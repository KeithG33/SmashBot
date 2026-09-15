"""smashbot.encode (torch-free numpy encoder) must match embed.from_state
byte-for-byte, and must be importable without torch."""

import subprocess
import sys

import numpy as np
import pytest
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


def _assert_identical(a, b):
    flat_a, flat_b = tree.flatten(a), tree.flatten(b)
    assert len(flat_a) == len(flat_b)
    for x, y in zip(flat_a, flat_b):
        assert x.dtype == y.dtype, (x.dtype, y.dtype)
        assert x.shape == y.shape
        assert np.array_equal(x, y)


@pytest.mark.parametrize("shape", [(), (7,), (3, 5)])
def test_game_encoder_matches_from_state(shape):
    game = embed_lib.EmbedConfig().make_game_embedding()
    enc = encode.build(game.spec())
    rng = np.random.default_rng(0)
    for _ in range(20):
        raw = _random_raw(game, rng, shape)
        _assert_identical(enc.from_state(raw), game.from_state(raw))


def test_controller_encoder_matches_from_state():
    ctrl = embed_lib.ControllerConfig().make_embedding()
    enc = encode.build(ctrl.spec())
    rng = np.random.default_rng(1)
    for _ in range(20):
        raw = _random_raw(ctrl, rng, (4,))
        _assert_identical(enc.from_state(raw), ctrl.from_state(raw))


def test_error_policy_raises_identically():
    game = embed_lib.EmbedConfig().make_game_embedding()
    enc = encode.build(game.spec())
    rng = np.random.default_rng(2)
    raw = _random_raw(game, rng, (), stress_policies=False)
    # force an out-of-range value into an ERROR-policy one-hot (character)
    bad = raw._replace(p0=raw.p0._replace(character=np.int64(10**6)))
    with pytest.raises(ValueError):
        game.from_state(bad)
    with pytest.raises(ValueError):
        enc.from_state(bad)


def test_spec_is_plain_data_and_picklable():
    import pickle
    spec = embed_lib.EmbedConfig().make_game_embedding().spec()
    again = pickle.loads(pickle.dumps(spec))
    assert again == spec


def test_encode_module_never_imports_torch():
    # cheap-import modules stay torch-free (fast CLI boot; nothing forks
    # env processes anymore, but the property is worth keeping)
    code = (
        "import sys; import smashbot.encode, smashbot.rl.config, "
        "smashbot.eval.dolphin_setup, smashbot.rl.train_rl; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_typed_flat_round_trip_matches_worker_encode():
    """env-side flatten_typed -> worker-side unflatten_typed_torch equals
    today's per-leaf stack+from_numpy path exactly (values, dtypes, shapes)
    for a batch of random frames."""
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
    flats = [encode.flatten_typed(f) for f in frames]
    b, i, f = (torch.from_numpy(np.stack([fl[k] for fl in flats])) for k in range(3))
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
    """controller_rows -> controller_from_flat reproduces each row's struct
    exactly (stick floats bit-equal, buttons as bools), incl. the neutral
    controller; leaf order is the Controller tree order."""
    from smashbot.eval.agent import _neutral_controller
    from smashbot.rl.agent import _split_rows

    ctrl = embed_lib.ControllerConfig().make_embedding()
    rng = np.random.default_rng(9)
    decoded = ctrl.decode(ctrl.from_state(_random_raw(ctrl, rng, (6,))))
    rows = encode.controller_rows(decoded)
    assert rows.shape == (6, len(tree.flatten(decoded))) and rows.dtype == np.float32
    for i, ref in enumerate(_split_rows(decoded, 6)):
        got = encode.controller_from_flat(rows[i])
        assert type(got) is type(ref)
        for x, y in zip(tree.flatten(got), tree.flatten(ref)):
            assert np.asarray(x).dtype == np.asarray(y).dtype, (x, y)
            assert np.array_equal(x, y)
    neutral = _neutral_controller()
    row = encode.controller_rows(tree.map_structure(lambda x: np.asarray(x)[None], neutral))[0]
    got = encode.controller_from_flat(row)
    for x, y in zip(tree.flatten(got), tree.flatten(neutral)):
        assert np.array_equal(x, y)


def test_row_encoder_matches_slow_path():
    """RowEncoder == flatten_typed(build(spec).from_state(asarray-mapped
    raw)) bit for bit: random games plus adversarial leaves (negative /
    huge ints for the wrap and one-hot policies, float edge values)."""
    import tree

    from smashbot import embed as embed_lib, encode
    from smashbot.tests.test_rollouts import _rand_raw_game

    game = embed_lib.EmbedConfig().make_game_embedding()
    spec = game.spec()
    slow = encode.build(spec)
    fast = encode.RowEncoder(spec)
    assert sum(fast.counts.values()) == 122
    rng = np.random.default_rng(0)

    def both(raw):
        a = encode.flatten_typed(slow.from_state(tree.map_structure(np.asarray, raw)))
        b = fast.encode(raw)
        for x, y in zip(a, b):
            assert x.dtype == y.dtype and x.shape == y.shape
            np.testing.assert_array_equal(x, y)

    for _ in range(200):
        both(_rand_raw_game(game, (), rng))
    # python-scalar leaves (libmelee hands us ints/floats/bools, not arrays)
    raw = _rand_raw_game(game, (), rng)
    both(tree.map_structure(lambda x: x.item() if hasattr(x, "item") else x, raw))
    # adversarial integer leaves: every int leaf pushed out of range, both
    # signs (int64 so numpy accepts them); the two paths must agree on the
    # result OR both raise the one-hot ERROR-policy ValueError
    def outcome(fn):
        try:
            return fn()
        except ValueError as e:
            return ("raised", "input" in str(e))

    # leaves under the ERROR policy raise on both paths; everything else
    # (CLAMP / EXTRA / plain casts incl. uint8 wrap) must agree on values
    def error_paths(sp, path=()):
        if sp[0] == "struct":
            return [q for k, sub in sp[1] for q in error_paths(sub, path + (k,))]
        return [path] if sp[0] == "onehot" and sp[1] == "ERROR" else []

    errs = set(error_paths(spec))
    assert errs  # the policy exists in this embedding
    for bad in (-1, -300, 2 ** 31 - 1, 70000):
        for spare_errors in (False, True):
            adv = tree.map_structure_with_path(
                lambda p_, x: (
                    np.asarray(bad, dtype=np.int64)
                    if np.asarray(x).dtype.kind in "iu" and not (spare_errors and p_ in errs)
                    else x
                ),
                raw,
            )
            a = outcome(lambda: encode.flatten_typed(slow.from_state(tree.map_structure(np.asarray, adv))))
            b = outcome(lambda: fast.encode(adv))
            if isinstance(a[0], str):  # ("raised", ...)
                assert a == b, bad
                assert not spare_errors
            else:
                assert spare_errors
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x, y)

