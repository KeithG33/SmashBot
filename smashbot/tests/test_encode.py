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
    # everything an env process (and the __main__ it re-imports) touches
    code = (
        "import sys; import smashbot.encode, smashbot.rl.config, "
        "smashbot.eval.dolphin_setup, smashbot.rl.env_process, "
        "smashbot.rl.train_rl; "
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
