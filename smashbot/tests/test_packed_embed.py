"""PackedStructForward equivalence contract vs the per-leaf reference path.

Bitwise-identical everywhere except the item-MLP block, where batching the 15
slot MLPs into one call changes BLAS summation order: deviations there are
float32 reassociation noise, measured <= 2e-7 absolute (vs ~1e-3 bf16
resolution). Tests below encode exactly that contract.
"""

import numpy as np
import pytest
import torch
import tree

from slippi_ai.types import StateAction

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy


def _rand_input(emb, shape, rng):
    """Random raw inputs, valid per each leaf's from_state contract."""
    if isinstance(emb, embed_lib.StructEmbedding):
        return emb.builder(
            {k: _rand_input(e, shape, rng) for k, e in emb.embedding}
        )
    if isinstance(emb, embed_lib.MLPWrapper):
        return _rand_input(emb._embed, shape, rng)
    if isinstance(emb, embed_lib.BoolEmbedding):
        return torch.from_numpy(rng.integers(0, 2, shape).astype(np.bool_))
    if isinstance(emb, embed_lib.OneHotEmbedding):
        if emb.one_hot_policy is embed_lib.OneHotPolicy.EMPTY:
            hi = emb.size + 2  # exercise the invalid -> all-zeros path
        elif emb.one_hot_policy is embed_lib.OneHotPolicy.EXTRA:
            hi = emb.size  # from_state may emit the extra bucket
        else:
            hi = emb.input_size
        return torch.from_numpy(rng.integers(0, hi, shape).astype(emb.dtype))
    if isinstance(emb, embed_lib.FloatEmbedding):
        return torch.from_numpy(
            rng.uniform(-300.0, 300.0, shape).astype(np.float32)
        )
    raise TypeError(emb)


def _make_embedding(num_names=16):
    cfg = embed_lib.EmbedConfig()
    return embed_lib.get_state_action_embedding(
        embed_game=cfg.make_game_embedding(),
        embed_action=cfg.controller.make_embedding(),
        num_names=num_names,
    )


def _mlp_col_mask(packed, size):
    mask = torch.zeros(size, dtype=torch.bool)
    for g in packed._groups:
        mask[g["start"] : g["start"] + g["k"] * g["out_size"]] = True
    return mask


@pytest.mark.parametrize("shape", [(7,), (3, 5)])
def test_packed_matches_reference_bitwise(shape):
    sae = _make_embedding()
    packed = embed_lib.PackedStructForward(sae)
    mlp_cols = _mlp_col_mask(packed, sae.size)
    rng = np.random.default_rng(0)
    for _ in range(5):
        sa = _rand_input(sae, shape, rng)
        ref = sae(sa)
        fast = packed(sa)
        assert ref.shape == fast.shape == (*shape, sae.size)
        assert torch.equal(ref[..., ~mlp_cols], fast[..., ~mlp_cols])
        assert torch.allclose(ref[..., mlp_cols], fast[..., mlp_cols], atol=1e-6, rtol=0)


def test_packed_matches_with_int64_inputs():
    """The play path upcasts all int leaves to int64; result must not change."""
    sae = _make_embedding()
    packed = embed_lib.PackedStructForward(sae)
    rng = np.random.default_rng(1)
    sa = _rand_input(sae, (4,), rng)

    def upcast(t):
        if t.dtype in (torch.uint8, torch.int32):
            return t.long()
        return t

    sa64 = tree.map_structure(upcast, sa)
    mlp_cols = _mlp_col_mask(packed, sae.size)
    ref, fast = sae(sa), packed(sa64)
    assert torch.equal(ref[..., ~mlp_cols], fast[..., ~mlp_cols])
    assert torch.allclose(ref[..., mlp_cols], fast[..., mlp_cols], atol=1e-6, rtol=0)


def test_packed_registers_no_state():
    sae = _make_embedding()
    packed = embed_lib.PackedStructForward(sae)
    assert list(packed.state_dict().keys()) == []
    assert list(packed.parameters()) == []


def test_packed_item_mlp_grads_match():
    sae = _make_embedding()
    packed = embed_lib.PackedStructForward(sae)
    mlp_params = [p for g in packed._groups for p in g["mlp"].parameters()]
    assert mlp_params, "expected the shared items MLP to be trainable"
    rng = np.random.default_rng(2)
    sa = _rand_input(sae, (6,), rng)
    weight = torch.randn(sae.size)  # random linear functional of the output

    (sae(sa) @ weight).sum().backward()
    ref_grads = [p.grad.clone() for p in mlp_params]
    for p in mlp_params:
        p.grad = None
    (packed(sa) @ weight).sum().backward()
    for p, ref in zip(mlp_params, ref_grads):
        assert torch.allclose(p.grad, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", ["tx_like", "transformer", "sgu"])
def test_policy_paths_match(name):
    torch.manual_seed(0)
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(name=name, num_layers=2, window=8),
        head_config=configs.ControllerHeadConfig(),
        policy_config=configs.PolicyConfig(delay=2),
        num_names=16,
    )
    net = policy.network
    assert net.packed_embed is not None
    rng = np.random.default_rng(3)
    sa = _rand_input(net.embed_state_action, (3, 12), rng)
    reset = torch.zeros(3, 12, dtype=torch.bool)
    reset[1, 4] = True
    state = net.initial_state(3)

    out_fast, _ = net.unroll(sa, reset, state)
    packed_ref, net.packed_embed = net.packed_embed, None
    out_ref, _ = net.unroll(sa, reset, state)
    net.packed_embed = packed_ref
    # item-MLP ulp noise propagates through the core: tight allclose, not equal
    assert torch.allclose(out_fast, out_ref, atol=1e-5, rtol=1e-5)
