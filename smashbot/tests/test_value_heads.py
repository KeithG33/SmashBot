"""Checkpoints from before value nets took the policy's attention heads (they
kept one) load widened to the policy's count, computing exactly what they did:
weights, Adam moments and the carried attention cache."""
import copy
import dataclasses

import numpy as np
import torch
import tree

from smashbot import configs, saving
from smashbot.train_bc import TrainConfig
from smashbot.value import build_value_function


def _config(attn_heads):
    cfg = TrainConfig()
    cfg.network = configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=2, layout="sl", window=8,
                                        attn_heads=attn_heads, attn_head_dim=8, qk_norm=True)
    cfg.value = configs.ValueConfig(hidden_size=16, num_layers=2, layout="sl", window=6, inputs="policy")
    return dataclasses.asdict(cfg)


def _inputs(value_fn, B, T, seed):
    to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x))
    dim = value_fn.network.embed_sa(tree.map_structure(to_t, value_fn.network.embed_state_action.dummy((B, 1)))).shape[-1]
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, dim, generator=g), torch.rand(B, T, generator=g) < 0.1


def test_one_head_value_net_widens_to_the_policys_heads_exactly():
    torch.manual_seed(0)
    old = build_value_function(_config(1), "cpu")
    with torch.no_grad():
        for p in old.parameters():
            p.normal_(0.0, 0.3)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    x, reset = _inputs(old, 4, 12, seed=1)
    out, hidden = old.network.core.unroll(x, reset, old.network.core.initial_state(4))
    old.head(out).square().mean().backward()
    opt.step()
    zero = torch.zeros(4, dtype=torch.long)
    ckpt = {"config": _config(2), "version": 1, "state": {
        "value": old.state_dict(), "value_opt": opt.state_dict(),
        "value_hidden": {"core": tree.map_structure(lambda t: t.detach(), hidden), "tech_mask": (zero, zero)}}}

    up = saving.upgrade_checkpoint(copy.deepcopy(ckpt))
    new = build_value_function(up["config"], "cpu")
    new.load_state_dict(up["state"]["value"])
    torch.optim.Adam(new.parameters(), lr=1e-3).load_state_dict(up["state"]["value_opt"])
    assert (old.network.core.blocks[0].attn_heads, new.network.core.blocks[0].attn_heads, up["version"]) == (1, 2, 2)

    x, reset = _inputs(old, 4, 12, seed=2)
    with torch.no_grad():
        v_old = old.head(old.network.core.unroll(x, reset, ckpt["state"]["value_hidden"]["core"])[0])
        v_new = new.head(new.network.core.unroll(x, reset, up["state"]["value_hidden"]["core"])[0])
    assert torch.allclose(v_old, v_new, atol=1e-6), (v_old - v_new).abs().max()
