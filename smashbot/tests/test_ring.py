"""Serving ring (SGUCore ring mode + the student agent's captured carry) vs the
canonical shifted-cache path, for pure SGU and for the hybrid's interleaved
LSTM layers. The blocks initialize their history projections to zero or
identity, so every block parameter is randomized: otherwise history has no
effect on outputs and a wrong read is invisible."""
import os

import numpy as np
import pytest
import torch
import tree

from smashbot.networks import SGUCore

B, D, H, W = 8, 24, 32, 16
FRAMES = 3 * (W - 1) + 7  # three full wraps and a bit
LAYOUTS = ["sss", "sls"]


def _randomize(blocks):
    with torch.no_grad():
        for p in blocks.parameters():
            p.normal_(0, 0.3)


def _core(layout, seed=0):
    torch.manual_seed(seed)
    core = SGUCore(D, H, len(layout), W, layout=layout)
    _randomize(core.blocks)
    return core


def _stream(seed=1):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(B, D, generator=g) for _ in range(FRAMES)]
    rs = [torch.rand(B, generator=g) < 0.08 for _ in range(FRAMES)]
    rs[0][:] = True
    return xs, rs


def _ring_step(core, x, reset, ring):
    """What BatchedPolicyAgent._ring_carry does: read via step_with_reset, write
    each SGU layer's new slot at the pointer, carry kv, recurrent states and
    cache_len, advance the pointer."""
    y, nxt = core.step_with_reset(x, reset, ring)
    layers = []
    for layer, new in zip(ring["layers"], nxt["layers"]):
        if isinstance(layer, tuple):
            layer[0].index_copy_(1, ring["ptr"].view(1), new[0].unsqueeze(1))
            layers.append((layer[0], new[1]))
        else:
            layers.append(new)
    ring = {"cache_len": nxt["cache_len"], "layers": layers, "ptr": (ring["ptr"] + 1) % (W - 1)}
    return y, ring


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("read,tol", [("gather", 0.0), ("roll", 1e-4)])
def test_ring_matches_canonical_across_wraps_and_resets(layout, read, tol):
    """gather (the student's read) is exact; roll (the grid's: rotated weights,
    slot order) sums in another order, so it agrees to rounding."""
    core = _core(layout)
    core.ring_read = read
    xs, rs = _stream()
    can, ring = core.initial_state(B), core.initial_ring_state(B)
    with torch.no_grad():
        for t, (x, reset) in enumerate(zip(xs, rs)):
            yc, can = core.step_with_reset(x, reset, can)
            yr, ring = _ring_step(core, x, reset, ring)
            assert (yc - yr).abs().max().item() <= tol, f"output differs at frame {t}"
            exported = core.canonical_state(ring)   # what the learner receives
            for a, b in zip(tree.flatten(can), tree.flatten(exported)):
                assert (a.float() - b.float()).abs().max().item() <= tol, f"state differs at frame {t}"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_ring_reset_clears_history_only_for_reset_rows(layout):
    core = _core(layout)
    ring = core.initial_ring_state(B)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        ring["cache_len"] = torch.randint(1, W - 1, (B,), generator=g)
        ring["layers"] = [tuple(torch.randn(t.shape, generator=g) for t in layer)
                          if isinstance(layer, tuple) else torch.randn(layer.shape, generator=g)
                          for layer in ring["layers"]]
        before = tree.map_structure(torch.clone, ring["layers"])
        reset = torch.tensor([True, False] * (B // 2))
        x = torch.randn(B, D, generator=g)
        _, reset_rows = core.step_with_reset(x, reset, ring)
        _, fresh = core.step_with_reset(x, torch.ones(B, dtype=torch.bool), core.initial_ring_state(B))
    assert torch.equal(reset_rows["cache_len"][reset], torch.ones(int(reset.sum()), dtype=torch.long))
    assert torch.equal(reset_rows["cache_len"][~reset], ring["cache_len"][~reset] + 1)
    for layer, prior, new, fresh_layer in zip(ring["layers"], before, reset_rows["layers"], fresh["layers"]):
        if isinstance(layer, tuple):
            assert torch.equal(layer[0], prior[0])        # the ring is never rewritten on reset
            kv, kv_before = new[1], prior[1]
            assert torch.equal(kv[reset][:, :-1], torch.zeros_like(kv[reset][:, :-1]))
            assert torch.equal(kv[~reset][:, :-1], kv_before[~reset][:, 1:])
        else:   # a recurrent layer: reset rows restart from zero, the others carry on
            assert torch.equal(new[reset], fresh_layer[reset])
            assert not torch.equal(new[~reset], fresh_layer[~reset])


GPU = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SMASHBOT_GPU_TESTS")),
    reason="cuda-only; set SMASHBOT_GPU_TESTS=1 on an IDLE gpu (never beside a live training run)")


def _policy(layout, dev, seed=0):
    from smashbot import configs, embed as embed_lib
    from smashbot.policy import build_policy
    torch.manual_seed(seed)
    pol = build_policy(
        embed_config=embed_lib.EmbedConfig(), controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(name="sgu", num_layers=len(layout), layout=layout,
                                             hidden_size=64, window=W),
        head_config=configs.ControllerHeadConfig(), policy_config=configs.PolicyConfig(),
        num_names=16).to(dev)
    _randomize(pol.network.core.blocks)
    pol.eval()
    pol.requires_grad_(False)
    return pol


@GPU
@pytest.mark.parametrize("layout", LAYOUTS)
def test_student_serving_matches_eager_under_capture_and_ring(layout):
    """BatchedPolicyAgent as RL serves the student (fp16, captured graph, ring)
    against the same agent uncaptured and captured without the ring: the
    recurrent snapshot handed to the learner, every frame, over three wraps
    with staggered resets. Open loop: every agent sees the same previous
    action, so the snapshots are a function of the inputs alone."""
    from smashbot import embed as embed_lib
    from smashbot.rl.agent import BatchedPolicyAgent
    from scripts.bench_agent_step import _rand_raw
    dev, N = "cuda", 16
    pol = _policy(layout, dev)
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(1)
    to_dev = lambda x: torch.from_numpy(np.ascontiguousarray(
        x.astype(np.int64) if x.dtype.kind in "iu" else x)).to(dev)
    frames = [tree.map_structure(to_dev, game.from_state(_rand_raw(game, rng, N))) for _ in range(FRAMES)]
    g = torch.Generator().manual_seed(2)
    resets = [(torch.rand(N, generator=g) < 0.08).to(dev) for _ in range(FRAMES)]
    resets[0][:] = True
    agents = [BatchedPolicyAgent(pol, N, name_code=1, device=dev, precision="fp16", capture=capture,
                                 state_dtype=torch.float16, ring=ring)
              for capture, ring in ((False, False), (True, False), (True, True))]
    assert [a._ring for a in agents] == [False, False, True]
    prev = tree.map_structure(torch.clone, agents[0]._prev_action)
    for t in range(FRAMES):
        snaps = []
        for a in agents:
            a._prev_action = tree.map_structure(torch.clone, prev)
            a.execute(torch.nonzero(resets[t]).flatten().tolist())
            snaps.append(a.infer(frames[t], resets[t], want_snapshot=True)[1])
        for name, other in (("capture", snaps[1]), ("ring", snaps[2])):
            for x, y in zip(tree.flatten(snaps[0]), tree.flatten(other)):
                assert torch.equal(x, y), f"{name} state differs at frame {t}"


@GPU
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("read,tol", [("gather", 0.0), ("roll", 5e-2)])
def test_grid_ring_matches_canonical_under_capture(layout, read, tol):
    """LeagueAgent (fp16 stacked slices, captured vmap) with the ring vs without:
    every slice's recurrent state over three wraps, staggered resets, two seat
    moves and a reload of a different member's weights. roll sums in another
    order in fp16 (opponent seats only; nothing enters a loss)."""
    from smashbot import embed as embed_lib, encode
    from smashbot.rl import sim_env
    from smashbot.rl.agent import LeagueAgent
    from scripts.bench_agent_step import _rand_raw
    dev, S, N = "cuda", 2, 8
    sd, other = _policy(layout, dev).state_dict(), _policy(layout, dev, seed=7).state_dict()
    game = embed_lib.EmbedConfig().make_game_embedding()
    ff = sim_env.FlatFrames(dev)
    rng = np.random.default_rng(1)

    def frame():
        enc = game.from_state(_rand_raw(game, rng, S * N))
        flats = tuple(t.view(S, N, t.shape[-1])
                      for t in ff.to_device(encode.flatten_typed_batched(enc, S * N)))
        return ff.view(flats), flats

    g = torch.Generator().manual_seed(2)
    frames = [frame() for _ in range(FRAMES)]
    resets = [(torch.rand(S, N, generator=g) < 0.08).to(dev) for _ in range(FRAMES)]
    resets[0][:] = True
    moves, reload_at = {9: ((0, 1), (1, 5)), 23: ((1, 0), (1, 7))}, 31

    def make(ring):
        a = LeagueAgent(_policy(layout, dev), S, N, 1, dev, capture=True,
                        weights_dtype=torch.float16, state_dtype=torch.float16, ring=ring)
        if ring:
            a._core.ring_read = read
        for s in range(S):
            a.load_slice(s, sd)
        a.set_flat_inputs(ff.view)
        return a

    canonical, ringed = make(False), make(True)
    assert ringed._ring and not canonical._ring
    prev = tree.map_structure(torch.clone, canonical._prev)
    for t in range(FRAMES):
        views, flats = frames[t]
        for a in (canonical, ringed):
            a._prev = tree.map_structure(torch.clone, prev)   # open loop
            if t in moves:
                a.move_cell(*moves[t])
            if t == reload_at:
                a.load_slice(1, other)
            a.execute()
            a.infer(views, resets[t], flats=flats)
        flat = lambda h: {k: v for k, v in h.items() if k != "ptr"} | {
            "layers": [tuple(x.reshape(S * N, *x.shape[2:]) for x in l) if isinstance(l, tuple)
                       else l.reshape(S * N, *l.shape[2:]) for l in h["layers"]],
            "cache_len": h["cache_len"].reshape(-1)}
        want = flat(canonical._out_hidden)
        got = ringed._core.canonical_state({**flat(ringed._in_hidden), "ptr": ringed._in_hidden["ptr"][0]})
        assert torch.equal(want["cache_len"], got["cache_len"]), f"cache_len differs at frame {t}"
        for x, y in zip(tree.flatten(want["layers"]), tree.flatten(got["layers"])):
            diff = (x.float() - y.float()).abs().max().item()
            assert diff <= tol, f"state differs at frame {t}: {diff:.2e}"
