"""Serving-side state handling across the three core layouts: SGU window
caches, hybrid (SGU + recurrent) and torch-RNN (tx_like) state.

- fp16 serving stores only the window caches in fp16; recurrent memory
  (LSTM h/c, GRU h) stays fp32.
- LeagueAgent.move_cell moves one seat for every layout (torch RNN state is
  [layers, N, H], not batch-first).
- BatchedPolicyAgent.reset_env resets one env in place for every layout.
"""
import numpy as np
import pytest
import torch
import tree

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy
from smashbot.rl.agent import BatchedPolicyAgent, LeagueAgent
from smashbot.tests.test_ppo import _rand_states


def _policy(name, layout="", seed=0):
    torch.manual_seed(seed)
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(
            name=name, num_layers=2, hidden_size=32, num_heads=1, window=4, layout=layout),
        head_config=configs.ControllerHeadConfig(residual_size=16, component_depth=0),
        policy_config=configs.PolicyConfig(delay=1),
        num_names=2,
    )
    policy.train_value_head = False
    return policy.eval()


LAYOUTS = [("sgu", ""), ("sgu", "sl"), ("sgu", "sg"), ("tx_like", "")]


@pytest.mark.parametrize("name,layout", LAYOUTS)
def test_cache_state_keeps_recurrent_memory_fp32(name, layout):
    policy = _policy(name, layout)
    state = policy.network.cache_state(policy.initial_state(3), torch.float16)
    leaves = [t for t in tree.flatten(state) if isinstance(t, torch.Tensor) and t.is_floating_point()]
    core = policy.network.core
    if name == "tx_like":
        assert all(t.dtype == torch.float32 for t in leaves)
    else:
        for block, layer in zip(core.blocks, state["layers"]):
            if isinstance(layer, tuple):
                assert all(t.dtype == torch.float16 for t in layer), "window caches go to fp16"
            else:
                assert layer.dtype == torch.float32, "recurrent memory stays fp32"


@pytest.mark.parametrize("name,layout", LAYOUTS)
def test_move_cell_moves_one_seat(name, layout):
    policy = _policy(name, layout)
    S, N = 2, 3
    agent = LeagueAgent(policy, S, N, name_code=1, device="cpu", capture=False)
    for s in range(S):
        agent.load_slice(s, policy.state_dict())
    rng = np.random.default_rng(0)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]
    views = _rand_states(game_embed, (S, N), rng)
    resets = torch.zeros(S, N, dtype=torch.bool)
    for _ in range(3):   # some frames of state
        agent.infer(views, resets)
    before = tree.map_structure(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, agent._hidden)

    def seat(state, s, n):
        def pick(t):
            x = t[s]
            return x[n] if x.shape[0] == N else x[:, n]
        return [pick(t).clone() for t in tree.flatten(state) if isinstance(t, torch.Tensor)]

    agent.move_cell((0, 2), (1, 1))
    after = agent._hidden
    for a, b in zip(seat(after, 1, 1), seat(before, 0, 2)):
        assert torch.equal(a, b)
    for n in (0, 2):   # the other seats of the destination slice are untouched
        for a, b in zip(seat(after, 1, n), seat(before, 1, n)):
            assert torch.equal(a, b)


@pytest.mark.parametrize("name,layout", LAYOUTS)
def test_reset_env_resets_one_env_in_place(name, layout):
    policy = _policy(name, layout)
    N = 3
    agent = BatchedPolicyAgent(policy, N, name_code=1, device="cpu")
    rng = np.random.default_rng(0)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]
    st = _rand_states(game_embed, (N,), rng)
    for _ in range(3):
        agent.infer(st, torch.zeros(N, dtype=torch.bool), want_snapshot=False)
    ids = [t.data_ptr() for t in tree.flatten(agent.hidden) if isinstance(t, torch.Tensor)]
    before = tree.map_structure(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, agent.hidden)
    agent.reset_env(1)
    fresh = policy.initial_state(N)
    for t_after, t_before, t_fresh in zip(*(
            [t for t in tree.flatten(x) if isinstance(t, torch.Tensor)] for x in (agent.hidden, before, fresh))):
        if t_after.dim() >= 1 and t_after.shape[0] == N:
            row = lambda t, i: t[i]
        else:   # torch RNN state [layers, N, H]
            row = lambda t, i: t[:, i]
        if t_after.is_floating_point():
            assert torch.equal(row(t_after, 1), row(t_fresh, 1).to(t_after.dtype))
        for i in (0, 2):
            assert torch.equal(row(t_after, i), row(t_before, i))
    assert [t.data_ptr() for t in tree.flatten(agent.hidden) if isinstance(t, torch.Tensor)] == ids, \
        "reset must reuse the buffers a captured graph points at"
