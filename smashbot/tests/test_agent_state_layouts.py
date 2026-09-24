"""Serving-side state handling across the three core layouts: SGU window
caches, hybrid (SGU + recurrent) and torch-RNN (tx_like) state.

- fp16 serving stores only the window caches in fp16; recurrent memory
  (LSTM h/c, GRU h) stays fp32.
- LeagueAgent.move_cell moves one seat for every layout (torch RNN state is
  [layers, N, H], not batch-first).
"""
import numpy as np
import pytest
import torch
import tree

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy
from smashbot.rl.agent import LeagueAgent
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
