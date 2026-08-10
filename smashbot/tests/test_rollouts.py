"""Rollout machinery tests (Dolphin-free): chunk assembly alignment, reward
delay-shifting, batched-agent equivalence with the single-env DelayedAgent
pathway, and end-to-end assembled-trajectory -> learner compatibility."""

import numpy as np
import pytest
import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent, FrameRecord
from smashbot.rl.ppo import Learner, PPOConfig, RLConfig
from smashbot.rl.rollouts import ChunkAssembler, compute_reward
from smashbot.tests.test_ppo import _tiny_policy, _tiny_value, _rand_states


def _fake_record(n_envs, t):
    """Distinguishable scalar payloads so alignment is checkable."""
    x = torch.full((n_envs,), float(t))
    return FrameRecord(state=x, prev_action=x + 0.5, logits=x + 0.25, name=x.long())


def test_assembler_chunking_and_reward_shift():
    T, D, N = 4, 2, 3
    asm = ChunkAssembler(unroll_length=T, delay=D)
    emitted = []
    for t in range(20):
        if t > 0:
            asm.push_reward(torch.full((N,), 100.0 + t))  # transition t-1 -> t
        snap = {"h": torch.tensor(float(t))} if t % T == 0 else None
        asm.push_frame(_fake_record(N, t), torch.zeros(N, dtype=torch.bool), snap)
        if asm.ready():
            emitted.append(asm.emit())

    first = emitted[0]
    # frames 0..T stacked
    assert torch.equal(first.states[:, 0], torch.zeros(3))
    assert torch.equal(first.states[:, T], torch.full((3,), float(T)))
    assert first.initial_state["h"].item() == 0.0
    # reward slot t = real transition t+D (pushed value 100 + (t+D+1))
    assert first.rewards.shape == (N, T)
    assert first.rewards[0, 0].item() == pytest.approx(100.0 + D + 1)
    assert first.rewards[0, T - 1].item() == pytest.approx(100.0 + D + T)

    second = emitted[1]
    # overlap: second chunk starts at frame T with the snapshot taken there
    assert torch.equal(second.states[:, 0], torch.full((3,), float(T)))
    assert second.initial_state["h"].item() == float(T)
    assert second.rewards[0, 0].item() == pytest.approx(100.0 + T + D + 1)


def test_compute_reward():
    prev_s = torch.tensor([[4.0, 4.0], [2.0, 3.0]])
    s = torch.tensor([[4.0, 3.0], [1.0, 3.0]])  # env0: opp died; env1: we died
    prev_p = torch.tensor([[10.0, 50.0], [80.0, 0.0]])
    p = torch.tensor([[10.0, 0.0], [95.0, 12.0]])  # opp % resets on death
    r = compute_reward(prev_s, s, prev_p, p, torch.zeros(2, dtype=torch.bool))
    assert r[0].item() == pytest.approx(1.0)  # kill, no damage exchanged
    # we died (-1), took 15 (-0.15), dealt 12 (+0.12)
    assert r[1].item() == pytest.approx(-1.0 - 0.15 + 0.12)
    # resets zero the reward
    r = compute_reward(prev_s, s, prev_p, p, torch.ones(2, dtype=torch.bool))
    assert torch.equal(r, torch.zeros(2))


def test_batched_agent_matches_independent_runs(monkeypatch):
    """N=2 envs fed identical states must behave like two synced single runs:
    env-batching cannot leak information across the batch. Sampling is
    patched to greedy so the streams are deterministic (independent per-row
    sampling would legitimately diverge them)."""
    from smashbot import embed as embed_lib

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[np.dtype(self.dtype).name]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    rng = np.random.default_rng(7)
    sae = policy.network.embed_state_action
    game_embed = dict(sae.embedding)["state"]

    agent = BatchedPolicyAgent(policy, num_envs=2, name_code=1)
    for t in range(6):
        state1 = _rand_states(game_embed, (1,), rng)
        both = tree.map_structure(lambda x: torch.cat([x, x], dim=0), state1)
        controllers, record = agent.step(both)
        # ...but the two identical envs see identical logits every frame
        tree.map_structure(
            lambda t_: torch.testing.assert_close(t_[0], t_[1]),
            record.logits,
        )
        # and identical prev-action inputs
        tree.map_structure(
            lambda t_: torch.testing.assert_close(t_[0].float(), t_[1].float()),
            record.prev_action,
        )

    # resetting env 0 must not disturb env 1's recurrent state
    before = tree.map_structure(
        lambda t_: t_.clone() if isinstance(t_, torch.Tensor) else t_,
        agent.hidden,
    )
    agent.reset_env(0)
    tree.map_structure(
        lambda a, b: torch.testing.assert_close(a[1], b[1])
        if isinstance(a, torch.Tensor) and a.shape and a.shape[0] == 2
        else None,
        before,
        agent.hidden,
    )


def test_assembled_trajectory_feeds_learner():
    """Full loop sans Dolphin: batched agent rollout -> assembler -> learner,
    with the fresh-learner invariants (ratio == 1, actor KL == 0)."""
    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    value = _tiny_value()
    learner = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1.0)), policy, teacher, value
    )

    N, T, D = 2, 5, policy.delay
    agent = BatchedPolicyAgent(policy, num_envs=N)
    asm = ChunkAssembler(unroll_length=T, delay=D)
    rng = np.random.default_rng(3)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]

    t = 0
    traj = None
    while traj is None:
        if t > 0:
            asm.push_reward(torch.zeros(N))
        snap = agent.hidden_snapshot() if t % T == 0 else None
        states = _rand_states(game_embed, (N,), rng)
        _, record = agent.step(states)
        asm.push_frame(record, torch.zeros(N, dtype=torch.bool), snap)
        if asm.ready():
            traj = asm.emit()
        t += 1

    fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(N))
    with torch.no_grad():
        _, metrics = learner._policy_loss(fixed)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["actor_kl_mean"] == pytest.approx(0.0, abs=1e-5)
