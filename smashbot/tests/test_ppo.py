"""PPO learner tests: clip math, KL/entropy leaf math, and full update steps
on synthetic trajectories rolled out by a real (tiny) policy."""

import numpy as np
import pytest
import torch
import tree

from slippi_ai.types import StateAction

from smashbot import configs, embed as embed_lib
from smashbot.networks import build_embed_network
from smashbot.policy import build_policy
from smashbot.rl.ppo import (
    ActionData,
    Learner,
    PPOConfig,
    RLConfig,
    Trajectory,
    clipped_surrogate,
)
from smashbot.value import ValueFunction


# ---------------------------------------------------------------- pure math


def test_clipped_surrogate_trust_region():
    eps = 0.01
    adv = torch.ones(4)
    # inside the region: objective follows the ratio exactly
    lr = torch.tensor([0.0, 0.005, -0.005, 0.009], requires_grad=True)
    obj = clipped_surrogate(lr, adv, eps)
    assert torch.allclose(obj, lr.exp() * adv)

    # beyond +eps with positive advantage: clipped -> zero gradient
    lr = torch.tensor([0.05], requires_grad=True)
    clipped_surrogate(lr, torch.ones(1), eps).sum().backward()
    assert lr.grad.abs().item() == 0.0

    # beyond +eps with NEGATIVE advantage: unclipped branch is the minimum,
    # gradient stays live (pessimistic bound punishes harmful drift)
    lr = torch.tensor([0.05], requires_grad=True)
    clipped_surrogate(lr, -torch.ones(1), eps).sum().backward()
    assert lr.grad.abs().item() > 0.0


def test_leaf_kl_entropy_match_torch_distributions():
    torch.manual_seed(0)
    # Bernoulli (BoolEmbedding): logits shaped [..., 1]
    b = embed_lib.BoolEmbedding()
    x, y = torch.randn(5, 1), torch.randn(5, 1)
    db = torch.distributions.Bernoulli(logits=x.squeeze(-1))
    db2 = torch.distributions.Bernoulli(logits=y.squeeze(-1))
    assert torch.allclose(b.logits_entropy(x), db.entropy(), atol=1e-6)
    assert torch.allclose(
        b.logits_kl(x, y), torch.distributions.kl_divergence(db, db2), atol=1e-6
    )

    # Categorical (OneHotEmbedding/DiscreteEmbedding)
    o = embed_lib.DiscreteEmbedding(16)
    p, q = torch.randn(5, 17), torch.randn(5, 17)
    dc = torch.distributions.Categorical(logits=p)
    dc2 = torch.distributions.Categorical(logits=q)
    assert torch.allclose(o.logits_entropy(p), dc.entropy(), atol=1e-6)
    assert torch.allclose(
        o.logits_kl(p, q), torch.distributions.kl_divergence(dc, dc2), atol=1e-6
    )
    # KL(p, p) == 0
    assert torch.allclose(o.logits_kl(p, p), torch.zeros(5), atol=1e-6)


# ------------------------------------------------------- synthetic rollouts


def _tiny_policy(seed=0):
    torch.manual_seed(seed)
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(
            name="sgu", num_layers=1, hidden_size=64, num_heads=1, window=4
        ),
        head_config=configs.ControllerHeadConfig(residual_size=32, component_depth=0),
        policy_config=configs.PolicyConfig(delay=2),
        num_names=4,
    )
    policy.train_value_head = False  # separate value net, as in production
    return policy


def _tiny_value(seed=1):
    torch.manual_seed(seed)
    return ValueFunction(
        build_embed_network(
            embed_config=embed_lib.EmbedConfig(),
            controller_embedding=embed_lib.ControllerConfig().make_embedding(),
            num_names=4,
            network_config=configs.NetworkConfig(
                name="sgu", num_layers=1, hidden_size=32, num_heads=1, window=4
            ),
        )
    )


def _rand_states(embed_state, shape, rng):
    """Random encoded Game struct (reuses the packed-embed test generator)."""
    from smashbot.tests.test_packed_embed import _rand_input

    return _rand_input(embed_state, shape, rng)


def _rollout(policy, B=3, T=8, seed=0) -> Trajectory:
    """Roll the policy itself over random states to get a coherent trajectory
    (actions and logits actually produced by the network)."""
    rng = np.random.default_rng(seed)
    game_embed = None
    # states leaf of the state_action embedding tree
    sae = policy.network.embed_state_action
    for key, e in sae.embedding:
        if key == "state":
            game_embed = e
    states = _rand_states(game_embed, (B, T + 1), rng)
    name = torch.zeros(B, T + 1, dtype=torch.int64)

    prev_stream, logits_list = [], []
    hidden = policy.initial_state(B)
    prev = tree.map_structure(
        lambda t: t[:, 0], _rand_action(policy, B, rng)
    )
    with torch.no_grad():
        for t in range(T + 1):
            state_t = tree.map_structure(lambda x: x[:, t], states)
            sa = StateAction(state=state_t, action=prev, name=name[:, t])
            prev_stream.append(prev)  # the agent's INPUT at frame t
            out, hidden = policy.sample(sa, hidden)
            act = tree.map_structure(
                lambda x: x.clone() if x.dtype == torch.bool else x.long().clone(),
                out.controller_state,
            )
            logits_list.append(out.logits)  # logits that sampled â_t
            prev = act

    stack = lambda seq: tree.map_structure(
        lambda *xs: torch.stack(xs, dim=1), *seq
    )
    return Trajectory(
        states=states,
        name=name,
        actions=ActionData(
            controller_state=stack(prev_stream), logits=stack(logits_list)
        ),
        rewards=torch.from_numpy(
            rng.normal(0, 0.1, (B, T)).astype(np.float32)
        ),
        is_resetting=torch.zeros(B, T + 1, dtype=torch.bool),
        initial_state=policy.initial_state(B),
    )


def _rand_action(policy, B, rng):
    embed_controller = policy.controller_head.controller_embedding
    from smashbot.tests.test_packed_embed import _rand_input

    return _rand_input(embed_controller, (B, 1), rng)


def _make_learner(**config_kwargs) -> tuple[Learner, Trajectory]:
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)  # same init: KL(policy || teacher) == 0
    value = _tiny_value()
    config = RLConfig(**config_kwargs)
    learner = Learner(config, policy, teacher, value)
    traj = _rollout(policy)
    return learner, traj


def test_first_step_ratios_are_one():
    """Before any update, the learner's policy == the actor: ratio == 1 and
    actor KL == 0, so the surrogate equals the mean advantage."""
    learner, traj = _make_learner()
    fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(3))
    with torch.no_grad():
        _, metrics = learner._policy_loss(fixed)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["actor_kl_mean"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["teacher_kl"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["surrogate"] == pytest.approx(
        fixed.advantages.mean().item(), abs=1e-4
    )


def test_step_runs_and_updates():
    learner, traj = _make_learner(
        ppo=PPOConfig(num_epochs=2, max_mean_actor_kl=1.0)  # never revert
    )
    before = copy_params(learner.policy)
    state, metrics = learner.step([traj], learner.initial_state(3))
    assert not metrics["reverted"]
    assert np.isfinite(metrics["post_update"]["loss"])
    assert any(
        not torch.equal(a, b) for a, b in zip(before, copy_params(learner.policy))
    ), "policy parameters should have moved"
    assert len(metrics["epochs"]) == 2


def test_oversized_step_reverts():
    learner, traj = _make_learner(
        learning_rate=1.0,  # absurd on purpose
        ppo=PPOConfig(num_epochs=1, max_mean_actor_kl=1e-6),
    )
    before = copy_params(learner.policy)
    _, metrics = learner.step([traj], learner.initial_state(3))
    assert metrics["reverted"]
    for a, b in zip(before, copy_params(learner.policy)):
        assert torch.equal(a, b), "revert must restore parameters exactly"


def test_sequential_steps_carry_state():
    """Two learner steps with carried recurrent state: the second must not
    backward into the first chunk's freed graph (regression: carried value
    state needed detaching)."""
    learner, traj = _make_learner(ppo=PPOConfig(max_mean_actor_kl=1.0))
    state = learner.initial_state(3)
    state, _ = learner.step([traj], state)
    state, metrics = learner.step([traj], state)  # crashed before the fix
    assert np.isfinite(metrics["post_update"]["loss"])


def test_teacher_stays_frozen():
    learner, traj = _make_learner(ppo=PPOConfig(max_mean_actor_kl=1.0))
    before = copy_params(learner.teacher)
    learner.step([traj], learner.initial_state(3))
    for a, b in zip(before, copy_params(learner.teacher)):
        assert torch.equal(a, b)


def copy_params(module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]
