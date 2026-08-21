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


def _poisoned_learner(N=2, T=5) -> tuple[Learner, Trajectory]:
    """Learner whose first policy parameter's gradient is NaN-poisoned via a
    backward hook, plus a real rollout-shaped trajectory to step on."""
    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    value = _tiny_value()
    learner = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1e9)), policy, teacher, value
    )

    param = next(policy.parameters())
    param.register_hook(lambda g: g * float("nan"))

    from smashbot.tests.test_rollouts import _fake_record  # noqa: F401
    from smashbot.rl.rollouts import ChunkAssembler
    from smashbot.rl.agent import BatchedPolicyAgent

    agent = BatchedPolicyAgent(policy, num_envs=N)
    asm = ChunkAssembler(unroll_length=T, delay=policy.delay)
    rng = np.random.default_rng(3)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]
    t, traj = 0, None
    while traj is None:
        if t > 0:
            asm.push_reward(torch.zeros(N))
        snap = agent.hidden_snapshot() if t % T == 0 else None
        states = _rand_states(game_embed, (N,), rng)
        _, records, _ = agent.step(states)
        asm.push_frame(records[0], torch.zeros(N, dtype=torch.bool), snap)
        if asm.ready():
            traj = asm.emit()
        t += 1
    return learner, traj


@pytest.mark.parametrize("scaled", [False, True], ids=["fp32", "fp16-scaler"])
def test_nonfinite_gradients_never_reach_weights(scaled):
    """A batch that produces NaN gradients (even with finite loss) must be
    skipped entirely — step 705 of rl-pool-v3 NaN'd every policy weight
    because only the LOSS was checked. Poison a gradient mid-update via a
    backward hook and assert weights stay finite and unchanged.

    The scaled arm exercises the fp16 GradScaler code path (cpu autocast
    fp16 is not the production path, but the scale->backward->unscale_->
    guard->update composition is device-agnostic): the guard must still
    catch the poison AFTER unscale_, and update() must halve the scale."""
    learner, traj = _poisoned_learner()
    if scaled:
        # attach the scaler exactly as fp16-on-cuda mode would create it
        learner.grad_scaler = torch.amp.GradScaler("cpu", init_scale=2.0 ** 16)

    import copy as copy_lib

    before = copy_lib.deepcopy(learner.policy.state_dict())
    state = learner.initial_state(2)
    learner.step([traj], state)
    for k, v in learner.policy.state_dict().items():
        assert torch.isfinite(v.float()).all(), f"nonfinite weight {k}"
        assert torch.equal(v, before[k]), f"weights changed despite NaN grads: {k}"
    if scaled:
        # unscale_ recorded found_inf; the guard skipped, update() reacted
        assert learner.grad_scaler.get_scale() == 2.0 ** 15


# ------------------------------------------------- learner precision (fp16)


def test_fp32_mode_is_golden():
    """precision="fp32" (and the default) must be the pre-fp16 code path
    byte-for-byte: no autocast object is ever constructed, no GradScaler
    exists, and a full step from identical seeds lands on bit-identical
    weights whether precision was defaulted or passed explicitly."""
    import contextlib

    results = []
    for kwargs in ({}, {"precision": "fp32"}):
        learner, traj = _make_learner(
            ppo=PPOConfig(max_mean_actor_kl=1.0), **kwargs
        )
        assert learner.precision == "fp32"
        assert learner.grad_scaler is None
        assert isinstance(learner._autocast(), contextlib.nullcontext)
        torch.manual_seed(123)
        _, metrics = learner.step([traj], learner.initial_state(3))
        results.append((learner.policy.state_dict(), metrics))
    (sd_a, m_a), (sd_b, m_b) = results
    assert m_a["post_update"]["loss"] == m_b["post_update"]["loss"]
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"fp32 golden mismatch: {k}"


def test_fp16_on_cpu_falls_back_loudly(capsys):
    """fp16 is the cuda production path only; on cpu the learner must fall
    back to fp32 with a printed warning (no autocast, no scaler)."""
    learner, traj = _make_learner(
        precision="fp16", ppo=PPOConfig(max_mean_actor_kl=1.0)
    )
    out = capsys.readouterr().out
    assert "WARNING" in out and "fp16" in out and "falling back to fp32" in out
    assert learner.precision == "fp32"
    assert learner.grad_scaler is None
    # and it still trains normally
    _, metrics = learner.step([traj], learner.initial_state(3))
    assert np.isfinite(metrics["post_update"]["loss"])


def test_unknown_precision_rejected():
    with pytest.raises(AssertionError):
        _make_learner(precision="bf16")


def test_precision_flag_plumbs_through_tyro():
    """train_rl's CLI must accept --learner.precision fp16."""
    import tyro

    from smashbot.rl.train_rl import Config

    cfg = tyro.cli(Config, args=["--learner.precision", "fp16"])
    assert cfg.learner.precision == "fp16"
    cfg = tyro.cli(Config, args=[])
    assert cfg.learner.precision == "fp32"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and __import__("os").environ.get("SMASHBOT_GPU_TESTS")),
    reason="cuda-only; set SMASHBOT_GPU_TESTS=1 on an IDLE gpu "
           "(never beside a live training run)",
)
def test_fp16_dtype_receipts_cuda():
    """Under precision="fp16" on cuda: policy/teacher unroll logits are
    float16 (autocast is real), while every sensitive quantity — log-probs,
    advantages, actor log-probs, the loss — and the ENTIRE value net
    (head in/out) stay float32."""
    policy = _tiny_policy(seed=0)
    traj = _rollout(policy)  # rollout on cpu fp32, as in production
    to_cuda = lambda s: tree.map_structure(
        lambda t: t.to("cuda") if isinstance(t, torch.Tensor) else t, s
    )
    traj = to_cuda(traj)
    policy = policy.to("cuda")
    teacher = _tiny_policy(seed=0).to("cuda")
    value = _tiny_value().to("cuda")
    learner = Learner(
        RLConfig(precision="fp16", ppo=PPOConfig(max_mean_actor_kl=1.0)),
        policy, teacher, value,
    )
    assert learner.precision == "fp16" and learner.grad_scaler is not None

    captured = {}
    orig_unroll = policy.unroll

    def _capturing_unroll(*a, **k):
        out = orig_unroll(*a, **k)
        captured["out"] = out
        return out

    policy.unroll = _capturing_unroll
    head_dtypes = []
    hook = value.head.register_forward_hook(
        lambda mod, inp, out: head_dtypes.append((inp[0].dtype, out.dtype))
    )
    try:
        fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(3, "cuda"))
        loss, _ = learner._policy_loss(fixed)
    finally:
        policy.__dict__.pop("unroll", None)
        hook.remove()

    out = captured["out"]
    assert tree.flatten(out.logits)[0].dtype == torch.float16
    assert tree.flatten(fixed.teacher_logits)[0].dtype == torch.float16
    assert out.log_probs.dtype == torch.float32
    assert fixed.advantages.dtype == torch.float32
    assert fixed.actor_log_probs.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert head_dtypes and all(
        din == torch.float32 and dout == torch.float32
        for din, dout in head_dtypes
    ), f"value net touched by autocast: {head_dtypes}"

    # and a full scaled step runs to completion with finite metrics
    _, metrics = learner.step([traj], learner.initial_state(3, "cuda"))
    assert np.isfinite(metrics["post_update"]["loss"])
    assert learner.grad_scaler.get_scale() > 0


@pytest.mark.parametrize("k", [2, 3])
def test_micro_batches_give_identical_update(k):
    """Chunked policy pass with exact valid-weighted accumulation produces the
    same parameters after one step as the full-batch pass (fp32, no scaler);
    the loss/metric means agree too."""
    torch.manual_seed(0)
    full, traj = _make_learner(learning_rate=1e-3, micro_batches=1)
    torch.manual_seed(0)
    chunked, _ = _make_learner(learning_rate=1e-3, micro_batches=k)
    # make the update non-trivial: perturb both policies identically away
    # from the actor so ratios != 1
    with torch.no_grad():
        for (a, b) in zip(full.policy.parameters(), chunked.policy.parameters()):
            noise = torch.randn_like(a) * 1e-2
            a.add_(noise); b.add_(noise)
    st_f = full.initial_state(traj.rewards.shape[0])
    st_c = chunked.initial_state(traj.rewards.shape[0])
    _, mf = full.step([traj], st_f)
    _, mc = chunked.step([traj], st_c)
    for pa, pb in zip(full.policy.parameters(), chunked.policy.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)
    assert mf["reverted"] == mc["reverted"]
    assert mf["post_update"]["actor_kl_mean"] == pytest.approx(
        mc["post_update"]["actor_kl_mean"], rel=1e-5, abs=1e-9)
