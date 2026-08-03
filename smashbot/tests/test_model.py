"""M3 gates: embeddings, tx_like network, autoregressive head, delay math."""

import numpy as np
import pytest
import torch
import tree

from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET
from slippi_ai.types import Frames, StateAction

from smashbot import configs, delay as delay_lib, embed as embed_lib
from smashbot.data import loader
from smashbot.heads import AutoRegressive
from smashbot.networks import SGUCore, TransformerCore, TransformerLike
from smashbot.policy import build_policy

DELAY = 4
UNROLL = 12


def make_policy(num_names: int = 2, delay: int = DELAY):
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(hidden_size=64, num_layers=2),
        head_config=configs.ControllerHeadConfig(residual_size=32, component_depth=1),
        policy_config=configs.PolicyConfig(delay=delay),
        num_names=num_names,
    )


@pytest.fixture(scope="module")
def toy_frames():
    cfg = configs.DataConfig(
        dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
        batch_size=4,
        unroll_length=UNROLL,
        num_workers=0,
        pin_memory=False,
    )
    sources = loader.make_sources(cfg, extra_frames=DELAY + 1)
    policy = make_policy()
    batch_with_meta, _ = next(sources.train)
    frames = loader.batch_to_frames(batch_with_meta.batch, policy.network)
    return policy, frames


def test_imitation_loss_smoke(toy_frames):
    policy, frames = toy_frames
    B, T = frames.is_resetting.shape
    assert T == UNROLL + DELAY + 1
    initial_state = policy.initial_state(B)
    loss, final_state, metrics = policy.imitation_loss(frames, initial_state)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    # buttons distance should be small-ish (mostly not pressed), sticks larger
    assert 0 < metrics["controller"]["buttons"].A < 2.0


def test_delay_slicing_indices():
    """Exact index semantics: state[k]=k, action[k]=k+D, reward[k]=k+D."""
    T, B = UNROLL + DELAY + 1, 2
    idx = np.arange(T, dtype=np.float32)[None, :].repeat(B, 0)
    idx_t = torch.from_numpy(idx)  # [B, T], value == time index

    frames = Frames(
        state_action=StateAction(state=idx_t.clone(), action=idx_t.clone(),
                                 name=idx_t.clone()),
        is_resetting=torch.zeros(B, T, dtype=torch.bool),
        reward=idx_t[:, :-1].clone(),
    )
    # state is normally a Game struct; slice_delayed_frames reads state.stage
    # for the length, so use a minimal NamedTuple (tree-compatible).
    import collections

    FakeState = collections.namedtuple("FakeState", ["stage"])
    frames = frames._replace(
        state_action=frames.state_action._replace(state=FakeState(stage=idx_t.clone()))
    )
    sliced = delay_lib.slice_delayed_frames(frames, DELAY)

    U1 = UNROLL + 1  # unroll length + overlap frame
    assert torch.equal(sliced.state_action.state.stage, idx_t[:, :U1])
    assert torch.equal(sliced.state_action.action, idx_t[:, DELAY:])
    assert torch.equal(sliced.state_action.name, idx_t[:, DELAY:])
    assert torch.equal(sliced.reward, idx_t[:, DELAY:-1])
    # states [0, U-1] predict actions [D+1, U+D] with prev actions [D, U+D-1]
    inputs_state = sliced.state_action.state.stage[:, :-1]
    prev_actions = sliced.state_action.action[:, :-1]
    target_actions = sliced.state_action.action[:, 1:]
    assert inputs_state[0, 0] == 0 and inputs_state[0, -1] == UNROLL - 1
    assert prev_actions[0, 0] == DELAY and prev_actions[0, -1] == UNROLL + DELAY - 1
    assert target_actions[0, 0] == DELAY + 1 and target_actions[0, -1] == UNROLL + DELAY


@pytest.mark.parametrize(
    "make_net",
    [
        lambda: TransformerLike(input_size=8, hidden_size=16, num_layers=2),
        lambda: TransformerCore(input_size=8, hidden_size=16, num_layers=2,
                                num_heads=2, window=24),
        lambda: SGUCore(input_size=8, hidden_size=16, num_layers=2, window=8),
    ],
    ids=["tx_like", "transformer", "sgu"],
)
def test_unroll_vs_step_equivalence(make_net):
    torch.manual_seed(0)
    net = make_net()
    T, B = 100, 3
    inputs = torch.randn(B, T, 8)
    reset = torch.zeros(B, T, dtype=torch.bool)
    reset[:, 0] = True
    reset[1, 57] = True  # mid-sequence reset for one element

    state = net.initial_state(B)
    unrolled, final_unroll = net.unroll(inputs, reset, state)

    state = net.initial_state(B)
    stepped = []
    for t in range(T):
        out, state = net.step_with_reset(inputs[:, t], reset[:, t], state)
        stepped.append(out)
    stepped = torch.stack(stepped, dim=1)

    torch.testing.assert_close(unrolled, stepped, atol=1e-5, rtol=1e-4)
    tree.map_structure(
        lambda a, b: torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4),
        final_unroll, state,
    )


def test_autoregressive_conditioning_direction():
    """Later components must depend on earlier targets, never vice versa."""
    torch.manual_seed(0)
    embed_controller = embed_lib.get_controller_embedding(axis_spacing=16)
    head = AutoRegressive(embed_controller, input_size=16, residual_size=8,
                          component_depth=1)
    # Decoders are zero-initialized (identity at init), which would mask the
    # conditioning path; randomize them so the coupling is observable.
    for block in head.res_blocks:
        torch.nn.init.normal_(block.decoder.weight, std=0.5)
    B = 5
    inputs = torch.randn(B, 16)
    prev = tree.map_structure(
        lambda x: torch.from_numpy(np.asarray(x)),
        embed_controller.dummy((B,)),
    )
    target = tree.map_structure(lambda t: t.clone(), prev)

    base = head.distance(inputs, prev, target)

    # Flip an EARLY component's target (button A): later logits must change.
    target_early = target._replace(
        buttons=target.buttons._replace(A=~target.buttons.A)
    )
    out_early = head.distance(inputs, prev, target_early)
    assert not torch.allclose(
        base.logits.main_stick.x, out_early.logits.main_stick.x
    ), "early target change must affect later component logits"

    # Flip the LAST component's target (shoulder): earlier logits must not change.
    target_late = target._replace(
        shoulder=(target.shoulder + 1) % 5
    )
    out_late = head.distance(inputs, prev, target_late)
    torch.testing.assert_close(base.logits.buttons.A, out_late.logits.buttons.A)
    torch.testing.assert_close(base.logits.main_stick.x, out_late.logits.main_stick.x)


def test_sample_distance_logit_consistency():
    """distance() with target == sampled controller reproduces sample()'s logits."""
    torch.manual_seed(0)
    embed_controller = embed_lib.get_controller_embedding(axis_spacing=16)
    head = AutoRegressive(embed_controller, input_size=16, residual_size=8)
    B = 4
    inputs = torch.randn(B, 16)
    prev = tree.map_structure(
        lambda x: torch.from_numpy(np.asarray(x)),
        embed_controller.dummy((B,)),
    )
    sampled = head.sample(inputs, prev)
    dist = head.distance(inputs, prev, sampled.controller_state)
    tree.map_structure(
        lambda a, b: torch.testing.assert_close(a, b),
        sampled.logits, dist.logits,
    )


def test_controller_roundtrip():
    from slippi_ai.types import Buttons, Controller, Stick

    embed_controller = embed_lib.get_controller_embedding(axis_spacing=16)
    rng = np.random.default_rng(0)
    bools = lambda: rng.random(7) < 0.5
    raw = Controller(
        main_stick=Stick(x=rng.random(7, dtype=np.float32),
                         y=rng.random(7, dtype=np.float32)),
        c_stick=Stick(x=rng.random(7, dtype=np.float32),
                      y=rng.random(7, dtype=np.float32)),
        shoulder=rng.random(7, dtype=np.float32),
        buttons=Buttons(*(bools() for _ in Buttons._fields)),
    )
    encoded = embed_controller.from_state(raw)
    decoded = embed_controller.decode(encoded)
    # sticks quantize to the 17-bin grid
    np.testing.assert_allclose(
        decoded.main_stick.x, np.round(raw.main_stick.x * 16) / 16, atol=1e-6
    )
    np.testing.assert_allclose(
        decoded.shoulder, np.round(raw.shoulder * 4) / 4, atol=1e-6
    )


def test_onehot_empty_policy_zeroes_invalid():
    e = embed_lib.OneHotEmbedding(
        "name", 4, dtype=np.int32, one_hot_policy=embed_lib.OneHotPolicy.EMPTY
    )
    t = torch.tensor([0, 3, 4, 7])
    out = e(t)
    assert out[0].sum() == 1 and out[1].sum() == 1
    assert out[2].sum() == 0 and out[3].sum() == 0


def test_param_count_production_size():
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(),  # 512 x 3, production
        head_config=configs.ControllerHeadConfig(),
        policy_config=configs.PolicyConfig(delay=18),
        num_names=16,
    )
    n = sum(p.numel() for p in policy.parameters())
    assert 3e6 < n < 30e6, f"unexpected param count {n}"


def test_transformer_window_horizon():
    """Events older than `window` frames must not influence the output."""
    torch.manual_seed(0)
    net = TransformerCore(input_size=4, hidden_size=16, num_layers=1,
                          num_heads=2, window=8)
    net.eval()
    B, T = 1, 30
    inputs_a = torch.randn(B, T, 4)
    inputs_b = inputs_a.clone()
    inputs_b[:, 0] += 100.0  # perturb a frame far outside the window
    reset = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        out_a, _ = net.unroll(inputs_a, reset, net.initial_state(B))
        out_b, _ = net.unroll(inputs_b, reset, net.initial_state(B))

    # last frame: frame 0 is 29 steps back, window is 8 -> identical outputs
    torch.testing.assert_close(out_a[:, -1], out_b[:, -1])
    # but frame 0 itself obviously differs
    assert not torch.allclose(out_a[:, 0], out_b[:, 0])


def test_sgu_hard_window_cutoff():
    """SGU's window is a strict per-layer horizon: with 1 layer / window 8,
    a frame 8+ steps back must have EXACTLY zero influence."""
    torch.manual_seed(0)
    net = SGUCore(input_size=4, hidden_size=16, num_layers=1, window=8)
    # zero-init makes mixing invisible; randomize so influence is observable
    for b in net.blocks:
        torch.nn.init.normal_(b.spatial.weight, std=0.5)
        torch.nn.init.normal_(b.mix_out.weight, std=0.5)
    net.eval()
    B, T = 1, 20
    inputs_a = torch.randn(B, T, 4)
    inputs_b = inputs_a.clone()
    inputs_b[:, 0] += 100.0
    reset = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        out_a, _ = net.unroll(inputs_a, reset, net.initial_state(B))
        out_b, _ = net.unroll(inputs_b, reset, net.initial_state(B))

    # frame 0 influences outputs 0..7 (window 8), and NOTHING after
    assert not torch.allclose(out_a[:, 7], out_b[:, 7])
    torch.testing.assert_close(out_a[:, 8], out_b[:, 8])
    torch.testing.assert_close(out_a[:, -1], out_b[:, -1])
