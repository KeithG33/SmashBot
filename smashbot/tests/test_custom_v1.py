"""The custom_v1 action space and the enhanced embed, as the ported big RL
Phillips use them (parity with slippi-ai's own forward is checked against
its checkout, not here)."""
import numpy as np
import torch
import tree

from slippi_ai.types import StateAction
from smashbot import configs, custom_v1, embed as embed_lib
from smashbot.networks import BACKWARD_TECH, FORWARD_TECH, NEUTRAL_TECH
from smashbot.policy import build_policy
from smashbot.rl.agent import LeagueAgent
from smashbot.tests.test_packed_embed import _rand_input


def all_labels():
    buttons, main_stick = np.meshgrid(np.arange(728), np.arange(85), indexing="ij")
    return custom_v1.ControllerV1(buttons.ravel().astype(np.uint16), main_stick.ravel().astype(np.uint16))


def test_every_label_survives_decode_then_bucket():
    bucketer = custom_v1.Config().create_bucketer()
    assert bucketer.sizes == (728, 85)
    labels = all_labels()
    back = bucketer.bucket(bucketer.decode(labels))
    assert np.array_equal(back.buttons, labels.buttons)
    assert np.array_equal(back.main_stick, labels.main_stick)


def test_decoded_sticks_stay_inside_the_gate():
    controller = custom_v1.Config().create_bucketer().decode(all_labels())
    for stick in (controller.main_stick, controller.c_stick):
        raw = np.hypot(custom_v1.stick_to_raw(stick.x), custom_v1.stick_to_raw(stick.y))
        assert raw.max() <= custom_v1.MAX_RADIUS + 1


def _big_phillip_like(hidden=16, joint_index_wraps=False, rating=2500.0, tech_mask_window=0):
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(type="custom_v1"),
        network_config=configs.NetworkConfig(
            name="tx_like", hidden_size=32, num_layers=2, ffw_multiplier=2, recurrent_layer="lstm",
            ln_eps=1e-6, gelu_approximate=True, embed="enhanced", embed_hidden_size=hidden,
            rating=rating, embed_joint_index_wraps=joint_index_wraps,
            tech_mask_window=tech_mask_window),
        head_config=configs.ControllerHeadConfig(residual_size=8, component_depth=2, controller_type="custom_v1"),
        policy_config=configs.PolicyConfig(delay=21),
        num_names=0,
    )


def test_enhanced_embed_layout():
    """Per player 2H + 14 and per Nana 2H + 15 (the learned action and
    character vectors plus the scalar and small one-hot leaves), stage and
    platforms 68, the summed items H, the rating 1, the previous custom_v1
    action 728 + 85."""
    hidden = 16
    policy = _big_phillip_like(hidden)
    assert policy.network.enhanced.output_size == 2 * (4 * hidden + 29) + 68 + hidden + 1 + 813


def test_joint_index_wraps_as_slippi_ai_computes_it():
    for wraps in (False, True):
        enhanced = _big_phillip_like(joint_index_wraps=wraps).network.enhanced
        with torch.no_grad():
            enhanced.embed_char_action.weight[:, 0] = torch.arange(len(enhanced.embed_char_action.weight))
            enhanced.embed_action.weight.zero_()
        char, action = torch.tensor([2, 9, 20]), torch.tensor([100, 14, 300])
        offset = char * 399 % 256 if wraps else char * 399
        player = enhanced._leaves["sa"]["state"].dummy((3,)).p0
        raw = tree.map_structure(torch.as_tensor, player)._replace(character=char, action=action)
        row = enhanced._player_or_nana(raw, enhanced._leaves["player"], nana=False)[:, 4]   # action block, column 0
        assert torch.equal(row, (offset + action).float())


def test_policy_samples_controllers_the_sim_can_take():
    policy = _big_phillip_like()
    encoded = policy.network.encode(StateAction(
        state=policy.network.embed_game.dummy((3, 5)),
        action=custom_v1.Config().create_bucketer().decode(
            custom_v1.ControllerV1(np.full((3, 5), 100, np.uint16), np.full((3, 5), 40, np.uint16))),
        name=np.zeros((3, 5), np.int32)))
    sa = tree.map_structure(torch.as_tensor, encoded)
    outputs, _ = policy.network.unroll(sa, torch.zeros(3, 5, dtype=torch.bool), policy.initial_state(3))
    assert outputs.shape == (3, 5, 32)
    step = tree.map_structure(lambda t: t[:, 0], sa)
    sample, _ = policy.sample(step, policy.initial_state(3))
    controller = policy.controller_head.controller_embedding.decode(
        tree.map_structure(lambda t: t.numpy(), sample.controller_state))
    assert controller.main_stick.x.shape == (3,) and controller.buttons.A.dtype == bool


def test_tech_mask_hides_the_direction_for_its_window():
    net = _big_phillip_like(tech_mask_window=4).network
    opponent = [14, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH,
                BACKWARD_TECH, NEUTRAL_TECH, 14]
    seen = []
    state = net.initial_state(1)["tech_mask"]
    game = tree.map_structure(torch.as_tensor, net.embed_game.dummy((1,)))
    for action in opponent:
        sa = StateAction(state=game._replace(p1=game.p1._replace(action=torch.tensor([action], dtype=torch.int32))),
                         action=None, name=None)
        sa, state = net._mask_tech(sa, *state)
        seen.append(int(sa.state.p1.action))
    assert seen == [14, NEUTRAL_TECH, NEUTRAL_TECH, NEUTRAL_TECH, NEUTRAL_TECH, FORWARD_TECH,
                    NEUTRAL_TECH, NEUTRAL_TECH, 14]


def test_grid_serves_each_phillip_as_it_would_alone():
    """Two big-Phillip-like slices on the league grid's forward (on CPU the
    same vmap that CUDA captures): each cell's logits are its own model's,
    run alone on the same frames with the prev action the grid used and the
    component the grid sampled; tech mask and recurrent state carried, a
    reset mid-game; sampled actions decode to controller rows."""
    torch.manual_seed(0)
    phillips = [_big_phillip_like(rating=r, tech_mask_window=4).eval() for r in (1400.0, 2800.0)]
    S, N = 2, 2
    grid = LeagueAgent(phillips[0], S, N, name_code=0, device="cpu")
    for s, p in enumerate(phillips):
        grid.load_slice(s, p.state_dict())
    rng = np.random.default_rng(1)
    solo = [[p.initial_state(1) for _ in range(N)] for p in phillips]
    opponent = [14, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH, FORWARD_TECH, 14, 14]
    for frame, action in enumerate(opponent):
        views = _rand_input(phillips[0].network.embed_game, (S, N), rng)
        views = views._replace(p1=views.p1._replace(action=torch.full((S, N), action, dtype=torch.int32)))
        resets = torch.zeros(S, N, dtype=torch.bool)
        resets[:] = frame == 0
        resets[1, 0] |= frame == 4
        if resets[1, 0] and frame:
            grid.reset_cell(1, 0)
        record = grid.launch(views, resets)
        grid.settle()
        for s, policy in enumerate(phillips):
            for n in range(N):
                row = s * N + n
                cell = lambda t: t[s, n][None]
                sa = StateAction(state=tree.map_structure(cell, views),
                                 action=tree.map_structure(lambda t: t[row][None], record.prev_action),
                                 name=torch.zeros(1, dtype=torch.int32))
                with torch.no_grad():
                    out, solo[s][n] = policy.network.step_with_reset(sa, resets[s, n][None], solo[s][n])
                    sampled = tree.map_structure(cell, grid._prev)
                    want = policy.controller_head.distance(out, sa.action, sampled).logits
                for got, ref in zip(tree.flatten(record.logits), tree.flatten(want)):
                    assert torch.allclose(got[row], ref[0], atol=1e-5), (frame, s, n)
    # one decoded controller row queued per frame; the reset refilled cell (1, 0)'s delay
    frames = [len(opponent), len(opponent), len(opponent) - 4, len(opponent)]
    assert [len(q) - phillips[0].delay for q in grid._queues] == frames
    assert grid._queues[0][-1].shape == grid._neutral_row.shape


def test_phillip_tiers_group_into_grids_by_architecture():
    from smashbot.rl.sim_league import group_by_architecture

    def small():
        return build_policy(
            embed_config=embed_lib.EmbedConfig(), controller_config=embed_lib.ControllerConfig(),
            network_config=configs.NetworkConfig(name="tx_like", hidden_size=32, num_layers=2),
            head_config=configs.ControllerHeadConfig(residual_size=8), policy_config=configs.PolicyConfig(delay=21),
            num_names=16)

    tiers = {"medium": small(), "gm-big": _big_phillip_like(rating=2500.0, tech_mask_window=4),
             "gold": small(), "super-gm": _big_phillip_like(rating=2800.0, tech_mask_window=4)}
    assert group_by_architecture(tiers) == [["medium", "gold"], ["gm-big", "super-gm"]]
