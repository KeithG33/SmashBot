"""The custom_v1 action space and the enhanced embed, as the ported big RL
Phillips use them (parity with slippi-ai's own forward is checked against
its checkout, not here)."""
import numpy as np
import torch
import tree

from slippi_ai.types import StateAction
from smashbot import configs, custom_v1, embed as embed_lib
from smashbot.policy import build_policy


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


def _big_phillip_like(hidden=16, joint_index_wraps=False):
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(type="custom_v1"),
        network_config=configs.NetworkConfig(
            name="tx_like", hidden_size=32, num_layers=2, ffw_multiplier=2, recurrent_layer="lstm",
            ln_eps=1e-6, gelu_approximate=True, embed="enhanced", embed_hidden_size=hidden,
            rating=2500.0, embed_joint_index_wraps=joint_index_wraps),
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
