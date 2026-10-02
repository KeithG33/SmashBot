"""EnhancedEmbed starts from upstream's flax initializers, not torch's."""
import torch

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy


def _enhanced(hidden):
    torch.manual_seed(0)
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(name="sgu", hidden_size=32, num_layers=1,
                                             embed="enhanced", embed_hidden_size=hidden),
        head_config=configs.ControllerHeadConfig(residual_size=8, component_depth=2),
        policy_config=configs.PolicyConfig(),
        num_names=0,
    ).network.enhanced


def test_tables_and_item_mlp_start_at_flax_scale():
    for hidden in (128, 384):
        enhanced = _enhanced(hidden)
        for table in (enhanced.embed_char, enhanced.embed_action):
            assert abs(table.weight.std().item() * hidden ** 0.5 - 1) < 0.1
        assert not enhanced.embed_char_action.weight.any()
        for layer in enhanced.item_mlp:
            if isinstance(layer, torch.nn.Linear):
                assert abs(layer.weight.std().item() * layer.weight.shape[1] ** 0.5 - 1) < 0.1
                assert not layer.bias.any()
