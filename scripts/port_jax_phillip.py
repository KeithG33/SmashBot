"""Port a JAX-era RL Phillip (a slippi-ai pickle from its jax trainer, like
models/gm-big) to a SmashBot checkpoint: tx_like with the enhanced embed and
the custom_v1 controller, conditioned on the rating its RL run fed it. Reads
the pickle's arrays directly (no JAX), refuses configs the port does not
cover, and records the tech mask its observations were trained under.

    python scripts/port_jax_phillip.py <pickle> <out.pt>
"""
import dataclasses
import pickle
import sys

import numpy as np
import torch

from smashbot import configs, embed as embed_lib
from smashbot.policy import build_policy
from smashbot.train_bc import TrainConfig

SUPPORTED_ENHANCED = dict(
    item_mlp_layers=2, use_self_nana=True, use_controller_rnn=False, use_learned_char=True,
    use_learned_action=True, use_char_action_joint=True, use_item_sum=True, use_items=True,
    hybrid_embed=False)
SUPPORTED_PLAYER = dict(xy_scale=0.05, shield_scale=0.01, with_speeds=False, with_controller=False,
                        with_nana=True, legacy_jumps_left=False)
SUPPORTED_CUSTOM_V1 = dict(c_stick_config=dict(n_radius_buckets=2, n_angle_buckets=[4, 8]),
                           main_stick_config=dict(n_radius_buckets=3, n_angle_buckets=[4, 16, 64]))


class _Opaque:
    """Stands in for the slippi-ai classes a pickle names (its enums)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Opaque


def load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return _Unpickler(f).load()


def _require(what: str, got: dict, want: dict) -> None:
    wrong = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if wrong:
        raise ValueError(f"{what} not covered by this port (got, supported): {wrong}")


def train_config(saved: dict, rating: float) -> TrainConfig:
    """Our config for the pickle's architecture, after checking it is one we port."""
    net = saved["network"]
    tx = net["tx_like"]
    enhanced = net["embed"]["enhanced"]
    if net["name"] != "tx_like" or tx["recurrent_layer"] != "lstm" or net["embed"]["name"] != "enhanced":
        raise ValueError(f"not a tx_like LSTM with the enhanced embed: {net['name']}, {net['embed']['name']}")
    _require("tx_like", tx, dict(activation="gelu", layer_norm="nnx"))
    _require("enhanced embed", enhanced, SUPPORTED_ENHANCED)
    _require("player embed", saved["embed"]["player"], SUPPORTED_PLAYER)
    _require("embed", saved["embed"], dict(with_randall=True, with_fod=True, with_rating=True))
    controller = saved["embed"]["controller"]
    if controller["type"] != "custom_v1":
        raise ValueError(f"controller {controller['type']} is not custom_v1")
    _require("custom_v1", controller["custom_v1"], SUPPORTED_CUSTOM_V1)
    if saved["observation"]["frame_skip"]["skip"]:
        raise ValueError("frame-skipped observations are not covered")
    head = saved["controller_head"]["autoregressive"]
    return TrainConfig(
        network=configs.NetworkConfig(
            name="tx_like", hidden_size=tx["hidden_size"], num_layers=tx["num_layers"],
            ffw_multiplier=tx["ffw_multiplier"], recurrent_layer="lstm",
            ln_eps=1e-6,   # flax nnx.LayerNorm's default epsilon
            gelu_approximate=tx["gelu_approximate"], embed="enhanced",
            embed_hidden_size=enhanced["hidden_size"], rating=rating,
            embed_joint_index_wraps=True),   # as slippi-ai computed it in training
        head=configs.ControllerHeadConfig(residual_size=head["residual_size"],
                                          component_depth=head["component_depth"],
                                          controller_type="custom_v1"),
        policy=configs.PolicyConfig(delay=saved["policy"]["delay"]),
        data=dataclasses.replace(TrainConfig().data, max_names=saved["max_names"]),
    )


def _get(tree: dict, *keys):
    for key in keys:
        tree = tree[key] if key in tree else tree[int(key)]
    return tree


def torch_state(policy: dict, num_layers: int) -> dict[str, torch.Tensor]:
    """The pickle's policy arrays under our names: flax kernels are [in, out],
    torch weights [out, in]; flax's LSTM has input kernels without bias and
    hidden kernels with, per gate (i, f, g, o in torch's order)."""
    w = lambda a: torch.from_numpy(np.ascontiguousarray(np.asarray(a).T))
    a = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x)))
    sd = {}

    def linear(name: str, node: dict):
        sd[name + ".weight"], sd[name + ".bias"] = w(node["kernel"]), a(node["bias"])

    embed = _get(policy, "network", "_embed_module")
    for ours, theirs in (("embed_action", ("_embed_action",)), ("embed_char", ("_embed_char",)),
                         ("embed_char_action", ("_embed_char_action", "_embed"))):
        sd[f"network.enhanced.{ours}.weight"] = a(_get(embed, *theirs, "embedding"))
    for i in ("0", "2"):
        linear(f"network.enhanced.item_mlp.{i}", _get(embed, "_item_mlp", "layers", i))

    layers = _get(policy, "network", "_network", "_layers")
    linear("network.core._layers.0._module", _get(layers, "0", "_module"))
    for i in range(1, 2 * num_layers + 1):
        prefix = f"network.core._layers.{i}"
        if i % 2:
            cell = _get(layers, str(i), "_net", "_core")
            gates = ("i", "f", "g", "o")
            sd[f"{prefix}._net._core.weight_ih_l0"] = torch.cat(
                [w(cell["if_" if g == "f" else "i" + g]["kernel"]) for g in gates])
            sd[f"{prefix}._net._core.weight_hh_l0"] = torch.cat([w(cell["h" + g]["kernel"]) for g in gates])
            sd[f"{prefix}._net._core.bias_hh_l0"] = torch.cat([a(cell["h" + g]["bias"]) for g in gates])
            sd[f"{prefix}._net._core.bias_ih_l0"] = torch.zeros_like(sd[f"{prefix}._net._core.bias_hh_l0"])
        else:
            block = _get(layers, str(i), "_module")
            sd[f"{prefix}._module.block.0.weight"] = a(block["layernorm"]["scale"])
            sd[f"{prefix}._module.block.0.bias"] = a(block["layernorm"]["bias"])
            linear(f"{prefix}._module.block.1", block["linear1"])
            linear(f"{prefix}._module.block.3", block["linear2"])

    head = _get(policy, "_controller_head")
    linear("controller_head.to_residual", head["to_residual"])
    blocks = head["res_blocks"]
    for j in range(len(blocks)):
        block = _get(blocks, str(j))
        for k in ("0", "2", "4"):
            linear(f"controller_head.res_blocks.{j}.encoder.{k}", _get(block, "_encoder", "layers", k))
        linear(f"controller_head.res_blocks.{j}.decoder", block["decoder"])
    return sd


def port(src: str, dst: str) -> None:
    saved = load_pickle(src)
    rating = saved["rl_config"]["agent"]["rating"]
    config = train_config(saved["config"], rating)
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(type="custom_v1"),
        network_config=config.network, head_config=config.head,
        policy_config=config.policy, num_names=config.data.max_names)
    weights = torch_state(saved["state"]["policy"], config.network.num_layers)
    policy.load_state_dict({**weights, "network.enhanced.rating": policy.network.enhanced.rating}, strict=True)
    torch.save({
        "config": {**dataclasses.asdict(config),
                   "observation": {"tech_mask_window": saved["config"]["observation"]["animation"]["tech_mask_window"]}},
        "state": {"policy": policy.state_dict(), "name_map": {}, "step": saved["step"], "ported_from": src},
        "best_eval_loss": None,
        "version": 1,
    }, dst)
    print(f"{src} -> {dst}: step {saved['step']}, rating {rating}, delay {config.policy.delay}, "
          f"{sum(v.numel() for v in policy.state_dict().values()) / 1e6:.1f}M params")


if __name__ == "__main__":
    port(*sys.argv[1:3])
