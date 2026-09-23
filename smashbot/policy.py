"""Policy: embed -> tx_like core -> autoregressive controller head.

PyTorch port of slippi_ai/tf/policies.py, using slippi-ai's Frames/StateAction
NamedTuples with torch tensors as leaves. All sequence tensors are batch-major
(B, T, ...).
"""

import typing as tp

import torch
import tree
from torch import nn

from slippi_ai.types import Frames, StateAction

from smashbot import delay as delay_lib
from smashbot.heads import ControllerHead, SampleOutputs
from smashbot.networks import RecurrentState, StateActionNetwork


class UnrollOutputs(tp.NamedTuple):
    log_probs: torch.Tensor  # [B, T]
    distances: tp.Any  # controller struct of [B, T]
    final_state: RecurrentState
    logits: tp.Any = None  # controller struct of [B, T, ...]; used by RL


class Policy(nn.Module):
    def __init__(
        self,
        network: StateActionNetwork,
        controller_head: ControllerHead,
        delay: int = 0,
    ):
        super().__init__()
        self.network = network
        self.controller_head = controller_head
        self.delay = delay
        # value is a separate network; checkpoints from the built-in head's era
        # (and the ported Phillips) still carry its two tensors
        self.register_load_state_dict_pre_hook(
            lambda module, state, prefix, *_: [
                state.pop(k) for k in list(state) if k.startswith(prefix + "value_head.")])

    def initial_state(self, batch_size: int, device=None) -> RecurrentState:
        return self.network.initial_state(batch_size, device)

    def unroll(
        self,
        frames: Frames,
        initial_state: RecurrentState,
    ) -> UnrollOutputs:
        """Frames must already be delay-aligned (see delay.slice_delayed_frames)
        and include one extra overlap frame at the end."""
        inputs = tree.map_structure(lambda t: t[:, :-1], frames.state_action)
        outputs, final_state = self.network.unroll(
            inputs, frames.is_resetting[:, :-1], initial_state
        )

        action = frames.state_action.action
        prev_action = tree.map_structure(lambda t: t[:, :-1], action)
        next_action = tree.map_structure(lambda t: t[:, 1:], action)

        distance_outputs = self.controller_head.distance(outputs, prev_action, next_action)
        policy_loss = sum(tree.flatten(distance_outputs.distance))
        return UnrollOutputs(
            log_probs=-policy_loss,
            distances=distance_outputs.distance,
            final_state=final_state,
            logits=distance_outputs.logits,
        )

    def imitation_loss(
        self,
        frames: Frames,
        initial_state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        """frames: [B, U + D + 1] raw (not yet delay-aligned)."""
        delayed = delay_lib.slice_delayed_frames(frames, self.delay)
        outputs = self.unroll(delayed, initial_state)

        total_loss = -outputs.log_probs.mean()
        metrics = {
            "policy_loss": total_loss.item(),
            "controller": tree.map_structure(
                lambda d: d.mean().item(), outputs.distances._asdict()
            ),
        }
        metrics["controller_flat"] = {
            "buttons": sum(metrics["controller"]["buttons"]) / len(metrics["controller"]["buttons"]),
            "main_x": metrics["controller"]["main_stick"].x,
            "main_y": metrics["controller"]["main_stick"].y,
            "c_x": metrics["controller"]["c_stick"].x,
            "c_y": metrics["controller"]["c_stick"].y,
            "shoulder": metrics["controller"]["shoulder"],
        }

        return total_loss, outputs.final_state, metrics

    @torch.no_grad()
    def forward(
        self,
        state_action: StateAction,
        initial_state: RecurrentState,
        is_resetting: tp.Optional[torch.Tensor] = None,
        temperature: tp.Optional[float] = None,
    ) -> tuple[SampleOutputs, RecurrentState]:
        """forward == sample, so torch.func.functional_call (which dispatches
        to forward) can run inference with stacked per-slot parameters.
        Calls the CLASS method, not self.sample: train_rl monkeypatches
        instances with torch.compile'd wrappers, and vmap tracing through a
        compiled wrapper blows dynamo's per-code-object cache (which is shared
        with the student's compiled sample) and stalls cudagraph trees —
        live-caught as a 22GB OOM. The capture path wants pure eager here."""
        return Policy.sample(self, state_action, initial_state, is_resetting, temperature)

    def sample(
        self,
        state_action: StateAction,  # [B], encoded
        initial_state: RecurrentState,
        is_resetting: tp.Optional[torch.Tensor] = None,
        temperature: tp.Optional[float] = None,
    ) -> tuple[SampleOutputs, RecurrentState]:
        if is_resetting is None:
            stage = state_action.state.stage
            is_resetting = torch.zeros(
                stage.shape[0], dtype=torch.bool, device=stage.device
            )

        output, final_state = self.network.step_with_reset(
            state_action, is_resetting, initial_state
        )
        next_action = self.controller_head.sample(
            output, state_action.action, temperature=temperature
        )
        return next_action, final_state


def build_policy_from_config(cfg: dict) -> Policy:
    """A policy from a checkpoint's saved config dict."""
    from smashbot import configs, embed as embed_lib

    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=cfg["head"]["axis_spacing"],
            shoulder_spacing=cfg["head"]["shoulder_spacing"],
        ),
        network_config=configs.from_dict(configs.NetworkConfig, cfg["network"]),
        head_config=configs.from_dict(configs.ControllerHeadConfig, cfg["head"]),
        policy_config=configs.from_dict(configs.PolicyConfig, cfg["policy"]),
        num_names=cfg["data"]["max_names"],
    )


def build_policy(
    embed_config,
    controller_config,
    network_config,
    head_config,
    policy_config,
    num_names: int,
) -> Policy:
    from smashbot.networks import build_embed_network

    controller_embedding = controller_config.make_embedding()
    network = build_embed_network(
        embed_config=embed_config,
        controller_embedding=controller_embedding,
        num_names=num_names,
        network_config=network_config,
    )
    from smashbot.heads import AutoRegressive

    head = AutoRegressive(
        embed_controller=controller_embedding,
        input_size=network.core.output_size,
        residual_size=head_config.residual_size,
        component_depth=head_config.component_depth,
    )
    return Policy(
        network=network,
        controller_head=head,
        delay=policy_config.delay,
    )
