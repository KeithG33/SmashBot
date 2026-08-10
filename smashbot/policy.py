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
    value_loss: tp.Optional[torch.Tensor]  # [B, T]; None when value head disabled
    value_metrics: dict
    final_state: RecurrentState
    logits: tp.Any = None  # controller struct of [B, T, ...]; used by RL


class Policy(nn.Module):
    def __init__(
        self,
        network: StateActionNetwork,
        controller_head: ControllerHead,
        delay: int = 0,
        train_value_head: bool = True,
    ):
        super().__init__()
        self.network = network
        self.controller_head = controller_head
        self.delay = delay
        self.train_value_head = train_value_head
        self.value_head = nn.Linear(network.core.output_size, 1)

    def initial_state(self, batch_size: int, device=None) -> RecurrentState:
        return self.network.initial_state(batch_size, device)

    def _value_outputs(
        self,
        outputs: torch.Tensor,  # [B, T, H], t in [0, T-1]
        last_input: StateAction,  # t = T
        is_resetting: torch.Tensor,  # [B, T+1]
        final_state: RecurrentState,
        rewards: torch.Tensor,  # [B, T]
        discount: float,
    ) -> tuple[torch.Tensor, dict]:
        if not self.train_value_head:
            outputs = outputs.detach()
        values = self.value_head(outputs).squeeze(-1)
        last_output, _ = self.network.step_with_reset(
            last_input, is_resetting[:, -1], final_state
        )
        last_value = self.value_head(last_output).squeeze(-1)

        discounts = torch.where(
            is_resetting[:, 1:], 0.0, torch.as_tensor(discount, device=rewards.device)
        )
        value_targets = delay_lib.discounted_returns(
            rewards=rewards, discounts=discounts, bootstrap=last_value
        ).detach()
        value_loss = torch.square(value_targets - values)

        uev = value_loss.mean() / (value_targets.var() + 1e-8)
        metrics = {
            "loss": value_loss.mean().item(),
            "uev": uev.item(),  # unexplained variance
        }
        return value_loss, metrics

    def unroll(
        self,
        frames: Frames,
        initial_state: RecurrentState,
        discount: float = 0.99,
    ) -> UnrollOutputs:
        """Frames must already be delay-aligned (see delay.slice_delayed_frames)
        and include one extra overlap frame at the end."""
        inputs = tree.map_structure(lambda t: t[:, :-1], frames.state_action)
        last_input = tree.map_structure(lambda t: t[:, -1], frames.state_action)
        outputs, final_state = self.network.unroll(
            inputs, frames.is_resetting[:, :-1], initial_state
        )

        action = frames.state_action.action
        prev_action = tree.map_structure(lambda t: t[:, :-1], action)
        next_action = tree.map_structure(lambda t: t[:, 1:], action)

        distance_outputs = self.controller_head.distance(outputs, prev_action, next_action)
        policy_loss = sum(tree.flatten(distance_outputs.distance))
        log_probs = -policy_loss

        # With a separate value network (production config), skip the built-in head.
        if self.train_value_head:
            value_loss, value_metrics = self._value_outputs(
                outputs, last_input, frames.is_resetting, final_state,
                frames.reward, discount,
            )
        else:
            value_loss, value_metrics = None, {}

        return UnrollOutputs(
            log_probs=log_probs,
            distances=distance_outputs.distance,
            value_loss=value_loss,
            value_metrics=value_metrics,
            final_state=final_state,
            logits=distance_outputs.logits,
        )

    def imitation_loss(
        self,
        frames: Frames,
        initial_state: RecurrentState,
        discount: float = 0.99,
        value_cost: float = 0.5,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        """frames: [B, U + D + 1] raw (not yet delay-aligned)."""
        delayed = delay_lib.slice_delayed_frames(frames, self.delay)
        outputs = self.unroll(delayed, initial_state, discount=discount)

        total_loss = -outputs.log_probs.mean()
        metrics = {
            "policy_loss": total_loss.item(),
            "value": outputs.value_metrics,
            "controller": tree.map_structure(
                lambda d: d.mean().item(), outputs.distances._asdict()
            ),
        }
        if self.train_value_head:
            total_loss = total_loss + value_cost * outputs.value_loss.mean()
        metrics["total_loss"] = total_loss.item()
        metrics["controller_flat"] = {
            "buttons": sum(metrics["controller"]["buttons"]) / 8,
            "main_x": metrics["controller"]["main_stick"].x,
            "main_y": metrics["controller"]["main_stick"].y,
            "c_x": metrics["controller"]["c_stick"].x,
            "c_y": metrics["controller"]["c_stick"].y,
            "shoulder": metrics["controller"]["shoulder"],
        }

        return total_loss, outputs.final_state, metrics

    @torch.no_grad()
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
