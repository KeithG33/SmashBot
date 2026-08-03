"""Separate value network (production slippi-ai config).

A small independent network (default: 1-layer tx_like/512) with its own
embedding, predicting discounted returns from the same delay-sliced frames the
policy trains on. Kept separate from the policy so value gradients never shape
the policy trunk; at M9 this becomes the RL advantage estimator.
"""

import typing as tp

import torch
import tree
from torch import nn

from slippi_ai.types import Frames

from smashbot import delay as delay_lib
from smashbot.networks import RecurrentState, StateActionNetwork


class ValueFunction(nn.Module):
    def __init__(self, network: StateActionNetwork):
        super().__init__()
        self.network = network
        self.head = nn.Linear(network.core.output_size, 1)

    def initial_state(self, batch_size: int, device=None) -> RecurrentState:
        return self.network.initial_state(batch_size, device)

    def loss(
        self,
        frames: Frames,  # delay-sliced, [B, U+1]
        initial_state: RecurrentState,
        discount: float,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        inputs = tree.map_structure(lambda t: t[:, :-1], frames.state_action)
        last_input = tree.map_structure(lambda t: t[:, -1], frames.state_action)
        outputs, final_state = self.network.unroll(
            inputs, frames.is_resetting[:, :-1], initial_state
        )
        values = self.head(outputs).squeeze(-1)
        last_output, _ = self.network.step_with_reset(
            last_input, frames.is_resetting[:, -1], final_state
        )
        last_value = self.head(last_output).squeeze(-1)

        # Return recursion and regression in fp32 even under bf16 autocast:
        # an 80-step serial accumulation is where low precision actually hurts.
        with torch.autocast(values.device.type, enabled=False):
            values = values.float()
            last_value = last_value.float()
            rewards = frames.reward.float()
            discounts = torch.where(
                frames.is_resetting[:, 1:], 0.0,
                torch.as_tensor(discount, device=values.device),
            )
            targets = delay_lib.discounted_returns(
                rewards=rewards, discounts=discounts, bootstrap=last_value
            ).detach()
            loss = torch.square(targets - values).mean()
            uev = loss / (targets.var() + 1e-8)

        metrics = {
            "loss": loss.item(),
            "uev": uev.item(),
            "return_mean": targets.mean().item(),
            "reward_mean": frames.reward.mean().item(),
        }
        return loss, final_state, metrics
