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

from shinebot import delay as delay_lib
from shinebot.networks import RecurrentState, StateActionNetwork


class ValueFunction(nn.Module):
    def __init__(self, network: StateActionNetwork):
        super().__init__()
        self.network = network
        self.head = nn.Linear(network.core.output_size, 1)

    def initial_state(self, batch_size: int, device=None) -> RecurrentState:
        return self.network.initial_state(batch_size, device)

    def loss(
        self,
        frames: Frames,  # delay-sliced, [U+1, B]
        initial_state: RecurrentState,
        discount: float,
    ) -> tuple[torch.Tensor, RecurrentState, dict]:
        inputs = tree.map_structure(lambda t: t[:-1], frames.state_action)
        last_input = tree.map_structure(lambda t: t[-1], frames.state_action)
        outputs, final_state = self.network.unroll(
            inputs, frames.is_resetting[:-1], initial_state
        )
        values = self.head(outputs).squeeze(-1)
        last_output, _ = self.network.step_with_reset(
            last_input, frames.is_resetting[-1], final_state
        )
        last_value = self.head(last_output).squeeze(-1)

        discounts = torch.where(
            frames.is_resetting[1:], 0.0,
            torch.as_tensor(discount, device=values.device),
        )
        targets = delay_lib.discounted_returns(
            rewards=frames.reward, discounts=discounts, bootstrap=last_value
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
