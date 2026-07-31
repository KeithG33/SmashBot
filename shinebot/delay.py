"""Single source of truth for all reaction-delay index math.

The policy acts with a fixed delay D: the action executed at frame t was
decided from state at frame t - D. slippi-ai's convention (verbatim from
their Policy.imitation_loss):

  With delay D and unroll length U, a trajectory chunk has U + D + 1 frames.
  States [0, U-1] predict actions [D+1, U+D], with previous actions
  [D, U+D-1]. The final hidden state comes from states [0, U-1]; state U
  would bootstrap the value function.

`slice_delayed_frames` performs exactly that slicing; the resulting Frames
are then consumed by Policy.unroll, which shifts actions by one internally
(prev = action[:-1], target = action[1:]).
"""

import typing as tp

import torch
import tree

from slippi_ai.types import Frames, StateAction


def slice_delayed_frames(frames: Frames, delay: int) -> Frames:
    """Aligns states with delayed actions. Input frames are time-major [T, B]."""
    state_action = frames.state_action
    total_frames = state_action.state.stage.shape[0]
    unroll_length = total_frames - delay  # includes the +1 overlap frame

    return Frames(
        state_action=StateAction(
            state=tree.map_structure(lambda t: t[:unroll_length], state_action.state),
            action=tree.map_structure(lambda t: t[delay:], state_action.action),
            name=state_action.name[delay:],
        ),
        is_resetting=frames.is_resetting[:unroll_length],
        # Only use rewards that follow actions.
        reward=frames.reward[delay:],
    )


def discounted_returns(
    rewards: torch.Tensor,  # [T, B]
    discounts: torch.Tensor,  # [T, B]
    bootstrap: torch.Tensor,  # [B]
) -> torch.Tensor:
    """returns[t] = rewards[t] + discounts[t] * returns[t+1], seeded by bootstrap."""
    returns = torch.empty_like(rewards)
    acc = bootstrap
    for t in reversed(range(rewards.shape[0])):
        acc = rewards[t] + discounts[t] * acc
        returns[t] = acc
    return returns
