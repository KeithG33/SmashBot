"""Policy: embed -> tx_like core -> autoregressive controller head.

PyTorch port of slippi_ai/tf/policies.py, using slippi-ai's Frames/StateAction
NamedTuples with torch tensors as leaves. All sequence tensors are batch-major
(B, T, ...).
"""

import functools
import typing as tp

import numpy as np
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
    sticks: tp.Any = None  # stick name -> joint log p(x, y) [B, T, X, Y]; eval


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
        joint_sticks: bool = False,
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
            sticks=(self.controller_head.stick_log_probs(outputs, prev_action, next_action)
                    if joint_sticks else None),
        )

    def imitation_loss(
        self,
        frames: Frames,
        initial_state: RecurrentState,
    ) -> tuple[torch.Tensor, RecurrentState, tp.Any]:
        """frames: [B, U + D + 1] raw (not yet delay-aligned). The loss, the
        final state and the per-component distances, all left on device:
        imitation_metrics reads them out when they are wanted."""
        delayed = delay_lib.slice_delayed_frames(frames, self.delay)
        outputs = self.unroll(delayed, initial_state)
        return -outputs.log_probs.mean(), outputs.final_state, outputs.distances

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

    # the league grids sample in fp32 behind an fp16 trunk: the ported
    # super-gm's logits reach ~1400, where fp16 rounds in steps of 1
    fp32_head: bool = False

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
        if self.fp32_head:
            with torch.autocast(output.device.type, enabled=False):
                next_action = self.controller_head.sample(
                    output.float(), state_action.action, temperature=temperature
                )
        else:
            next_action = self.controller_head.sample(
                output, state_action.action, temperature=temperature
            )
        return next_action, final_state


def imitation_metrics(loss: torch.Tensor, distances) -> dict:
    """imitation_loss's loss and each controller component's mean distance as
    floats, read in one host transfer."""
    components = distances._asdict()
    means = [d.mean().float() for d in tree.flatten(components)]
    values = torch.stack([loss.detach().float()] + means).tolist()
    controller = tree.unflatten_as(components, values[1:])
    buttons = controller["buttons"]
    return {
        "policy_loss": values[0],
        "controller": controller,
        "controller_flat": {
            "buttons": sum(buttons) / len(buttons),
            "main_x": controller["main_stick"].x,
            "main_y": controller["main_stick"].y,
            "c_x": controller["c_stick"].x,
            "c_y": controller["c_stick"].y,
            "shoulder": controller["shoulder"],
        },
    }


@functools.lru_cache
def _stick_tables(size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """For every pair of `size` axis buckets: which distinct stick value Melee
    reads from it, and the distance between each two pairs' read values. A
    bucket decodes to a -80..80 position; the game clamps it to the unit
    circle, then reads |k| <= 22 on either axis as 0."""
    position = np.arange(size) * 160 / (size - 1) - 80
    xy = np.stack(np.meshgrid(position, position, indexing="ij"), -1).reshape(-1, 2)
    radius = np.hypot(xy[:, 0], xy[:, 1])[:, None]
    read = np.trunc(np.where(radius > 80, xy * 80 / radius, xy))
    read = np.where(np.abs(read) <= 22, 0, read)
    _, outcome = np.unique(read, axis=0, return_inverse=True)
    read = torch.tensor(read, dtype=torch.float32, device=device)
    return torch.tensor(outcome.reshape(-1), device=device), torch.cdist(read, read)


def stick_scores(sticks: dict, target) -> dict:
    """Per stick, from unroll's joint log p(x, y) and the human's controller:
    [top-1, same outcome, miss distance] averaged over the frames, on device.
    Top-1: the model's most likely bucket pair is the human's. Same outcome:
    the game reads it as the same stick value as the human's. Miss distance:
    the probability-weighted distance between the value the game reads from
    each bucket pair and from the human's (full tilt = 80)."""
    scores = {}
    for name, log_p in sticks.items():
        size = log_p.shape[-1]
        outcome, distance = _stick_tables(size, log_p.device)
        stick = getattr(target, name)
        human = stick.x.long() * size + stick.y.long()
        flat = log_p.flatten(-2)
        guess = flat.argmax(-1)
        scores[name] = torch.stack([
            (guess == human).float().mean(),
            (outcome[guess] == outcome[human]).float().mean(),
            (flat.exp() * distance[human]).sum(-1).mean(),
        ])
    return scores


def stick_metrics(batches: list) -> dict:
    """stick_scores of equal-sized batches, averaged and named, read in one
    host transfer."""
    if not batches or not batches[0]:
        return {}
    names = list(batches[0])
    values = torch.stack([torch.stack([b[n] for b in batches]).mean(0) for n in names]).tolist()
    return {f"{name}/{score}": v for name, row in zip(names, values)
            for score, v in zip(("top1", "same_outcome", "miss_distance"), row)}


def build_policy_from_config(cfg: dict) -> Policy:
    """A policy from a checkpoint's saved config dict."""
    from smashbot import configs, embed as embed_lib

    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=cfg["head"]["axis_spacing"],
            shoulder_spacing=cfg["head"]["shoulder_spacing"],
            type=cfg["head"].get("controller_type", "default"),
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
