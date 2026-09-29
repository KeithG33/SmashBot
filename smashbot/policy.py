"""Policy: embed -> tx_like core -> autoregressive controller head.

PyTorch port of slippi_ai/tf/policies.py, using slippi-ai's Frames/StateAction
NamedTuples with torch tensors as leaves. All sequence tensors are batch-major
(B, T, ...).
"""

import typing as tp

import numpy as np
import torch
import tree
from torch import nn

from slippi_ai.types import Frames, StateAction

from smashbot import delay as delay_lib
from smashbot.embed import JointStickEmbedding, stick_read, stick_regions
from smashbot.heads import ControllerHead, SampleOutputs
from smashbot.networks import RecurrentState, StateActionNetwork


class UnrollOutputs(tp.NamedTuple):
    log_probs: torch.Tensor  # [B, T]
    distances: tp.Any  # controller struct of [B, T]
    final_state: RecurrentState
    logits: tp.Any = None  # controller struct of [B, T, ...]; used by RL
    sticks: tp.Any = None  # stick name -> log-probs over its buckets [B, T, K]; eval


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
        energy: tp.Optional["EnergyScore"] = None,
        energy_weight: float = 0.0,
    ) -> tuple[torch.Tensor, RecurrentState, tp.Any]:
        """frames: [B, U + D + 1] raw (not yet delay-aligned). The loss, the
        final state and the per-component distances, all left on device:
        imitation_metrics reads them out when they are wanted. With `energy`,
        the loss adds energy_weight x the joint sticks' energy score; the
        distances stay the plain negative log-probs."""
        delayed = delay_lib.slice_delayed_frames(frames, self.delay)
        outputs = self.unroll(delayed, initial_state)
        loss = -outputs.log_probs.mean()
        if energy is not None:
            target = tree.map_structure(lambda t: t[:, 1:], delayed.state_action.action)
            loss = loss + energy_weight * energy(outputs.logits, target)
        return loss, outputs.final_state, outputs.distances

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

    def stick(name, short):   # a grid stick's two axes, or a joint stick's one choice
        value = controller[name]
        return {f"{short}_x": value.x, f"{short}_y": value.y} if hasattr(value, "x") else {short: value}

    return {
        "policy_loss": values[0],
        "controller": controller,
        "controller_flat": {
            "buttons": sum(buttons) / len(buttons),
            **stick("main_stick", "main"),
            **stick("c_stick", "c"),
            "shoulder": controller["shoulder"],
        },
    }


def bucket_reads(stick) -> np.ndarray:
    """The position Melee reads from each of a stick's buckets [K, 2]: a
    joint stick's decode points, or a grid's bucket pairs x-major (a grid
    bucket decodes to a multiple of 160 / (size - 1) on the -80..80 scale)."""
    if isinstance(stick, JointStickEmbedding):
        return stick_read(stick.positions)
    position = np.arange(stick.x.size) * 160 / (stick.x.size - 1) - 80
    return stick_read(np.stack(np.meshgrid(position, position, indexing="ij"), -1).reshape(-1, 2))


class StickScorer:
    """Eval's stick scores, per stick, from unroll's log-probabilities over
    its buckets, the human's bucket and the human's exact position (replay
    sticks are the game's own reads):
      top1: the model's most likely bucket is the human's;
      same_action: the model's most likely bucket reads on the same side of
        every line the game checks as the human's stick (stick_regions);
      same_read: the game reads the model's most likely bucket exactly as it
        read the human's stick;
      read_distance: the probability-weighted distance between what the game
        reads from each bucket and the human's stick (full tilt = 80).
    same_action, same_read and read_distance don't depend on the encoding,
    so they compare encodings."""

    def __init__(self, controller_head, device):
        struct = controller_head.embed_struct
        self.sticks = {name: getattr(struct, name) for name in ("main_stick", "c_stick") if hasattr(struct, name)}
        self.reads = {name: torch.tensor(bucket_reads(stick), dtype=torch.float32, device=device)
                      for name, stick in self.sticks.items()}
        self.regions = {name: torch.tensor(stick_regions(name), dtype=torch.long, device=device)
                        for name in self.sticks}

    def _bucket(self, name, human):
        if isinstance(self.sticks[name], JointStickEmbedding):
            return human.long()
        return human.x.long() * self.sticks[name].x.size + human.y.long()

    def score(self, log_probs: dict, target, exact: dict) -> dict:
        """[top1, same_action, same_read, read_distance] per stick, averaged
        over the frames, on device."""
        scores = {}
        for name, log_p in log_probs.items():
            reads, position = self.reads[name], exact[name].float()
            region = lambda xy: self.regions[name][xy[..., 0].long() + 80, xy[..., 1].long() + 80]
            guess = log_p.argmax(-1)
            miss = torch.linalg.vector_norm(reads - position.unsqueeze(-2), dim=-1)
            scores[name] = torch.stack([
                (guess == self._bucket(name, getattr(target, name))).float().mean(),
                (region(reads[guess]) == region(exact[name])).float().mean(),
                (reads[guess] == position).all(-1).float().mean(),
                (log_p.exp() * miss).sum(-1).mean(),
            ])
        return scores


class EnergyScore:
    """Each joint stick's energy score against the human's bucket (Gneiting
    and Raftery 2007): E|X - y| - E|X - X'| / 2 under the model's
    distribution over bucket decode points, Euclidean distance in full-tilt
    units. Probability near the human's bucket costs less than probability
    far away, and the score is still lowest at the true distribution. With
    Q = P @ D, E|X - y| is Q at the human's bucket and E|X - X'| is P . Q."""

    def __init__(self, controller_head, device):
        struct = controller_head.embed_struct
        points = {name: getattr(struct, name) for name in ("main_stick", "c_stick")
                  if isinstance(getattr(struct, name, None), JointStickEmbedding)}
        if not points:
            raise ValueError("the energy score needs joint sticks: a STICK_TABLES controller type")
        self.distance = {}
        for name, stick in points.items():
            xy = torch.tensor(stick.positions, dtype=torch.float32, device=device) / 80
            self.distance[name] = torch.cdist(xy, xy)

    def __call__(self, logits, target) -> torch.Tensor:
        """The sticks' energy scores, each averaged over the frames, summed."""
        total = 0.0
        for name, distance in self.distance.items():
            p = torch.softmax(getattr(logits, name).float(), dim=-1)
            q = p @ distance
            human = getattr(target, name).long().unsqueeze(-1)
            total = total + (q.gather(-1, human).squeeze(-1) - 0.5 * (p * q).sum(-1)).mean()
        return total


def stick_metrics(batches: list) -> dict:
    """StickScorer.score of equal-sized batches, averaged and named, read in
    one host transfer."""
    if not batches or not batches[0]:
        return {}
    names = list(batches[0])
    values = torch.stack([torch.stack([b[n] for b in batches]).mean(0) for n in names]).tolist()
    return {f"{name}/{score}": v for name, row in zip(names, values)
            for score, v in zip(("top1", "same_action", "same_read", "read_distance"), row)}


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
