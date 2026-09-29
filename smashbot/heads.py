"""PyTorch port of slippi-ai's controller heads (vendor: slippi_ai/tf/controller_heads.py).

The AutoRegressive head samples controller components in the declared order of
the controller StructEmbedding (buttons, main_stick x, y, c_stick x, y,
shoulder), each conditioned on previously sampled components via a residual
stream. Training uses teacher forcing (the target feeds the residual).
"""

import abc
import typing as tp

import torch
from torch import nn

from smashbot.embed import Embedding, StructEmbedding


class SampleOutputs(tp.NamedTuple):
    controller_state: tp.Any  # sampled controller struct (encoded)
    logits: tp.Any  # struct of logits


class DistanceOutputs(tp.NamedTuple):
    distance: tp.Any  # struct of negative log-probs
    logits: tp.Any


class ControllerHead(nn.Module, abc.ABC):
    @abc.abstractmethod
    def sample(self, inputs, prev_controller_state, temperature=None) -> SampleOutputs:
        ...

    @abc.abstractmethod
    def distance(self, inputs, prev_controller_state, target_controller_state) -> DistanceOutputs:
        ...

    @property
    @abc.abstractmethod
    def controller_embedding(self) -> StructEmbedding:
        ...


def _make_mlp(input_size: int, hidden_size: int, depth: int, output_size: int) -> nn.Module:
    layers: list[nn.Module] = []
    in_size = input_size
    for _ in range(depth):
        layers.append(nn.Linear(in_size, hidden_size))
        layers.append(nn.ReLU())
        in_size = hidden_size
    layers.append(nn.Linear(in_size, output_size))
    return nn.Sequential(*layers)


class AutoRegressiveComponent(nn.Module):
    """One controller component in the residual stream."""

    def __init__(self, embedder: Embedding, residual_size: int, depth: int = 0):
        super().__init__()
        self.embedder = embedder
        self.encoder = _make_mlp(
            residual_size + embedder.size, residual_size, depth, embedder.size
        )
        # a single Linear decoding a one-hot has full expressive power
        self.decoder = nn.Linear(embedder.size, residual_size)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def _logits(self, residual, prev_raw):
        prev_embedding = self.embedder(prev_raw)
        return self.encoder(torch.cat([residual, prev_embedding], dim=-1))

    def sample(self, residual, prev_raw, temperature=None):
        logits = self._logits(residual, prev_raw)
        sample = self.embedder.sample(logits, temperature=temperature)
        residual = residual + self.decoder(self.embedder(sample))
        return residual, SampleOutputs(controller_state=sample, logits=logits)

    def distance(self, residual, prev_raw, target_raw):
        logits = self._logits(residual, prev_raw)
        distance = self.embedder.distance(logits, target_raw)
        # auto-regress on the target (teacher forcing)
        residual = residual + self.decoder(self.embedder(target_raw))
        return residual, DistanceOutputs(distance=distance, logits=logits)


class AutoRegressive(ControllerHead):
    """Samples components sequentially, conditioned on past samples."""

    def __init__(
        self,
        embed_controller: StructEmbedding,
        input_size: int,
        residual_size: int = 128,
        component_depth: int = 2,
    ):
        super().__init__()
        self.embed_controller = embed_controller
        self.to_residual = nn.Linear(input_size, residual_size)
        self.embed_struct = embed_controller.map(lambda e: e)
        self.embed_flat = list(embed_controller.flatten(self.embed_struct))
        self.res_blocks = nn.ModuleList(
            [
                AutoRegressiveComponent(e, residual_size, component_depth)
                for e in self.embed_flat
            ]
        )

    @property
    def controller_embedding(self) -> StructEmbedding:
        return self.embed_controller

    def sample(self, inputs, prev_controller_state, temperature=None):
        residual = self.to_residual(inputs)
        prev_flat = self.embed_controller.flatten(prev_controller_state)

        sample_outputs: list[SampleOutputs] = []
        for res_block, prev in zip(self.res_blocks, prev_flat):
            residual, sample = res_block.sample(residual, prev, temperature=temperature)
            sample_outputs.append(sample)

        samples, logits = zip(*sample_outputs)
        return SampleOutputs(
            controller_state=self.embed_controller.unflatten(iter(samples)),
            logits=self.embed_controller.unflatten(iter(logits)),
        )

    def distance(self, inputs, prev_controller_state, target_controller_state):
        residual = self.to_residual(inputs)
        prev_flat = self.embed_controller.flatten(prev_controller_state)
        target_flat = self.embed_controller.flatten(target_controller_state)

        distance_outputs: list[DistanceOutputs] = []
        for res_block, prev, target in zip(self.res_blocks, prev_flat, target_flat):
            residual, distance = res_block.distance(residual, prev, target)
            distance_outputs.append(distance)

        distances, logits = zip(*distance_outputs)
        return DistanceOutputs(
            distance=self.embed_controller.unflatten(iter(distances)),
            logits=self.embed_controller.unflatten(iter(logits)),
        )

    def stick_log_probs(self, inputs, prev_controller_state, target_controller_state) -> dict:
        """Each stick's joint log p(x, y) over every pair of axis buckets,
        [..., x, y], with the components before it teacher-forced as in
        distance. distance only scores y given the human's x; this runs the y
        component once per x, so a metric can score the pair the model
        would pick."""
        sticks = {name: getattr(self.embed_struct, name) for name in ("main_stick", "c_stick")
                  if hasattr(self.embed_struct, name)}
        first_axis = {next(i for i, e in enumerate(self.embed_flat) if e is stick.x): name
                      for name, stick in sticks.items()}
        residual = self.to_residual(inputs)
        prev_flat = list(self.embed_controller.flatten(prev_controller_state))
        target_flat = list(self.embed_controller.flatten(target_controller_state))
        joint = {}
        for i, (block, prev, target) in enumerate(zip(self.res_blocks, prev_flat, target_flat)):
            if i in first_axis:
                y_block, y_prev = self.res_blocks[i + 1], prev_flat[i + 1]
                assert y_block.embedder is sticks[first_axis[i]].y
                n = block.embedder.size
                every_x = block.decoder(block.embedder(torch.arange(n, device=residual.device)))
                y_logits = y_block._logits(
                    residual.unsqueeze(-2) + every_x, y_prev.unsqueeze(-1).expand(*y_prev.shape, n))
                log_px = torch.log_softmax(block._logits(residual, prev).float(), dim=-1)
                joint[first_axis[i]] = log_px.unsqueeze(-1) + torch.log_softmax(y_logits.float(), dim=-1)
            residual = residual + block.decoder(block.embedder(target))
        return joint
