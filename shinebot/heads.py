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

from shinebot.embed import Embedding, StructEmbedding


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

    def dummy_controller(self, shape):
        return self.controller_embedding.dummy(shape)

    def decode_controller(self, controller_state):
        return self.controller_embedding.decode(controller_state)


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
