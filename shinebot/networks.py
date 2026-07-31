"""PyTorch port of slippi-ai's recurrent cores (vendor: slippi_ai/tf/networks.py).

All tensors are time-major: inputs [T, B, D], reset [T, B], states [B, ...].
The data pipeline guarantees resets only at chunk boundaries, but `unroll`
handles resets at arbitrary timesteps by segmenting the sequence, so each
segment still runs as one cuDNN call.
"""

import abc
import typing as tp

import torch
from torch import nn

RecurrentState = tp.Any


def _mask_state(reset: torch.Tensor, initial, prev):
    """Replace state with initial where reset is True. reset: [B]."""

    def where(init: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # states may have a leading layer dim: [B, H] or [L, B, H]
        mask = reset
        while mask.dim() < state.dim():
            mask = mask.unsqueeze(-1)
        if state.dim() == 3:  # [L, B, H]: batch is dim 1
            mask = reset.view(1, -1, 1)
        return torch.where(mask, init, state)

    return torch.utils._pytree.tree_map(where, initial, prev)


class Network(nn.Module, abc.ABC):
    @abc.abstractmethod
    def initial_state(self, batch_size: int, device=None) -> RecurrentState:
        ...

    @abc.abstractmethod
    def step(self, inputs: torch.Tensor, prev_state) -> tuple[torch.Tensor, RecurrentState]:
        """inputs: [B, D] -> (outputs [B, D'], next_state)."""

    def step_with_reset(self, inputs, reset, prev_state):
        initial = self.initial_state(reset.shape[0], device=reset.device)
        return self.step(inputs, _mask_state(reset, initial, prev_state))

    def unroll(self, inputs, reset, initial_state):
        """inputs: [T, B, D], reset: [T, B] -> (outputs [T, B, D'], final_state).

        Default implementation steps one frame at a time; recurrent wrappers
        override with segmented cuDNN calls.
        """
        outputs = []
        state = initial_state
        for t in range(inputs.shape[0]):
            out, state = self.step_with_reset(inputs[t], reset[t], state)
            outputs.append(out)
        return torch.stack(outputs), state


class FFWWrapper(Network):
    """Stateless module applied over the whole sequence at once."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self._module = module

    def initial_state(self, batch_size, device=None):
        return ()

    def step(self, inputs, prev_state):
        return self._module(inputs), ()

    def step_with_reset(self, inputs, reset, prev_state):
        return self._module(inputs), ()

    def unroll(self, inputs, reset, initial_state):
        return self._module(inputs), ()


class RecurrentWrapper(Network):
    """Wraps nn.LSTM / nn.GRU (single layer, time-major)."""

    def __init__(self, core: nn.Module):
        super().__init__()
        assert isinstance(core, (nn.LSTM, nn.GRU))
        assert not core.batch_first
        self._core = core

    def initial_state(self, batch_size, device=None):
        h = torch.zeros(1, batch_size, self._core.hidden_size, device=device)
        if isinstance(self._core, nn.LSTM):
            return (h, h.clone())
        return h

    def step(self, inputs, prev_state):
        out, next_state = self._core(inputs.unsqueeze(0), prev_state)
        return out.squeeze(0), next_state

    def unroll(self, inputs, reset, initial_state):
        # Segment at timesteps where any element resets; one cuDNN call each.
        reset_any = reset.any(dim=1)
        boundaries = torch.nonzero(reset_any).squeeze(-1).tolist()

        outputs = []
        state = initial_state
        pos = 0
        T = inputs.shape[0]
        for b in boundaries + [T]:
            if pos < b:
                out, state = self._core(inputs[pos:b], state)
                outputs.append(out)
                pos = b
            if b < T:
                initial = self.initial_state(reset.shape[1], device=inputs.device)
                state = _mask_state(reset[b], initial, state)
        # note: a boundary at t masks the state, then t joins the next segment
        return torch.cat(outputs) if len(outputs) > 1 else outputs[0], state


class ResidualWrapper(Network):
    def __init__(self, net: Network):
        super().__init__()
        self._net = net

    def initial_state(self, batch_size, device=None):
        return self._net.initial_state(batch_size, device)

    def step(self, inputs, prev_state):
        outputs, next_state = self._net.step(inputs, prev_state)
        return inputs + outputs, next_state

    def step_with_reset(self, inputs, reset, prev_state):
        outputs, next_state = self._net.step_with_reset(inputs, reset, prev_state)
        return inputs + outputs, next_state

    def unroll(self, inputs, reset, initial_state):
        outputs, final_state = self._net.unroll(inputs, reset, initial_state)
        return inputs + outputs, final_state


class Sequential(Network):
    def __init__(self, layers: list[Network]):
        super().__init__()
        self._layers = nn.ModuleList(layers)

    def initial_state(self, batch_size, device=None):
        return [layer.initial_state(batch_size, device) for layer in self._layers]

    def step(self, inputs, prev_state):
        next_states = []
        for layer, state in zip(self._layers, prev_state):
            inputs, next_state = layer.step(inputs, state)
            next_states.append(next_state)
        return inputs, next_states

    def step_with_reset(self, inputs, reset, prev_state):
        next_states = []
        for layer, state in zip(self._layers, prev_state):
            inputs, next_state = layer.step_with_reset(inputs, reset, state)
            next_states.append(next_state)
        return inputs, next_states

    def unroll(self, inputs, reset, prev_state):
        final_states = []
        for layer, state in zip(self._layers, prev_state):
            inputs, final_state = layer.unroll(inputs, reset, state)
            final_states.append(final_state)
        return inputs, final_states


class ResBlock(nn.Module):
    """Pre-LayerNorm residual FFW block with zero-initialized output."""

    def __init__(self, residual_size: int, hidden_size: int | None = None, activation="relu"):
        super().__init__()
        out = nn.Linear(hidden_size or residual_size, residual_size)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.block = nn.Sequential(
            nn.LayerNorm(residual_size),
            nn.Linear(residual_size, hidden_size or residual_size),
            {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation],
            out,
        )

    def forward(self, residual):
        return residual + self.block(residual)


class TransformerLike(Sequential):
    """Transformer block layout with self-attention replaced by a recurrent layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        ffw_multiplier: int = 2,
        recurrent_layer: str = "lstm",
        activation: str = "gelu",
    ):
        recurrent_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[recurrent_layer]

        layers: list[Network] = [FFWWrapper(nn.Linear(input_size, hidden_size))]
        for _ in range(num_layers):
            layers.append(
                ResidualWrapper(RecurrentWrapper(recurrent_cls(hidden_size, hidden_size)))
            )
            layers.append(
                FFWWrapper(
                    ResBlock(hidden_size, hidden_size * ffw_multiplier, activation)
                )
            )
        super().__init__(layers)
        self.output_size = hidden_size


class StateActionNetwork(Network):
    """Embeds StateAction structs, then runs the core network."""

    def __init__(self, embed_game, embed_state_action, core: Network):
        super().__init__()
        self.embed_game = embed_game
        self.embed_state_action = embed_state_action
        self.core = core

    def encode(self, state_action):
        """numpy Batch structs -> encoded numpy structs (data thread / inference)."""
        return self.embed_state_action.from_state(state_action)

    def encode_game(self, game):
        return self.embed_game.from_state(game)

    def initial_state(self, batch_size, device=None):
        return self.core.initial_state(batch_size, device)

    def step(self, state_action, prev_state):
        return self.core.step(self.embed_state_action(state_action), prev_state)

    def step_with_reset(self, state_action, reset, prev_state):
        return self.core.step_with_reset(
            self.embed_state_action(state_action), reset, prev_state
        )

    def unroll(self, state_action, reset, initial_state):
        return self.core.unroll(self.embed_state_action(state_action), reset, initial_state)


def build_embed_network(
    embed_config,
    controller_embedding,
    num_names: int,
    network_config,
) -> StateActionNetwork:
    from shinebot import embed as embed_lib

    embed_game = embed_config.make_game_embedding()
    embed_state_action = embed_lib.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=controller_embedding,
        num_names=num_names,
    )
    core = TransformerLike(
        input_size=embed_state_action.size,
        hidden_size=network_config.hidden_size,
        num_layers=network_config.num_layers,
        ffw_multiplier=network_config.ffw_multiplier,
        recurrent_layer=network_config.recurrent_layer,
    )
    return StateActionNetwork(embed_game, embed_state_action, core)
