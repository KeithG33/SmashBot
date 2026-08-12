"""PyTorch port of slippi-ai's recurrent cores (vendor: slippi_ai/tf/networks.py).

All sequence tensors are batch-major: inputs [B, T, D], reset [B, T]; recurrent
states are batched with no time axis (LSTM tuples keep torch's [layers, B, H]).
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
    """Replace state with initial where reset is True. reset: [B].

    State leaves are batch-first ([B, ...], e.g. KV caches) except torch RNN
    states, which are [layers, B, H] — disambiguated by which dim matches B.
    """
    B = reset.shape[0]

    def where(init: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if state.dim() >= 1 and state.shape[0] == B:
            mask = reset
            while mask.dim() < state.dim():
                mask = mask.unsqueeze(-1)
        else:  # [layers, B, H] torch RNN convention
            mask = reset.view(1, -1, *([1] * (state.dim() - 2)))
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
        """inputs: [B, T, D], reset: [B, T] -> (outputs [B, T, D'], final_state).

        Default implementation steps one frame at a time; recurrent wrappers
        override with segmented cuDNN calls.
        """
        outputs = []
        state = initial_state
        for t in range(inputs.shape[1]):
            out, state = self.step_with_reset(inputs[:, t], reset[:, t], state)
            outputs.append(out)
        return torch.stack(outputs, dim=1), state


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
    """Wraps nn.LSTM / nn.GRU (single layer, batch_first)."""

    def __init__(self, core: nn.Module):
        super().__init__()
        assert isinstance(core, (nn.LSTM, nn.GRU))
        assert core.batch_first
        self._core = core

    def initial_state(self, batch_size, device=None):
        h = torch.zeros(1, batch_size, self._core.hidden_size, device=device)
        if isinstance(self._core, nn.LSTM):
            return (h, h.clone())
        return h

    def step(self, inputs, prev_state):
        out, next_state = self._core(inputs.unsqueeze(1), prev_state)
        return out.squeeze(1), next_state

    def unroll(self, inputs, reset, initial_state):
        # Segment at timesteps where any element resets; one cuDNN call each.
        reset_any = reset.any(dim=0)  # [T]
        boundaries = torch.nonzero(reset_any).squeeze(-1).tolist()

        outputs = []
        state = initial_state
        pos = 0
        T = inputs.shape[1]
        for b in boundaries + [T]:
            if pos < b:
                out, state = self._core(inputs[:, pos:b], state)
                outputs.append(out)
                pos = b
            if b < T:
                initial = self.initial_state(reset.shape[0], device=inputs.device)
                state = _mask_state(reset[:, b], initial, state)
        # note: a boundary at t masks the state, then t joins the next segment
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], state


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

    def __init__(
        self,
        residual_size: int,
        hidden_size: int | None = None,
        activation="relu",
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        out = nn.Linear(hidden_size or residual_size, residual_size)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.block = nn.Sequential(
            # slippi-ai's hand-rolled LayerNorm has no epsilon; checkpoints
            # ported from TF set ln_eps=0.0 for exact equivalence.
            nn.LayerNorm(residual_size, eps=ln_eps),
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
        ln_eps: float = 1e-5,
    ):
        recurrent_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[recurrent_layer]

        layers: list[Network] = [FFWWrapper(nn.Linear(input_size, hidden_size))]
        for _ in range(num_layers):
            layers.append(
                ResidualWrapper(
                    RecurrentWrapper(
                        recurrent_cls(hidden_size, hidden_size, batch_first=True)
                    )
                )
            )
            layers.append(
                FFWWrapper(
                    ResBlock(
                        hidden_size,
                        hidden_size * ffw_multiplier,
                        activation,
                        ln_eps=ln_eps,
                    )
                )
            )
        super().__init__(layers)
        self.output_size = hidden_size


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0):
    """Rotary embedding. x: [B, T, heads, head_dim], positions: [B, T] (absolute)."""
    hd = x.shape[-1]
    freqs = theta ** (
        -torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd
    )
    angles = positions.float()[..., None] * freqs  # [B, T, hd/2]
    cos = angles.cos()[:, :, None, :]  # [B, T, 1, hd/2]
    sin = angles.sin()[:, :, None, :]
    x1, x2 = x.float()[..., 0::2], x.float()[..., 1::2]
    out = torch.empty_like(x, dtype=torch.float32)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(x.dtype)


class TransformerBlock(nn.Module):
    """Pre-RMSNorm causal attention + SwiGLU, zero-init output projections."""

    def __init__(self, d: int, num_heads: int):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.attn_norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)  # QK-norm: attention stability
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.attn_out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.attn_out.weight)

        self.ffw_norm = nn.RMSNorm(d)
        hidden = int(8 * d / 3 / 64) * 64  # SwiGLU sizing, 64-aligned
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def attend(
        self,
        x: torch.Tensor,  # [B, T, d]
        positions: torch.Tensor,  # [B, T] absolute positions of x
        k_cache: torch.Tensor,  # [B, W, d] (rotated keys, newest right-aligned)
        v_cache: torch.Tensor,  # [B, W, d]
        cache_len: torch.Tensor,  # [B] valid entries in the cache
    ):
        B, T, d = x.shape
        W = k_cache.shape[1]
        h = self.num_heads

        qkv = self.qkv(self.attn_norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, h, self.head_dim))
        k = self.k_norm(k.view(B, T, h, self.head_dim))
        q = _rope(q, positions)
        k = _rope(k, positions)
        k_flat = k.reshape(B, T, d)

        keys = torch.cat([k_cache, k_flat], dim=1)  # [B, W+T, d]
        values = torch.cat([v_cache, v], dim=1)

        # mask [B, 1, T, W+T]. A key is attendable iff it is causal AND at
        # most `W` frames older than the query — the same horizon the step
        # path's rolling cache enforces structurally. Cache slot w holds the
        # frame W - w steps before the chunk (age t + W - w for query t), so
        # it stays visible iff w >= t; chunk key t' has age t - t' <= W.
        slot = torch.arange(W, device=x.device)
        t = torch.arange(T, device=x.device)
        cache_valid = slot[None, :] >= (W - cache_len)[:, None]  # [B, W]
        cache_in_window = slot[None, :] >= t[:, None]  # [T, W]
        causal_window = (t[None, :] <= t[:, None]) & (
            t[:, None] - t[None, :] <= W
        )  # [T(query), T(key)]
        mask = torch.cat(
            [
                cache_valid[:, None, :] & cache_in_window[None, :, :],
                causal_window[None, :, :].expand(B, T, T),
            ],
            dim=2,
        ).unsqueeze(1)  # [B, 1, T, W+T]

        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),  # [B, h, T, hd]
            keys.view(B, W + T, h, self.head_dim).transpose(1, 2),
            values.view(B, W + T, h, self.head_dim).transpose(1, 2),
            attn_mask=mask,
        )
        x = x + self.attn_out(out.transpose(1, 2).reshape(B, T, d))

        gate, up = self.gate_up(self.ffw_norm(x)).chunk(2, dim=-1)
        x = x + self.down(torch.nn.functional.silu(gate) * up)

        # slide the cache: keep the last W of [cache + new]
        new_k = torch.cat([k_cache, k_flat], dim=1)[:, -W:]
        new_v = torch.cat([v_cache, v], dim=1)[:, -W:]
        return x, new_k, new_v


class TransformerCore(Network):
    """Sliding-window causal transformer. Recurrent state = per-layer KV cache
    (last `window` frames) + per-element absolute position / cache length.
    Memory horizon is exactly `window` frames — a deliberate contrast to the
    LSTM's unbounded carry."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        window: int = 256,
    ):
        super().__init__()
        self.d = hidden_size
        self.window = window
        self.encoder = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(hidden_size)
        self.output_size = hidden_size

    def initial_state(self, batch_size, device=None):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return {
            "pos": torch.zeros(batch_size, dtype=torch.long, device=device),
            "cache_len": torch.zeros(batch_size, dtype=torch.long, device=device),
            "kv": [
                (z(batch_size, self.window, self.d), z(batch_size, self.window, self.d))
                for _ in self.blocks
            ],
        }

    def _forward(self, inputs, state):
        """inputs: [B, T, D_in] (one reset-free segment)."""
        T = inputs.shape[1]
        x = self.encoder(inputs)
        positions = state["pos"][:, None] + torch.arange(T, device=inputs.device)[None]
        new_kv = []
        for block, (k_cache, v_cache) in zip(self.blocks, state["kv"]):
            x, nk, nv = block.attend(x, positions, k_cache, v_cache, state["cache_len"])
            new_kv.append((nk, nv))
        next_state = {
            "pos": state["pos"] + T,
            "cache_len": torch.clamp(state["cache_len"] + T, max=self.window),
            "kv": new_kv,
        }
        return self.final_norm(x), next_state

    def step(self, inputs, prev_state):
        out, state = self._forward(inputs[:, None], prev_state)
        return out[:, 0], state

    def unroll(self, inputs, reset, initial_state):
        reset_any = reset.any(dim=0)  # [T]
        boundaries = torch.nonzero(reset_any).squeeze(-1).tolist()

        outputs = []
        state = initial_state
        pos = 0
        T = inputs.shape[1]
        for b in boundaries + [T]:
            if pos < b:
                out, state = self._forward(inputs[:, pos:b], state)
                outputs.append(out)
                pos = b
            if b < T:
                initial = self.initial_state(reset.shape[0], device=inputs.device)
                state = _mask_state(reset[:, b], initial, state)
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], state



class SGUBlock(nn.Module):
    """aMLP-style causal Spatial Gating Unit (right-aligned window / Toeplitz):
    norm -> project to (gate u, value v); v mixed by causal depthwise conv over
    the last `window` frames; a causal windowed TINY ATTENTION (single head,
    dk=64) feeds the gate per the aMLP variant: out = u * (v_mixed + attn).

    Identity at init: conv weights 0 with bias 1 (v_mixed==1), attention output
    projection zero-init (a==0), sublayer out-projection zero-init.
    """

    ATTN_DK = 64

    def __init__(self, d: int, window: int):
        super().__init__()
        self.window = window
        self.mix_norm = nn.RMSNorm(d)
        self.uv = nn.Linear(d, 2 * d, bias=False)
        self.spatial = nn.Conv1d(d, d, kernel_size=window, groups=d)
        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.spatial.bias)

        # tiny attention (aMLP): single head over the same causal window
        self.attn_qkv = nn.Linear(d, 3 * self.ATTN_DK, bias=False)
        self.attn_out = nn.Linear(self.ATTN_DK, d, bias=False)
        nn.init.zeros_(self.attn_out.weight)

        self.mix_out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.mix_out.weight)

        self.ffw_norm = nn.RMSNorm(d)
        hidden = int(8 * d / 3 / 64) * 64
        self.gate_up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def mix(self, x, v_cache, kv_cache, cache_len):
        """x: [B, T, d]; v_cache: [B, W-1, d]; kv_cache: [B, W-1, 2*dk];
        cache_len: [B] valid entries. Returns (out, new_v_cache, new_kv_cache)."""
        B, T, _ = x.shape
        W = self.window
        xn = self.mix_norm(x)
        u, v = self.uv(xn).chunk(2, dim=-1)

        # static mixing: causal depthwise conv over [cache || current]
        v_full = torch.cat([v_cache, v], dim=1)  # [B, W-1+T, d]
        if T == 1:
            # Grouped conv with one output position is just a per-channel
            # weighted sum over the window; conv kernels handle B=1/groups=d
            # badly (and defeat inductor fusion) on the play path.
            w = self.spatial.weight.squeeze(1)  # [d, W]
            v_mixed = (v_full * w.t()).sum(dim=1, keepdim=True) + self.spatial.bias
        else:
            v_mixed = self.spatial(v_full.transpose(1, 2)).transpose(1, 2)

        # tiny attention over the same causal window
        qkv = self.attn_qkv(xn)  # [B, T, 3*dk]
        q, k_new, va_new = qkv.chunk(3, dim=-1)
        kv_new = torch.cat([k_new, va_new], dim=-1)
        kv_full = torch.cat([kv_cache, kv_new], dim=1)  # [B, W-1+T, 2*dk]
        keys, vals = kv_full.chunk(2, dim=-1)

        # Same windowed-causal rule as the conv: key attendable iff at most
        # W-1 frames older than the query (cache slot w ages out when w < t).
        slot = torch.arange(W - 1, device=x.device)
        t = torch.arange(T, device=x.device)
        cache_valid = slot[None, :] >= (W - 1 - cache_len)[:, None]  # [B, W-1]
        cache_in_window = slot[None, :] >= t[:, None]  # [T, W-1]
        causal_window = (t[None, :] <= t[:, None]) & (
            t[:, None] - t[None, :] <= W - 1
        )  # [T, T]
        mask = torch.cat(
            [cache_valid[:, None, :] & cache_in_window[None, :, :],
             causal_window[None].expand(B, T, T)], dim=2,
        ).unsqueeze(1)  # [B, 1, T, W-1+T]
        a = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), keys.unsqueeze(1), vals.unsqueeze(1), attn_mask=mask
        ).squeeze(1)  # [B, T, dk]

        x = x + self.mix_out(u * (v_mixed + self.attn_out(a)))

        gate, up = self.gate_up(self.ffw_norm(x)).chunk(2, dim=-1)
        x = x + self.down(torch.nn.functional.silu(gate) * up)

        return x, v_full[:, -(W - 1):], kv_full[:, -(W - 1):]


class SGUCore(Network):
    """Stack of aMLP/SGU blocks. State per layer = ring of last window-1
    v-vectors (conv) + kv pairs (tiny attention), plus a shared cache_len.
    Hard per-layer horizon of `window` frames."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        window: int = 8,
    ):
        super().__init__()
        self.d = hidden_size
        self.window = window
        self.encoder = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [SGUBlock(hidden_size, window) for _ in range(num_layers)]
        )
        self.final_norm = nn.RMSNorm(hidden_size)
        self.output_size = hidden_size

    def initial_state(self, batch_size, device=None):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return {
            "cache_len": torch.zeros(batch_size, dtype=torch.long, device=device),
            "layers": [
                (
                    z(batch_size, self.window - 1, self.d),
                    z(batch_size, self.window - 1, 2 * SGUBlock.ATTN_DK),
                )
                for _ in self.blocks
            ],
        }

    def _forward(self, inputs, state):
        T = inputs.shape[1]
        x = self.encoder(inputs)
        new_layers = []
        for block, (v_cache, kv_cache) in zip(self.blocks, state["layers"]):
            x, nv, nkv = block.mix(x, v_cache, kv_cache, state["cache_len"])
            new_layers.append((nv, nkv))
        next_state = {
            "cache_len": torch.clamp(state["cache_len"] + T, max=self.window - 1),
            "layers": new_layers,
        }
        return self.final_norm(x), next_state

    def step(self, inputs, prev_state):
        out, state = self._forward(inputs[:, None], prev_state)
        return out[:, 0], state

    def unroll(self, inputs, reset, initial_state):
        reset_any = reset.any(dim=0)
        boundaries = torch.nonzero(reset_any).squeeze(-1).tolist()

        outputs = []
        state = initial_state
        pos = 0
        T = inputs.shape[1]
        for b in boundaries + [T]:
            if pos < b:
                out, state = self._forward(inputs[:, pos:b], state)
                outputs.append(out)
                pos = b
            if b < T:
                initial = self.initial_state(reset.shape[0], device=inputs.device)
                state = _mask_state(reset[:, b], initial, state)
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], state


class StateActionNetwork(Network):
    """Embeds StateAction structs, then runs the core network."""

    def __init__(self, embed_game, embed_state_action, core: Network, packed: bool = True):
        super().__init__()
        from smashbot import embed as embed_lib

        self.embed_game = embed_game
        self.embed_state_action = embed_state_action
        self.core = core
        self.packed_embed = (
            embed_lib.PackedStructForward(embed_state_action) if packed else None
        )

    def embed_sa(self, state_action) -> torch.Tensor:
        if self.packed_embed is not None:
            return self.packed_embed(state_action)
        return self.embed_state_action(state_action)

    def encode(self, state_action):
        """numpy Batch structs -> encoded numpy structs (data thread / inference)."""
        return self.embed_state_action.from_state(state_action)

    def encode_game(self, game):
        return self.embed_game.from_state(game)

    def initial_state(self, batch_size, device=None):
        return self.core.initial_state(batch_size, device)

    def step(self, state_action, prev_state):
        return self.core.step(self.embed_sa(state_action), prev_state)

    def step_with_reset(self, state_action, reset, prev_state):
        return self.core.step_with_reset(
            self.embed_sa(state_action), reset, prev_state
        )

    def unroll(self, state_action, reset, initial_state):
        return self.core.unroll(self.embed_sa(state_action), reset, initial_state)


def build_embed_network(
    embed_config,
    controller_embedding,
    num_names: int,
    network_config,
) -> StateActionNetwork:
    from smashbot import embed as embed_lib

    embed_game = embed_config.make_game_embedding()
    embed_state_action = embed_lib.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=controller_embedding,
        num_names=num_names,
    )
    name = getattr(network_config, "name", "tx_like")
    if name == "tx_like":
        core = TransformerLike(
            input_size=embed_state_action.size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            ffw_multiplier=network_config.ffw_multiplier,
            recurrent_layer=network_config.recurrent_layer,
            ln_eps=getattr(network_config, "ln_eps", 1e-5),
        )
    elif name == "transformer":
        core = TransformerCore(
            input_size=embed_state_action.size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            num_heads=network_config.num_heads,
            window=network_config.window,
        )
    elif name == "sgu":
        core = SGUCore(
            input_size=embed_state_action.size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            window=network_config.window,
        )
    else:
        raise ValueError(f"unknown network name: {name}")
    return StateActionNetwork(
        embed_game,
        embed_state_action,
        core,
        packed=getattr(embed_config, "packed", True),
    )
