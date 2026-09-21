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


class RMSNorm(nn.RMSNorm):
    """nn.RMSNorm computed in fp32 under autocast (fused kernel, full-
    precision statistics; same state_dict keys; no-op under fp32)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32:
            return super().forward(x)
        return super().forward(x.float()).to(x.dtype)


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
        # match the carried state's dtype (fp32 zeros would promote the
        # masked state to fp32 — permanent pool residency under capture)
        if init.dtype != state.dtype and init.is_floating_point():
            init = init.to(state.dtype)
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
        if getattr(self, "manual_step", False):
            # Hand-rolled LSTM cell (cuDNN's fused step has no vmap rule;
            # the stacked-weights phillip grid needs one). Same parameters
            # and equations as nn.LSTM (gate order i,f,g,o); ~5e-5 off
            # cuDNN (fusion order). Training unrolls never take this path.
            assert isinstance(self._core, nn.LSTM), "manual_step is LSTM-only"
            h, c = prev_state                       # each [1, B, H]
            gates = (
                inputs @ self._core.weight_ih_l0.t() + self._core.bias_ih_l0
                + h[0] @ self._core.weight_hh_l0.t() + self._core.bias_hh_l0
            )
            i, f, g, o = gates.chunk(4, -1)
            c2 = torch.sigmoid(f) * c[0] + torch.sigmoid(i) * torch.tanh(g)
            h2 = torch.sigmoid(o) * torch.tanh(c2)
            return h2, (h2.unsqueeze(0), c2.unsqueeze(0))
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


_FFW_IN_RENAMES = {"ffw_norm.weight": "ffw_in.0.weight", "gate_up.weight": "ffw_in.1.weight"}


def current_names(state_dict: dict) -> dict:
    """A state dict saved under older submodule names, under today's names."""
    def rename(key: str) -> str:
        for old, new in _FFW_IN_RENAMES.items():
            if key.endswith("." + old):
                return key[: -len(old)] + new
        return key
    return {rename(k): v for k, v in state_dict.items()}


def _accept_renamed(module: nn.Module, renames: dict[str, str]) -> None:
    """load_state_dict keeps accepting checkpoints saved under the old names."""

    def rename(module, state_dict, prefix, *_):
        for old, new in renames.items():
            if prefix + old in state_dict:
                state_dict[prefix + new] = state_dict.pop(prefix + old)

    module.register_load_state_dict_pre_hook(rename)


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

        self.attn_norm = RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.head_dim)  # QK-norm: attention stability
        self.k_norm = RMSNorm(self.head_dim)
        self.attn_out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.attn_out.weight)

        hidden = int(8 * d / 3 / 64) * 64  # SwiGLU sizing, 64-aligned
        self.ffw_in = nn.Sequential(RMSNorm(d), nn.Linear(d, 2 * hidden, bias=False))
        _accept_renamed(self, _FFW_IN_RENAMES)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def _attend(self, x, positions, k_cache, v_cache, mask):
        B, T, d = x.shape
        W = k_cache.shape[1]
        h = self.num_heads
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        q = _rope(self.q_norm(q.view(B, T, h, self.head_dim)), positions)
        k = _rope(self.k_norm(k.view(B, T, h, self.head_dim)), positions)
        keys = torch.cat([k_cache, k.reshape(B, T, d)], dim=1)  # [B, W+T, d]
        values = torch.cat([v_cache, v], dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            keys.view(B, W + T, h, self.head_dim).transpose(1, 2),
            values.view(B, W + T, h, self.head_dim).transpose(1, 2),
            attn_mask=mask,
        )
        return self.attn_out(out.transpose(1, 2).reshape(B, T, d)), keys[:, -W:], values[:, -W:]

    def attend(self, x, positions, k_cache, v_cache, mask):
        attn, new_k, new_v = self._attend(x, positions, k_cache, v_cache, mask)
        x = x + attn
        gate, up = self.ffw_in(x).chunk(2, dim=-1)
        x = x + self.down(torch.nn.functional.silu(gate) * up)
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
        self.final_norm = RMSNorm(hidden_size)
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

    def _attn_mask(self, T, cache_len, B, device):
        # a key is attendable iff causal and at most W frames older than the
        # query; cache slot w holds the frame W-w steps before the chunk
        W = self.window
        slot = torch.arange(W, device=device)
        t = torch.arange(T, device=device)
        cache_valid = slot[None, :] >= (W - cache_len)[:, None]
        cache_in_window = slot[None, :] >= t[:, None]
        causal_window = (t[None, :] <= t[:, None]) & (t[:, None] - t[None, :] <= W)
        return torch.cat(
            [cache_valid[:, None, :] & cache_in_window[None, :, :],
             causal_window[None, :, :].expand(B, T, T)], dim=2,
        ).unsqueeze(1)  # [B, 1, T, W+T]

    def _forward(self, inputs, state):
        T = inputs.shape[1]
        x = self.encoder(inputs)
        positions = state["pos"][:, None] + torch.arange(T, device=inputs.device)[None]
        mask = self._attn_mask(T, state["cache_len"], inputs.shape[0], inputs.device)
        new_kv = []
        for block, (k_cache, v_cache) in zip(self.blocks, state["kv"]):
            x, nk, nv = block.attend(x, positions, k_cache, v_cache, mask)
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
    the last `window` frames; a causal windowed TINY ATTENTION (attn_heads x attn_head_dim;
    aMLP's is one head of 64) feeds the gate per the aMLP variant: out = u * (v_mixed + attn).

    Identity at init: conv weights 0 with bias 1 (v_mixed==1), attention output
    projection zero-init (a==0), sublayer out-projection zero-init.
    """

    def __init__(self, d: int, window: int, attn_heads: int = 1, attn_head_dim: int = 64):
        super().__init__()
        self.window = window
        self.attn_heads = attn_heads
        self.attn_width = attn_heads * attn_head_dim
        self.mix_norm = RMSNorm(d)
        self.uv = nn.Linear(d, 2 * d, bias=False)
        self.spatial = nn.Conv1d(d, d, kernel_size=window, groups=d)
        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.spatial.bias)

        # tiny attention (aMLP) over the same causal window
        self.attn_qkv = nn.Linear(d, 3 * self.attn_width, bias=False)
        self.attn_out = nn.Linear(self.attn_width, d, bias=False)
        nn.init.zeros_(self.attn_out.weight)

        self.mix_out = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.mix_out.weight)

        hidden = int(8 * d / 3 / 64) * 64
        self.ffw_in = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, 2 * hidden, bias=False)
        )
        _accept_renamed(self, _FFW_IN_RENAMES)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def _attend(self, xn, kv_cache, attn_mask):
        W = self.window
        q, k_new, va_new = self.attn_qkv(xn).chunk(3, dim=-1)
        kv_new = torch.cat([k_new, va_new], dim=-1)
        kv_full = torch.cat([kv_cache.to(kv_new.dtype), kv_new], dim=1)
        keys, vals = kv_full.chunk(2, dim=-1)
        heads = lambda t: t.unflatten(-1, (self.attn_heads, -1)).transpose(1, 2)   # [B, h, T, dk]
        a = torch.nn.functional.scaled_dot_product_attention(
            heads(q), heads(keys), heads(vals), attn_mask=attn_mask,
        ).transpose(1, 2).flatten(-2)
        return self.attn_out(a), kv_full[:, -(W - 1):].contiguous()

    def _spatial(self, v, v_cache):
        W = self.window
        v_cache = v_cache.to(v.dtype)
        if v.shape[1] == 1:
            # one output position is a per-channel weighted sum, not a conv:
            # 2.4x faster at n=400, +30% at n=1. Don't "simplify" it away.
            w = self.spatial.weight.squeeze(1)
            v_mixed = (
                (v_cache * w[:, : W - 1].t()).sum(dim=1)
                + v[:, 0] * w[:, W - 1]
                + self.spatial.bias
            ).unsqueeze(1)
            return v_mixed, torch.cat([v_cache[:, 1:], v], dim=1)
        v_full = torch.cat([v_cache, v], dim=1)
        v_mixed = self.spatial(v_full.transpose(1, 2)).transpose(1, 2)
        return v_mixed, v_full[:, -(W - 1):].contiguous()

    def _spatial_ring(self, v, v_ring, idx, valid):
        # gather-by-age is fused into the reduction by inductor (no window
        # materialized; measured), so the summation order matches _spatial
        W = self.window
        w = self.spatial.weight.squeeze(1)
        hist = torch.where(valid[:, :, None], v_ring.index_select(1, idx).to(v.dtype), 0.0)
        v_mixed = (
            (hist * w[:, : W - 1].t()).sum(dim=1)
            + v[:, 0] * w[:, W - 1]
            + self.spatial.bias
        ).unsqueeze(1)
        return v_mixed, v[:, 0]

    def mix_ring(self, x, v_ring, kv_cache, attn_mask, idx, valid):
        """Serving with the v-cache as a ring: returns the NEW slot [B, d]
        instead of a shifted cache; the caller writes it in place."""
        xn = self.mix_norm(x)
        u, v = self.uv(xn).chunk(2, dim=-1)
        v_mixed, v_new = self._spatial_ring(v, v_ring, idx, valid)
        attn, new_kv = self._attend(xn, kv_cache, attn_mask)
        x = x + self.mix_out(u * (v_mixed + attn))

        gate, up = self.ffw_in(x).chunk(2, dim=-1)
        x = x + self.down(torch.nn.functional.silu(gate) * up)

        return x, v_new, new_kv

    def mix(self, x, v_cache, kv_cache, attn_mask):
        xn = self.mix_norm(x)
        u, v = self.uv(xn).chunk(2, dim=-1)
        v_mixed, new_v = self._spatial(v, v_cache)
        attn, new_kv = self._attend(xn, kv_cache, attn_mask)
        x = x + self.mix_out(u * (v_mixed + attn))

        gate, up = self.ffw_in(x).chunk(2, dim=-1)
        x = x + self.down(torch.nn.functional.silu(gate) * up)

        return x, new_v, new_kv


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
        attn_heads: int = 1,
        attn_head_dim: int = 64,
    ):
        super().__init__()
        self.d = hidden_size
        self.window = window
        self.attn_width = attn_heads * attn_head_dim
        self.encoder = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [SGUBlock(hidden_size, window, attn_heads, attn_head_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.output_size = hidden_size

    def initial_state(self, batch_size, device=None):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return {
            "cache_len": torch.zeros(batch_size, dtype=torch.long, device=device),
            "layers": [
                (
                    z(batch_size, self.window - 1, self.d),
                    z(batch_size, self.window - 1, 2 * self.attn_width),
                )
                for _ in self.blocks
            ],
        }

    def _attn_mask(self, T, cache_len, B, device):
        W = self.window
        slot = torch.arange(W - 1, device=device)
        cache_valid = slot[None, :] >= (W - 1 - cache_len)[:, None]
        if T == 1:
            ones = torch.ones(B, 1, dtype=torch.bool, device=device)
            return torch.cat([cache_valid, ones], dim=1)[:, None, None, :]
        t = torch.arange(T, device=device)
        cache_in_window = slot[None, :] >= t[:, None]  # [T, W-1]
        causal_window = (t[None, :] <= t[:, None]) & (
            t[:, None] - t[None, :] <= W - 1
        )  # [T, T]
        return torch.cat(
            [cache_valid[:, None, :] & cache_in_window[None, :, :],
             causal_window[None].expand(B, T, T)], dim=2,
        ).unsqueeze(1)  # [B, 1, T, W-1+T]

    # ---- serving ring: the v-cache is a ring written in place by the agent
    # (kv stays canonical — small, and cat+SDPA beats a ring read for it).
    # Ring mode is keyed by "ptr" in the state; the learner never sees it. ----
    def initial_ring_state(self, batch_size, device=None):
        s = self.initial_state(batch_size, device)
        s["ptr"] = torch.zeros((), dtype=torch.long, device=device)
        return s

    def _ring_index(self, ptr, cache_len, device):
        W = self.window
        i = torch.arange(W - 1, device=device)
        idx = (i + ptr) % (W - 1)                 # canonical position i -> ring slot
        valid = i[None, :] >= (W - 1 - cache_len)[:, None]
        return idx, valid

    def canonical_state(self, state):
        """Ring state -> the chronological state the learner expects."""
        idx, valid = self._ring_index(state["ptr"], state["cache_len"], state["cache_len"].device)
        layers = [
            (torch.where(valid[:, :, None], v_ring.index_select(1, idx), 0.0), kv)
            for v_ring, kv in state["layers"]
        ]
        return {"cache_len": state["cache_len"], "layers": layers}

    def step_with_reset(self, inputs, reset, prev_state):
        if "ptr" not in prev_state:
            return super().step_with_reset(inputs, reset, prev_state)
        # stale ring slots are masked at read time by cache_len; never rewrite the ring
        initial = self.initial_state(reset.shape[0], device=reset.device)
        state = {
            "cache_len": torch.where(reset, 0, prev_state["cache_len"]),
            "ptr": prev_state["ptr"],
            "layers": [(v_ring, _mask_state(reset, ikv, kv))
                       for (v_ring, kv), (_, ikv) in zip(prev_state["layers"], initial["layers"])],
        }
        return self.step(inputs, state)

    def _forward(self, inputs, state):
        T = inputs.shape[1]
        x = self.encoder(inputs)
        mask = self._attn_mask(T, state["cache_len"], inputs.shape[0], inputs.device)
        ring = "ptr" in state
        if ring:
            assert T == 1, "ring mode is the serving path"
            idx, valid = self._ring_index(state["ptr"], state["cache_len"], inputs.device)
        new_layers = []
        for block, (v_cache, kv_cache) in zip(self.blocks, state["layers"]):
            if ring:
                x, v_new, nkv = block.mix_ring(x, v_cache, kv_cache, mask, idx, valid)
                new_layers.append((v_new, nkv))
            else:
                x, nv, nkv = block.mix(x, v_cache, kv_cache, mask)
                new_layers.append((nv, nkv))
        next_state = {
            "cache_len": torch.clamp(state["cache_len"] + T, max=self.window - 1),
            "layers": new_layers,
        }
        if ring:
            next_state["ptr"] = state["ptr"]      # advanced by the agent after the write
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
            attn_heads=network_config.attn_heads,
            attn_head_dim=network_config.attn_head_dim,
        )
    else:
        raise ValueError(f"unknown network name: {name}")
    return StateActionNetwork(
        embed_game,
        embed_state_action,
        core,
        packed=getattr(embed_config, "packed", True),
    )
