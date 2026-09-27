"""PyTorch port of slippi-ai's recurrent cores (vendor: slippi_ai/tf/networks.py).

All sequence tensors are batch-major: inputs [B, T, D], reset [B, T]; recurrent
states are batched with no time axis (LSTM tuples keep torch's [layers, B, H]).
The data pipeline guarantees resets only at chunk boundaries, but `unroll`
handles resets at arbitrary timesteps by segmenting the sequence, so each
segment still runs as one cuDNN call.
"""

import abc
import dataclasses
import typing as tp

import torch
import torch.utils.checkpoint
from torch import nn

from smashbot.causal_conv import causal_conv, causal_conv_ring, ring_taps

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

    def cache_state(self, state, dtype):
        """`state` with its window caches stored in `dtype` (serving keeps
        them half precision: they are the resident term). Recurrent memory
        (LSTM h/c, GRU h) stays fp32: it is tiny, and rounding it every
        frame is what drifts (scripts/check_serving_precision.py)."""
        return state

    # BC's loader resets rows only at a chunk's first frame (batch_to_frames
    # refuses anything else), so its unrolls mask the state once and run the
    # chunk whole, without reading the reset frames back to the host
    chunk_start_resets: bool = False

    def _segmented_unroll(self, run, inputs, reset, initial_state):
        """run(inputs, state) -> (outputs, state) over the segments between
        reset frames; a reset at t masks the state, then t starts a segment."""
        initial = lambda: self.initial_state(reset.shape[0], device=inputs.device)
        if self.chunk_start_resets:
            return run(inputs, _mask_state(reset[:, 0], initial(), initial_state))
        boundaries = torch.nonzero(reset.any(dim=0)).squeeze(-1).tolist()
        outputs, state, pos, T = [], initial_state, 0, inputs.shape[1]
        for b in boundaries + [T]:
            if pos < b:
                out, state = run(inputs[:, pos:b], state)
                outputs.append(out)
                pos = b
            if b < T:
                state = _mask_state(reset[:, b], initial(), state)
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0], state

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


def _gates(rnn, x, h):
    """x @ W_ih^T + b_ih and h @ W_hh^T + b_hh in fp32, with the matmuls in
    the weights' dtype: stacked fp16 grid weights are used as stored (fp32
    accumulation inside the GEMM) instead of being cast every frame."""
    gi = (x.to(rnn.weight_ih_l0.dtype) @ rnn.weight_ih_l0.t()).float() + rnn.bias_ih_l0.float()
    gh = (h.to(rnn.weight_hh_l0.dtype) @ rnn.weight_hh_l0.t()).float() + rnn.bias_hh_l0.float()
    return gi, gh


def _lstm_cell(rnn: nn.LSTM, x, h, c):
    """nn.LSTM's equations for one frame (gate order i, f, g, o)."""
    gi, gh = _gates(rnn, x, h)
    i, f, g, o = (gi + gh).chunk(4, -1)
    c = torch.sigmoid(f) * c.float() + torch.sigmoid(i) * torch.tanh(g)
    return torch.sigmoid(o) * torch.tanh(c), c


def _gru_cell(rnn: nn.GRU, x, h):
    """nn.GRU's equations for one frame (gate order r, z, n)."""
    gi, gh = _gates(rnn, x, h)
    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h.float()


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

    # Serving: the one-frame step written out from the cell's weights. cuDNN's
    # fused step has no vmap rule (the stacked-weights grids need one) and its
    # fp16 path drifts from fp32 (97.6% action agreement vs 99.996% for the
    # written-out cell; scripts/check_serving_precision.py). Training unrolls
    # never take this path.
    manual_step: bool = False

    def step(self, inputs, prev_state):
        if self.manual_step:
            if isinstance(self._core, nn.LSTM):
                h, c = prev_state                       # each [1, B, H]
                h2, c2 = _lstm_cell(self._core, inputs, h[0], c[0])
                return h2, (h2.unsqueeze(0), c2.unsqueeze(0))
            h2 = _gru_cell(self._core, inputs, prev_state[0])
            return h2, h2.unsqueeze(0)
        out, next_state = self._core(inputs.unsqueeze(1), prev_state)
        return out.squeeze(1), next_state

    def unroll(self, inputs, reset, initial_state):
        return self._segmented_unroll(self._core, inputs, reset, initial_state)   # one cuDNN call per segment


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
# the (u, v) projection became Sequential(Linear, GELU) when the paper block
# went unconditional; only a checkpoint trained WITH that GELU may take the rename
_UV_RENAMES = {"uv.weight": "uv.0.weight"}
_RENAMES = {**_FFW_IN_RENAMES, **_UV_RENAMES}


def current_names(state_dict: dict) -> dict:
    """A state dict saved under older submodule names, under today's names."""
    def rename(key: str) -> str:
        for old, new in _RENAMES.items():
            if key.endswith("." + old):
                return key[: -len(old)] + new
        return key
    return {rename(k): v for k, v in state_dict.items()}


def check_loadable(network_cfg: dict, state_dict: dict) -> None:
    """A SGU checkpoint saved before the GELU on (u, v) and the norm on v
    became unconditional computes a different network: refuse it rather than
    load its weights into today's block. Only a full checkpoint's own saved
    config can vouch for old names; for bare weights pass {}."""
    if any(k.endswith(".uv.weight") for k in state_dict) and not (
            network_cfg.get("gate_gelu") and network_cfg.get("v_norm")):
        raise ValueError("checkpoint predates the unconditional paper block (no GELU on uv "
                         "/ no v norm); current code cannot run it")


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
        gelu_approximate: bool = False,
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
            {"relu": nn.ReLU(), "gelu": nn.GELU(approximate="tanh" if gelu_approximate else "none")}[activation],
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
        gelu_approximate: bool = False,
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
                        gelu_approximate=gelu_approximate,
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
        _accept_renamed(self, _RENAMES)
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

    def cache_state(self, state, dtype):
        return {**state, "kv": [(k.to(dtype), v.to(dtype)) for k, v in state["kv"]]}

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

    inputs_encoded: bool = False   # the learner's StateActionNetwork applied the encoder (use_packed_encoder)

    def _forward(self, inputs, state):
        T = inputs.shape[1]
        x = inputs if self.inputs_encoded else self.encoder(inputs)
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
        return self._segmented_unroll(self._forward, inputs, reset, initial_state)



def _recomputed_in_backward(fn, *args):
    """fn(*args) without keeping fn's activations for backward: they are
    recomputed there. Worth it for cheap ops with large outputs (the SGU's
    GELU and v norm hold +1.3 GiB at 6/576, batch 512; recomputing costs 1.4%)."""
    if torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class SGUBlock(nn.Module):
    """aMLP-style causal Spatial Gating Unit (right-aligned window / Toeplitz):
    norm -> project to (gate u, value v); v mixed by causal depthwise conv over
    the last `window` frames; a causal windowed TINY ATTENTION (attn_heads x attn_head_dim;
    aMLP's is one head of 64) feeds the gate per the aMLP variant: out = u * (v_mixed + attn).

    Identity at init: conv weights 0 with bias 1 (v_mixed==1), attention output
    projection zero-init (a==0), sublayer out-projection zero-init.

    As in the published gMLP block, the (u, v) projection is followed by a
    GELU and v is normalized before the temporal mix.
    """

    def __init__(self, d: int, window: int, attn_heads: int = 1, attn_head_dim: int = 64):
        super().__init__()
        self.window = window
        self.attn_heads = attn_heads
        self.attn_width = attn_heads * attn_head_dim
        self.mix_norm = RMSNorm(d)
        self.uv = nn.Sequential(
            nn.Linear(d, 2 * d, bias=False),
            nn.GELU()
        )
        # no gain: a per-channel scale folds into the depthwise filter
        self.v_norm = RMSNorm(d, elementwise_affine=False)
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
        _accept_renamed(self, _RENAMES)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    def _uv(self, xn):
        u, v = self.uv(xn).chunk(2, dim=-1)
        return u, self.v_norm(v)

    def _qkv(self, xn):
        q, k, va = self.attn_qkv(xn).chunk(3, dim=-1)
        return q, torch.cat([k, va], dim=-1)

    def _attend_over(self, q, kv, attn_mask):
        keys, vals = kv.chunk(2, dim=-1)
        heads = lambda t: t.unflatten(-1, (self.attn_heads, -1)).transpose(1, 2)   # [B, h, T, dk]
        a = torch.nn.functional.scaled_dot_product_attention(
            heads(q), heads(keys), heads(vals), attn_mask=attn_mask,
        ).transpose(1, 2).flatten(-2)
        return self.attn_out(a)

    def _attend(self, xn, kv_cache, attn_mask):
        q, kv_new = self._qkv(xn)
        kv_full = torch.cat([kv_cache.to(kv_new.dtype), kv_new], dim=1)
        return self._attend_over(q, kv_full, attn_mask), kv_full[:, -(self.window - 1):].contiguous()

    def _attend_ring(self, xn, kv_ring, kv_ptr, attn_mask):
        """Serving: the frame's K/V goes into its ring slot, then the frame
        attends over the whole ring. The attention has no positions, so slot
        order only changes the summation order. A reset leaves the finished
        game's entries for the mask, and a masked NaN still poisons SDPA
        (0 * NaN): a non-finite entry is stored as 0."""
        q, kv_new = self._qkv(xn)
        kv_new = torch.nan_to_num(kv_new.to(kv_ring.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        # tensor indices on both dims: a Python index reads kv_ptr on the host
        # (no capture), and vmap loops index_copy_ over the grid's slices
        rows = torch.arange(kv_ring.shape[0], device=kv_ring.device)
        kv_ring.index_put_((rows, kv_ptr.expand(rows.shape[0])), kv_new[:, 0])
        return self._attend_over(q, kv_ring, attn_mask)

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
        if v.is_cuda:
            T = v.shape[1]
            v_mixed = causal_conv(v_cache, v, self.spatial.weight, self.spatial.bias)
            v_new = torch.cat([v_cache[:, T:], v], dim=1) if T < W - 1 else v[:, T - (W - 1):]
            return v_mixed, v_new.contiguous()
        v_full = torch.cat([v_cache, v], dim=1)
        v_mixed = self.spatial(v_full.transpose(1, 2)).transpose(1, 2)
        return v_mixed, v_full[:, -(W - 1):].contiguous()

    def _spatial_ring(self, v, v_ring, where, read):
        # gather-by-age: inductor fuses the gather into the reduction (no window
        # materialized) and the summation order matches _spatial: exact in
        # eager, within the compile-vs-eager fp16 floor under inductor. roll:
        # eager paths can't fuse a gather, so rotate the (tiny) weights and
        # read the ring in slot order (different order -> fp16-ULP class).
        # fused: roll's read as one kernel (fp32 accumulation), for the grid's
        # eager vmap, where nothing fuses the masked copy, product and sum.
        if read == "fused":
            ptr, cache_len = where
            v_mixed = causal_conv_ring(v_ring, v[:, 0], self.spatial_taps, self.spatial.bias,
                                       cache_len, ptr)
            return v_mixed.unsqueeze(1), v[:, 0]
        idx, valid = where
        W = self.window
        w = self.spatial.weight.squeeze(1)[:, : W - 1]
        if read == "roll":
            hist = torch.where(valid[:, :, None], v_ring.to(v.dtype), 0.0)
            w = w.index_select(1, idx)
        else:
            hist = torch.where(valid[:, :, None], v_ring.index_select(1, idx).to(v.dtype), 0.0)
        v_mixed = (
            (hist * w.t()).sum(dim=1)
            + v[:, 0] * self.spatial.weight.squeeze(1)[:, W - 1]
            + self.spatial.bias
        ).unsqueeze(1)
        return v_mixed, v[:, 0]

    def mix_ring(self, x, v_ring, kv_ring, attn_mask, where, kv_ptr, read="gather"):
        """Serving with both caches as rings: returns the NEW v slot [B, d]
        for the caller to write (the conv reads the frame's own v apart), and
        the attention ring, which this frame's K/V was written into first."""
        xn = self.mix_norm(x)
        u, v = _recomputed_in_backward(self._uv, xn)
        v_mixed, v_new = self._spatial_ring(v, v_ring, where, read)
        attn = self._attend_ring(xn, kv_ring, kv_ptr, attn_mask)
        x = x + self.mix_out(u * (v_mixed + attn))

        x = _swiglu(self, x)

        return x, v_new, kv_ring

    def mix(self, x, v_cache, kv_cache, attn_mask):
        xn = self.mix_norm(x)
        u, v = _recomputed_in_backward(self._uv, xn)
        v_mixed, new_v = self._spatial(v, v_cache)
        attn, new_kv = self._attend(xn, kv_cache, attn_mask)
        x = x + self.mix_out(u * (v_mixed + attn))

        x = _swiglu(self, x)

        return x, new_v, new_kv


def _swiglu(block, x):
    gate, up = block.ffw_in(x).chunk(2, dim=-1)
    return x + block.down(torch.nn.functional.silu(gate) * up)


def use_chunk_start_resets(module: nn.Module) -> None:
    """For BC, whose chunks reset rows only at their first frame."""
    for m in module.modules():
        if isinstance(m, Network):
            m.chunk_start_resets = True


def use_packed_encoder(module: nn.Module) -> None:
    """For the learners: each network's packed input embedding and its core's
    encoder run as one (PackedStructForward.encode), so the one-hot input is
    never built. Serving copies keep the two (the grids vmap them)."""
    for m in module.modules():
        if isinstance(m, StateActionNetwork) and m.packed_embed is not None and hasattr(m.core, "inputs_encoded"):
            m.packed_encoder = m.core.inputs_encoded = True


def use_manual_recurrent_step(module: nn.Module) -> None:
    """Serve every LSTM/GRU in `module` through its hand-rolled one-frame
    step (vmap-able, fp16-faithful) instead of cuDNN."""
    for m in module.modules():
        if isinstance(m, (RecurrentWrapper, RecurrentBlock)):
            m.manual_step = True


class RecurrentBlock(nn.Module):
    """An SGUBlock with its conv + attention mixing replaced by a residual
    GRU or LSTM, as in slippi-ai's tx_like layers: game-long memory in one
    fp32 tensor per row ([d], or [2, d] = (h, c) for the LSTM) instead of a
    window cache. The cell runs in fp32 whatever the autocast (recurrent
    state loses too much in half precision)."""

    def __init__(self, d: int, cell: str):
        super().__init__()
        self.norm = RMSNorm(d)
        self.rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[cell](d, d, batch_first=True)
        self.state_shape = (d,) if cell == "gru" else (2, d)

        hidden = int(8 * d / 3 / 64) * 64
        self.ffw_in = nn.Sequential(
            RMSNorm(d),
            nn.Linear(d, 2 * hidden, bias=False)
        )
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.zeros_(self.down.weight)

    manual_step: bool = False   # serving: see RecurrentWrapper.manual_step

    def mix(self, x, h):
        """x [B, T, d], h [B, *state_shape] fp32 -> (x, h)."""
        xn = self.norm(x)
        with torch.autocast(x.device.type, enabled=False):
            h = h.float()
            if self.manual_step and x.shape[1] == 1:
                if isinstance(self.rnn, nn.LSTM):
                    hn, cn = _lstm_cell(self.rnn, xn[:, 0], *h.unbind(1))
                    out, h = hn[:, None], torch.stack([hn, cn], dim=1)
                else:
                    h = _gru_cell(self.rnn, xn[:, 0], h)
                    out = h[:, None]
            elif isinstance(self.rnn, nn.LSTM):
                out, (hn, cn) = self.rnn(xn.float(), tuple(t[None].contiguous() for t in h.unbind(1)))
                h = torch.stack([hn[0], cn[0]], dim=1)
            else:
                out, hn = self.rnn(xn.float(), h[None].contiguous())
                h = hn[0]
        return _swiglu(self, x + out.to(x.dtype)), h


class SGUCore(Network):
    """Stack of aMLP/SGU blocks, optionally interleaved with recurrent blocks
    (`layout`, one letter per layer: s = SGU, g = GRU, l = LSTM; a recurrent
    layer's state is one fp32 tensor). SGU layer state = ring of last window-1
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
        layout: str = "",
    ):
        super().__init__()
        layout = layout or "s" * num_layers
        assert len(layout) == num_layers and set(layout) <= {"s", "g", "l"}, layout
        self.d = hidden_size
        self.window = window
        self.attn_width = attn_heads * attn_head_dim
        self.encoder = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [SGUBlock(hidden_size, window, attn_heads, attn_head_dim) if kind == "s"
             else RecurrentBlock(hidden_size, {"g": "gru", "l": "lstm"}[kind])
             for kind in layout]
        )
        self.final_norm = RMSNorm(hidden_size)
        self.output_size = hidden_size
        self.ring_read = "gather"   # "roll" or "fused" (serve_fused_ring) for eager (vmap) serving paths

    def serve_fused_ring(self, dtype):
        """Read the ring with one kernel per layer (causal_conv_ring), from
        conv taps kept slot-major in the serving dtype, so no frame re-lays
        them out. Whoever loads new weights refreshes them
        (LeagueAgent.load_slice)."""
        self.ring_read = "fused"
        for block in self.blocks:
            if isinstance(block, SGUBlock):
                block.register_buffer("spatial_taps", ring_taps(block.spatial.weight.detach()).to(dtype),
                                      persistent=False)

    def initial_state(self, batch_size, device=None):
        z = lambda *shape: torch.zeros(*shape, device=device)
        return {
            "cache_len": torch.zeros(batch_size, dtype=torch.long, device=device),
            "layers": [
                (
                    z(batch_size, self.window - 1, self.d),
                    z(batch_size, self.window - 1, 2 * self.attn_width),
                ) if isinstance(block, SGUBlock) else z(batch_size, *block.state_shape)
                for block in self.blocks
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

    def cache_state(self, state, dtype):
        return {
            **state,
            "layers": [tuple(t.to(dtype) for t in layer) if isinstance(layer, tuple) else layer
                       for layer in state["layers"]],
        }

    # ---- serving rings: the v-cache (W-1 slots, written by the agent at ptr)
    # and the attention cache (W slots: the frame's own K/V is written at
    # kv_ptr before it attends). Ring mode is keyed by "ptr" in the state;
    # the learner never sees it (canonical_state). ----
    def initial_ring_state(self, batch_size, device=None):
        s = self.initial_state(batch_size, device)
        s["layers"] = [(layer[0], torch.zeros(batch_size, self.window, 2 * self.attn_width, device=device))
                       if isinstance(layer, tuple) else layer for layer in s["layers"]]
        s["ptr"] = torch.zeros((), dtype=torch.long, device=device)
        s["kv_ptr"] = torch.zeros((), dtype=torch.long, device=device)
        return s

    def _ring_attn_mask(self, kv_ptr, cache_len):
        """[B, 1, 1, W]: slot s holds the entry (kv_ptr - s) mod W frames old,
        valid within cache_len (the frame's own, at kv_ptr, always)."""
        age = (kv_ptr - torch.arange(self.window, device=cache_len.device)) % self.window
        return (age[None, :] <= cache_len[:, None])[:, None, None, :]

    def _ring_index(self, ptr, cache_len, device, roll=False):
        W = self.window
        i = torch.arange(W - 1, device=device)
        if roll:   # slot order: slot s holds canonical position (s - ptr) mod (W-1)
            canon = (i - ptr) % (W - 1)
            return canon, canon[None, :] >= (W - 1 - cache_len)[:, None]
        idx = (i + ptr) % (W - 1)                 # canonical position i -> ring slot
        valid = i[None, :] >= (W - 1 - cache_len)[:, None]
        return idx, valid

    def canonical_state(self, state):
        """Ring state -> the chronological state the learner expects."""
        idx, valid = self._ring_index(state["ptr"], state["cache_len"], state["cache_len"].device)
        W = self.window
        kv_idx = (torch.arange(W - 1, device=idx.device) + state["kv_ptr"] + 1) % W   # the latest W-1, oldest first
        layers = [
            (torch.where(valid[:, :, None], layer[0].index_select(1, idx), 0.0),
             torch.where(valid[:, :, None], layer[1].index_select(1, kv_idx), 0.0))
            if isinstance(layer, tuple) else layer
            for layer in state["layers"]
        ]
        return {"cache_len": state["cache_len"], "layers": layers}

    def step_with_reset(self, inputs, reset, prev_state):
        if "ptr" not in prev_state:
            return super().step_with_reset(inputs, reset, prev_state)
        # stale v-ring and attention-cache entries are masked at read time by
        # cache_len: a reset rewrites neither (a full pass over every cache,
        # every frame), only the recurrent layers' state
        state = {
            "cache_len": torch.where(reset, 0, prev_state["cache_len"]),
            "ptr": prev_state["ptr"],
            "kv_ptr": prev_state["kv_ptr"],
            "layers": [layer if isinstance(layer, tuple)
                       else _mask_state(reset, torch.zeros_like(layer), layer)
                       for layer in prev_state["layers"]],
        }
        return self.step(inputs, state)

    inputs_encoded: bool = False   # the learner's StateActionNetwork applied the encoder (use_packed_encoder)

    def _forward(self, inputs, state):
        T = inputs.shape[1]
        x = inputs if self.inputs_encoded else self.encoder(inputs)
        ring = "ptr" in state
        read = self.ring_read
        if ring:
            assert T == 1, "ring mode is the serving path"
            mask = self._ring_attn_mask(state["kv_ptr"], state["cache_len"])
            where = ((state["ptr"], state["cache_len"]) if read == "fused" else
                     self._ring_index(state["ptr"], state["cache_len"], inputs.device, read == "roll"))
        else:
            mask = self._attn_mask(T, state["cache_len"], inputs.shape[0], inputs.device)
        new_layers = []
        for block, layer in zip(self.blocks, state["layers"]):
            if isinstance(block, RecurrentBlock):
                x, h = block.mix(x, layer)
                new_layers.append(h)
                continue
            v_cache, kv_cache = layer
            if ring:
                x, v_new, nkv = block.mix_ring(x, v_cache, kv_cache, mask, where, state["kv_ptr"], read)
                new_layers.append((v_new, nkv))
            else:
                x, nv, nkv = block.mix(x, v_cache, kv_cache, mask)
                new_layers.append((nv, nkv))
        next_state = {
            "cache_len": torch.clamp(state["cache_len"] + T, max=self.window - 1),
            "layers": new_layers,
        }
        if ring:
            next_state["ptr"] = state["ptr"]      # both advanced by the agent after the frame
            next_state["kv_ptr"] = state["kv_ptr"]
        return self.final_norm(x), next_state

    def step(self, inputs, prev_state):
        out, state = self._forward(inputs[:, None], prev_state)
        return out[:, 0], state

    def unroll(self, inputs, reset, initial_state):
        return self._segmented_unroll(self._forward, inputs, reset, initial_state)


class StateActionNetwork(Network):
    """Embeds StateAction structs, then runs the core network."""

    def __init__(self, embed_game, embed_state_action, core: Network, packed: bool = True,
                 enhanced=None):
        super().__init__()
        from smashbot import embed as embed_lib

        self.embed_game = embed_game
        self.embed_state_action = embed_state_action
        self.core = core
        self.enhanced = enhanced   # an EnhancedEmbed in place of the leaves' embedding
        self.packed_embed = (
            embed_lib.PackedStructForward(embed_state_action) if packed and enhanced is None else None
        )

    packed_encoder: bool = False   # see use_packed_encoder

    def embed_sa(self, state_action) -> torch.Tensor:
        """The core's input: the embedded struct, or with packed_encoder the
        encoder's output already."""
        if self.packed_encoder:
            return self.packed_embed.encode(state_action, self.core.encoder)
        if self.enhanced is not None:
            return self.enhanced(state_action)
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

    def cache_state(self, state, dtype):
        return self.core.cache_state(state, dtype)

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

    enhanced = getattr(network_config, "embed", "simple") == "enhanced"
    if enhanced:   # the enhanced embed runs the items' MLP itself
        embed_config = dataclasses.replace(
            embed_config, items=embed_lib.ItemsConfig(type=embed_lib.ItemsType.FLAT))
    embed_game = embed_config.make_game_embedding()
    embed_state_action = embed_lib.get_state_action_embedding(
        embed_game=embed_game,
        embed_action=controller_embedding,
        num_names=num_names,
    )
    enhanced = embed_lib.EnhancedEmbed(
        embed_state_action, network_config.embed_hidden_size, rating=network_config.rating,
        joint_index_wraps=network_config.embed_joint_index_wraps,
    ) if enhanced else None
    input_size = enhanced.output_size if enhanced is not None else embed_state_action.size
    name = getattr(network_config, "name", "tx_like")
    if name == "tx_like":
        core = TransformerLike(
            input_size=input_size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            ffw_multiplier=network_config.ffw_multiplier,
            recurrent_layer=network_config.recurrent_layer,
            ln_eps=getattr(network_config, "ln_eps", 1e-5),
            gelu_approximate=getattr(network_config, "gelu_approximate", False),
        )
    elif name == "transformer":
        core = TransformerCore(
            input_size=input_size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            num_heads=network_config.num_heads,
            window=network_config.window,
        )
    elif name == "sgu":
        core = SGUCore(
            input_size=input_size,
            hidden_size=network_config.hidden_size,
            num_layers=network_config.num_layers,
            window=network_config.window,
            attn_heads=network_config.attn_heads,
            attn_head_dim=network_config.attn_head_dim,
            layout=network_config.layout,
        )
    else:
        raise ValueError(f"unknown network name: {name}")
    return StateActionNetwork(
        embed_game,
        embed_state_action,
        core,
        packed=getattr(embed_config, "packed", True),
        enhanced=enhanced,
    )
