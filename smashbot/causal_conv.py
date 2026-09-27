"""The SGU block's training convolution: the depthwise causal conv
y[b, t, c] = bias[c] + sum_k w[c, k] * x[b, t + k, c] over x = [cache | chunk]
(cache: the window - 1 frames before the chunk; a reset row's cache reads as
zeros). Per channel it is one matmul
against the chunk's Toeplitz matrix of that channel's taps, so the conv and
both its gradients run as batched matmuls on tensor cores: inputs in the
activation dtype and fp32 accumulation, as the direct conv kernels had them,
in half their time for the 256-tap window (measured). The matmuls take the
channels first; a tiled Triton kernel transposes (PyTorch's permute copy runs
these 20x slower); the backward reuses the forward's channels-first operand.
Registered as custom ops so a compiled core calls them as single ops."""
from typing import Optional

import torch
import triton
import triton.language as tl

_INDEX: dict = {}


def _toeplitz_index(W, T, device):
    """The matmuls run over [cache | a zero column | chunk], W + T columns,
    so both parts start aligned when W and T are multiples of 8 (cuBLAS then
    runs its aligned kernels). For each (column, output) pair: its tap
    (clamped) and whether one reaches, [W + T, T]; and each tap's entries in
    that flattened matrix, [W, T]."""
    key = (W, T, device)
    if key not in _INDEX:
        col = torch.arange(W + T, device=device)[:, None]
        t = torch.arange(T, device=device)
        k = torch.where(col < W - 1, col, col - 1) - t[None, :]
        pos = torch.arange(W, device=device)[:, None] + t[None, :]
        entries = (pos + (pos >= W - 1).long()) * T + t[None, :]
        _INDEX[key] = (k.clamp(0, W - 1), (col != W - 1) & (k >= 0) & (k < W), entries)
    return _INDEX[key]


def _toeplitz(weight, T, dtype):
    """[C, 1, W] -> [C, W + T, T] in dtype: entry (column, t) is the tap from
    that input column to output t."""
    C, _, W = weight.shape
    tap, reaches, _ = _toeplitz_index(W, T, weight.device)
    return torch.where(reaches, weight.reshape(C, W).to(dtype)[:, tap], 0)


@triton.jit
def _relayout(src, dst, reset, L, C, s_b, s_l, s_c, d_b, d_l, d_c,
              RESET: tl.constexpr, BL: tl.constexpr, BC: tl.constexpr):
    """dst[b, l, c] = src[b, l, c] through each side's strides, in tiles so
    that a transpose between channels-last and channels-first reads and
    writes contiguous runs on both sides; 0 for a reset row b."""
    l = tl.program_id(0) * BL + tl.arange(0, BL)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    b = tl.program_id(2)
    m = (l < L)[:, None] & (c < C)[None, :]
    if RESET:
        m = m & (tl.load(reset + b) == 0)
    x = tl.load(src + b * s_b + l[:, None] * s_l + c[None, :] * s_c, mask=m, other=0)
    m = (l < L)[:, None] & (c < C)[None, :]
    tl.store(dst + b * d_b + l[:, None] * d_l + c[None, :] * d_c, x.to(dst.dtype.element_ty), mask=m)


def _to_channels_first(src, dst, reset=None, BL=64, BC=64):
    """src [B, L, C] into dst [C, B, L] (either may be a strided view)."""
    B, L, C = src.shape
    _relayout[(triton.cdiv(L, BL), triton.cdiv(C, BC), B)](
        src, dst, src if reset is None else reset, L, C, src.stride(0), src.stride(1), src.stride(2),
        dst.stride(1), dst.stride(2), dst.stride(0), RESET=reset is not None, BL=BL, BC=BC)
    return dst


def _to_channels_last(src, dst, reset=None, BL=64, BC=64):
    """src [C, B, L] into dst [B, L, C]."""
    C, B, L = src.shape
    _relayout[(triton.cdiv(L, BL), triton.cdiv(C, BC), B)](
        src, dst, src if reset is None else reset, L, C, src.stride(1), src.stride(2), src.stride(0),
        dst.stride(0), dst.stride(1), dst.stride(2), RESET=reset is not None, BL=BL, BC=BC)
    return dst


def _row(M, T):
    """The operand's row length: M + 1 + T padded to an odd multiple of 8
    elements, aligned for cuBLAS, where a multiple of 16 halves the
    transposes' write bandwidth (measured)."""
    row = -(-(M + 1 + T) // 8) * 8
    return row + 8 * (row // 8 % 2 == 0)


def _channels_first(cache, v, reset):
    """[cache | 0 | v] as the matmuls' operand, [C, B, _row] in v's dtype
    (the first M + 1 + T columns used)."""
    B, T, C = v.shape
    M = cache.shape[1]
    x = v.new_empty(C, B, _row(M, T))
    _to_channels_first(cache, x[:, :, :M], reset)
    x[:, :, M] = 0
    _to_channels_first(v, x[:, :, M + 1:M + 1 + T])
    return x


def _forward(cache, v, weight, bias, reset):
    B, T, C = v.shape
    x = _channels_first(cache, v, reset)
    toe = _toeplitz(weight, T, v.dtype)   # conv under autocast: weights in the input dtype
    y = torch.baddbmm(bias.to(v.dtype)[:, None, None], x[:, :, :toe.shape[1]], toe)
    return _to_channels_last(y, torch.empty_like(v)), x


def _backward(gy, x, weight, reset, M, cache_grad):
    B, T, C = gy.shape
    toe = _toeplitz(weight, T, gy.dtype)
    g = _to_channels_first(gy, gy.new_empty(C, B, T))
    gx = torch.bmm(g, toe[:, 0 if cache_grad else M + 1:].transpose(1, 2))
    gv = _to_channels_last(gx[:, :, -T:], torch.empty_like(gy))
    gcache = _to_channels_last(gx[:, :, :M], gy.new_empty(B, M, C), reset) if cache_grad else gy.new_empty(0)
    fp32_out = {} if gy.dtype == torch.float32 else {"out_dtype": torch.float32}
    gtoe = torch.bmm(x[:, :, :M + 1 + T].transpose(1, 2), g, **fp32_out)   # [C, W + T, T]
    _, _, diagonals = _toeplitz_index(M + 1, T, gy.device)
    gw = gtoe.flatten(1)[:, diagonals].sum(-1)   # each tap's T entries
    return gcache, gv, gw.reshape(weight.shape).to(weight.dtype), gy.float().sum((0, 1)).to(weight.dtype)


@torch.library.custom_op("smashbot::causal_conv", mutates_args=())
def _causal_conv(cache: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                 reset: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward(cache, v, weight, bias, reset)


@_causal_conv.register_fake
def _(cache, v, weight, bias, reset):
    B, T, C = v.shape
    return v.new_empty(v.shape), v.new_empty(C, B, _row(cache.shape[1], T))


@torch.library.custom_op("smashbot::causal_conv_backward", mutates_args=())
def causal_conv_backward(gy: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, reset: Optional[torch.Tensor],
                         M: int, cache_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward(gy, x, weight, reset, M, cache_grad)


@causal_conv_backward.register_fake
def _(gy, x, weight, reset, M, cache_grad):
    B, T, C = gy.shape
    return (gy.new_empty(B, M, C) if cache_grad else gy.new_empty(0), gy.new_empty(gy.shape),
            weight.new_empty(weight.shape), weight.new_empty(weight.shape[0]))


def _setup_context(ctx, inputs, output):
    cache, _, weight, _, reset = inputs
    ctx.mark_non_differentiable(output[1])
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(output[1], weight, reset)
    ctx.M = cache.shape[1]


def _grads(ctx, gy, _):
    x, weight, reset = ctx.saved_tensors
    cache_grad = ctx.needs_input_grad[0]
    gcache, gv, gw, gb = causal_conv_backward(gy, x, weight, reset, ctx.M, cache_grad)
    return (gcache if cache_grad else None), gv, gw, gb, None


_causal_conv.register_autograd(_grads, setup_context=_setup_context)


def causal_conv(cache: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                reset: Optional[torch.Tensor] = None) -> torch.Tensor:
    """cache [B, W-1, C], v [B, T, C] (channels contiguous), weight [C, 1, W],
    bias [C], reset [B] bool or None -> [B, T, C] in v's dtype."""
    reset = None if reset is None else reset.contiguous()   # the kernels read reset[b] at offset b
    return _causal_conv(cache, v, weight, bias, reset)[0]


@triton.jit
def _ring(ring, v, w, bias, cache_len, ptr, y, C,
          ring_ss, ring_sb, ring_sm, v_ss, v_sb, w_ss, bias_ss, cl_ss, cl_sb, ptr_ss, y_ss, y_sb,
          M: tl.constexpr, MM: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    b = tl.program_id(1)
    sl = tl.program_id(2)
    cm = c < C
    p0 = tl.load(ptr + sl * ptr_ss)
    first = M - tl.load(cache_len + sl * cl_ss + b * cl_sb)   # oldest valid position
    acc = tl.zeros((BC,), dtype=tl.float32)
    for s0 in range(0, M, MM):
        s = s0 + tl.arange(0, MM)
        pos = (s - p0 + M) % M                                  # slot s holds this position
        m = ((s < M) & (pos >= first))[:, None] & cm[None, :]
        x = tl.load(ring + sl * ring_ss + b * ring_sb + s[:, None] * ring_sm + c[None, :], mask=m, other=0.0)
        k = tl.load(w + sl * w_ss + pos[:, None] * C + c[None, :], mask=m, other=0.0)
        acc += tl.sum(x.to(tl.float32) * k.to(tl.float32), axis=0)
    now = tl.load(v + sl * v_ss + b * v_sb + c, mask=cm, other=0.0).to(tl.float32)
    acc += now * tl.load(w + sl * w_ss + M * C + c, mask=cm, other=0.0).to(tl.float32)
    acc += tl.load(bias + sl * bias_ss + c, mask=cm, other=0.0).to(tl.float32)
    tl.store(y + sl * y_ss + b * y_sb + c, acc.to(y.dtype.element_ty), mask=cm)


def ring_taps(weight: torch.Tensor) -> torch.Tensor:
    """A conv weight [..., C, 1, W] slot-major, as causal_conv_ring reads it:
    [..., W, C]."""
    return weight.squeeze(-2).transpose(-1, -2).contiguous()


def _ring_forward(ring, v, taps, bias, cache_len, ptr, MM=32, BC=64):
    """Over a leading slice dim: ring [S, B, W-1, C], v [S, B, C], taps
    [S, W, C], bias [S, C], cache_len [S, B], ptr [S] -> [S, B, C]."""
    S, B, M, C = ring.shape
    taps = taps.to(v.dtype).contiguous()
    bias = bias.to(v.dtype)
    y = v.new_empty(S, B, C)
    _ring[(triton.cdiv(C, BC), B, S)](
        ring, v, taps, bias, cache_len, ptr, y, C,
        ring.stride(0), ring.stride(1), ring.stride(2), v.stride(0), v.stride(1), taps.stride(0),
        bias.stride(0), cache_len.stride(0), cache_len.stride(1), ptr.stride(0), y.stride(0), y.stride(1),
        M=M, MM=MM, BC=BC)
    return y


@torch.library.custom_op("smashbot::causal_conv_ring", mutates_args=())
def causal_conv_ring(ring: torch.Tensor, v: torch.Tensor, taps: torch.Tensor, bias: torch.Tensor,
                     cache_len: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """One serving frame of the conv, read from the v-cache ring in slot order:
    ring [B, W-1, C] (slot s holds window position (s - ptr) mod (W-1);
    positions below W-1-cache_len are stale), v [B, C] the frame's own v,
    taps [W, C] (ring_taps of the conv weight), bias [C], cache_len [B],
    ptr [] -> [B, C] in v's dtype. Serving only (no backward)."""
    return _ring_forward(ring[None], v[None], taps[None], bias[None], cache_len[None], ptr[None])[0]


@causal_conv_ring.register_fake
def _(ring, v, taps, bias, cache_len, ptr):
    return v.new_empty(v.shape)


@causal_conv_ring.register_vmap
def _(info, in_dims, ring, v, taps, bias, cache_len, ptr):
    """The grid's slices in one launch: batch dims to the front, unbatched
    inputs broadcast over the slices."""
    lead = lambda t, d: t.movedim(d, 0) if d is not None else t.expand(info.batch_size, *t.shape)
    args = (ring, v, taps, bias, cache_len, ptr)
    return _ring_forward(*(lead(t, d) for t, d in zip(args, in_dims))), 0
