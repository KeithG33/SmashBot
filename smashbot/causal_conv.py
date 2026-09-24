"""The SGU block's training convolution in Triton: the depthwise causal conv
y[b, t, c] = bias[c] + sum_k w[c, k] * x[b, t + k, c] over x = [cache | chunk]
(cache: the window - 1 frames before the chunk), reading cache and chunk in
place instead of concatenating them. nn.Conv1d runs this 256-tap depthwise
conv on PyTorch's generic kernels, whose backward took a third of the BC step;
these run it 6x faster. Registered as custom ops so a compiled core calls
them as single ops."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(cache, v, w, bias, y, T, C, cache_sb, cache_ss, v_sb, v_ss, y_sb, y_ss,
         W: tl.constexpr, TT: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    b = tl.program_id(1)
    t = tl.program_id(2) * TT + tl.arange(0, TT)
    cm = c < C
    tm = t < T
    acc = tl.zeros((TT, BC), dtype=tl.float32)
    for k in range(W):
        s = t + k
        hist = s < W - 1
        x = tl.load(cache + b * cache_sb + s[:, None] * cache_ss + c[None, :],
                    mask=(hist & tm)[:, None] & cm[None, :], other=0.0)
        x += tl.load(v + b * v_sb + (s - (W - 1))[:, None] * v_ss + c[None, :],
                     mask=(~hist & tm)[:, None] & cm[None, :], other=0.0)
        acc += x.to(tl.float32) * tl.load(w + k * C + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    acc += tl.load(bias + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    tl.store(y + b * y_sb + t[:, None] * y_ss + c[None, :], acc.to(y.dtype.element_ty),
             mask=tm[:, None] & cm[None, :])


@triton.jit
def _bwd_x(gy, w, gcache, gv, T, C, gy_sb, gy_ss, gc_sb, gc_ss, gv_sb, gv_ss, s_start,
           W: tl.constexpr, SS: tl.constexpr, BC: tl.constexpr, CACHE_GRAD: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    b = tl.program_id(1)
    s0 = s_start + tl.program_id(2) * SS
    s = s0 + tl.arange(0, SS)
    cm = c < C
    sm = s < W - 1 + T
    acc = tl.zeros((SS, BC), dtype=tl.float32)
    for k in range(tl.maximum(s0 - (T - 1), 0), tl.minimum(s0 + SS, W)):   # taps that reach this block
        t = s - k
        m = (t >= 0) & (t < T) & sm
        g = tl.load(gy + b * gy_sb + t[:, None] * gy_ss + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        acc += g.to(tl.float32) * tl.load(w + k * C + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    hist = s < W - 1
    if CACHE_GRAD:
        tl.store(gcache + b * gc_sb + s[:, None] * gc_ss + c[None, :], acc.to(gcache.dtype.element_ty),
                 mask=(hist & sm)[:, None] & cm[None, :])
    tl.store(gv + b * gv_sb + (s - (W - 1))[:, None] * gv_ss + c[None, :], acc.to(gv.dtype.element_ty),
             mask=(~hist & sm)[:, None] & cm[None, :])


@triton.jit
def _bwd_w(gy, cache, v, part, B, T, C, B_PER, gy_sb, gy_ss, cache_sb, cache_ss, v_sb, v_ss,
           W: tl.constexpr, KK: tl.constexpr, BC: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    k = tl.program_id(1) * KK + tl.arange(0, KK)
    pb = tl.program_id(2)
    cm = c < C
    km = k < W
    acc = tl.zeros((KK, BC), dtype=tl.float32)
    for bi in range(B_PER):
        b = pb * B_PER + bi
        bm = b < B
        for t in range(T):
            g = tl.load(gy + b * gy_sb + t * gy_ss + c, mask=cm & bm, other=0.0).to(tl.float32)
            s = t + k
            hist = s < W - 1
            x = tl.load(cache + b * cache_sb + s[:, None] * cache_ss + c[None, :],
                        mask=(hist & km & bm)[:, None] & cm[None, :], other=0.0)
            x += tl.load(v + b * v_sb + (s - (W - 1))[:, None] * v_ss + c[None, :],
                         mask=(~hist & km & bm)[:, None] & cm[None, :], other=0.0)
            acc += x.to(tl.float32) * g[None, :]
    tl.store(part + pb * W * C + k[:, None] * C + c[None, :], acc, mask=km[:, None] & cm[None, :])


def _forward(cache, v, weight, bias, TT=16, BC=64):
    B, T, C = v.shape
    W = weight.shape[-1]
    wt = weight.reshape(C, W).to(v.dtype).t().contiguous()   # conv under autocast: weights in the input dtype
    y = v.new_empty(B, T, C)
    _fwd[(triton.cdiv(C, BC), B, triton.cdiv(T, TT))](
        cache, v, wt, bias.to(v.dtype), y, T, C, cache.stride(0), cache.stride(1),
        v.stride(0), v.stride(1), y.stride(0), y.stride(1), W=W, TT=TT, BC=BC)
    return y


def _backward(gy, cache, v, weight, cache_grad, SS=16, BC=64, KK=32, WBC=64, B_PER=16):
    B, T, C = v.shape
    W = weight.shape[-1]
    wt = weight.reshape(C, W).to(v.dtype).t().contiguous()
    gv = v.new_empty(B, T, C)
    gcache = cache.new_empty(cache.shape) if cache_grad else cache.new_empty(0)
    s_start = 0 if cache_grad else W - 1
    _bwd_x[(triton.cdiv(C, BC), B, triton.cdiv(W - 1 + T - s_start, SS))](
        gy, wt, gcache if cache_grad else gv, gv, T, C, gy.stride(0), gy.stride(1),
        gcache.stride(0) if cache_grad else 0, gcache.stride(1) if cache_grad else 0,
        gv.stride(0), gv.stride(1), s_start, W=W, SS=SS, BC=BC, CACHE_GRAD=cache_grad)
    nb = triton.cdiv(B, B_PER)
    part = torch.empty(nb, W, C, device=v.device, dtype=torch.float32)
    _bwd_w[(triton.cdiv(C, WBC), triton.cdiv(W, KK), nb)](
        gy, cache, v, part, B, T, C, B_PER, gy.stride(0), gy.stride(1), cache.stride(0), cache.stride(1),
        v.stride(0), v.stride(1), W=W, KK=KK, BC=WBC)
    gw = part.sum(0).t().contiguous().reshape(weight.shape).to(weight.dtype)
    return gcache, gv, gw, gy.float().sum((0, 1)).to(weight.dtype)


@torch.library.custom_op("smashbot::causal_conv", mutates_args=())
def causal_conv(cache: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """cache [B, W-1, C], v [B, T, C] (channels contiguous), weight [C, 1, W],
    bias [C] -> [B, T, C] in v's dtype."""
    return _forward(cache, v, weight, bias)


@causal_conv.register_fake
def _(cache, v, weight, bias):
    return v.new_empty(v.shape)


@torch.library.custom_op("smashbot::causal_conv_backward", mutates_args=())
def causal_conv_backward(gy: torch.Tensor, cache: torch.Tensor, v: torch.Tensor, weight: torch.Tensor,
                         cache_grad: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward(gy if gy.stride(-1) == 1 else gy.contiguous(), cache, v, weight, cache_grad)


@causal_conv_backward.register_fake
def _(gy, cache, v, weight, cache_grad):
    return (cache.new_empty(cache.shape) if cache_grad else cache.new_empty(0), v.new_empty(v.shape),
            weight.new_empty(weight.shape), weight.new_empty(weight.shape[0]))


def _setup_context(ctx, inputs, output):
    cache, v, weight, _ = inputs
    ctx.save_for_backward(cache, v, weight)


def _grads(ctx, gy):
    cache, v, weight = ctx.saved_tensors
    cache_grad = ctx.needs_input_grad[0]
    gcache, gv, gw, gb = causal_conv_backward(gy, cache, v, weight, cache_grad)
    return (gcache if cache_grad else None), gv, gw, gb


causal_conv.register_autograd(_grads, setup_context=_setup_context)
