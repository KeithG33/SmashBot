"""The Triton SGU conv equals nn.Conv1d on [cache | chunk]: the output and every
gradient, in fp32 and bf16, including chunks longer than the window; a reset
row's cache reads as zeros and gets no gradient."""
import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SMASHBOT_GPU_TESTS")),
    reason="cuda-only; set SMASHBOT_GPU_TESTS=1 on an IDLE gpu (never beside a live training run)")


@pytest.mark.parametrize("B,T,C,W", [(3, 80, 70, 256), (2, 300, 64, 256), (4, 5, 64, 256), (2, 17, 33, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_grad", [False, True])
@pytest.mark.parametrize("with_reset", [False, True])
def test_matches_conv1d(B, T, C, W, dtype, cache_grad, with_reset):
    from smashbot.causal_conv import causal_conv
    torch.manual_seed(0)
    reset = (torch.arange(B, device="cuda") % 2 == 0) if with_reset else None
    cache = torch.randn(B, W - 1, C, device="cuda").to(dtype).requires_grad_(cache_grad)
    v = torch.randn(B, T, C, device="cuda").to(dtype).requires_grad_()
    weight = (torch.randn(C, 1, W, device="cuda") * 0.1).requires_grad_()
    bias = torch.randn(C, device="cuda").requires_grad_()
    gy = torch.randn(B, T, C, device="cuda").to(dtype)
    # float64 reference, weights in the input dtype as under autocast
    ref_in = [t.detach().double().requires_grad_() for t in (cache, v, weight.to(dtype), bias.to(dtype))]
    ref_cache = ref_in[0] if reset is None else ref_in[0].masked_fill(reset[:, None, None], 0)
    ref = F.conv1d(torch.cat([ref_cache, ref_in[1]], 1).transpose(1, 2), ref_in[2], ref_in[3],
                   groups=C).transpose(1, 2)
    (ref * gy.double()).sum().backward()
    y = causal_conv(cache, v, weight, bias, reset)
    y.backward(gy)
    got = [y, v.grad, weight.grad, bias.grad] + ([cache.grad] if cache_grad else [])
    want = [ref] + [t.grad for t in ref_in[1:]] + ([ref_in[0].grad] if cache_grad else [])
    tol = 1e-5 if dtype == torch.float32 else 5e-3
    for a, r in zip(got, want):
        assert ((a.double() - r).abs().max() / r.abs().max()).item() < tol
    if cache_grad and with_reset:
        assert cache.grad[reset].eq(0).all()
