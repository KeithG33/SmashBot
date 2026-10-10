"""tx_like's LSTM/GRU runs at its weights' dtype whatever the autocast (fp32,
where autocast alone gives cuDNN's RNNs fp16), or bf16 when BC opts in."""
import os

import pytest
import torch
import tree
from torch import nn

from smashbot import networks

# cuda = the cuDNN kernels, where autocast's fp16 RNN lives; opt-in like every
# GPU test, so a routine run never touches a GPU that a live training run may own
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and os.environ.get("SMASHBOT_GPU_TESTS") else [])


def _wrapper(cell, device):
    torch.manual_seed(0)
    return networks.RecurrentWrapper(cell(16, 16, batch_first=True)).to(device)


def _core(device):
    torch.manual_seed(0)
    return networks.TransformerLike(input_size=16, hidden_size=32, num_layers=2).to(device)


def _unroll(core, x):
    reset = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    reset[1, 7] = True
    return core.unroll(x, reset, core.initial_state(x.shape[0], x.device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cell", [nn.LSTM, nn.GRU])
def test_fp32_cell_is_the_module_call(device, cell):
    wrapper = _wrapper(cell, device)
    x = torch.randn(3, 12, 16, device=device)
    state = wrapper.initial_state(3, device)
    with torch.no_grad():
        out, next_state = wrapper._run(x, state)
        ref_out, ref_state = wrapper._core(x, state)
    assert torch.equal(out, ref_out)
    for a, b in zip(tree.flatten(next_state), tree.flatten(ref_state)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_autocast_leaves_the_cells_in_fp32(device):
    core = _core(device)
    x = torch.randn(3, 12, 16, device=device)
    with torch.no_grad():
        ref, _ = _unroll(core, x)
        with torch.autocast(device, dtype=torch.bfloat16):
            mixed, state = _unroll(core, x)
            wrapper = core._layers[1]._net
            cell_out, _ = wrapper._run(x.new_empty(3, 12, 32).normal_(), wrapper.initial_state(3, device))
    assert cell_out.dtype == torch.float32
    assert all(t.dtype == torch.float32 for t in tree.flatten(state))
    assert (mixed.float() - ref).norm() / ref.norm() < 3e-2   # bf16 linears, fp32 cells


@pytest.mark.parametrize("device", DEVICES)
def test_bf16_cells_track_fp32_and_train_the_fp32_weights(device):
    x = torch.randn(3, 12, 16, device=device)
    runs = {}
    for bf16 in (False, True):
        core = _core(device)
        if bf16:
            networks.use_bf16_cells(core)
        wrappers = [m for m in core.modules() if isinstance(m, networks.RecurrentWrapper)]
        assert len(wrappers) == 2 and all(w.cell_dtype == (torch.bfloat16 if bf16 else None)
                                          for w in wrappers)
        with torch.autocast(device, dtype=torch.bfloat16):
            out, state = _unroll(core, x)
        out.float().square().mean().backward()
        grads = torch.cat([p.grad.flatten() for w in wrappers for p in w.parameters()])
        assert all(p.dtype == torch.float32 for w in wrappers for p in w.parameters())
        assert all(t.dtype == torch.float32 for t in tree.flatten(state))
        runs[bf16] = out.float(), grads
    (out32, g32), (out16, g16) = runs[False], runs[True]
    assert (out16 - out32).norm() / out32.norm() < 3e-2
    assert torch.isfinite(g16).all() and torch.cosine_similarity(g16, g32, dim=0) > 0.99
