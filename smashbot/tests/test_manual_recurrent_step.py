"""RecurrentBlock's hand-rolled one-frame step must match cuDNN's, for both
cells, frame by frame and against a whole unroll (the serving path)."""
import pytest
import torch

from smashbot import networks

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])   # cuda = the cuDNN kernels


def _core(layout, device):
    torch.manual_seed(0)
    c = networks.SGUCore(input_size=32, hidden_size=64, num_layers=2, window=8,
                         attn_heads=2, attn_head_dim=16, layout=layout).eval()
    for p in c.parameters():   # zero-init outputs would hide the cell entirely
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    return c.to(device)


def _serve(core, x, r):
    s = core.initial_state(x.shape[0], x.device)
    outs = []
    for t in range(x.shape[1]):
        o, s = core.step_with_reset(x[:, t], r[:, t], s)
        outs.append(o)
    return torch.stack(outs, 1), s


@pytest.mark.parametrize("device", DEVICES)
def test_manual_step_matches_cudnn_and_unroll(device):
    B, T = 3, 24
    x = torch.randn(B, T, 32, device=device)
    r = torch.zeros(B, T, dtype=torch.bool, device=device)
    r[1, 9] = True
    for layout in ("sl", "sg"):
        core = _core(layout, device)
        with torch.no_grad():
            unrolled, _ = core.unroll(x, r, core.initial_state(B, device))
            cudnn, s_cudnn = _serve(core, x, r)
            networks.use_manual_recurrent_step(core)
            assert core.blocks[1].manual_step
            manual, s_manual = _serve(core, x, r)
        assert torch.allclose(manual, cudnn, atol=1e-5), layout
        assert torch.allclose(manual, unrolled, atol=1e-5), layout
        assert torch.allclose(s_manual["layers"][1], s_cudnn["layers"][1], atol=1e-5), layout


def test_manual_step_only_takes_single_frames():
    core = _core("sl", "cpu")
    networks.use_manual_recurrent_step(core)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        a, _ = core.unroll(x, torch.zeros(2, 5, dtype=torch.bool), core.initial_state(2))
        core.blocks[1].manual_step = False
        b, _ = core.unroll(x, torch.zeros(2, 5, dtype=torch.bool), core.initial_state(2))
    assert torch.equal(a, b)   # multi-frame chunks still go through cuDNN
