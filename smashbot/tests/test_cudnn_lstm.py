"""The cuDNN LSTM custom op equals nn.LSTM bit for bit (outputs, final
states, every gradient) in training and inference mode, and a compiled
function using it traces as one graph (fullgraph) across sequence lengths."""
import os

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and os.environ.get("SMASHBOT_GPU_TESTS")),
    reason="cuda-only; set SMASHBOT_GPU_TESTS=1 on an IDLE gpu (never beside a live training run)")


def _inputs(T, B, D, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(*shape, device="cuda", generator=g).requires_grad_()
            for shape in ((T, B, D), (1, B, D), (1, B, D))]


@pytest.mark.parametrize("T,B", [(80, 7), (1, 3), (240, 5)])
@pytest.mark.parametrize("train", [True, False])
def test_matches_nn_lstm(T, B, train):
    from smashbot.cudnn_lstm import cudnn_lstm
    torch.manual_seed(0)
    rnn = nn.LSTM(48, 48).cuda().train(train)
    results = []
    def module(x, h, c):
        out, (hy, cy) = rnn(x, (h, c))
        return [out, hy, cy]

    for run in (module, lambda x, h, c: list(cudnn_lstm(rnn, x, h, c))):
        rnn.zero_grad()
        ins = _inputs(T, B, 48, seed=1)
        outs = run(*ins)
        if train:
            torch.autograd.backward(outs, [torch.ones_like(o) for o in outs])
        results.append([o.detach() for o in outs] + ([t.grad for t in ins] + [p.grad.clone() for p in rnn.parameters()]
                                                    if train else []))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def _block_run(block, x, h):
    block.zero_grad()
    x, h = x.clone().requires_grad_(), h.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, hn = block.mix(x, h)
    (out.float().sum() + hn.sum()).backward()
    return [out, hn, x.grad, h.grad] + [p.grad.clone() for p in block.parameters()]


def test_lstm_block_with_op_matches_module():
    """The hybrid's LSTM block (bf16 autocast, as BC runs it) is unchanged by
    use_cudnn_lstm_op: outputs, state and every gradient bit for bit."""
    from smashbot.networks import RecurrentBlock, use_cudnn_lstm_op
    torch.manual_seed(0)
    block = RecurrentBlock(64, "lstm").cuda()
    x, h = torch.randn(5, 80, 64, device="cuda"), torch.randn(5, 2, 64, device="cuda")
    module = _block_run(block, x, h)
    use_cudnn_lstm_op(block)
    op = _block_run(block, x, h)
    assert all(torch.equal(a, b) for a, b in zip(module, op))


def test_lstm_block_with_op_compiles_as_one_graph():
    from smashbot.networks import RecurrentBlock, use_cudnn_lstm_op
    block = RecurrentBlock(64, "lstm").cuda()
    use_cudnn_lstm_op(block)
    mix = torch.compile(block.mix, dynamic=True, fullgraph=True)
    for T in (80, 33, 1):
        x = torch.randn(4, T, 64, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, hn = mix(x, torch.randn(4, 2, 64, device="cuda"))
        (out.float().sum() + hn.sum()).backward()
        assert x.grad is not None and block.rnn.weight_hh_l0.grad is not None


@pytest.mark.parametrize("autocast", [False, True])
def test_compiles_as_one_graph(autocast):
    """Also under fp16 autocast, as the learner calls it: the op stays fp32."""
    from smashbot.cudnn_lstm import cudnn_lstm
    torch._dynamo.config.allow_rnn = True
    rnn = nn.LSTM(32, 32).cuda()
    f = torch.compile(lambda x, h, c: cudnn_lstm(rnn, x.sin(), h, c)[0].cos(), dynamic=True, fullgraph=True)
    for T in (40, 17, 1):
        x, h, c = _inputs(T, 4, 32, seed=T)
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out = f(x, h, c)
        assert out.dtype == torch.float32
        out.sum().backward()
        assert x.grad is not None and rnn.weight_hh_l0.grad is not None


def _wrapper_run(wrapper, x, state):
    wrapper.zero_grad()
    x = x.clone().requires_grad_()
    state = tuple(t.clone().requires_grad_() for t in state)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, (hy, cy) = wrapper._run(x, state)
    (out.float().sum() + hy.sum() + cy.sum()).backward()
    return [out, hy, cy, x.grad, *(t.grad for t in state)] + [p.grad.clone() for p in wrapper.parameters()]


@pytest.mark.parametrize("cell_dtype", [None, torch.bfloat16])
def test_tx_like_wrapper_with_op_matches_torch_lstm(cell_dtype):
    """tx_like's LSTM through the op equals its torch.lstm path, fp32 and bf16
    cells: outputs, states and every gradient bit for bit."""
    from smashbot.networks import RecurrentWrapper, use_cudnn_lstm_op
    torch.manual_seed(0)
    wrapper = RecurrentWrapper(nn.LSTM(64, 64, batch_first=True)).cuda()
    wrapper.cell_dtype = cell_dtype
    x = torch.randn(5, 80, 64, device="cuda")
    state = tuple(torch.randn(1, 5, 64, device="cuda") for _ in range(2))
    plain = _wrapper_run(wrapper, x, state)
    use_cudnn_lstm_op(wrapper)
    op = _wrapper_run(wrapper, x, state)
    assert all(a.dtype == b.dtype and torch.equal(a, b) for a, b in zip(plain, op))


@pytest.mark.parametrize("cell_dtype", [None, torch.bfloat16])
def test_tx_like_core_with_op_compiles_as_one_graph(cell_dtype):
    """With the op, a tx_like core's whole unroll traces as one graph across
    chunk lengths and matches eager up to the fused glue's rounding."""
    from smashbot.networks import TransformerLike, use_bf16_cells, use_chunk_start_resets, use_cudnn_lstm_op
    torch.manual_seed(0)
    core = TransformerLike(input_size=48, hidden_size=64, num_layers=2).cuda()
    use_chunk_start_resets(core)
    use_cudnn_lstm_op(core)
    if cell_dtype is not None:
        use_bf16_cells(core)
    unroll = torch.compile(core.unroll, dynamic=True, fullgraph=True)
    for T in (80, 33, 80):
        x = torch.randn(4, T, 48, device="cuda", requires_grad=True)
        reset = torch.zeros(4, T, dtype=torch.bool, device="cuda")
        reset[1, 0] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out, _ = unroll(x, reset, core.initial_state(4, "cuda"))
            ref, _ = core.unroll(x, reset, core.initial_state(4, "cuda"))
        assert (out.float() - ref.float()).norm() / ref.float().norm() < 2e-2
        out.float().sum().backward()
        assert x.grad is not None and all(p.grad is not None for p in core.parameters())
