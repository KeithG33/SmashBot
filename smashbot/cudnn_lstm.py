"""cuDNN's LSTM as custom ops, so a compiled core traces the hybrid's LSTM
layers as single nodes instead of breaking its graph around them (dynamo can't
trace nn.LSTM; its breaks split SGUCore._forward into pieces that the policy,
teacher and value net share, whose variants multiplied past the recompile
limit). The ops call the kernels nn.LSTM runs, aten._cudnn_rnn and
aten._cudnn_rnn_backward, wired as PyTorch's own autograd formula for
_cudnn_rnn wires them (the backward takes the forward's output, reserve and
weight_buf): bit-identical to nn.LSTM, outputs and gradients. One layer,
unidirectional, time-major, fp32 whatever the autocast."""
import torch
from torch import Tensor

_LSTM = 2   # cuDNN's mode id


@torch.library.custom_op("smashbot::cudnn_lstm", mutates_args=())
def _cudnn_lstm(x: Tensor, h0: Tensor, c0: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor,
                train: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    with torch.autocast(x.device.type, enabled=False):   # fp32 whatever the caller's autocast (it casts _cudnn_rnn)
        out, hy, cy, reserve, weight_buf = torch.ops.aten._cudnn_rnn(
            x, [w_ih, w_hh, b_ih, b_hh], 4, None, h0, c0, _LSTM, w_hh.shape[1], 0, 1, False, 0.0, train, False, [],
            None)
    return out, hy, cy, reserve, weight_buf


@_cudnn_lstm.register_fake
def _(x, h0, c0, w_ih, w_hh, b_ih, b_hh, train):
    T, B, _ = x.shape
    reserve = x.new_empty(torch.library.get_ctx().new_dynamic_size(), dtype=torch.uint8)   # cuDNN sizes it
    weight_buf = x.new_empty(sum(w.numel() for w in (w_ih, w_hh, b_ih, b_hh)))
    return x.new_empty(T, B, w_hh.shape[1]), h0.new_empty(h0.shape), c0.new_empty(c0.shape), reserve, weight_buf


@torch.library.custom_op("smashbot::cudnn_lstm_backward", mutates_args=())
def _cudnn_lstm_backward(x: Tensor, h0: Tensor, c0: Tensor, w_ih: Tensor, w_hh: Tensor, b_ih: Tensor, b_hh: Tensor,
                         out: Tensor, reserve: Tensor, weight_buf: Tensor, g_out: Tensor, g_hy: Tensor, g_cy: Tensor
                         ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    with torch.autocast(x.device.type, enabled=False):
        gx, gh0, gc0, gw = torch.ops.aten._cudnn_rnn_backward(
            x, [w_ih, w_hh, b_ih, b_hh], 4, weight_buf, h0, c0, out, g_out, g_hy, g_cy,
            _LSTM, w_hh.shape[1], 0, 1, False, 0.0, True, False, [], None, reserve, [True, True, True, True])
    return (gx, gh0, gc0, *(g.clone() for g in gw))   # views of one buffer: an op's outputs may not alias


@_cudnn_lstm_backward.register_fake
def _(x, h0, c0, w_ih, w_hh, b_ih, b_hh, out, reserve, weight_buf, g_out, g_hy, g_cy):
    return tuple(t.new_empty(t.shape) for t in (x, h0, c0, w_ih, w_hh, b_ih, b_hh))


def _setup_context(ctx, inputs, output):
    *tensors, ctx.train = inputs
    out, _, _, reserve, weight_buf = output
    ctx.mark_non_differentiable(reserve, weight_buf)
    ctx.set_materialize_grads(False)   # no zeros for the buffers; the three cuDNN needs are made below
    ctx.save_for_backward(*tensors, out, reserve, weight_buf)


def _grads(ctx, g_out, g_hy, g_cy, _reserve, _weight_buf):
    if not ctx.train:   # as nn.LSTM: an inference-mode forward keeps no reserve
        raise RuntimeError("cudnn RNN backward can only be called in training mode")
    x, h0, c0, w_ih, w_hh, b_ih, b_hh, out, reserve, weight_buf = ctx.saved_tensors
    fill = lambda g, like: torch.zeros_like(like) if g is None else g.contiguous()
    grads = _cudnn_lstm_backward(x, h0, c0, w_ih, w_hh, b_ih, b_hh, out, reserve, weight_buf,
                                 fill(g_out, out), fill(g_hy, h0), fill(g_cy, c0))
    return (*grads, None)


_cudnn_lstm.register_autograd(_grads, setup_context=_setup_context)


def cudnn_lstm(rnn: torch.nn.LSTM, x: Tensor, h0: Tensor, c0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """nn.LSTM(x, (h0, c0)) on CUDA, time-major: x [T, B, d], h0/c0 [1, B, H]
    -> (out [T, B, H], hy, cy). Training mode as the module's (as nn.LSTM)."""
    out, hy, cy, _, _ = _cudnn_lstm(x, h0, c0, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0,
                                    rnn.training)
    return out, hy, cy
