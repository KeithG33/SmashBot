"""A checkpoint with QK-norm on its SGU tiny attention (network.qk_norm), the
new per-head q and k gains fitted so each head's attention distribution on
real frames matches the original's; optionally a wider value window, its new
conv taps zero at the oldest lags so the conv computes what it did.

A head sharper than --cap (mean top-minus-mean logit) is fitted to its
attention at the temperature that brings it to the cap: copying a saturated
head (the hybrid's block 0: logits ~6300) would rebuild the saturation.
Reports the step-0 policy and value loss of the original and the converted
model on held-out frames.

    python scripts/add_qk_norm.py <in.pt> <out.pt> --data-dir D --meta-path M \
        [--value-window 80] [--cap 30]
"""
from __future__ import annotations

import argparse
import copy
import dataclasses

import torch
from slippi_ai.data import DatasetConfig

from smashbot import configs, saving
from smashbot.data import loader
from smashbot.delay import slice_delayed_frames
from smashbot.networks import SGUBlock
from smashbot.policy import build_policy_from_config
from smashbot.value import build_value_function


def frame_batches(cfg: dict, name_map: dict, data_dir: str, meta_path: str, rows: int, n: int, seed: int):
    """n consecutive raw batches of `rows` test-split games, each row seated
    mid-game."""
    data = configs.from_dict(configs.DataConfig, {
        **cfg["data"], "dataset": dataclasses.replace(
            configs.from_dict(DatasetConfig, cfg["data"]["dataset"]), data_dir=data_dir, meta_path=meta_path)})
    data = dataclasses.replace(data, batch_size=rows, num_workers=0)
    delay = cfg["policy"]["delay"]
    sources = loader.make_sources(data, delay + 1, name_map=name_map)
    loader.seat_mid_game(sources.test, span=(n + 1) * (data.unroll_length + delay + 1), seed=seed, num_workers=0)
    batches = [next(sources.test)[0].batch for _ in range(n)]
    sources.test.shutdown()
    sources.train.shutdown()
    return batches


def widen_value_window(state: dict, value) -> dict:
    """state's conv taps zero-padded at the oldest lags to value's window."""
    target = value.state_dict()
    out = dict(state)
    for k, v in state.items():
        if k.endswith("spatial.weight") and v.shape != target[k].shape:
            out[k] = torch.cat([v.new_zeros(*v.shape[:-1], target[k].shape[-1] - v.shape[-1]), v], dim=-1)
    return out


def run(models: list, batches: list, delay: int, discount: float, warm: int) -> list:
    """Each (policy, value) pair's mean policy loss and value loss over the
    batches after the first `warm` (which only warm the recurrent state)."""
    out = []
    for policy, value in models:
        hp, hv = policy.initial_state(len(batches[0].game.stage)), value.initial_state(len(batches[0].game.stage))
        pl, vl = [], []
        for i, b in enumerate(batches):
            frames = slice_delayed_frames(loader.batch_to_frames(b, policy.network), delay)
            o = policy.unroll(frames, hp)
            hp = o.final_state
            loss, hv, _ = value.loss(frames, hv, discount)
            if i >= warm:
                pl.append(-o.log_probs.mean().item())
                vl.append(loss.item())
        out.append((sum(pl) / len(pl), sum(vl) / len(vl)))
    return out


class AttentionRecorder:
    """Records every SGU block's raw q, keys and mask (as its _attend_over
    sees them) while `on`."""

    def __init__(self):
        self.on, self.seen = False, {}
        original = SGUBlock._attend_over
        recorder = self

        def attend_over(block, q, kv, attn_mask):
            if recorder.on:
                keys, _ = kv.chunk(2, dim=-1)
                heads = lambda t: t.unflatten(-1, (block.attn_heads, -1)).transpose(1, 2).float()
                recorder.seen.setdefault(block, []).append((heads(q), heads(keys), attn_mask))
            return original(block, q, kv, attn_mask)

        SGUBlock._attend_over = attend_over


def _masked(logits, mask):
    if mask is None:
        return logits
    return logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.finfo(x.dtype).eps)


def fit_gains(samples: list, cap: float, steps: int = 300) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Per-head q and k gains [h, d] whose normalized attention matches the
    raw attention in `samples` [(q, keys, mask)], each head's target softened
    to the cap. Returns the gains and the fit's report."""
    scale = samples[0][0].shape[-1] ** -0.5
    old = [_masked(q @ k.transpose(-1, -2) * scale, m) for q, k, m in samples]
    unit = [(_rms(q), _rms(k), m) for q, k, m in samples]

    def spread(logits):   # mean top-minus-mean logit per head
        finite = torch.isfinite(logits)
        top = logits.masked_fill(~finite, -1e30).amax(-1)
        mean = logits.masked_fill(~finite, 0).sum(-1) / finite.sum(-1)
        return (top - mean).mean(dim=(0, 2))

    old_spread = torch.stack([spread(lg) for lg in old]).mean(0)
    temperature = (old_spread / cap).clamp_min(1.0)
    targets = [torch.softmax(lg / temperature[None, :, None, None], -1) for lg in old]

    unit_spread = torch.stack([spread(_masked(q @ k.transpose(-1, -2) * scale, m)) for q, k, m in unit]).mean(0)
    start = (old_spread / temperature / unit_spread).sqrt()
    h, d = samples[0][0].shape[1], samples[0][0].shape[-1]
    g_q = (start[:, None] * torch.ones(h, d)).requires_grad_()
    g_k = (start[:, None] * torch.ones(h, d)).requires_grad_()
    opt = torch.optim.Adam([g_q, g_k], lr=0.05)

    def kl():
        total = 0.0
        for (q, k, m), p in zip(unit, targets):
            logits = _masked((q * g_q[None, :, None]) @ (k * g_k[None, :, None]).transpose(-1, -2) * scale, m)
            logp = torch.log_softmax(logits, -1)
            total = total + (p * (torch.log(p.clamp_min(1e-30)) - logp)).nan_to_num().sum(-1).mean()
        return total / len(unit)

    start_kl = kl().item()
    for _ in range(steps):
        opt.zero_grad()
        loss = kl()
        loss.backward()
        opt.step()
    with torch.no_grad():
        new_spread = torch.stack([
            spread(_masked((q * g_q[None, :, None]) @ (k * g_k[None, :, None]).transpose(-1, -2) * scale, m))
            for q, k, m in unit]).mean(0)
    return g_q.detach(), g_k.detach(), {
        "old_spread": old_spread.tolist(), "temperature": temperature.tolist(),
        "new_spread": new_spread.tolist(), "kl_start": start_kl, "kl_end": kl().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--meta-path", required=True)
    ap.add_argument("--value-window", type=int, default=0)
    ap.add_argument("--cap", type=float, default=30.0)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--warm", type=int, default=4, help="batches that only warm the state (window 256 = 3.2)")
    ap.add_argument("--fit-batches", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(8)

    ckpt = saving.load_checkpoint(args.src)
    cfg = ckpt["config"]
    new_cfg = copy.deepcopy(cfg)
    new_cfg["network"]["qk_norm"] = True
    if args.value_window:
        new_cfg["value"]["window"] = args.value_window
    delay = cfg["policy"]["delay"]
    discount = 0.5 ** (1 / (cfg["value"]["reward_halflife"] * 60))

    policy, value = build_policy_from_config(cfg), build_value_function(cfg, "cpu")
    policy.load_state_dict(ckpt["state"]["policy"])
    value.load_state_dict(ckpt["state"]["value"])
    new_policy, new_value = build_policy_from_config(new_cfg), build_value_function(new_cfg, "cpu")
    for model, state in ((new_policy, ckpt["state"]["policy"]),
                         (new_value, widen_value_window(ckpt["state"]["value"], new_value))):
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.endswith(("q_norm.weight", "k_norm.weight")) for k in missing), (missing, unexpected)
    for m in (policy, value, new_policy, new_value):
        m.eval()
        m.requires_grad_(False)

    name_map = ckpt["state"]["name_map"]
    fit = frame_batches(cfg, name_map, args.data_dir, args.meta_path, args.rows, args.warm + args.fit_batches, seed=11)
    held_out = frame_batches(cfg, name_map, args.data_dir, args.meta_path, args.rows, args.warm + args.eval_batches, seed=29)

    recorder = AttentionRecorder()
    with torch.no_grad():
        for model, hidden_of in ((policy, policy.initial_state), (value, value.initial_state)):
            hidden = hidden_of(args.rows)
            for i, b in enumerate(fit):
                frames = slice_delayed_frames(loader.batch_to_frames(b, policy.network), delay)
                recorder.on = i >= args.warm
                if model is policy:
                    hidden = model.unroll(frames, hidden).final_state
                else:
                    _, hidden, _ = model.loss(frames, hidden, discount)
            recorder.on = False

    report = {}
    for model, new_model, label in ((policy, new_policy, "policy"), (value, new_value, "value")):
        old_blocks = [m for m in model.modules() if isinstance(m, SGUBlock)]
        new_blocks = [m for m in new_model.modules() if isinstance(m, SGUBlock)]
        for i, (old_b, new_b) in enumerate(zip(old_blocks, new_blocks)):
            g_q, g_k, rep = fit_gains(recorder.seen[old_b], args.cap)
            new_b.q_norm.weight.data.copy_(g_q)
            new_b.k_norm.weight.data.copy_(g_k)
            report[f"{label}_sgu{i}"] = rep
            print(f"{label} SGU block {i}: spread {[round(x, 1) for x in rep['old_spread']]} -> "
                  f"{[round(x, 1) for x in rep['new_spread']]} (temperature {[round(x, 1) for x in rep['temperature']]}), "
                  f"KL {rep['kl_start']:.3f} -> {rep['kl_end']:.4f}", flush=True)

    with torch.no_grad():
        (p0, v0), (p1, v1) = run([(policy, value), (new_policy, new_value)], held_out, delay, discount, args.warm)
    report.update(policy_loss_before=p0, policy_loss_after=p1, value_loss_before=v0, value_loss_after=v1, cap=args.cap)
    print(f"held-out policy loss {p0:.4f} -> {p1:.4f} ({p1 - p0:+.4f}), value loss {v0:.4f} -> {v1:.4f}", flush=True)

    state = {**ckpt["state"], "policy": new_policy.state_dict(), "value": new_value.state_dict(),
             "qk_norm": report}
    torch.save({**ckpt, "config": new_cfg, "state": state}, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
