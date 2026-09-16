"""Micro-benchmark of BatchedPolicyAgent.step (rollout inference wrapper):
ms per call at a given batch size with a real compiled policy, plus an
optional cProfile of the call. No Dolphins, no worker."""

import argparse
import cProfile
import pstats
import time

import numpy as np
import torch
import tree

from smashbot import configs
from smashbot import embed as embed_lib
from smashbot.eval.game import load_policy
from smashbot.policy import build_policy
from smashbot.rl.agent import BatchedPolicyAgent


def _rand_raw(embedding, rng, n):
    def gen(e):
        if isinstance(e, embed_lib.MLPWrapper):
            return e._embed.map(gen)
        if isinstance(e, embed_lib.DiscreteEmbedding):
            return rng.random((n,), dtype=np.float32)
        if isinstance(e, embed_lib.OneHotEmbedding):
            return rng.integers(0, e.input_size, size=(n,), dtype=np.int64)
        if isinstance(e, embed_lib.BoolEmbedding):
            return rng.integers(0, 2, size=(n,)).astype(bool)
        return (rng.standard_normal((n,)) * 20).astype(np.float32)
    return embedding.map(gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/kage/drive2/ShineBot/models/rl-v4-teacher-frozen-ev07736.pt")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile-mode", default="reduce-overhead",
                    help="torch.compile mode (reduce-overhead = cudagraph trees; max-autotune adds Triton autotuning)")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="autocast dtype for the forward (match the training precision)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--torch-profile", action="store_true",
                    help="torch.profiler: top CUDA kernels by self time over 50 steps")
    # config-spec mode: build a random-init policy of a given architecture
    # instead of loading --ckpt (weights don't affect timing).
    ap.add_argument("--arch", default="", help="tx_like | transformer | sgu; "
                    "if set, build a random policy from --layers/--hidden/etc")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--ffw", type=int, default=2)
    ap.add_argument("--rec", default="lstm")
    args = ap.parse_args()
    device = "cuda"
    if args.arch:
        policy = build_policy(
            embed_config=embed_lib.EmbedConfig(),
            controller_config=embed_lib.ControllerConfig(),
            network_config=configs.NetworkConfig(
                name=args.arch, num_layers=args.layers, hidden_size=args.hidden,
                window=args.window, ffw_multiplier=args.ffw,
                recurrent_layer=args.rec),
            head_config=configs.ControllerHeadConfig(),
            policy_config=configs.PolicyConfig(),
            num_names=16,
        ).to(device)
        label = f"{args.arch} {args.layers}L/{args.hidden}H/w{args.window}"
    else:
        policy, _, _ = load_policy(args.ckpt, device)
        label = args.ckpt.split("/")[-1]
    policy.train_value_head = False
    policy.requires_grad_(False)
    policy.eval()
    if args.compile:
        policy.sample = torch.compile(policy.sample, mode=args.compile_mode)
    agent = BatchedPolicyAgent(policy, args.n, name_code=1, device=device, batch_steps=1)
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)

    def state():
        enc = game.from_state(_rand_raw(game, rng, args.n))
        return tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x)).to(device), enc)
    states = [state() for _ in range(4)]
    import contextlib
    ac = (contextlib.nullcontext() if args.precision == "fp32" else torch.autocast(
        "cuda", dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16))
    ac.__enter__()
    resets = torch.zeros(args.n, dtype=torch.bool, device=device)
    for i in range(30):
        agent.step(states[i % 4], resets)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):
        agent.step(states[i % 4], resets)
    torch.cuda.synchronize()
    ac.__exit__(None, None, None)
    print(f"[{label}] {args.precision} n={args.n} compile={args.compile_mode if args.compile else False}: {(time.perf_counter() - t0) / args.steps * 1e3:.3f} ms/step")
    if args.torch_profile:
        from torch.profiler import profile, ProfilerActivity
        ac.__enter__()
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for i in range(50):
                agent.step(states[i % 4], resets)
            torch.cuda.synchronize()
        ac.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))
    if args.profile:
        pr = cProfile.Profile(); pr.enable()
        for i in range(100):
            agent.step(states[i % 4], resets)
        torch.cuda.synchronize(); pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(16)


if __name__ == "__main__":
    main()
