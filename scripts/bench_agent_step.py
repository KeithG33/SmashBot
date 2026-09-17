"""Micro-benchmark of the rollout serving frame (BatchedPolicyAgent
execute + infer, flat controllers — the worker's per-frame pattern): ms per
frame at a given batch size with a real compiled policy."""

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
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"],
                    help="agent forward precision (production serves fp16)")
    ap.add_argument("--flats", action="store_true",
                    help="feed the worker's three typed flats (FlatFrames) instead of a leaf struct")
    ap.add_argument("--state-fp16", action="store_true",
                    help="fp16 carried-state buffers (requires --precision fp16)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="manual static-buffer CUDA graph (no per-frame state clone)")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="skip the chunk-boundary state clone (production takes "
                         "it every unroll_length frames, not every frame)")
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
        # a manual graph cannot contain cudagraph trees: compile for kernels only
        policy.sample = torch.compile(
            policy.sample, mode=None if args.capture else args.compile_mode)
    agent = BatchedPolicyAgent(policy, args.n, name_code=1, device=device,
                               batch_steps=1, capture=args.capture,
                               precision=args.precision,
                               state_dtype=torch.float16 if args.state_fp16 else None)
    agent.set_flat_controllers(True)
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)

    ff = None
    if args.flats:
        from smashbot import encode
        from smashbot.rl import sim_env
        ff = sim_env.FlatFrames(device)
        agent.set_flat_inputs(ff.view)

    def state():
        enc = game.from_state(_rand_raw(game, rng, args.n))
        if ff is not None:
            flats = ff.to_device(encode.flatten_typed_batched(enc, args.n))
            return ff.view(flats), flats
        return tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x)).to(device), enc), None
    states = [state() for _ in range(4)]
    resets = torch.zeros(args.n, dtype=torch.bool, device=device)
    snap = not args.no_snapshot

    def frame(i):  # the worker's per-frame pattern: pop, then infer
        agent.execute(())
        st, fl = states[i % 4]
        agent.infer(st, resets, want_snapshot=snap, flats=fl)

    for i in range(30):
        frame(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(args.steps):
        frame(i)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / args.steps * 1e3
    print(f"[{label}] {args.precision}{'+state16' if args.state_fp16 else ''}{'+flats' if args.flats else ''} n={args.n} "
          f"compile={args.compile_mode if args.compile else False} capture={args.capture} "
          f"snapshot={snap}: {ms:.3f} ms/step")
    print(f"[{label}] n={args.n} vram: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
          f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    if args.torch_profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for i in range(50):
                frame(i)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))
    if args.profile:
        pr = cProfile.Profile(); pr.enable()
        for i in range(100):
            frame(i)
        torch.cuda.synchronize(); pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(16)


if __name__ == "__main__":
    main()
