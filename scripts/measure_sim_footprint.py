"""Real combined rollout+learner VRAM footprint for the sim backend.

Builds the actual Learner (v10 config), the sim league rollout (student +
phillips + bounded PFSP pool + harvest), collects one real unroll chunk, runs
learner.step, and reports the VRAM breakdown exactly like train_rl's [vram]
line. This is the number that decides num_envs on the 3090.

Student forward is compiled reduce-overhead (training's mode). Opponents run
eager here (real training also compiles them reduce-overhead -> a bit more
cudagraph-pool VRAM; noted).
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

import melee_sim as msl
from smashbot import saving
from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.rl.config import RLConfig, PPOConfig
from smashbot.rl.ppo import Learner
from smashbot.rl.sim_league import SimLeague, MultiOpponentSimWorker
from smashbot.rl.train_rl import build_value_function

MODELS = "/home/kage/drive2/ShineBot/models"
# tier -> (torch ckpt, frac of N)   (medium weakest 4% ... gm strongest 10%)
PHILLIPS = [
    ("medium", f"{MODELS}/medium-v2-torch.pt", 0.04),
    ("plat",   f"{MODELS}/plat-torch.pt",      0.06),
    ("diamond",f"{MODELS}/diamond-torch.pt",   0.07),
    ("master", f"{MODELS}/master-torch.pt",    0.08),
    ("gm",     f"{MODELS}/gm-torch.pt",        0.10),
]
FOX = {
    "import:s9000": f"{MODELS}/rl-v3-tournament1st-step0009000.pt",
    "import:s10000": f"{MODELS}/rl-best-step0010000-phillip56.pt",
    "import:s9500": "/home/kage/drive2/ShineBot/runs/rl-pool-v3/snapshots/snapshot-0009500.pt",
}


def v10_learner_config(unroll, micro_batches=4):
    return RLConfig(
        learning_rate=3e-5, kl_teacher_weight=0.1206522, kl_teacher_weight_final=0.025,
        entropy_weight=1e-4, reward_halflife=4.0, max_grad_norm=1.0,
        precision="fp16", micro_batches=micro_batches, grad_scaler_growth_interval=500,
        ppo=PPOConfig(num_epochs=1, epsilon=1e-2, beta=0.0,
                      max_mean_actor_kl=1e-4, log_rho_clamp=10.0),
        imitation_rows=-1, imitation_beta=1.0, imitation_w_cap=20.0,
        imitation_lambda=0.01, imitation_lambda_final_frac=0.2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/kage/drive2/ShineBot/runs/rl-pool-v10/latest.pt")
    ap.add_argument("--snapshot-dir", default="/home/kage/drive2/ShineBot/runs/rl-pool-v10/snapshots")
    ap.add_argument("--num-envs", type=int, default=283)
    ap.add_argument("--unroll", type=int, default=240)
    ap.add_argument("--max-pfsp", type=int, default=8)
    ap.add_argument("--micro-batches", type=int, default=4)
    ap.add_argument("--data-dir", default=os.environ.get("MSL_DATA_DIR"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    N, T = args.num_envs, args.unroll

    # --- policies (student / teacher / value), as train_rl.main does ---
    policy, name_map, step = load_policy(args.ckpt, dev)
    policy.train_value_head = False
    teacher, _, _ = load_policy(args.ckpt, dev); teacher.train_value_head = False
    ckpt = saving.load_checkpoint(args.ckpt)
    value_fn = build_value_function(ckpt["config"], dev)
    value_fn.load_state_dict(ckpt["state"]["value"])
    sc = resolve_name_code(name_map, "Master Player")

    learner = Learner(v10_learner_config(T, args.micro_batches), policy, teacher, value_fn)
    # training compiles the rollout student reduce-overhead (train_rl:199)
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 128
    policy.sample = torch.compile(policy.sample, mode="reduce-overhead")

    # --- phillips (full ckpts), compiled as the launch does ---
    phillips = {}
    for tier, path, frac in PHILLIPS:
        if not os.path.exists(path):
            print(f"  (skip phillip {tier}: {path} missing)"); continue
        pol, pnm, _ = load_policy(path, dev); pol.eval(); pol.requires_grad_(False)
        pol.sample = torch.compile(pol.sample, mode="reduce-overhead")
        phillips[tier] = (pol, frac, resolve_name_code(pnm, "Master Player"))
    fox = {k: v for k, v in FOX.items() if os.path.exists(v)}

    self_frac = 1.0 - sum(f for _, _, f in phillips.values()) - 0.35  # phillips + 35% pfsp
    lg = SimLeague(policy, args.snapshot_dir, phillips=phillips, fox_imports=fox,
                   self_frac=self_frac, device=dev,
                   config_from=args.ckpt, self_name_code=sc,
                   compile_fn=lambda s: torch.compile(s, mode="reduce-overhead"))

    rng = random.Random(0)
    part = lg.partition(N, rng, max_pfsp_members=args.max_pfsp)
    print(f"num_envs={N} unroll={T} | groups={len(part)} "
          f"(self+{len(phillips)}phil+{len(part)-1-len(phillips)}pfsp)")
    opponents = []
    slot = 0
    for key, idx in part.items():
        if key == "self" or key.startswith("phillip:"):
            pol, nc = lg.get(key)
        else:
            pol, nc = lg.get(key, slot=slot)
            slot += 1
        opponents.append((key, pol, idx, key != "self", nc))

    chars = [msl.Character.FOX, msl.Character.FALCO, msl.Character.MARTH,
             msl.Character.FALCON, msl.Character.JIGGLYPUFF, msl.Character.PEACH]
    char_pairs = [(rng.choice(chars), rng.choice(chars)) for _ in range(N)]

    w = MultiOpponentSimWorker(policy, opponents, N, T, args.data_dir,
                               msl.Stage.FINAL_DESTINATION, char_pairs,
                               name_code=sc, device=dev, precision="fp16",
                               record_fn=lambda i, gid, s0, s1: lg.record(gid, s0 > s1))
    state = learner.initial_state(N, dev)

    # warmup: trigger compile / cudagraph capture, fill one chunk
    ppo, imit = [], []
    while len(ppo) < 1:
        p, i = w.collect(30); ppo += p; imit += i
    torch.cuda.synchronize()

    # ---- measure: baseline (resident) -> collect -> learner.step (peak) ----
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ppo, imit = [], []
    while len(ppo) < 1:
        p, i = w.collect(30); ppo += p; imit += i
    roll_peak = torch.cuda.max_memory_allocated()
    trajs = ppo + imit
    state, metrics = learner.step(trajs, state, progress=step / 40000)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    resv = torch.cuda.memory_reserved()
    print(f"[vram] N={N}: resident {base/2**30:.2f} GiB "
          f"(weights+KV+graph pools) | rollout peak {roll_peak/2**30:.2f} | "
          f"LEARNER peak {peak/2**30:.2f} GiB | activations {(peak-base)/2**30:.2f} | "
          f"reserved {resv/2**30:.2f} GiB")
    print(f"        ppo_trajs={len(ppo)} imit_trajs={len(imit)} "
          f"| metrics finite={all(np.isfinite(v) for v in metrics.values() if isinstance(v,(int,float)))}")

    # ---- throughput: timed steady-state cycles (sequential = the floor;
    # learner_overlap in training hides the shorter of the two phases) ----
    import time
    for cyc in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ppo, imit = [], []
        while len(ppo) < 1:
            p, i = w.collect(30); ppo += p; imit += i
        torch.cuda.synchronize(); t1 = time.perf_counter()
        state, _m = learner.step(ppo + imit, state, progress=step / 40000)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        frames = N * T
        print(f"[cycle {cyc}] collect {t1-t0:.1f}s ({frames/(t1-t0):,.0f} fps) | "
              f"learner {t2-t1:.1f}s | sequential {frames/(t2-t0):,.0f} fps | "
              f"overlapped-> {frames/max(t1-t0, t2-t1):,.0f} fps", flush=True)
    w.close()


if __name__ == "__main__":
    main()
