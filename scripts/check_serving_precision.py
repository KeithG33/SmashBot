"""Open-loop lockstep check: does serving a policy in half precision change
its play?

Feeds the IDENTICAL recorded observation stream (open loop, argmax actions)
to an fp32 reference and to each serving path we use, and reports greedy
action agreement (Bernoulli for buttons, argmax for bins) and KL to the
reference over whole games so recurrent drift has time to build. Paths: the
league grid (3 stacked fp16 slices, fp16 window caches, hand-rolled cells),
the production student (CUDA-graph capture, fp16 static buffers), the eager
student with and without the hand-rolled cells, and bf16. Each arm samples
its own actions (temperature 1e-3), so after a near-tie flip an arm's later
inputs differ from the reference's: agreement measures play, not numerics
alone. The fp32 cuDNN arm is the floor for that effect. scripts/check_fp16_state.py compared fp16
vs fp32 STATE with the forward in fp16 both times; this compares against a
true fp32 forward.

    PYTHONPATH=.:vendor/melee-sim-light MSL_DATA_DIR=/home/kage/drive2/ShineBot/msl-data \
        python scripts/check_serving_precision.py <checkpoint.pt | phillip tier> [frames]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import tree

import melee_sim as msl
from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.networks import use_manual_recurrent_step
from smashbot.rl.agent import BatchedPolicyAgent, LeagueAgent
from smashbot.rl import sim_env
from smashbot.rl.sim_env import states_to_torch as _states_to_torch

DEV = "cuda"
MODELS = "/home/kage/drive2/ShineBot/models"
DRIVER = f"{MODELS}/master-torch.pt"   # any loadable policy: it only makes the observation stream realistic
NENV = 8


def record_stream(frames):
    """Drive player 0 with a Phillip tier; record player 1's view (the
    seat a Phillip would sit in) and the reset masks. Seeded: the driver
    samples, and agreement counts depend on which near-ties a stream hits."""
    torch.manual_seed(0)
    policy, nm, _ = load_policy(DRIVER, DEV)
    policy.eval()
    agent = BatchedPolicyAgent(policy, NENV, name_code=resolve_name_code(nm, "Master Player"),
                               device=DEV, precision="fp16")
    agent.set_flat_controllers(True)
    env = msl.EnvBatch(batch_size=NENV, length=64, data_dir=os.environ["MSL_DATA_DIR"])
    env.configure_match(stage=msl.Stage.FINAL_DESTINATION,
                        players=[msl.PlayerConfig(msl.Character.FOX),
                                 msl.PlayerConfig(msl.Character.MARTH)])
    env.reset_all()
    stream, resets = [], []
    reset_np = np.ones(NENV, dtype=bool)
    item_slots = sim_env.ItemSlots(NENV)
    for _ in range(frames):
        if env.t >= env.length:
            env.reset_cursor()
        obs = env.current_frame
        items = item_slots.place(obs["items"], reset_np)
        stream.append(tree.map_structure(np.copy, sim_env.encode_obs(obs, items, self_slot=1, opp_slot=0)))
        resets.append(reset_np.copy())
        st = _states_to_torch(sim_env.encode_obs(obs, items), DEV)
        agent.infer(st, torch.as_tensor(reset_np, device=DEV), want_snapshot=False)
        rows = np.stack(agent.execute(np.nonzero(reset_np)[0].tolist()))
        sim_env.write_controller_rows(env, rows, player=0)
        is_r, _ = env.step_and_reset()
        reset_np = np.asarray(is_r, dtype=bool)
    env.close()
    del agent, policy
    torch.cuda.empty_cache()
    return stream, resets


def league_arm(policy, name_code, weights_dtype, state_dtype, slices=1):
    """The league grid's serving path (its template gets the hand-rolled
    cells); `slices` > 1 exercises the vmap over stacked weights, with the
    member loaded in every slice and slice 0 compared."""
    a = LeagueAgent(policy, slices, NENV, name_code, DEV, temperature=1e-3,
                    weights_dtype=weights_dtype, state_dtype=state_dtype)
    for s in range(slices):
        a.load_slice(s, policy.state_dict())
    return lambda st, r: tree.map_structure(
        lambda t: t[:NENV],
        a.infer(tree.map_structure(lambda t: t[None].expand(slices, *t.shape).contiguous(), st),
                r[None].expand(slices, -1).contiguous()).logits)


def batched_arm(policy, name_code, precision, state_dtype, manual, capture=False):
    """The student's serving path; `manual` = hand-rolled recurrent cells,
    `capture` = the production CUDA-graph replay on static buffers."""
    import copy
    policy = copy.deepcopy(policy)
    if manual:
        use_manual_recurrent_step(policy)
    a = BatchedPolicyAgent(policy, NENV, name_code=name_code, device=DEV, precision=precision,
                           state_dtype=state_dtype, temperature=1e-3, capture=capture)
    return lambda st, r: a.infer(st, r, want_snapshot=False)[0][0].logits


def _greedy_and_kl(lr, la):
    """Greedy action and KL(ref|arm) per row: Bernoulli for a button's single
    logit, categorical over the bins otherwise."""
    if lr.shape[-1] == 1:
        lr, la = lr[..., 0], la[..., 0]
        p = torch.sigmoid(lr)
        kl = p * (torch.nn.functional.logsigmoid(lr) - torch.nn.functional.logsigmoid(la)) \
            + (1 - p) * (torch.nn.functional.logsigmoid(-lr) - torch.nn.functional.logsigmoid(-la))
        return lr > 0, la > 0, kl
    p = torch.softmax(lr, -1)
    kl = (p * (torch.log_softmax(lr, -1) - torch.log_softmax(la, -1))).sum(-1)
    return lr.argmax(-1), la.argmax(-1), kl


def compare(name, ref_logits, arm_logits, stats):
    s = stats.setdefault(name, {"agree": 0, "total": 0, "kl": 0.0, "frames_diff": 0, "l1": 0.0, "n": 0})
    frame_diff = False
    for lr, la in zip(tree.flatten(ref_logits), tree.flatten(arm_logits)):
        lr, la = lr.float(), la.float()
        s["l1"] += (lr - la).abs().mean().item(); s["n"] += 1
        gr, ga, kl = _greedy_and_kl(lr, la)
        s["kl"] += kl.mean().item()
        agree = (gr == ga).sum().item()
        total = gr.numel()
        s["agree"] += agree; s["total"] += total
        frame_diff |= agree < total
    s["frames_diff"] += frame_diff


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "gm"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
    path = target if target.endswith(".pt") else f"{MODELS}/{target}-torch.pt"
    stream, resets = record_stream(frames)
    print(f"recorded {len(stream)} frames x {NENV} envs")

    pol, pnm, _ = load_policy(path, DEV)
    pol.requires_grad_(False).eval()
    nc = resolve_name_code(pnm, "Master Player")
    layout = getattr(pol.network.core, "blocks", None)
    print(f"policy: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
          + (f" | blocks {[type(b).__name__ for b in layout]}" if layout is not None else ""))

    ref = league_arm(pol, nc, torch.float32, None)
    arms = {
        "league grid: fp16 weights + fp16 caches, 3 slices": league_arm(pol, nc, torch.float16, torch.float16, slices=3),
        "student (production): captured, fp16, fp16 caches": batched_arm(pol, nc, "fp16", torch.float16, True, capture=True),
        "student: fp16 autocast, manual cells, fp32 state": batched_arm(pol, nc, "fp16", None, True),
        "student: fp16 autocast, cuDNN cells": batched_arm(pol, nc, "fp16", None, False),
        "student: bf16 autocast + bf16 caches, cuDNN cells": batched_arm(pol, nc, "bf16", torch.bfloat16, False),
        "fp32 eager cuDNN cells (harness floor)": batched_arm(pol, nc, "fp32", None, False),
    }

    stats = {}
    for enc, rst in zip(stream, resets):
        st = _states_to_torch(enc, DEV)
        r = torch.as_tensor(rst, device=DEV)
        lref = ref(st, r)
        for name, arm in arms.items():
            compare(name, lref, arm(st, r), stats)

    print(f"\nvs fp32 reference (league path, fp32 weights, manual cells), {frames} frames x {NENV} envs, argmax actions")
    print(f"{'arm':52s} {'component agreement':>20s} {'frames w/ any diff':>19s} {'mean KL(ref|arm)':>17s} {'mean |d|':>9s}")
    for name, s in stats.items():
        print(f"{name:52s} {100 * s['agree'] / s['total']:19.3f}% {s['frames_diff']:19d} {s['kl'] / s['n']:17.2e} {s['l1'] / s['n']:9.4f}")
    print("SERVING_PRECISION_CHECK_DONE")


if __name__ == "__main__":
    main()
