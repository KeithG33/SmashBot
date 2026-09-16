"""Open-loop lockstep check: does fp16 recurrent-state storage change an
opponent's play?

Two LeagueAgents serve the SAME member with fp16 stacked weights (v10's
league setting); one stores carried state fp32, the other fp16. Both are fed
the IDENTICAL pre-recorded observation stream (open loop — a divergent
sample can't cascade), temperature ~0 so sampling is argmax. Reports
per-frame logit deltas and action agreement across a full unroll.

If fp16 state were behaviorally meaningful, argmax actions would diverge;
the KV entries themselves are computed in fp16 under autocast either way,
so the only true perturbation is the LSTM h/c rounding.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import tree

import melee_sim as msl
from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.rl.agent import BatchedPolicyAgent, LeagueAgent
from smashbot.rl import sim_env
from smashbot.rl.sim_league import SimLeague
from smashbot.rl.sim_env import states_to_torch as _states_to_torch

DEV = "cuda"
V10 = "/home/kage/drive2/ShineBot/runs/rl-pool-v10/latest.pt"
MEMBER = "/home/kage/drive2/ShineBot/runs/rl-pool-v10/snapshots/snapshot-0030000.pt"
NENV, FRAMES = 8, 300


def record_stream():
    """Drive player-0 with the student, player-1 neutral; record the
    swapped-view encodes + reset masks (a realistic obs stream)."""
    policy, nm, _ = load_policy(V10, DEV)
    policy.eval()
    sc = resolve_name_code(nm, "Master Player")
    agent = BatchedPolicyAgent(policy, NENV, name_code=sc, device=DEV, precision="fp16")
    agent.set_flat_controllers(True)
    env = msl.EnvBatch(batch_size=NENV, length=64,
                       data_dir=os.environ["MSL_DATA_DIR"])
    env.configure_match(stage=msl.Stage.FINAL_DESTINATION,
                        players=[msl.PlayerConfig(msl.Character.FOX),
                                 msl.PlayerConfig(msl.Character.MARTH)])
    env.reset_all()
    stream, resets = [], []
    reset_np = np.ones(NENV, dtype=bool)
    for _ in range(FRAMES):
        if env.t >= env.length:
            env.reset_cursor()
        obs = env.current_frame
        stream.append(tree.map_structure(np.copy,
                      sim_env.encode_obs(obs, self_slot=1, opp_slot=0)))
        resets.append(reset_np.copy())
        st = _states_to_torch(sim_env.encode_obs(obs), DEV)
        agent.infer(st, torch.as_tensor(reset_np, device=DEV), want_snapshot=False)
        rows = np.stack(agent.execute(np.nonzero(reset_np)[0].tolist()))
        sim_env.write_controller_rows(env, rows, player=0)
        is_r, _ = env.step_and_reset()
        reset_np = np.asarray(is_r, dtype=bool)
    env.close()
    del agent, policy
    torch.cuda.empty_cache()
    return stream, resets


def build_arm(template, state_dtype):
    a = LeagueAgent(template, 1, NENV, 1, DEV, temperature=1e-3,
                    weights_dtype=torch.float16, state_dtype=state_dtype)
    a.load_slice(0, torch.load(MEMBER, map_location=DEV, weights_only=True))
    return a


def main():
    stream, resets = record_stream()
    print(f"recorded {len(stream)} frames x {NENV} envs")

    lg = SimLeague(None, os.path.dirname(MEMBER), phillips={}, fox_imports={},
                   config_from=V10, device=DEV)
    fp32_arm = build_arm(lg.make_grid_template(), None)
    fp16_arm = build_arm(lg.make_grid_template(), torch.float16)

    n_frames = n_agree = 0
    max_rel = 0.0
    disagree_by_frame = []
    for f, (enc, rst) in enumerate(zip(stream, resets)):
        views = tree.map_structure(
            lambda t: t[None], _states_to_torch(enc, DEV))   # [1, NENV, ...]
        r = torch.as_tensor(rst[None], device=DEV)
        rec32 = fp32_arm.infer(views, r)
        rec16 = fp16_arm.infer(views, r)
        both = zip(tree.flatten(rec32.logits), tree.flatten(rec16.logits))
        for l32, l16 in both:
            d = (l32.float() - l16.float()).abs().max().item()
            s = l32.float().abs().max().item() + 1e-6
            max_rel = max(max_rel, d / s)
        agree = 0
        total = 0
        for l32, l16 in zip(tree.flatten(rec32.logits), tree.flatten(rec16.logits)):
            agree += (l32.float().argmax(-1) == l16.float().argmax(-1)).sum().item()
            total += l32.shape[0] * (l32.shape[1] if l32.dim() > 2 else 1)
        n_agree += agree
        n_frames += total
        if agree < total:
            disagree_by_frame.append(f)
    print(f"frames: {FRAMES} | argmax action-component agreement: "
          f"{n_agree}/{n_frames} = {100 * n_agree / n_frames:.3f}%")
    print(f"max relative logit delta: {max_rel:.2e}")
    print(f"frames with any component disagreement: {len(disagree_by_frame)}"
          + (f" (first at frame {disagree_by_frame[0]})" if disagree_by_frame else ""))

    # ---- seat-B twin: BatchedPolicyAgent fp32-state vs fp16-state ----
    del fp32_arm, fp16_arm
    torch.cuda.empty_cache()
    pol, _, _ = load_policy(V10, DEV)
    pol.eval()
    arms = [BatchedPolicyAgent(pol, NENV, name_code=1, device=DEV,
                               precision="fp16", state_dtype=sd,
                               temperature=1e-3)
            for sd in (None, torch.float16)]
    b_agree = b_total = 0
    b_rel = 0.0
    for enc, rst in zip(stream, resets):
        st = _states_to_torch(enc, DEV)
        r = torch.as_tensor(rst, device=DEV)
        recs = [a.infer(st, r, want_snapshot=False)[0][0] for a in arms]
        for l32, l16 in zip(tree.flatten(recs[0].logits), tree.flatten(recs[1].logits)):
            d = (l32.float() - l16.float()).abs().max().item()
            b_rel = max(b_rel, d / (l32.float().abs().max().item() + 1e-6))
            b_agree += (l32.float().argmax(-1) == l16.float().argmax(-1)).sum().item()
            b_total += l32.shape[0] * (l32.shape[1] if l32.dim() > 2 else 1)
    print(f"seat-B (BatchedPolicyAgent): agreement {b_agree}/{b_total} = "
          f"{100 * b_agree / max(1, b_total):.3f}% | max rel logit delta {b_rel:.2e}")
    print("FP16_STATE_CHECK_DONE")


if __name__ == "__main__":
    main()
