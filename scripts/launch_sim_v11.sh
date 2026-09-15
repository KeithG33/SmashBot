#!/usr/bin/env bash
# Sim-backend league run v11: resume v10 weights/optimizer/pfsp -> 100k steps.
# Locked settings (2026-09-15, v10-parity grid serving, REAL-pool overlapped
# measurements after the harness self_frac fix):
#   num_envs 496 / micro_batches 14 -- 3,585 fps (1.63x v10's ~2,200),
#     co-peak ~17.9 GiB stable (ceiling ~20 per Keith's 2-3GB buffer).
#     Fallbacks: 480/mb14=3,530, 448/mb12=3,390. 512 OOMs (needs learner-state fp16).
#   leash: kl-teacher 0.025 flat (v10's final value, no decay)
#   imitation: lambda 0.01 flat, all rows (imitation_rows=-1)
#   pool: self 30% / phillips 35% (medium 4, plat 6, diamond 7, master 8,
#         gm 10) / PFSP 35% (v10 ghosts + imp9000, imp10000, s9500 fox)
#   snapshots: every 1500, keep all; PFSP hard 0.25 explore 0.075 (v10)
#   NEW wandb id (rl-sim-v11) -- v10's history stays untouched.
# DO NOT run without Keith's go-ahead.
set -euo pipefail

REPO=/home/kage/smashbot_workspace/SmashBot-sim
SHINE=/home/kage/drive2/ShineBot
export PYTHONPATH=$REPO:/home/kage/smashbot_workspace/SmashBot/vendor/melee-sim-light
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $REPO
exec /home/kage/smashbot_workspace/SmashBot/.venv/bin/python -m smashbot.rl.train_rl \
  --backend sim \
  --ckpt $SHINE/models/rl-v4-teacher-frozen-ev07736.pt \
  --runtime.device cuda \
  --runtime.run-dir $SHINE/runs --runtime.tag rl-sim-v11 \
  --runtime.restore $SHINE/runs/rl-pool-v10/latest.pt \
  --runtime.steps 100000 \
  --runtime.checkpoint-interval 25 \
  --runtime.teacher-check-interval 100 \
  --learner.learning-rate 3e-5 \
  --learner.precision fp16 \
  --learner.micro-batches 14 \
  --learner.kl-teacher-weight 0.025 \
  --learner.kl-teacher-weight-final -1 \
  --learner.entropy-weight 1e-4 \
  --learner.imitation-rows -1 \
  --learner.imitation-lambda 0.01 \
  --learner.imitation-lambda-final-frac 1.0 \
  --sim.num-envs 496 \
  --sim.unroll-length 240 \
  --sim.snapshot-interval 1500 \
  --sim.seed-snapshots-from $SHINE/runs/rl-pool-v10/snapshots \
  "$@"
