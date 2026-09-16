#!/usr/bin/env bash
# Sim-backend league run v11: resume v10 weights/optimizer/pfsp -> 100k steps.
# Locked settings (2026-09-15, v10-parity grid serving, REAL-pool overlapped
# measurements after the harness self_frac fix):
#   308 envs = 400 learner rows (self 30% of envs, both seats) / 40 pfsp slices (108 pfsp envs,
#   2.7 per slice) / mb 12. Dry run: first-step peak 18.6 GiB, reserved 21.2 (the edge). 449 rows OOM'd at the first learner
#   step twice (21.4-22.0 GiB allocated) — rows are the only lever that scales every term.
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
  --runtime.run-dir $SHINE/runs --runtime.tag rl-sim-v12 --runtime.wandb-id rl-sim-v12b \
  --runtime.restore "$(
    [ -f $SHINE/runs/rl-sim-v12/latest.pt ] \
      && echo auto \
      || echo $SHINE/runs/rl-pool-v10/latest.pt
  )" \
  --runtime.steps 100000 \
  --runtime.checkpoint-interval 25 \
  --learner.learning-rate 3e-5 \
  --learner.precision fp16 \
  --learner.micro-batches 12 \
  --learner.kl-teacher-weight 0.025 \
  --learner.kl-teacher-weight-final -1 \
  --learner.entropy-weight 1e-4 \
  --learner.imitation-rows -1 \
  --learner.imitation-lambda 0.01 \
  --learner.imitation-lambda-final-frac 1.0 \
  --sim.num-envs 308 \
  --sim.pfsp-slices 40 \
  --sim.unroll-length 240 \
  --sim.snapshot-interval 1500 \
  --sim.seed-snapshots-from $SHINE/runs/rl-pool-v10/snapshots \
  "$@"
