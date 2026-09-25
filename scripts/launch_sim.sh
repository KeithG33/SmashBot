#!/usr/bin/env bash
# Sim-backend league run from TEACHER, the BC checkpoint it starts from: a
# fresh start, and a relaunch resumes the tag's latest.pt. TAG overrides the
# run name.
# Learner rows = envs, one per game (a self-play env's second seat is served,
# not learned from). v12 fit 400 learner rows / 40 pfsp slices / mb 12 on the
# pre-paper 6/576 SGU (first-step peak 18.6 GiB of 21.2 reserved; 449 rows
# OOM'd), with self-play feeding both seats. 400 envs keeps 400 learner rows
# and serves 520; the slslsl hybrid changes the memory: dry-run before the
# first launch, and size the slices there (140 pfsp envs / 40 = 3.5 per slice).
set -euo pipefail

REPO=/home/kage/smashbot_workspace/SmashBot
SHINE=/home/kage/drive2/ShineBot
TEACHER=${TEACHER:?set TEACHER to the BC checkpoint the run starts from}
TAG=${TAG:-rl-sim-v13}
export PYTHONPATH=$REPO:$REPO/vendor/melee-sim-light
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $REPO
exec $REPO/.venv/bin/python -m smashbot.rl.train_rl \
  --ckpt "$TEACHER" \
  --runtime.device cuda \
  --runtime.run-dir $SHINE/runs --runtime.tag "$TAG" \
  --runtime.restore "$([ -f "$SHINE/runs/$TAG/latest.pt" ] && echo auto || true)" \
  --runtime.steps 50000 \
  --runtime.checkpoint-interval 25 \
  --learner.learning-rate 3e-5 \
  --learner.precision fp16 \
  --learner.micro-batches 6 \
  --learner.kl-teacher-weight 0.075 \
  --learner.kl-teacher-weight-final 0.01 \
  --learner.entropy-weight 1e-4 \
  --learner.imitation-rows -1 \
  --learner.imitation-lambda 0.05 \
  --learner.imitation-lambda-final-frac 0.2 \
  --sim.num-envs 400 \
  --sim.pfsp-slices 40 \
  --sim.unroll-length 240 \
  --sim.snapshot-interval 1500 \
  "$@"
