#!/usr/bin/env bash
# Sim-backend league run from the current BC policy: a fresh start, and a
# relaunch resumes the tag's latest.pt. TEACHER (default: train_rl's --ckpt,
# paths.DEFAULT_POLICY) and TAG override.
# The env/row/slice sizes are v12's (2026-09-15), fit for the pre-paper 6/576
# SGU: 308 envs = 400 learner rows (self 30% of envs, both seats) / 40 pfsp
# slices (108 pfsp envs, 2.7 per slice) / mb 12, first-step peak 18.6 GiB of
# 21.2 reserved; 449 rows OOM'd at the first learner step. The paper block and
# the slslsl layout change the memory: dry-run before the first launch.
set -euo pipefail

REPO=/home/kage/smashbot_workspace/SmashBot
SHINE=/home/kage/drive2/ShineBot
TAG=${TAG:-rl-sim-v13}
export PYTHONPATH=$REPO:$REPO/vendor/melee-sim-light
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd $REPO
exec $REPO/.venv/bin/python -m smashbot.rl.train_rl \
  ${TEACHER:+--ckpt "$TEACHER"} \
  --runtime.device cuda \
  --runtime.run-dir $SHINE/runs --runtime.tag "$TAG" \
  --runtime.restore "$([ -f "$SHINE/runs/$TAG/latest.pt" ] && echo auto || true)" \
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
  "$@"
