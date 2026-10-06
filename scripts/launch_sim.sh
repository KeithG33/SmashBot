#!/usr/bin/env bash
# Sim-backend league run from TEACHER, the BC checkpoint it starts from: a
# fresh start, and a relaunch resumes the tag's latest.pt. TAG overrides the
# run name.
# Learner rows = envs, one per game (a self-play env's second seat is served,
# not learned from): 400 envs train 400 rows and serve 500. Rows are the VRAM
# budget; 100 pfsp envs over 40 slices is 2.5 per slice.
set -euo pipefail

REPO=/home/kage/smashbot_workspace/SmashBot
SHINE=/home/kage/drive2/ShineBot
TEACHER=${TEACHER:?set TEACHER to the BC checkpoint the run starts from}
TAG=${TAG:-rl-sim-v13}
export PYTHONPATH=$REPO:$REPO/vendor/melee-sim-light
# The learner's stream caches its freed memory where serving (default stream)
# can't reuse it: at 560 envs reserved reached 21.7 GiB for a 15.3 GiB peak.
# A cap with garbage collection keeps reserved at 18.8 GiB (step 8.65 -> 8.88 s).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export SMASHBOT_MEM_FRACTION=${SMASHBOT_MEM_FRACTION:-0.80}

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
  --learner.kl-teacher-weight 0.025 \
  --learner.kl-teacher-weight-final 0.0025 \
  --learner.reverse-kl-teacher-weight 0.025 \
  --learner.reverse-kl-teacher-weight-final 0.0025 \
  --learner.entropy-weight 1e-4 \
  --learner.ppo.max-mean-actor-kl 3e-4 \
  --learner.imitation-rows -1 \
  --learner.imitation-lambda 0.05 \
  --learner.imitation-lambda-final-frac 0.2 \
  --sim.num-envs 400 \
  --sim.sim-shards 4 \
  --sim.pfsp-slices 40 \
  --sim.unroll-length 240 \
  --sim.snapshot-interval 1000 \
  "$@"
