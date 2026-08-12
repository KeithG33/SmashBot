#!/bin/bash
# Supervised big-train launcher: relaunches with --runtime.restore auto on any
# crash (watchdog trip, env death), so a hang-turned-crash self-heals in ~20min
# instead of losing a night. First launch starts fresh (restore auto tolerates
# a missing checkpoint). Stop for real with Ctrl-C twice (10s window).
set -u
cd /home/kage/smashbot_workspace/SmashBot
while true; do
  OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python -m smashbot.rl.train_rl \
    --ckpt /home/kage/drive2/ShineBot/models/mega-best.pt \
    --runtime.tag rl-pool-v2 --runtime.steps 20000 \
    --runtime.device cuda --runtime.checkpoint-interval 50 \
    --learner.learning-rate 3e-5 \
    --rollouts.num-envs 128 --rollouts.cpu-envs 8 --rollouts.teacher-envs 32 \
    --rollouts.ref-envs 32 --rollouts.snapshot-slots 4 \
    --rollouts.ref-shard-size 16 --rollouts.no-double-buffer \
    --runtime.restore auto
  code=$?
  if [ $code -eq 0 ]; then echo "run completed (steps done)"; break; fi
  echo "train exited code $code; cleaning up and restoring in 30s (Ctrl-C now to stop for good)"
  sleep 10
  pkill -9 -f "AppImage -e" 2>/dev/null
  pkill -9 -f "ref_server.py" 2>/dev/null
  sleep 20
done
