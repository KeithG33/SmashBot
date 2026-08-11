#!/bin/bash
# Atomically refresh the local RL teacher checkpoint from the BC training box.
# scp overwrite is NOT atomic — fetch to .tmp, then mv (atomic on same fs),
# so train_rl's TeacherWatcher never sees a torn file. Cron-able.
#
# Usage: pull_teacher.sh [dest] (default: drive2 models/mega-best.pt)
set -euo pipefail
BOX="root@194.228.55.129"
PORT=36713
REMOTE="/workspace/runs/smashbot-sgu576w256-batch512-12char/best.pt"
DEST="${1:-/home/kage/drive2/ShineBot/models/mega-best.pt}"

# snapshot remotely first so we never copy best.pt mid-write on the box side
ssh -p "$PORT" "$BOX" "cp $REMOTE /tmp/teacher-snapshot.pt"
scp -q -P "$PORT" "$BOX:/tmp/teacher-snapshot.pt" "$DEST.tmp"
mv "$DEST.tmp" "$DEST"
echo "teacher refreshed: $DEST ($(du -h "$DEST" | cut -f1))"
