#!/bin/bash
# Disk-full recovery: parse queued Raw zips ONE at a time (smallest first),
# deleting each after it's recorded as processed, so space frees incrementally.
set -u
ROOT=/home/kage/drive2/ShineBot/data/full/Root
STAGING=/home/kage/drive2/ShineBot/data/full/staging
VENV=/home/kage/smashbot_workspace/ShineBot/.venv/bin/python
PARSE=/home/kage/smashbot_workspace/ShineBot/vendor/slippi-ai/slippi_db/parse_local.py

mkdir -p "$STAGING"
rm -f "$ROOT"/Raw/*.zip.tmp                     # incomplete repacks: junk
mv "$ROOT"/Raw/*.zip "$STAGING"/ 2>/dev/null

# smallest first: builds headroom before tackling the 40GB platinum zips
for z in $(ls -Sr "$STAGING"); do
  avail=$(df --output=avail /home/kage/drive2 | tail -1)
  echo "[drain] next: $z ($(du -h "$STAGING/$z" | cut -f1)), free: $((avail/1024/1024))G"
  if [ "$avail" -lt 2000000 ]; then
    echo "[drain] ABORT: under 2GB free"
    exit 1
  fi
  mv "$STAGING/$z" "$ROOT/Raw/"
  "$VENV" "$PARSE" --root="$ROOT" --threads=40 >/dev/null 2>&1
  if grep -q "\"$z\"" "$ROOT/raw.json" && grep -B1 "\"$z\"" "$ROOT/raw.json" | grep -q true; then
    rm -f "$ROOT/Raw/$z"
    echo "[drain] done: $z"
  else
    echo "[drain] WARNING: $z not marked processed; leaving in Raw"
  fi
done
rmdir "$STAGING" 2>/dev/null
echo "[drain] COMPLETE"
df -h /home/kage/drive2 | tail -1