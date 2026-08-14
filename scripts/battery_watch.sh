#!/usr/bin/env bash
# battery_watch: every 30 min, look for snapshot-*.pt in the live run's
# snapshot pool that has no matching battery result, and run the battery on
# it. Snapshots are BARE policy state_dicts (SnapshotPool.save), so battery.py
# is invoked in --snapshot mode with --config-from the run's latest.pt.
#
# Result pairing is by step string: snapshot-0001250.pt <-> step-0001250.json.
# A failed battery leaves no JSON and will be retried next cycle.
#
# Deliberately NOT started automatically -- run it by hand when you want a
# standing battery ticker next to the training run:
#   nohup scripts/battery_watch.sh >> /tmp/battery_watch.log 2>&1 &
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUN_DIR="${RUN_DIR:-/home/kage/drive2/ShineBot/runs/rl-pool-v3}"
RESULTS="$ROOT/scripts/battery_results"
INTERVAL="${INTERVAL:-1800}"  # seconds between scans

mkdir -p "$RESULTS"

while true; do
    for snap in "$RUN_DIR"/snapshots/snapshot-*.pt; do
        [ -e "$snap" ] || continue  # glob matched nothing
        base="$(basename "$snap" .pt)"
        step="${base#snapshot-}"
        out="$RESULTS/step-${step}.json"
        if [ ! -f "$out" ]; then
            echo "[battery_watch $(date +%H:%M:%S)] running battery on $snap"
            OMP_NUM_THREADS=4 "$PY" "$ROOT/scripts/battery.py" \
                --snapshot "$snap" \
                --config-from "$RUN_DIR/latest.pt" \
                --device cpu \
                || echo "[battery_watch] battery FAILED for $snap (will retry next cycle)"
        fi
    done
    sleep "$INTERVAL"
done
