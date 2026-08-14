#!/usr/bin/env bash
# battery_all: ONE-SHOT battery runner, then exit. No daemon, no backlog
# crawl -- the battery answers "how strong is the bot NOW".
#
#   DEFAULT (no args):  battery the LATEST snapshot (highest step) in the
#                       live run's snapshot pool.
#   EXPLICIT (args):    battery exactly the named snapshots, sequentially.
#                       Each arg is a snapshot path OR a bare step number
#                       (e.g. "750" -> $RUN_DIR/snapshots/snapshot-0000750.pt).
#
#   cd /home/kage/smashbot_workspace/SmashBot
#   nohup bash scripts/battery_all.sh >> /tmp/battery_all.log 2>&1 &          # latest
#   nohup bash scripts/battery_all.sh 500 1000 >> /tmp/battery_all.log 2>&1 & # history
#
# Snapshots are BARE policy state_dicts (SnapshotPool.save), so battery.py is
# invoked in --snapshot mode with --config-from the run's latest.pt; CPU-only
# inference protects the live train's GPU margin. Result pairing is by step
# string: snapshot-0001250.pt <-> step-0001250.json (both zero-padded, so the
# glob's lexical order IS step order). A snapshot whose result JSON already
# exists is skipped -- delete the JSON to force a re-run.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUN_DIR="${RUN_DIR:-/home/kage/drive2/ShineBot/runs/rl-pool-v3}"
RESULTS="$ROOT/scripts/battery_results"

mkdir -p "$RESULTS"

targets=()
if [ "$#" -eq 0 ]; then
    latest=""
    for snap in "$RUN_DIR"/snapshots/snapshot-*.pt; do
        [ -e "$snap" ] && latest="$snap"  # lexical glob order = step order
    done
    if [ -z "$latest" ]; then
        echo "[battery_all] no snapshots in $RUN_DIR/snapshots/" >&2
        exit 1
    fi
    targets+=("$latest")
else
    for arg in "$@"; do
        case "$arg" in
            *.pt) targets+=("$arg") ;;
            *)    targets+=("$RUN_DIR/snapshots/$(printf 'snapshot-%07d.pt' "$arg")") ;;
        esac
    done
fi

ran=0 failed=0 skipped=0
for snap in "${targets[@]}"; do
    if [ ! -e "$snap" ]; then
        echo "[battery_all] no such snapshot: $snap" >&2
        failed=$((failed + 1))
        continue
    fi
    base="$(basename "$snap" .pt)"
    step="${base#snapshot-}"
    out="$RESULTS/step-${step}.json"
    if [ -f "$out" ]; then
        echo "[battery_all] $base already graded ($out); delete to re-run"
        skipped=$((skipped + 1))
        continue
    fi
    echo "[battery_all $(date +%H:%M:%S)] running battery on $snap"
    if OMP_NUM_THREADS=4 "$PY" "$ROOT/scripts/battery.py" \
        --snapshot "$snap" \
        --config-from "$RUN_DIR/latest.pt" \
        --device cpu; then
        ran=$((ran + 1))
    else
        failed=$((failed + 1))
        echo "[battery_all] FAILED for $snap (rerun battery_all.sh to retry)"
    fi
done
echo "[battery_all $(date +%H:%M:%S)] done: $ran run, $skipped skipped," \
     "$failed failed"
[ "$failed" -eq 0 ]
