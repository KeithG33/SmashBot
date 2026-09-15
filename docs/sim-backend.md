# Sim training backend (melee-sim-light)

RL rollouts run on [melee-sim-light](https://github.com/kyhavlov/melee-sim-light)
(a deterministic batched SSBM sim built on the decompilation) instead of the
Dolphin fleet. **Training only** — watching, playing, tournaments, and eval
stay on Dolphin; a sim-trained model transfers because the sim is ~1:1 parity
with the game. The Dolphin RL path (`DolphinRolloutWorker` + `LeagueRuntime`)
remains the proven fallback until a sim run has validated end-to-end.

## Setup (once per machine)

```bash
git clone https://github.com/kyhavlov/melee-sim-light vendor/melee-sim-light
cd vendor/melee-sim-light
make python-release PY=/path/to/.venv/bin/python   # NOT python-library (-O0, ~2x slower)
cp build/melee_core/python-release/libmelee_core.so melee_sim/libmelee_core.so
# game data, extracted once from our ISO:
#   MSL_DATA_DIR=/home/kage/drive2/ShineBot/msl-data
```

`vendor/` is gitignored (pristine upstream clone + 720M of build artifacts).
Add `vendor/melee-sim-light` to `PYTHONPATH` (the launch script does).

## Architecture

```
train_rl --backend sim                        (rl/train_rl.py dispatch)
  └─ rl/train_sim.py   run(): learner loop — reuses Learner, overlap
     │                 pipeline, checkpoint schema, TeacherWatcher
     ├─ SimLeagueWorker: period lifecycle — SimLeague.partition() →
     │                   MultiOpponentSimWorker per period, coarse
     │                   re-partition every repartition_interval steps,
     │                   GameTracker per class, outcome recording
     └─ rl/sim_league.py
        ├─ SimLeague: pool + assignment. Shares: self (no harvest) /
        │             phillip tiers / PFSP (SnapshotPool draw + pfsp.json).
        │             Bare-state members (snapshots, fox imports) built
        │             from config_from. max_pfsp_members bounds resident
        │             policies per period.
        └─ MultiOpponentSimWorker: per-frame loop — student on player-0,
                       one forward per opponent group on the slot-swapped
                       view, harvest-all-but-self as kind="imitation"
                       (phillip chunks re-encoded through the student
                       embedding + name reconditioned: make_reencoder),
                       rewards, ChunkAssembler, step_and_reset.
  rl/sim_env.py       obs → encoded Game struct; decoded controllers → sim
  rl/sim_rollout.py   single-opponent reference worker (benchmarks)
```

## Launch

`scripts/launch_sim_v11.sh` — resume v10 → 100k with the locked pool
(self 30% / phillips 35%: medium 4, plat 6, diamond 7, master 8, gm 10 /
PFSP 35%: v10 ghosts + imp9000, imp10000, s9500). Seeds the new run's
snapshot dir from v10 (symlinks + pfsp.json with ghost keys rewritten).

## Memory + throughput (RTX 3090, 24 GB — scripts/measure_sim_footprint.py)

Serving is v10's league stack ported to the sim: one swapped-view encode +
GPU tree-slices; PFSP slots on ONE LeagueAgent grid (stacked fp16 weights +
fp16 carried state, captured vmap forward, in-place load_slice on member
swap); phillips on 5 constant-shape reduce-overhead graphs (their LSTM has
no vmap rule); self on the student's compiled graph; vectorized [N,13] row
controller writes. fp16 opponent state verified BIT-IDENTICAL to fp32
storage over a 300-frame lockstep stream (scripts/check_fp16_state.py) —
the forward computes fp16 under autocast either way.

REAL-pool overlapped measurements (`--overlap` = the actual training
pipeline; an earlier harness bug had collapsed the pool to pfsp-only and
those numbers were retracted):

| num_envs | micro_batches | fps (overlapped) | co-peak | reserved |
|---:|---:|---:|---:|---:|
| **496 (launch)** | **14** | **3,585** | **~17.9 GiB** | stable |
| 480 | 14 | 3,530 | 17.2 GiB | stable |
| 448 (fallback) | 12 | 3,390 | 16.4 GiB | stable |
| 512 | 14-16 | OOM at overlap co-peak | — | — |

v10's Dolphin backend ran ~2,200 fps at 283 envs — 448 is 1.54x that with
the full pool (self 139 / phillips 157 / 8 grid slots x 19).

To unlock 512+: the learner's carried policy/teacher recurrent states
(fp32, ~7 MB/env each) are the remaining batch-proportional term — storing
them fp16 needs a precision-probe pass (they enter loss computation, unlike
opponent state). The self-play opponent seat's fp32 KV is also convertible
under the (now-verified) opponent-state argument.
