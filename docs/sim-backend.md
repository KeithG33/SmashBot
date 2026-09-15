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

With the optimized serving (one-pass swapped-view encode + every opponent on
a constant-shape cudagraph via slot skeletons, fp16 rollouts):

| num_envs | micro_batches | fps (overlapped) | learner peak | reserved |
|---:|---:|---:|---:|---:|
| **320 (launch)** | **6** | **3,277** | **12.5 GiB** | **16.0 GiB** |
| 448 | 8 | 3,743 | 16.8 GiB | 20.7 GiB |

v10's Dolphin backend ran ~2,200 fps at 283 envs — 320 is 1.5x that. 448 is
rejected for launch: training's learner_overlap holds the in-flight
trajectory set on top of these sequential-cycle numbers, leaving ~1 GiB.
(Pre-optimization, per-group encodes + eager opponents measured 1,700 fps /
20.8 GiB reserved at 320, and 384 OOM'd — the serving rework bought speed
and memory at once.)

The remaining batch-proportional term is the KV recurrent state of the
windowed-attention "sgu" net (6 layers x window 256 x 576 ≈ 7 MB/env/seat,
fp32) held by the rollout seats AND the learner's carried policy/teacher
states — untouchable by micro_batches. Storing those caches fp16 is the
next memory lever (~halves ≈10 GB of KV at N=320); needs a precision-probe
pass first. Collect still dominates (23s vs 4s learner); merging the ~14
graph replays into one stacked-vmap forward (LeagueAgent proper) is the
next speed lever.
