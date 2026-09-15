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

## Memory (RTX 3090, 24 GB — measured via scripts/measure_sim_footprint.py)

| num_envs | micro_batches | learner peak | reserved |
|---:|---:|---:|---:|
| 283 (v10) | 4 | 14.8 GiB | 18.5 GiB |
| **320 (launch)** | **6** | **12.5 GiB** | **16.0 GiB** |
| 352 | 8 | 17.9 GiB | 21.1 GiB |
| 384 | 8 | OOM | — |

The binding term is the KV recurrent state of the windowed-attention "sgu"
net (6 layers x window 256 x 576 ≈ 7 MB/env/seat, fp32) held by the rollout
seats AND the learner's carried policy/teacher states — batch-proportional,
untouchable by micro_batches. Storing those caches fp16 is the known next
lever (~halves ≈10 GB of KV at N=320); needs a precision-probe pass first.

Throughput saturates by ~batch 512 in the serialized loop (9.5k env-frames/s
student-only; +8% at 768); the sim (32.5k frames/s/core) over-feeds the GPU,
so GPU inference is the bottleneck, not env stepping.
