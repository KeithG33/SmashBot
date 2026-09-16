# Sim training backend (melee-sim-light)

RL rollouts AND evaluation run on
[melee-sim-light](https://github.com/kyhavlov/melee-sim-light) (a
deterministic batched SSBM sim built on the decompilation). **Dolphin
remains for humans only** — play.py and watch_live.py via eval/game.py —
and serves as the final eyeball transfer check on sim-trained models.
The Dolphin RL/eval fleet (DolphinRolloutWorker, env_process, league grid)
was removed once training and eval both validated sim-side; git history
has it.

Eval tools (all sim, CPU-friendly, deterministic slates):
  scripts/battery.py     student vs the phillip/fox slate; --grid = the
                         full 144-pair baseline mode with per-pair results
  scripts/tournament.py  checkpoint round-robin, standings
  smashbot/eval/sim_arena.py  the shared MatchSet engine

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
train_rl (backend sim)                         (rl/train_rl.py dispatch)
  └─ rl/train_sim.py   run(): learner loop — Learner, overlap pipeline,
     │                 checkpoint schema; learner rows = envs + self envs
     ├─ SimLeagueWorker: STATIC env layout (self / phillip tiers / pfsp),
     │                   both grids, per-match PFSP routing, trackers,
     │                   match draws (chars, stage, ports, seed, 8-min timer)
     ├─ rl/league.py:  v5's per-match routing — LeagueSeats (slices are a
     │                 weight cache; weights only load into EMPTY slices,
     │                 compaction via move_cell), League (member_now /
     │                 member_next drawn a game ahead, seat at the env's own
     │                 game boundary, counted fallbacks), MemberWeights
     │                 (LRU + background warm)
     └─ rl/sim_league.py
        ├─ SimLeague: pool (SnapshotPool PFSP draw + pfsp.json), layout(),
        │             bare-state member loading (config_from)
        ├─ PfspGrid: S slices x Nc cells on ONE LeagueAgent; seat/unseat/
        │            move keep the cell->env gather map; a cell harvests
        │            only if occupied for the whole chunk
        └─ MultiOpponentSimWorker: per-frame loop — one student forward
                       over all learner rows (every env's seat A + self
                       envs' seat B), phillip grid + PFSP grid forwards,
                       harvest of every non-self seat, rewards, events,
                       game-end handling (_on_done: record, re-seat, redraw)
  rl/sim_env.py       obs -> encoded Game struct (flat 3-tensor path)
```

Every game runs to its natural end; nothing resets an env except the sim's
own game-over. (The first sim port re-partitioned all envs every 25 steps
with reset_all — 96% of games truncated, trackers biased; fixed 2026-09-15
by porting the v5 design above.)

## Launch

`scripts/launch_sim_v12.sh` — fresh run from v10 weights → 100k. Shares of
envs: self 30% (both seats are learner rows) / phillips 35% / PFSP 35%
(the 30 hardest v10 ghosts from 17k+ plus imp9000/imp10000/imp9500), 60
PFSP slices, snapshots every 1500. Seeds the run's snapshot dir from v10
(symlinks + pfsp.json with ghost keys rewritten).

## Memory + throughput

Measure with the REAL run, never a harness (a re-implemented loop drifted
from the launch path three times and caused a launch OOM): run the launch
script with `--runtime.steps <start+6> --runtime.wandb-mode disabled` and
read the `[vram]` lines (first sequential learner step peak, then the
overlapped steps' peak/reserved) and the `fps` in the step ticker.

v12 (345 envs = 449 rows, 48 pfsp slices, mb 24): see the launch log.
