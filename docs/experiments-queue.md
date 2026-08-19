# Experiments queue

Queued experiments with enough context to run them cold. Each gets a
branch-from-shared-checkpoint A/B and a battery verdict before adoption.

## Learner precision: fp32 vs bf16 vs fp16 (v4-boundary experiment)

**Why**: GPU activation memory gates the learner batch (~110MB/row at fp32;
120-row ceiling on the 3090). Half precision plausibly unlocks ~160+ rows,
feeding the box's ~20 idle CPU threads (more dolphins) — the "more rows"
door. GPU compute is NOT the motive (util ~22%); memory is.

**Why not obvious**: RL is numerically touchier than supervised training —
PPO ratios exp(logpi-logpi_old) amplify logit noise, advantages are
cancellation-prone differences, errors feed back through data collection,
and our aKL diagnostic (~1e-5, feeds the revert backstop) sits below bf16's
noise floor. Our net already trains loss-neutral in bf16 for BC, so the
forward pass itself is proven.

**fp16 vs bf16**: Unsloth's RL guide argues fp16 > bf16 for RL — bf16's
7-bit mantissa corrupts the rollout-vs-learner logprob match that PPO
ratios depend on; fp16's 10-bit mantissa preserves ratio fidelity at the
cost of range (needs loss scaling / overflow care).
https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation/fp16-vs-bf16-for-rl

**Design (three arms from one shared RL checkpoint)**:
1. fp32 (control, current behavior)
2. bf16 autocast: network unroll under torch.autocast(cuda, bf16); fp32
   master weights; losses/log-probs/ratios/KL/advantages in fp32 (autocast
   already keeps softmax/log_softmax/losses fp32)
3. fp16 autocast + GradScaler, same fp32-sensitive-path rules

**Measure per arm** (few hundred steps each): nonfinite-guard cadence
(baseline ~1/300-400 steps), aKL noise floor (must stay ~1e-5, not jump),
tKL behavior, ratio_mean == 1 invariant on fresh learner, learner peak
memory (rows headroom), fps; then a battery each vs the shared baseline.

**Adopt if**: equal-or-better battery + no guard-cadence regression; then
raise rows to the new ceiling at the v4 launch.

## Ticker redesign for the league era

kill@/die@ are hardwired to teacher-kind games (often benched under PFSP =
stuck at 0); per-kind columns generally assume fixed partitions. Rethink:
per-serving-member stats, aggregate-over-policy-opponents, or rotating
display keyed to the current auction. (User: "plan a better logging
statement" — don't just re-point at one member.)

## Char-axis redraw prioritization (optional signal boost)

Per-character win-rate EMAs driving per-game redraws by f_hard (applies to
Phillip + league members): concentrates games on hard characters without
touching env allocation. Per-ghost slot selection already integrates the
char mixture (~100-game memory), so this is about reshaping what we play,
not measuring better.

## Also queued (older)

- Advantage imitation A/B (machinery landed, dormant: imitation-slots /
  imitation-lambda; see docs/idea-opponent-learning.md)
- 12-char whitelist flip = v4 all-character student
- 705-era dump-on-skip forensics (save the culprit batch when a nonfinite
  guard fires)
- Netplay bot account (direct-connect only)
- Playback-of-ExiAI-replays question for vladfi (evidence bundle in memory;
  live watching via scripts/watch_live.py meanwhile)
- GC-adapter LD_PRELOAD shim (headless dolphins blind to the WUP-028)
