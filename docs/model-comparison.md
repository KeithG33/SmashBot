# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.

| Architecture            | Steps         | L / H  | Batch | Params | Precision | Eval Ploss        | ms @1 | ms @32 | ms @128 | ms @256 | ms @400 |
|-------------------------|--------------:|:------:|------:|-------:|:---------:|:-----------------:|------:|-------:|--------:|--------:|--------:|
| Transformer             | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.907             | —†    | —†     | —†      | —†      | —†      |
| SGU                     | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.909             | 2.76   | 3.07   | 4.07   | 5.60   | 6.98   |
| ffw+lstm                | 30k           | 3/512  | 512   | 11.3M  | fp32      | 0.923             | —†    | —†     | —†      | —†      | —†      |
|                         |               |        |       |        |           |                   |       |        |         |         |         |
| SGU (scaled)            | 30k/100k      | 6/576  | 512   | 25.7M  | bf16      | 0.873/0.829       | 3.00   | 3.51   | 5.22   | 7.07   | 9.36   |
| Transformer (scaled)    | 30k/100k      | 6/576  | 352*   | 25.9M  | bf16      | 0.887/0.867       | —†    | —†     | —†      | —†      | —†      |
| ffw+lstm (Phillip)      | 30k/100k      | 3/768  | 512   | 23.9M  | bf16      | 0.936/0.904       | —     | —      | —       | —       | —       |
| ffw+lstm (Phillip) fp32 | 30k/100k      | 3/768  | 512   | 23.9M  | fp32      | pending           | 2.75   | 2.86   | 3.13   | 3.43   | 3.87   |

\* Largest batch that fit in memory  
The fp32 Phillip row is the from-scratch run `txlike768w256-fp32-b512-12char-100k-v2`
(in progress); the LSTM needs fp32 training (a contaminated fp32 attempt reached 0.868 @74k).  
<br>
- Precision = training precision. Latency is measured at the SERVING precision: fp16 for
  SGU/Transformer (what the rollout runs), fp32 for the LSTM (it needs it).
- † not re-measured on the fixed benchmark (2026-09-16); the earlier numbers were fp32
  on a non-production code path and are not comparable — see the latency note.
- Both blocks train on the **same 12 characters** (fox, falco, marth, sheik,
jigglypuff, cptfalcon, peach, yoshi, popo, luigi, pikachu, samus).
- Top block is a small-scale experiment with 20k replays and smaller networks (under 5ms)
- Bottom block is scaled up nets (and actual Phillip) using full 841,682 replay dataset
## Notes

- **The two blocks see data very differently.** At 30k steps the bake-off had
  already made ~3.6 passes over its 20k-replay subset, so it was well into
  repeating data. The scaled runs at 100k steps have seen only ~0.20 of one
  epoch of the full set. Bake-off eval numbers are subset-limited and the two
  blocks' absolute losses should not be read against each other.
- **Matched at 100k, SGU wins outright: 0.829 vs Transformer 0.867 vs ffw+lstm
  0.904.** All three scaled runs are now trained to the same step count on the
  same 12-character data at bf16, so this is the clean architecture comparison.
  The ordering is the same at 30k (0.873 / 0.887 / 0.936). Comparison closed.
- **The scaled Transformer had to drop to batch 352** to fit VRAM; the other two
  ran 512. That is a real confound in its favour on a per-step basis (smaller
  batch = more steps per epoch) and against it on wall-clock — it still loses to
  SGU on both. Its run is named `transformer576-b382`, but the recorded config
  says 352; the name is wrong.
- **Latency (re-measured 2026-09-16 on the FIXED benchmark; one method for every
  cell).** `scripts/bench_agent_step.py --compile --capture --no-snapshot`: the
  worker's per-frame path (`execute` + `infer`, flat controllers, one packed
  controller D2H), manual static-buffer CUDA graph with fp16 state buffers, idle
  3090, 30 warmup + 300 timed frames. The previous benchmark had three defects
  (found in a Codex review): `--precision` never reached the forward, so every
  "bf16" cell was fp32; it drove the non-production `step()` path (a GPU sync per
  frame); and its profiler cloned the state every frame, so the "54% `aten::copy_`"
  that motivated the capture work was an artifact. Attribution ladder, scaled SGU
  @400, one change per rung: 21.1 ms (old bench) -> 16.7 (production path) ->
  12.9 (fp16 forward) -> 12.6 (packed D2H) -> 9.6 (fp16 static buffers). The first
  two are measurement corrections; the last two shipped (`capture_serving=True`,
  student `state_dtype=fp16`). Production fps on the new path: not yet measured (the launcher still ran the
  stale SmashBot-sim worktree; see TODO).
- **Capture vs cudagraph trees at fp16:** with fp32 state buffers the captured
  graph LOSES to trees (12.9 vs 12.0 ms @400) — it upcasts every layer's fp16
  cache into the buffer and casts it back next replay. That is why the earlier
  real-rollout A/B saw nothing. With fp16 buffers capture wins (9.4). At n=1
  trees still edge it (2.83 vs 3.00; launch-bound either way).
- **ffw+lstm is still ~2.5x faster than scaled SGU at the serving batch** (3.87 vs
  9.36 ms @400) — the same ratio as the old table; both rows moved by the same
  path correction. It is nearly flat in batch (2.75 -> 3.87 from n=1 to 400)
  while SGU grows ~3x. At n=1 they tie (2.75 vs 3.00). NOTE: live play runs on
  CPU (`eval/play.py --device cpu`), where only SGU has been measured (~7 ms
  compiled); the LSTM's CPU batch-1 cost is unmeasured. SGU was chosen at the
  Dolphin-era rollout batch (n=32) under different conditions (trees, CPU,
  uncompiled) — none of which hold now.
- **Scaling SGU 4/512 -> 6/576** costs 2.76 -> 3.00 ms @1 and 6.98 -> 9.36 @400
  for eval 0.909 -> 0.829 (different data regimes; see above).
- Remaining serving lever for the windowed cores: the per-layer window shift
  (`cat` + contiguous rebuild of the caches). Plain in-place `index_copy_` gave
  nothing (inductor functionalizes it back to a copy; 3 variants measured); it
  needs a fused kernel that reads the window, masks by slot age and writes one
  slot — and must still export chronological state at chunk boundaries.
- `ffw+lstm` = tx_like = slippi-ai's / Phillip's architecture (LSTM recurrent core).
- Params are totals (network + controller head + value head).
