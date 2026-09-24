# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.

### Small-scale bake-off (20k replays, 30k steps)

| Architecture            | Steps         | L / H  | Batch | Params | Precision | Eval Ploss        | ms @1 | ms @32 | ms @128 | ms @256 | ms @400 |
|-------------------------|--------------:|:------:|------:|-------:|:---------:|:-----------------:|------:|-------:|--------:|--------:|--------:|
| Transformer             | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.907             | —†    | —†     | —†      | —†      | —†      |
| SGU                     | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.909             | 1.63   | 1.87   | 2.27   | 2.87   | 3.81   |
| ffw+lstm                | 30k           | 3/512  | 512   | 11.3M  | fp32      | 0.923             | —†    | —†     | —†      | —†      | —†      |

### Scaled networks + Phillip (full 841,682-replay dataset)

| Architecture            | Steps         | L / H  | Batch | Params | Precision | Eval Ploss        | ms @1 | ms @32 | ms @128 | ms @256 | ms @400 |
|-------------------------|--------------:|:------:|------:|-------:|:---------:|:-----------------:|------:|-------:|--------:|--------:|--------:|
| SGU (scaled)            | 30k/100k      | 6/576  | 512   | 25.7M  | bf16      | 0.873/0.829       | 1.88   | 2.04   | 2.70   | 3.68   | 4.85   |
| Transformer (scaled)    | 30k/100k      | 6/576  | 352*   | 25.9M  | bf16      | 0.887/0.867       | 1.81   | 2.98   | 6.49   | 11.4   | 16.8   |
| ffw+lstm (Phillip)      | 30k/100k      | 3/768  | 512   | 23.9M  | bf16      | 0.936/0.904       | 1.64   | 1.83   | 1.89   | 2.20   | 2.53   |
| ffw+lstm (Phillip) fp32 | 30k/100k      | 3/768  | 512   | 23.9M  | fp32      | 0.884/0.826       | 1.42   | 1.53   | 1.79   | 2.27   | 2.71   |

\* Largest batch that fit in memory  
The fp32 Phillip row is the from-scratch run `txlike768w256-fp32-b512-12char-100k-v2`
(clean: no restarts, 0.27 epoch continuous). The LSTM needs fp32 training: at 30k, on
identical data, fp32 is 0.884 vs bf16 0.9365.
<br>
- Precision = training precision. Latency is measured at the SERVING precision: fp16 for
  SGU/Transformer (what the rollout runs), fp32 for the LSTM (it needs it).
- † small-model rows not re-measured on the fixed benchmark; their earlier numbers were fp32
  on a non-production code path and are not comparable — see the latency note.
- Both tables train on the **same 12 characters** (fox, falco, marth, sheik,
jigglypuff, cptfalcon, peach, yoshi, popo, luigi, pikachu, samus).
- The first table is a small-scale experiment with 20k replays and smaller networks
- The second table is the scaled-up nets (and actual Phillip) on the full 841,682-replay dataset
## SGU depth/width at ~25M params (2026-09-17)

The 6/576 shape was chosen under conditions that no longer hold (cudagraph
trees, n=32, no ring). At fixed parameters the GEMM work is constant but the
window read and the ring/kv traffic scale with layers x width x window, and
batch-1 cost is a chain of per-layer kernels — so shallower-and-wider should be
faster. Measured, SGU only, one method (compile + manual capture + fp16 static
state + flat inputs, no-snapshot, idle 3090):

| shape | params | state/row | ms @1 | ms @128 | ms @400 | serving VRAM @400 | learner peak* | fps @25* |
|-------|-------:|----------:|------:|--------:|--------:|------------------:|--------------:|---------:|
| 3/768 | 23.1M | 1.37 MB | **1.64** | 2.31 | **3.67** | **2.23 GiB** | **12.42 GiB** | **5472** |
| 3/832 | 26.4M | 1.47 MB | 1.69 | 2.28 | 3.82 | 2.39 GiB | 13.35 GiB | 5365 |
| 4/704 | 25.4M | 1.70 MB | 1.72 | 2.48 | 4.20 | 2.75 GiB | — | — |
| 5/640 | 25.9M | 1.96 MB | 1.79 | 2.55 | 4.58 | 3.17 GiB | — | — |
| 6/576 (current) | 25.7M | 2.15 MB | 1.91 | 2.74 | 5.01 | 3.48 GiB | 15.53 GiB | 4914 |
| 8/512 | 26.7M | 2.61 MB | 1.92 | 3.07 | 5.84 | 4.23 GiB | — | — |

\* real entrypoint (`train_rl`), 400 rows / 40 slices / mb 12,
fresh start from a random-init checkpoint of the shape, empty league, no
imports, wandb disabled, 30 steps; identical settings for the three shapes, so
read relatively (the 6/576 peak matches the resume dry-runs' 15.51). fps is
cumulative-since-boot at step 25 and warmup-limited.

- Monotonic at every batch: 3/768 is -27% @400 and -14% @1 vs 6/576, 1.25 GiB
  lighter in serving and **3.1 GiB lighter at the learner peak** (rows are the
  budget). It is also the LSTM's shape (Phillip is 3/768) and 2.6M params smaller.
- Speed only. Whether three layers hold eval loss at these params is the open
  question — a BC run (fp32, batch 512 as 2x256) on the box after the 6/576
  fp32 run finishes.

## Notes

- **The two tables see data very differently.** At 30k steps the bake-off had
  already made ~3.6 passes over its 20k-replay subset, so it was well into
  repeating data. The scaled runs at 100k steps have seen only ~0.20 of one
  epoch of the full set. Bake-off eval numbers are subset-limited and the two
  tables' absolute losses should not be read against each other.
- **SGU precision check (2026-09-18):** `sgu576w256-fp32-b512x2-12char-100k` (fp32,
  batch 512 as 2x256 via grad accumulation) tracked the bf16 mega run within 0.002
  at every matched eval from 20k on (best 0.8282 @91k vs bf16 0.829 @100k) and was
  stopped at 94k as settled: precision is irrelevant for SGU (it is not for the LSTM).
  The direct shape comparison now runs on the box: SGU 3/768 bf16 (23.1M) vs Phillip
  3/768 fp32 (23.7M), both 0->250k.
- **At 100k: fp32 ffw+lstm 0.826 vs bf16 SGU 0.829 vs bf16 Transformer 0.867.**
  The LSTM's earlier 0.904 was a precision artifact (bf16 training hurts LSTM
  recurrence); in fp32 it matches SGU on loss while serving ~1.8x faster. SGU and
  the Transformer have only been trained in bf16 — SGU-fp32 (batch 512 as 2x256
  via grad accumulation, since a 512 micro-batch OOMs at fp32) is the run that
  decides the architecture. The bf16 LSTM run also restarted at 30k into the old
  dataloader bug (re-saw data); the SGU and Transformer 100k runs were clean.
- **The scaled Transformer had to drop to batch 352** to fit VRAM; the other two
  ran 512. That is a real confound in its favour on a per-step basis (smaller
  batch = more steps per epoch) and against it on wall-clock — it still loses to
  SGU on both. Its run is named `transformer576-b382`, but the recorded config
  says 352; the name is wrong.
- **Latency (2026-09-16, branch ring-serving; one method for every cell).**
  `scripts/bench_agent_step.py --compile --capture --no-snapshot --flats` (+
  `--precision fp16 --state-fp16` for the windowed cores): the worker's per-frame
  path (`execute` + `infer`, flat controllers, one packed controller D2H), static
  inputs = FlatFrames' three typed flats, manual static-buffer CUDA graph with
  fp16 state buffers and the SGU v-cache as an in-place ring, idle 3090, 30 warmup
  + 300 timed frames. Ladder for scaled SGU @400, one change per rung: 21.1 ms
  (old, broken bench: fp32 forward, non-production `step()` path) -> 16.7 (worker
  path) -> 12.9 (fp16) -> 12.6 (packed D2H) -> 9.6 (fp16 static buffers) -> 6.3
  (v-cache ring) -> 5.2 (flat static inputs). The first two rungs are measurement
  corrections; the rest shipped. Every rung bit-exact against the previous
  (open-loop logits + canonical snapshots, 0.00e+00).
- **Capture vs cudagraph trees at fp16:** with fp32 state buffers the captured
  graph LOST to trees (12.9 vs 12.0 ms @400) — it upcast every layer's fp16
  cache into the buffer and cast it back next replay; that is why the earlier
  real-rollout A/B saw nothing. With fp16 buffers capture wins, and the ring is
  capture-only (inductor functionalizes an in-place write back into a copy).
- **ffw+lstm is ~1.8x faster than scaled SGU at the serving batch** (2.71 vs
  4.85 ms @400; was 2.5x before the ring) and 1.3x at n=1 (1.42 vs 1.88). Both
  gained equally from the wrapper fixes (flats, packed D2H); the ring is SGU-only.
  The scaled Transformer, with no ring and full attention over W=256 per row, is
  16.8 ms @400 — 3.5x SGU. What remains of SGU's GPU frame is structural: the
  window read (0.94 ms at W=256 — the model reads 255x576 per row per layer),
  attention (0.69), the kv traffic (~0.7, ring-able), GEMMs (~0.7). NOTE: live play
  runs on CPU (`eval/play.py --device cpu`), where only SGU has been measured
  (~7 ms compiled); the LSTM's CPU batch-1 cost is unmeasured.
- **Scaling SGU 4/512 -> 6/576** costs 1.63 -> 1.88 ms @1 and 3.81 -> 4.85 @400
  for eval 0.909 -> 0.829 (different data regimes; see above).
- Remaining serving levers, in order of measured size: (1) the ~70 remaining
  outside-graph launches per frame — prev-action masking/clones and logit clones,
  13 leaves each (pack; unpack once per chunk in ChunkAssembler); (2) the kv ring
  (~0.7 ms of traffic; cat+SDPA beat a ring read for it, 0.129 vs 0.233 ms/layer,
  so it needs a different attention formulation); (3) the window itself — W=128
  halves the read, the ring and the kv cache, untested on eval loss.
- `ffw+lstm` = tx_like = slippi-ai's / Phillip's architecture (LSTM recurrent core).
- Params are totals (network + controller head + value head).
