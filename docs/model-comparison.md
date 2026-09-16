# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.

| Architecture            | Steps         | L / H  | Batch | Params | Precision | Eval Ploss        | ms @1 | ms @32 | ms @128 | ms @256 | ms @400 |
|-------------------------|--------------:|:------:|------:|-------:|:---------:|:-----------------:|------:|-------:|--------:|--------:|--------:|
| Transformer             | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.907             | 2.85   | 4.63   | 9.77   | 17.8   | 26.1   |
| SGU                     | 30k           | 4/512  | 512   | 14.3M  | bf16      | 0.909             | 2.81   | 3.64   | 5.96   | 10.6   | 15.6   |
| ffw+lstm                | 30k           | 3/512  | 512   | 11.3M  | fp32      | 0.923             | 2.57   | 2.89   | 3.33   | 5.93   | 6.74   |
|                         |               |        |       |        |           |                   |       |        |         |         |         |
| SGU (scaled)  | 30k/100k | 6/576  | 512   | 25.7M  | bf16      | 0.873/0.829 | 2.99   | 4.34   | 7.78   | 13.4   | 19.8   |
| Transformer (scaled)    | 30k/100k      | 6/576  | 352*   | 25.9M  | bf16      | 0.887/0.867       | 3.15   | 6.43   | 16.0   | 31.5   | 47.4   |
| ffw+lstm (Phillip)      | 30k/100k      | 3/768  | 512   | 23.9M  | bf16      | 0.936/0.904       | —     | —      | —       | —       | —       |
| ffw+lstm (Phillip) fp32 | 30k/100k      | 3/768  | 512   | 23.9M  | fp32      | pending           | 2.60   | 3.01   | 3.56   | 5.90   | 8.12   |

\* Largest batch that fit in memory  
The fp32 Phillip row is the from-scratch run `txlike768w256-fp32-b512-12char-100k-v2`
(in progress); the LSTM needs fp32 training (a contaminated fp32 attempt reached 0.868 @74k).  
<br>
- Precision = training and inference precision (the latency columns are measured at it).
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
- **Latency (re-measured 2026-09-16: matched precision, production snapshot
  cadence, best of cudagraph-trees vs manual static-buffer capture, idle 3090,
  30 warmup + 300 timed steps).** Capture wins at n>=128 for every architecture
  (-12 to -17% for the windowed cores at 400; -25 to -30% for the LSTMs at 128,
  where it removes launch overhead rather than a state clone) and loses at n=1.
  Our production batches are **400** (the student's rollout forward) and **1**
  (human play). ALL cells use the capture path — the serving default — so rows
  are internally comparable; mixing it with cudagraph trees produced a
  non-monotonic table.
- **ffw+lstm is the fastest architecture at EVERY batch**, including n=1
  (2.57 vs SGU 2.81). Its apparent batch-1 weakness in earlier tables
  (4.3-4.4 ms) was an artifact of cudagraph trees, not the architecture. SGU
  was originally chosen on latency measured at n=32 under trees, on CPU, before
  compilation — none of those conditions hold now. NOTE: live play runs on CPU
  (`eval/play.py --device cpu`), where only SGU has been measured (~7 ms
  compiled); the LSTM's CPU batch-1 cost is unmeasured. SGU was chosen at the Dolphin-era rollout batch
  (n=32), where it is fastest. At the sim's serving batch (~400 rows per student
  forward) the ranking inverts: ffw+lstm at fp32 is 10.1ms vs SGU 27.2ms vs
  Transformer 59.8ms — the LSTM's eager cuDNN path is launch-bound and nearly flat
  in batch, while the compiled models grow ~linearly beyond n≈32.
  `torch.compile(mode="max-autotune")` does not help (scaled SGU 9.8/18.8/28.3ms at
  128/256/400, Transformer 58.5 @400): the cost is the per-frame window shift
  (`torch.cat` rebuilding the [B, W, d] caches each step), i.e. memory traffic, not
  kernel choice. A ring-buffered state (write one slot in place, rotate the conv
  weights, mask by slot age) would remove it — candidate optimization.
- **The n=1 "tie" (~2.83ms both) is NOT a real equivalence** — it's a cudagraph
  launch-overhead floor. At batch 1 the compute is trivial, so both bottom out at
  the replay/launch cost. They diverge at any real batch, and the gap *widens*:
  transformer/SGU = 4.9/3.8ms @32 (+29%), 11.4/6.7ms @128 (+69%). SGU is genuinely
  faster once you're batching; only n=1 hides it.
- **ffw+lstm doesn't fully compile:** its LSTM graph-breaks under torch.compile and
  falls back to eager (~2× speedup vs SGU/Transformer's ~4×), so it's slowest
  compiled despite being smallest.
- **Scaling SGU was nearly free:** 14.3M → 25.7M params and eval 0.909 → 0.774 moved
  latency only 2.83 → 3.18ms @1.
- `ffw+lstm` = tx_like = slippi-ai's / Phillip's architecture (LSTM recurrent core).
- Params are totals (network + controller head + value head).
