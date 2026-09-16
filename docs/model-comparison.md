# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.

| Architecture            | Steps         | L / H  | Batch | Params | Eval Ploss        | ms @1 | ms @32 |
|-------------------------|--------------:|:------:|------:|-------:|:-----------------:|------:|-------:|
| Transformer             | 30k           | 4/512  | 512   | 14.3M  | 0.907             | 2.85  | 4.90   |
| SGU                     | 30k           | 4/512  | 512   | 14.3M  | 0.909             | 2.79  | 3.80   |
| ffw+lstm                | 30k           | 3/512  | 512   | 11.3M  | 0.923             | 4.38  | 4.56   |
|                         |               |        |       |        |                   |       |        |
| SGU (scaled)  | 30k/100k/1.8M | 6/576  | 512   | 25.7M  | 0.873/0.829/0.774 | 3.18  | 4.61   |
| Transformer (scaled)    | 30k/100k      | 6/576  | 352*   | 25.9M  | 0.887/0.867       | 3.39  | 7.42   |
| ffw+lstm (Phillip)      | 30k/100k      | 3/768  | 512   | 23.9M  | 0.936/0.904       | 4.43  | 7.32   |

\* Largest batch that fit in memory  
<br>
  
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
- **Latency is why SGU was chosen.** At the rollout batch (n=32), SGU is fastest and
  scales best; Transformer's attention is O(batch × window) and degrades hardest
  (scaled Transformer: 7.42ms @32 vs SGU's 4.61ms).
- **The n=1 "tie" (~2.83ms both) is NOT a real equivalence** — it's a cudagraph
  launch-overhead floor. At batch 1 the compute is trivial, so both bottom out at
  the replay/launch cost. They diverge at any real batch, and the gap *widens*:
  transformer/SGU = 4.9/3.8ms @32 (+29%), 11.4/6.7ms @128 (+69%). SGU is genuinely
  faster once you're batching; only n=1 hides it.
- **Params are near-identical but NOT equal.** Transformer 4/512 = 14,267,250;
  SGU 4/512 = 14,268,786 (1,536 apart). Different architectures (`TransformerCore`
  vs `SGUCore`), coincidentally within 0.01% at this size — both just round to 14.3M.
- **ffw+lstm doesn't fully compile:** its LSTM graph-breaks under torch.compile and
  falls back to eager (~2× speedup vs SGU/Transformer's ~4×), so it's slowest
  compiled despite being smallest.
- **Scaling SGU was nearly free:** 14.3M → 25.7M params and eval 0.909 → 0.774 moved
  latency only 2.83 → 3.18ms @1.
- `ffw+lstm` = tx_like = slippi-ai's / Phillip's architecture (LSTM recurrent core).
- **Two stale figures were corrected here (2026-09-15).** The scaled Transformer
  was previously marked never-trained; it finished at 100k. And "ffw+lstm 0.9946
  @30k" was actually its value @15k — the true 30k figure is 0.936.
- **Bake-off caveat:** the top ffw+lstm row (0.923) is run `exp-clip`, which was
  fp32 with grad clipping, while the Transformer and SGU rows are bf16 without.
  It is the best of the three tx_like variants at 30k (vs 0.925 for both the
  bf16 and the unclipped fp32 runs), so the row flatters ffw+lstm and it still
  loses.
- Params are totals (network + controller head + value head).
