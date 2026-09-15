# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.

| Architecture            | Steps      | L / H  | Params | Eval loss | ms @1 | ms @32 |
|-------------------------|-----------:|:------:|-------:|:---------:|------:|-------:|
| Transformer             | 30k        | 4/512  | 14.3M  | 0.907     | 2.85  | 4.90   |
| SGU                     | 30k        | 4/512  | 14.3M  | 0.909     | 2.79  | 3.80   |
| ffw+lstm                | 30k        | 3/512  | 11.3M  | 0.923     | 4.38  | 4.56   |
| SGU (scaled → teacher)  | 30k/1.8M      | 6/576  | 25.7M  | 0.873/0.774     | 3.18  | 4.61   |
| Transformer (scaled)    | —          | 6/576  | 25.9M  | —         | 3.39  | 7.42   |
| ffw+lstm (Phillip)      | 100k | 3/768  | 23.9M  | 0.904     | 4.43  | 7.32   |

## Notes

- **Matched bake-off (top 3 rows, all 30k steps, window 256):** Transformer ≈ SGU
  (tie within noise); both beat ffw+lstm on eval.
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
  The "scaled Transformer" row has blank steps/eval — never trained here;
  benchmarked for latency reference only.
- **Phillip-arch long run (2026-09-15): 0.9946 @30k -> 0.904 @100k — at 3.3x
  the steps it still trails scaled SGU's 0.873 @30k.** SGU wins per-step and
  per-ms; comparison closed.
- Params are totals (network + controller head + value head).


scaled 0.87257 best policy loss @ 30k steps
