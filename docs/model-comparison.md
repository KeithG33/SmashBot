# Network Architecture Comparison

Imitation-learning (behavior-cloning) architecture comparison. Metric is
`eval/best_policy_loss` (policy cross-entropy, lower = better). Inference latency
measured with cudagraph compilation (the setting we run) on GPU, in ms per call.


### Scaled networks + Phillip (full 841,682-replay dataset)

| Architecture            | L / H  | Batch | Params | Precision | Eval @100k | Eval @250k | Eval @500k | ms @1 | ms @32 | ms @128 | ms @256 | ms @512 |
|-------------------------|:------:|------:|-------:|:---------:|-----------:|-----------:|-----------:|------:|-------:|--------:|--------:|--------:|
| SGU (scaled)            | 6/576  | 512   | 26.5M  | bf16      | 0.833      | —          | —          | 1.79  | 2.12   | 2.43    | 3.15    | 4.55    |
| Transformer (scaled)    | 6/576  | 352*  | 25.9M  | bf16      | 0.877      | —          | —          | 1.84  | 2.96   | 6.45    | 11.5    | 22.5    |
| ffw+lstm (Phillip)      | 3/768  | 512   | 23.9M  | bf16      | 0.904      | —          | —          | —     | —      | —       | —       | —       |
| ffw+lstm (Phillip) fp32 | 3/768  | 512   | 23.9M  | fp32      | 0.836      | 0.822      | 0.806      | **1.65** | **1.82** | **1.92** | **2.27** | **2.78** |
| Hybrid `slslsl` + `sl` value | 6/576 | 512 | 31.3M  | bf16    | **0.829**  | **0.816**  | **0.801**  | 1.81  | 1.99   | 2.39    | 3.08    | 4.31    |

\* Largest batch that fit in memory
<br>

## Notes

- Both tables train on the **same 12 characters** (fox, falco, marth, sheik,
jigglypuff, cptfalcon, peach, yoshi, popo, luigi, pikachu, samus).
- Table is on the full 841,682-replay dataset
- Eval @Nk is the average of the ±3 evals around step N.
- **The scaled Transformer had to drop to batch 352** to fit VRAM; the other two
  ran 512. That is a real confound in its favour on a per-step basis (smaller
  batch = more steps per epoch) and against it on wall-clock — it still loses to
  SGU on both. Its run is named `transformer576-b382`, but the recorded config
  says 352; the name is wrong.
- `ffw+lstm` = tx_like = slippi-ai's / Phillip's architecture (LSTM recurrent core).
- Params are totals (network + controller head + value head).
