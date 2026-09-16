# TODO

## Serving cost of windowed models at rollout batch (SGU @400 rows = 27 ms)
Kernel profile (`bench_agent_step.py --torch-profile`, scaled SGU, bf16, n=400,
2026-09-16): ~54% of CUDA time is `aten::copy_` OUTSIDE the compiled graph —
the recurrent state (6 layers x [400, 255, 576+128]) is cloned by
`BatchedPolicyAgent` every frame (`self.hidden = clone(hidden)`; the returned
caches are strided views, so the clone hits the slow generic copy kernel) and
then copied back into the cudagraph-tree input placeholder next frame.
`LeagueAgent._captured_forward` does the equivalent `out -> in` copy in python
per leaf per frame. ~20% is the in-graph window shift (`cat` + reset `where`),
~10% matmuls, ~6% attention.

Fix is coupled (either half alone buys little):
1. Network: ring-buffered SGU state — caches stay in place, one slot written
   per frame, conv weights rotated by the pointer, attention keys need no
   rotation (mask by slot age); T>1 unroll un-rotates once. Numerics identical
   up to summation order (parity test vs current: tolerance, plus bit-exact
   for the unroll path).
2. Agents: hold the state in static buffers and let the graph update it in
   place — no per-frame clone/copy. LeagueAgent: carry inside the captured
   graph (or double-buffered graphs); BatchedPolicyAgent: manual capture of
   `policy.sample` with static in/out buffers instead of torch.compile.
Expected: SGU @400 from ~27 ms toward the LSTM's ~10 ms.

MEASURED (2026-09-16, branch `sgu-ring`, scaled SGU bf16 n=400, 3090):
| variant                                   | ms @400 | ms @32 |
|-------------------------------------------|--------:|-------:|
| baseline (cat-shifted window)             | 26.9    | 4.73   |
| ring buffer, out-of-place `index_copy`    | 27.8    | 4.82   |
| ring buffer, in-place `index_copy_`       | 28.2    | 4.66   |
| baseline + `max-autotune`                 | 28.3    | —      |

**Network-only changes cannot fix this.** `index_copy` is out-of-place, so it
rewrites the whole cache exactly like the `cat` it replaced; the in-place variant
is functionalized by inductor back into a copy. The 54% agent-side `copy_` is
untouched by anything in networks.py. Parity of the ring implementation was
verified (repo's own unroll-vs-step scenario: 8.9e-08, identical to baseline;
300 frames with staggered resets vs the old implementation: 1.9e-05), and it
additionally makes the state NON-CANONICAL — a ring state and a chronological
state encode the same history with different layouts, so the 7 tests that compare
state trees elementwise (`test_unroll_vs_step_equivalence`,
`test_fixed_pass_chunking[with_resets]`) fail on layout, not math. Reverted.

SECOND PASS over the SGU hot path (2026-09-16), three real inefficiencies, all
fixed and BIT-EXACT (0.00e+00 on outputs and state, elementwise, 126 tests pass):
mask built once per forward instead of once per layer (6x); the [B, W, d] window
no longer materialized on the serving path (the one-position grouped conv is a
per-channel weighted sum, so cache and current frame reduce separately); returned
caches made contiguous. Measured at n=1/32/128/400: 3.14 / 4.75 / 9.27 / 26.63 ms
vs baseline 3.13 / 4.73 / 9.34 / 27.23 — no gain beyond noise. Inductor was
already fusing these patterns; they were never the cost. Kept for clarity, not
speed.

Conclusion after two passes: **nothing inside networks.py moves this number.**
The remaining lever is agent-side (item 2 above): hold the recurrent state in
static buffers and let the captured graph update it in place, so there is no
per-frame clone and no copy into the graph's input placeholder. That is where
the 54% is. Do it in agent.py with a manual CUDA-graph capture of `policy.sample`
(LeagueAgent already works this way — its `out -> in` carry can move inside the
captured graph).

## Benchmark hygiene (2026-09-16)
`bench_agent_step.py` measures a full serving frame — delay-queue pop, state
clones, the compiled forward, the device->host copy of the sampled controller,
numpy decode, re-enqueue — i.e. what the rollout pays per frame, NOT just the
network. Its `agent.step()` defaulted to `want_snapshot=True`, cloning the whole
recurrent state EVERY frame; production takes that snapshot once per unroll
(240 frames). Measured cost of one full-state clone at n=400: SGU 3.43 ms,
LSTM 0 (its state is KB, not MB). So the published latency table overstates the
windowed models and not the LSTM — re-measure with `--no-snapshot` for a fair
architecture comparison.

Decomposition at n=400 (scaled SGU, bf16): 27.4 bench default -> 23.9 with one
clone (production) -> ~20.5 projected if the carried-state clone also goes.
The earlier "~12 ms" projection was too optimistic: the profiler's 54% `copy_`
includes the copies INTO the cudagraph input placeholders, which only the
static-buffer rework removes. LSTM @400 is 8.3 either way.

The T==1 branch in SGUBlock.mix is load-bearing: forcing the conv path for all T
costs 2.4x at n=400 (58.2 vs 23.9) and +30% at n=1.

## SGU optimization campaign — everything tried (2026-09-16)

| change | parity | ms @400 | verdict |
|---|---|---:|---|
| ring buffer, out-of-place `index_copy` | 1.9e-05 | 27.8 | no gain; state layout non-canonical -> reverted |
| ring buffer, in-place `index_copy_` | 1.9e-05 | 28.2 | inductor functionalizes it back to a copy -> reverted |
| `max-autotune` | n/a | 28.3 | no gain (bandwidth, not kernels) |
| mask hoisted 6x -> 1x, window not materialized, contiguous caches | 0.00e+00 | 26.6 | KEPT (strictly less work; no speedup) |
| drop the carried-state clone in BatchedPolicyAgent | — | — | torch REJECTS it: "accessing tensor output of CUDAGraphs that has been overwritten" -> reverted |
| remove the T==1 branch (always conv) | — | 58.2 | 2.4x worse @400, +43% @32, +30% @1 -> branch KEPT |
| T>1: conv vs unfold+einsum | 0.00e+00 | — | conv flat in T (0.19/0.13/0.13 ms at T=2/4/8) vs unfold linear (0.16/0.30/0.56) -> conv KEPT |

Production-equivalent SGU 6/576 @400 is **23.9 ms** (one state clone), not the
26-27 ms the bench default showed. Learner unroll B=32 T=240: 19.3 ms (80 us/frame).

**Nothing further is available inside networks.py.** The one remaining lever is a
manual static-buffer CUDA-graph capture of `policy.sample` in BatchedPolicyAgent
(LeagueAgent already works this way): the state lives in static buffers the graph
updates in place, removing the 3.4 ms/frame clone and the copies into the graph's
input placeholders. Worth ~14%+ of the student forward; it touches the live
serving path, so it wants its own branch and a lockstep parity run.
