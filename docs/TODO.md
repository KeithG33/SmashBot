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

The remaining lever is agent-side (item 2 above): hold the recurrent state in
static buffers and let the captured graph update it in place, so there is no
per-frame clone and no copy into the graph's input placeholder. That is where
the 54% is. Do it in agent.py with a manual CUDA-graph capture of `policy.sample`
(LeagueAgent already works this way — its `out -> in` carry can move inside the
captured graph).
