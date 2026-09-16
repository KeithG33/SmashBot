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
`max-autotune` measured: no gain (memory traffic, not kernel choice).
