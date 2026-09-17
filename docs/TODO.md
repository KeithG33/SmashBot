## Serving frame after the ring branch (2026-09-16 late) — where the time is NOW
Student @400 (bench, one method): 5.2 ms, GPU 4.3. **League grid (40 slices x 3
cells, captured eager vmap, fp16 weights): 9.6 ms/frame — the largest item.**
Phillip grid (PfspGrid: 5 tier slices x ~24 cells, tx_like with manual_step,
captured eager vmap, fp16 weights): 2.7 ms/frame (GPU 1.67) — already near the
LSTM's floor; fp32 weights would be 5.0. The PFSP grid is the biggest GPU item.
Profile: GEMMs 2.6 (40 weight sets = 2.05 GB fp16 read per frame; bandwidth floor
~2.3 — at the floor), EAGER elementwise ~2.3 (vmap forward is not compiled;
nothing fuses), window-shift cats 1.1, carry/clone copies ~1.5, conv reduce 0.4,
attention 0.15. Levers, both proven on the student: compile the vmap forward
(Codex #5) and ring the grid's v-cache. Flat static inputs (1b) done for the
grid: 10.8 -> 10.2. Phase 2 (pack prev-action/logits, ~0.4 ms) now ranks below
these.
Compile of the vmapped grid forward is a torch limitation: inductor raises
`Cannot access storage of BatchedTensorImpl` on vmap+functional_call over stacked
params, and vmap over a compiled fn is unsupported. Fusion for the grid means a
grid-native forward (explicit bmm over the S dim, no vmap) or cheaper eager ops.
Eager micro-bench at the grid shape [120,255,576] fp16, per layer: today's read
0.37 ms (2.2/frame — the floor under eager; bmm 0.54 and the ring's gather 0.54
are slower without inductor), reset `where` copy 0.13, window-shift cat 0.15,
carry copy ~0.13. So a grid ring with the ROLL-THE-WEIGHTS read (same cost as
today, ~1e-6 tolerance — acceptable for opponent seats) removes ~2.5 ms/frame:
9.6 -> ~7. Beyond that only a compiled grid-native forward fuses the read.

## DONE (branch ring-serving, worktree SmashBot-ring, 2026-09-16 late): SGU v-cache ring under capture
Frame @400 (fixed bench, fp16 + fp16 statics, capture): 9.4 -> 6.3 ms; GPU 8.08 ->
4.26 ms/frame — the window-shift cat (1.9) and in-graph carry (1.8) kernels are
gone; the fused window read (0.94) remains, as it must. Parity: open loop, both
paths under capture, 800 frames (3.1 wraps), staggered resets, logits AND the
canonical snapshot compared on every frame: 0.00e+00. Suite 126. NOT merged to
main; NOT run in production (launcher still points at the stale SmashBot-sim
worktree — see below).
Design (Codex review invariants all hold): ptr = next slot; read history (ages
1..W-1, valid iff age <= cache_len, where-select so a stale NaN can't leak; costs
~0.12 ms vs multiply, kept) before writing the current v; the compiled forward
returns [B, d] slots, the captured graph does the only writes (index_copy_ per
layer at a device ptr), carries kv + cache_len, advances ptr once; resets set
cache_len=0 and never touch the ring; hidden_snapshot gathers, zeroes, drops ptr,
clones — taken before the frame's replay. kv stays canonical (18% of traffic;
cat+SDPA 0.129 vs ring-attn 0.233 ms/layer measured).
Next lever, now visible: frame 6.3 ms vs GPU 4.3 — ~2 ms of CPU/launch/sync no
longer hidden under GPU time. ~190 outside-graph launches per frame, mostly the
122-leaf game-state struct copied leaf-by-leaf into the static inputs; production
states are views into FlatFrames' three flat tensors, so the static input could
be those three tensors (3 copies instead of 122). Then the kv ring if it earns it.

## CORRECTION 2026-09-16 (late): the serving-cost story below was built on a broken benchmark
A Codex review found, and I verified against the code, that `bench_agent_step.py`
(1) never passed `precision=` to the agent, whose inner `autocast(enabled=False)`
overrides any outer context — every "bf16" latency in the table was fp32;
(2) ran its profiler loops with `want_snapshot=True`, cloning the full state every
frame — the "~54% aten::copy_ outside the graph" below measured a cost production
pays once per 240 frames, so the agent-clone hunt (ring buffers, in-graph carry,
manual capture) was chasing an artifact; (3) timed the non-production `step()`
path (GPU `nonzero().tolist()` sync per frame, per-env Controller structs).
Fixed (c5d2279). Attribution ladder, scaled SGU @400: 21.1 (old bench) -> 16.7
(worker path) -> 12.9 (fp16) -> 12.6 (packed controller D2H, e53c775) -> 9.6
(fp16 static state buffers under capture, d5ab201). The real production defect
was capture's fp32 statics: at fp16 the graph paid an up/down cast per layer per
frame and LOST to cudagraph trees (12.9 vs 12.0) — hence the neutral A/B.
Shipped: capture ON + fp16 statics (bit-exact vs fp32 statics: logits, actions,
snapshots 0.00e+00 over 300x64 with resets). ffw+lstm on the same bench: 3.87 @400
(was 8.12 by the same path correction) — the 2.5x ratio to SGU is unchanged.
Production dry-run of the v12 resume: NOT DONE — the launcher's REPO points at the
SmashBot-sim worktree (still e93055a), so the dry-run measured the OLD code (Codex
caught it). Until that worktree is merged or REPO points at main, launches run
without any of the above. Also: `--runtime.restore auto` is tag-relative, so a
`-dryrun` tag starts FRESH; a resume dry-run must pass the explicit v12 path.
Measured and REVERTED: packing the controller inside the captured graph (Codex
follow-up) — n=1 2.94-3.01 vs ~3.00, n=400 9.43-9.77 vs 9.36-9.56: neutral.
Capture path verified faithful: OPEN-LOOP logits vs eager and vs compiled(mode=None)
are 0.00e+00 over 300x64 frames. A CLOSED-loop capture-vs-trees comparison is NOT
a valid parity test — the captured graph samples from graph-registered philox state,
so per-frame torch.manual_seed reseeds only the non-capture side; near-tie samples
differ and cascade (looked like 5.5k/19.2k row mismatches). Compare logits open-loop
(fixed prev_action stream), or compare two runs on the SAME path.
Not done from the review (measure-first): fuse `uv`+`attn_qkv` (same input;
changes state_dict keys -> needs a load pre-hook), one-hot->lookup in the head
decoder, compile the league's vmap forward, fused ring kernel (the lever above).

**Measured ceiling for the ring buffer (fixed profiler, scaled SGU @400, fp16 +
fp16 statics, capture; 50 frames): 8.1 ms GPU per 10.0 ms frame, of which ~5.3 ms
(65%) is cache memory traffic** — fused cat/slice for the window shift 1.92,
in-graph carry `hidden.copy_(new_hidden)` 1.79, kv cat/contiguous + DtoD + misc
to_copy ~1.6; the T==1 weighted-sum conv 0.93 (reads the window once —
unavoidable), attention 0.69, ALL GEMMs ~0.7. Consistent with pure bandwidth:
6 x [400,255,576] fp16 = 860 MB read+written for the shift and again for the
carry. Avoidable ~4.5 ms of the 9.4 ms frame (I earlier wrote "~1 ms" — wrong).
Design: the static cache buffer IS the ring (manual capture only — inductor/trees
functionalize in-place writes back into copies, which is why the three earlier
ring variants showed nothing): write one 2.8 MB slot per frame, roll the conv
weights by the shared pointer, mask attention by slot age (age < cache_len),
canonicalize only in hidden_snapshot() (once per 240 frames); the learner's T>1
path is untouched. At n=1 the frame is CPU-bound instead (190 outside-graph
launches: the 122-leaf state struct copied into the statics + prev/logit clones);
not a production path (play is CPU), so not worth code.

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

## DONE: manual static-buffer capture in BatchedPolicyAgent (2026-09-16)
`BatchedPolicyAgent(capture=True)` records one `policy.sample` into a manual
CUDA graph over static input buffers, and the graph's last op copies the new
recurrent state back into the buffer it read from. No per-frame clone, no
placeholder copies. `policy.sample` must be uncompiled or compiled WITHOUT
cudagraph trees (`torch.compile(fn)` default mode) — a graph inside a graph is
not capturable.

Parity: bit-identical (0.000e+00 on logits AND carried state) over 150 frames
with resets exercised inside the graph, against the clone path running the same
compiled kernels.

Measured (scaled SGU bf16, --no-snapshot), trees+clone -> capture:
n=1 2.89 -> 3.01 (WORSE, +4%) | n=32 4.49 -> 4.37 | n=128 8.93 -> 7.75 (-13%)
| n=400 23.87 -> 20.15 (-15.6%). LSTM @400 8.63 -> 8.29 (tiny state, as expected).
=> enable for the rollout batch, NOT for the batch-1 play path.

Still opt-in (`capture=False` default); nothing in training uses it yet.

### Collected in the sim rollout
1. DONE: `SimRolloutConfig.capture_serving` (default True) passes `capture=True`
   to the student agent and compiles `sample` without cudagraph trees. Parity at
   the exact production config (400 rows, precision="fp16", 120 frames with
   resets): bit-identical, 0.000e+00 on logits and carried state. GPU smoke of
   the real SimLeagueWorker: runs, finite trajectories, queues invariant.
2. NOT WORTH IT (measured): moving LeagueAgent's carry inside its captured graph
   gained nothing (11.755 -> 11.725 ms at the 40x4 PFSP grid). Its python carry
   was already a plain static-buffer copy, so the carry only MOVES — the copy
   itself is 0.862 ms for 329 MiB at 745 GiB/s either way. The student won
   because its clone was an ALLOCATION plus a copy, on top of inductor's
   placeholder copies. Reverted (it also changes move_cell's target buffer,
   i.e. risk for no gain).

Phase split of the PFSP grid at 40x4 (LeagueAgent.infer): forward 11.1 ms
(88%), record+to_cpu 1.35 ms, decode+queues 0.14 ms. GPU-bound, and the
per-cell cost (69 us) is worse than the student's per-row cost (50 us) because
a vmap over 40 separate weight sets is 40 skinny matmuls. Slice count is the
throughput<->fallback knob; see the routing notes.

### Remaining (unattempted)
The student's capture still pays one in-graph carry copy (~2.3 ms at 400 rows).
Removing it needs ping-pong buffers or in-place ring updates inside the manual
graph (no functionalization there, so `index_copy_` is legal). The ring's
non-canonical state layout is acceptable for SERVING state, which is never
compared or checkpointed — unlike the learner's.

## v13 design decisions (Keith, 2026-09-16 evening — NO action yet)
Launch only once the teacher question is settled; the teacher cannot change
mid-run, and three things are open at once (fp32 vs bf16; SGU vs tx_like now
that the latency table inverted; delay 18 vs 21).

- **Pool mix.** v12 runs self 46% / phillips 27% / PFSP 27% of learner ROWS,
  because `self_frac=0.30` is a share of ENVS and each self env contributes two
  rows (`2s/(1+s)`). That silently cut phillip signal from v11's 35%. Keith wants
  one learner row per self-play game plus a larger phillip share. Note the two
  routes are not equivalent:
    - `self_frac=0.176`, both seats -> 30% self rows, no extra envs, but the two
      rows of a game are CORRELATED (same trajectory, mirrored).
    - one row per self game -> independent rows, but needs ~400 envs for 400
      rows (~30% more sim CPU per frame).
- **Schedules.** v12 holds both flat (`kl-teacher-weight 0.025 --final -1`,
  `imitation-lambda 0.01 --final-frac 1.0`). v10 phase 2 held imitation at
  0.002, so v12 runs 5x that constantly. Decay both over the run (the machinery
  exists: `kl_teacher_weight_final`, `imitation_lambda_final_frac`).
- **Open question, cheap to test:** the phillips are LSTMs served under fp16
  autocast. If precision hurts LSTM recurrence (the BC result suggests it may),
  our opponents play below true strength and "winrate vs gm" is not calibrated.
  `check_fp16_state.py` does NOT cover this — it compared fp16 vs fp32 STATE
  STORAGE with the forward in fp16 both times. Test = lockstep fp32 vs fp16
  forward for one phillip, compare action agreement.
