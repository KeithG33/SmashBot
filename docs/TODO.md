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

## Expose Nana (Ice Climbers' follower) from melee-sim-light (Keith, 2026-09-23)
- The mismatch: replays (slippi_db.parse_peppi, Slippi records Nana as a follower) and
  Dolphin (libmelee `player.nana`) give the policy Nana's full state, and the embedding
  uses it (`with_nana=True`); Nana is present on ~71% of Popo frames in the replays. The
  sim simulates her exactly (upstream validates Ice Climbers replays bit-exact, incl.
  Nana), but `MslObservation` has only per-player slots and no follower fields, so
  `rl/sim_env.py` reports `nana.exists=0` on every frame.
- Consequence in sim RL and sim evals: a student playing Ice Climbers can't see its own
  Nana; against Ice Climbers, Nana is invisible but still hits. The KL teacher and the
  Phillip opponents get the same blind input; a sim-trained policy meets a visible Nana
  only in Dolphin. About 1 seat in 12 per player.
- Upstream (checked 2026-09-23, origin/main 8cf34043, 42 commits past our b7a9ed1c):
  still no follower in the observation (`melee_sim/dtypes.py` unchanged); no Python
  accessor for the core's follower lanes.
- Patch: `src/runtime/observation.c` writes each player with `write_player(...)` into
  `output->slots[...]`; a 1v1 leaves two of the four slots empty. Write Popo's follower
  fighter (`sub_character_entity`) there with the same writer, mark it as a follower
  (a flag in the slot dtype, `melee_sim/dtypes.py`), and read it into `nana` in
  `sim_env._player`. Rebuild (`make python-release`), then pin it in
  `smashbot/tests/test_sim_encoding.py` with an Ice Climbers fixture. Or ask upstream.
- Until then, leaving ICE_CLIMBERS out of the sim character lists (`rl/train_sim.py`,
  `eval/sim_arena.py`) removes the gap at the cost of the matchup.

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
Not done from the review (measure-first): fuse `uv`+`attn_qkv` — MEASURED
2026-09-17 inside a CUDA graph at the serving shapes: saves 54 us/frame @400,
15 us @1 (~1%); changes state_dict keys -> not worth a load hook. Skipped.
Still open: one-hot->lookup in the head decoder (unmeasured), league vmap
compile (impossible: BatchedTensorImpl), fused ring kernel (done as the ring).
Transformer (scaled) serving: 18.5 ms @400 — no ring, and its whole state is a
2x576-wide kv cache (~1.4 GB/frame shifted + carried) plus 8-head attention over
255 keys. A kv ring is EASIER than SGU's: attention is permutation-invariant over
keys given the mask, so ring slots need only an age mask (no gather-by-age, no
reduction-order question). Not done: it loses on loss (0.867) and would still be
~2x SGU. Do it only if an fp32 Transformer run wins; the agent side is generic.

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

## A/B: residual stream in fp32 under bf16/fp16 autocast (Keith, 2026-09-24)
- Today the encoder's output enters the core in bf16 (fp16 in RL), so every block's
  `x = x + block(x)` accumulates the residual sum in half precision (bf16 keeps ~3
  significant digits) and the rounding compounds over depth. Common mixed-precision
  practice keeps the residual stream fp32 and only the matmuls in half precision.
- Change: upcast the encoder output to fp32 once; `fp32 + bf16` then stays fp32 in every
  residual add while the blocks' matmuls still run in bf16. Costs some memory bandwidth,
  no extra compute. Not a bug: our runs train fine; this may buy a little loss or nothing.
- Test: the hybrid (slslsl, sl value) to 100k with the upcast vs the existing slslsl
  100k run, same seed and data; compare paired eval and train loss step for step.

## Controller bins aligned to Melee's thresholds, at the next from-scratch BC run (Keith, 2026-09-24)
- The problem (inherited from slippi-ai, whose TF and JAX embeddings do the same): stick
  axes are 17 evenly spaced bins (`ControllerConfig.axis_spacing` 16, one per 10 raw
  units on the +-80 scale) and the shoulder 5 (`shoulder_spacing` 4), encoded
  `round(x * n)` and decoded `bin / n`. The stick deadzone is ~+-22 raw, so a tilt at
  raw 23-24 (just past it) rounds to the raw-20 bin, which decodes inside the deadzone:
  the bot presses neutral. Light shield starts near 0.31, but presses of 0.31-0.37
  round to the 0.25 bin, below the threshold. ~0.3-1.3% of replay frames (review count).
- Fix: bins with explicit edges instead of a spacing, one edge on each gameplay
  threshold (deadzone, walk/dash, tilt/smash, crouch, light shield) and each bin
  decoded to a value inside it, so decoding never crosses a threshold. The code change
  is small (encode/decode take edges); it changes the model's inputs and outputs, so
  every checkpoint needs retraining: do it when a BC run starts from scratch anyway.
- First check: Slippi may record sticks after the game's deadzone processing (in-deadzone
  = exactly 0, real tilts >= 23). A histogram of stick and shoulder values from the parsed
  replays settles where the edges go.

## Try the Kron (Kronecker-factored) optimizer in the RL learner (Keith, 2026-09-23)
- Paper: "Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning",
  Creus Castanyer, Obando-Ceron, Li, Bacon, Berseth, Courville, Castro (Mila / DeepMind),
  NeurIPS 2025. https://papers.neurips.cc/paper_files/paper/2025/file/32375260090404f907ceae19f3564a7e-Paper-Conference.pdf
- Diagnosis: under non-stationarity (RL's moving data and targets) gradient norms
  collapse with depth and width, and deeper nets stop learning; stationary
  supervised training of the same nets is fine. Two interventions fix it:
  1. Multi-skip residuals: the encoder's features are fed directly into every
     later layer (the D2RL idea; see the value-net notes). Our blocks are
     pre-norm residual already, so this would be the encoder output added or
     concatenated at each block, not a new residual stream.
  2. Kron: Kronecker-factored preconditioning of the gradient (curvature-aware,
     like K-FAC / PSGD Kron) in place of Adam's diagonal scaling. Alone it keeps
     deep MLPs learning under PQN; with (1) PQN gains a median +83% over 57 ALE
     games and PPO +31% (better in 84% of games), stable across depths/widths;
     also helps Simba on DMC. They found alternative stabilizers (Sec. 4.4) did not.
- Relevance: the argument is about the non-stationary regime, i.e. our PPO
  learner, whose 6-block student is deeper than anything in the paper's
  baselines. BC is stationary; Adam is fine there and this is not a BC change.
- Cost: Kron keeps two preconditioner factors per weight matrix (m x m and n x n
  for an m x n weight), so for our shapes (576 x 1536 FFN, 2304 x 576 LSTM,
  128 x 576 attention) roughly 1-2x the parameter count extra, ~100-250 MB for a
  30M policy: nothing next to the learner's activations and recurrent state.
  Compute: preconditioner updates are amortised (every few steps); expect
  ~10-30% learner-step overhead. Implementation: PSGD Kron (`kron_torch`, or
  heavyball's Kron), as a drop-in for the policy optimizer only; keep Adam for
  the value net until measured.
- Test: a short league run, Kron vs Adam for the policy optimizer, same seed and
  opponent pool; compare learner loss curves, actor KL, and winrate vs the Phillip
  tiers at matched steps. Learning rate must be retuned (Kron's effective step
  differs from Adam's); start from the paper's / kron_torch defaults.

## Try AutoClip in place of a fixed clip norm for the next mega run (Keith, 2026-09-22)
- Code: https://github.com/pseeth/autoclip (Seetharaman et al., "AutoClip: Adaptive
  Gradient Clipping for Source Separation Networks", MLSP 2020, arXiv 2007.14469).
- Idea: keep a history of every step's gradient norm and clip to a percentile of
  it (the repo's default is the 10th) instead of a hand-picked constant. The
  threshold follows the run: tight early when norms are large and falling, loose
  later when they settle, so no single constant has to fit the whole run. A few
  lines: record `total_norm` each step, `clip_grad_norm_(params, np.percentile(history, p))`.
- Why here: `max_grad_norm 1.0` was a guess, and a constant threshold can only be
  right for one phase of a 1M+ step run. Clipping does real work for us early
  (~0.013 eval at 2.5k vs the unclipped fp32 6/576 run, gone by ~55k).
- Test: 6/576 bf16, same seed/data: AutoClip (p = 10) vs `max_grad_norm 1.0` vs
  none; readable by 10k steps. If it holds, use it for the next SGU mega run.
- If adaptive clipping helps and we want to go further: GradientStabilizer
  (Huang et al., arXiv 2502.17055) keeps the gradient direction and replaces its
  norm with a running estimate, so a spike never reaches Adam's moments. Aimed at
  LLM pre-training spikes, which our runs do not show, so AutoClip first.

## DONE, keep SwiGLU: standard FFN in place of the SwiGLU FFN in SGUBlock (Keith, 2026-09-20)
- Result (2026-09-22, 6/576 paper block, equal params, 72k steps, same seed/data): the
  4x GELU MLP was ~0.003 WORSE on eval throughout, for 0.86 GiB less memory and 1%
  more speed. Not worth it. Original sizing notes kept below.
- Equal params: SwiGLU has three d x h matrices (h = 1536 at d = 576), a standard
  FFN has two, so equal params means h' = 1.5 h = 2304 = exactly 4d. Same FLOPs.
- Training memory, activations kept for backward per token per layer: SwiGLU keeps
  gate+up (2h), silu(gate) (h) and the product (h) = 4h = 6144 floats; standard keeps
  pre- and post-activation = 2h' = 4608. 25% less; ~0.93 GB at 6/576, batch 512,
  T = 99, bf16, 6 layers (arithmetic from shapes, not measured). Serving unaffected.
- Prior on quality: Shazeer's "GLU Variants Improve Transformer" found SwiGLU slightly
  better than GELU/ReLU FFNs at equal params in LMs, so expect a small loss, if any.
- Test: 6/576, 2x64 heads, gate_gelu + v_norm (the paper block, the current best:
  ~0.0035 better than plain two-head and the gap grows with training), standard GELU
  FFN at h' = 2304, same seed/data, compare paired train loss and eval step for step.
- Related, further toward the paper: the published gMLP block has NO separate FFN; the
  channel expansion lives inside the gating block (d_ffn = 4d to 6d before the u/v
  split, vs our 2d followed by a SwiGLU FFN). Worth a variant once the above is read.
