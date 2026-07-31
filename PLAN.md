# ShineBot: Replicate slippi-ai in PyTorch (imitation → RL)

## Context

Fresh start on the SSBM bot. The old undergrad-era attempt (`SmashBot/`, custom transformer BC + half-built PPO) stays untouched for posterity. New goal: replicate **vladfi1/slippi-ai** (Phillip II) — imitation learning on Slippi replays, then PPO fine-tuning with a KL-to-teacher — via a **hybrid** approach: reuse slippi-ai's battle-tested data pipeline (`slippi_db` → parquet) and Dolphin infrastructure, but reimplement the model, losses, and training loops as our own clean **PyTorch** codebase. Faithful replication of their production architecture first (validated against known-good behavior), improvements after.

**Decisions made by user:** hybrid approach; Fox first; faithful `tx_like` architecture baseline; bring-up on a **small debug slice of the HuggingFace ranked dataset** (`erickfm/melee-ranked-replays` — the mirror of Fizzi's anonymized ranked dumps that Phillip trains on) before downloading the full thing. The old local .slp files (v3.16, below slippi-ai's 3.18 minimum) are **not used** — known-good HF data only.

**Machine facts (verified):** RTX 3090 24GB, 48 threads, 125GB RAM. The main NVMe (which holds the OS *and* `/home/kage`) is 99% full (11GB free) → **all heavy artifacts (datasets, venvs, caches, checkpoints) go to `/home/kage/drive2` (466GB free)**. System Python is 3.11 and **stays untouched**; `uv` installs a project-local Python 3.12 on drive2. **No system-level changes (sudo, system packages, system Python) will ever be made without telling the user first.** On disk already: Melee v1.02 NTSC ISO (`/home/kage/slippi/`), Slippi_Online + **ExiAI** AppImages (vladfi1's AI build with fast-forward gecko), working Dolphin configs, user's own replay archive (6,492 games — reserved for a future personal-style fine-tune, not the baseline).

## Key slippi-ai facts driving the design

- **Data**: `slippi_db/parse_local.py` pipeline: `Root/{Raw/ (archived .slp), raw.json, Parsed/ (one zlib'd parquet per game), parsed.sqlite, meta.json}`. Offline parser = `peppi-py-vladfi`; a matching **libmelee parser** (`slippi_db/parse_libmelee.py`) is used at inference so live gamestate encodes identically to training data (`preprocessing.assert_same_parse` cross-validates). Schema in `slippi_ai/types.py`.
- **Model (production)**: composable embeddings (player: percent×0.01, x/y×0.05, action one-hot 399 clamped, character 33, jumps 7, etc.; stage 64; items 15×shared MLP(128,32); name one-hot 16 conditioning, `DEFAULT_NAME='Master Player'`) → `tx_like` core: Linear→512, then 3×[Residual LSTM(512) → pre-LN ResBlock(ffw×2, GELU, zero-init out)] — a transformer layout with attention swapped for LSTM, O(1)/frame inference → **autoregressive controller head** (residual stream 128; order: buttons(8 Bernoulli), main stick x,y (17 bins each), c-stick x,y, shoulder (5 bins); teacher-forced in training). Loss = summed per-component NLL. Separate 1-layer value net, TD loss, γ=0.5^(1/240). Adam 1e-4, batch 512 × unroll 80.
- **Delay (the crux)**: `policy.delay=18` frames (~300ms, human-like reaction + netplay buffer). Training slicing: batch has U+D+1 frames; states[0..U-1] predict actions[D+1..U+D] with prev-actions[D..U+D-1]. Inference: `DelayedAgent` deque pre-filled with D no-ops; effective delay −= dolphin online_delay. **#1 source of silent bugs — off-by-one hell.**
- **RL**: PPO (log-space clip ε=1e-2) + **forward KL to frozen teacher** ("refine what humans do") + entropy bonus; reward zero-sum: −1/death, −0.01×damage taken, γ half-life 4s, optional ledge/stall penalties (`slippi_ai/reward.py` — reuse). ~96 headless ffw Dolphins, periodic restarts (memory leaks), CPU-bound not GPU-bound.
- Imitation ≈ "a few days to a week" on a 3080Ti — our 3090 is the same class.

## Top-level structure

- **Repo**: `/home/kage/smashbot_workspace/ShineBot/` (code is small; fine on root disk). Heavy stuff at `/home/kage/drive2/ShineBot/{venv, venv-ref, data, runs, models, hf-cache, uv}` with a `scripts/env.sh` exporting `UV_PYTHON_INSTALL_DIR, UV_CACHE_DIR, HF_HOME, WANDB_DIR, PIP_CACHE_DIR` → drive2, so no tool silently fills the nearly-full home drive with caches. Confirm with `df` after setup.
- **slippi-ai consumption**: git **submodule** at `vendor/slippi-ai`, pinned commit, installed `pip install --no-deps -e` so its TF/JAX deps never enter our training venv. Explicitly install only what the reused modules need (`peppi-py-vladfi>=0.9.2`, vladfi1's `melee` fork from his git, `pyarrow`, `dm-tree`, `absl-py`, `portpicker`, `py7zr`, …) — determined empirically at M0 by import-auditing `slippi_db.*`, `slippi_ai.{types,reward,dolphin,envs}`. Any shared module that drags in TF gets ported into our repo instead of reused (candidates: `observations.py`, `envs.py`).
- **Second "reference" venv** (`venv-ref`) with slippi-ai's full TF stack (use `tensorflow-cpu`): used only for parse cross-validation, dumping golden batches from their loader, and running their pretrained **medium-v2** model as an opponent.
- **Reuse as-is**: `slippi_db/*` (parsing), `slippi_ai/types.py` (schema), `slippi_ai/dolphin.py` (process mgmt, ExiAI/gecko/ffw — model-agnostic), `slippi_ai/reward.py`, `parse_libmelee` (inference-time encoding guarantee). **Reimplement in PyTorch**: data loader, embeddings, network, controller head, policy/value/losses, delay logic, DelayedAgent, BC + RL training loops.
- **Config**: dataclasses + `tyro` (typed nested configs mirroring theirs, e.g. `--policy.delay 18`). **Checkpoints**: `torch.save` of `{config, state, best_eval_loss, version}` with a versioned-upgrade map in `saving.py`.

**Working agreement:** after each completed milestone, give the user a high-level (50k-foot) summary of what was built and how it fits the pipeline, before continuing (brief updates after M0–M3). **Hard pause at the start of M4**: a bigger consolidated review of everything built (M0–M3) so the user is on the same footing before the debugging/usage phase (M4–M8) begins. Detailed full review at the end.

### Repo layout (new code)

```
shinebot/               # Python package (lowercase by import convention; repo dir is ShineBot/)
  paths.py configs.py types_bridge.py
  embed.py            # composable Embedding classes (port of tf/embed.py semantics)
  observations.py     # tech-animation masking (ported if theirs is TF-entangled)
  data/{dataset.py, trajectories.py, loader.py}   # TrajectoryManager(overlap=D+1), mp decode, BatchAccumulator
  networks.py heads.py policy.py value.py
  delay.py            # ALL delay index math in ONE place + deque logic
  saving.py train_bc.py
  eval/{agent.py, env.py, play.py, evaluate.py, vs_reference.py}
  rl/{ppo.py, rollouts.py, train_rl.py}
  tests/
scripts/{env.sh, setup_env.sh, parse_local.sh, download_hf_fox.sh, dump_reference_batches.py}
vendor/slippi-ai/     # pinned submodule, never edited
```

## Milestones (each with a hard verification gate)

**M0 — Environment + assets (~half day).** uv python 3.12, both venvs on drive2, torch cu12x, `--no-deps` install of the submodule + minimal dep set via import audit. Gates: CUDA sees the 3090; `slippi_db.parse_peppi`/`parse_libmelee`/`slippi_ai.dolphin` import clean in the training venv; Dolphin smoke test — launch ExiAI AppImage headless against the ISO via `slippi_ai.dolphin`, reach menu, clean shutdown. Strongly recommended: download their pretrained **medium-v2** (Dropbox link in their README) and run their own eval vs CPU lvl 9 in venv-ref — proves the whole Dolphin stack on this machine with a known-good agent before any of our code exists.

**M1 — Debug corpus from HuggingFace + parse through slippi_db (~1 day).** Download **one small Fox shard** (a few hundred games) from `erickfm/melee-ranked-replays` to `drive2/ShineBot/data/debug-fox/Root/Raw/`, run `parse_local.py` + `make_local_dataset.py`. Known-good modern data — no version hacks; this doubles as a dress rehearsal for the full M7 download. (Old local v3.16 replays are not used, per user.) Gates: parquet per game; spot-check decoded `Game` arrays (valid stage/action ids, plausible positions, frame counts); `assert_same_parse` (venv-ref) passes on a few games.

**M2 — PyTorch data loader (~2–4 days).** Mirror their design: `DatasetConfig(allowed_characters='fox', swap=True, test_ratio=0.1)`, game-level split before mirror, windows of U+D+1 frames, multiprocess parquet decode → prefetch thread → preallocated batches `(512, U+D+1)` pinned tensors. Gates: **golden-file test** — venv-ref script dumps their loader's decoded per-game arrays on `vendor/slippi-ai/slippi_ai/data/toy_dataset/`; ours must match bit-exactly per game (batch composition may differ — compare stats only); throughput benchmark (won't starve the 3090); unit tests for swap/split/overlap.

**M3 — Embeddings + network + head (~3–5 days).** Port embedding semantics exactly (scales, one-hot sizes, clamping, Struct field order). `tx_like` with cuDNN `nn.LSTM` (no torch.compile on the LSTM at first); both `unroll()` and O(1) `step()`. Autoregressive head exactly as specified. All delay slicing in `delay.py`. Gates (pytest): synthetic-delay test that reaches zero loss **only** at the configured D; unroll-vs-step equivalence over 100 frames; teacher-forced vs forced-sample path logit equality; autoregressive-order coupling test; embed round-trips; param-count sanity.

**M4 — Overfit sanity (~1–2 days)** — the gate the old attempt never had. **⏸ PAUSE POINT: before starting M4, stop and give the user a consolidated review of M0–M3** (the full lay of the land) — the debugging/usage phase starts here and the user wants to be on the same footing. (a) **Synthetic movement check** (the user's "moves left and right" idea): fake dataset through the real loader where the player alternates stick-left/stick-right every 30 frames → model reaches ~0 NLL and open-loop rollout reproduces the pattern shifted by exactly D frames. (b) Overfit 3–5 real games to ~0 NLL, per-component loss curves behave. Failure here = bug in M2/M3; do not proceed.

**M5 — End-to-end BC on debug corpus (~2–3 days).** Full `train_bc.py`: eval split (key metric: eval policy loss), separate value net + TD loss, checkpointing/resume, wandb. Gates: eval loss beats marginal-distribution baseline; no NaNs; checkpoint→reload→identical eval loss; measure frames/s to forecast M7 wall-clock.

**M6 — Live play (~3–5 days).** `eval/agent.py` `DelayedAgent` (deque of D no-ops, recurrent state, `delay -= online_delay`), `eval/play.py` vs human/CPU using `slippi_ai.dolphin` + `parse_libmelee` (or our thin `env.py` if their `envs.py` is TF-entangled per M0). Gates: **offline logit-match test first** — feed a recorded game through the inference path, assert logits match the training path on identical states (catches peppi↔libmelee mismatch and delay off-by-ones without Dolphin); then live vs CPU — moves purposefully, not standing still/spinning; NN step ≤ ~5ms/frame.

**M7 — Full Fox dataset + real run (~1 day setup + 3–7 days training).** `snapshot_download('erickfm/melee-ranked-replays', allow_patterns=[Fox shards])` to drive2 — **size-check via HF API against the 466GB budget first**; delete Raw archives after parsing if tight. Parse with 48 threads. Train Fox vs all, batch 512 × unroll 80, delay 18, condition on 'Master Player'. Gates: dataset stats report; sane loss curves; periodic Dolphin spot-checks (watch real movement/tech emerge); hourly checkpoints + best-eval retention.

**M8 — Evaluation harness (~2–4 days).** Headless ffw N-game matches; metrics from `reward.py` + stocks/damage; opponents: CPU 3/6/9, older self-checkpoints, and **their medium-v2**: run both stacks against each other in one Dolphin via a small `vs_reference.py` in venv-ref (their agent drives port 1 via their eval_lib, ours drives port 2 — the interface is the Dolphin loop, not their class hierarchy; ONNX-export our model if TF+torch fight in-process). Gates: reproducible JSON eval reports; beats CPU lvl 9 convincingly; non-embarrassing vs medium-v2 (losing is fine — it's RL-tuned).

**M9 — RL fine-tuning (~2–4 weeks, iterative).** `rl/`: teacher = frozen M7 checkpoint, policy init = same; PPO + forward-KL-to-teacher + entropy per their loss; reward via reused `reward.py`; rollout pool of headless ffw Dolphins starting at 64 envs scaling toward 96–128 (48 threads), `AsyncDelayedAgent` with `batch_steps=4`, rollout 240, periodic env restarts + 10-step burn-in; opponent schedule CPU lvl 9 → self-play snapshots. Gates: win rate vs frozen M7 climbs past 50%; reward-vs-KL curves; watch for reward hacking (enable ledge/stall penalties if it appears); no throughput decay over 12h.

## Risks / gotchas

1. **Delay off-by-ones** — all index math in `delay.py`; synthetic-delay + offline logit-match tests are the guards.
2. **peppi vs libmelee mismatch** — reuse their `parse_libmelee` verbatim; `assert_same_parse` at M1; M6 logit-match is the end-to-end check.
3. **Autoregressive order coupling** — one ordering constant, asserted everywhere, unit-tested.
4. **Hidden TF imports in shared modules** — M0 import audit decides reuse-vs-port per module.
5. **Disk space** — the main NVMe (OS + home dir) has only 11GB free; by default pip/HF/wandb caches land in `~/.cache` on that drive and would fill it. Fix: redirect all caches/venvs/datasets to drive2 via env vars in `scripts/env.sh` (user-level config only — no system changes); confirm with `df` after setup.
6. **Dolphin leaks/zombies in RL** — periodic restarts, portpicker, watchdog (their `dolphin.py` handles much of it).
7. **`peppi-py-vladfi` cp312 wheel** may not exist → user-local rust toolchain on drive2 (no system install; user will be told if this becomes necessary).
8. **Checkpoint/config drift** — versioned configs from day 1.

## Post-baseline fun (future work, after M8 A/B infra exists)

custom_v1-style compact action space; Mamba/sliding-window core replacing the LSTM stack; bf16 + torch.compile throughput; GAE/bigger value net; personal-style fine-tune on the user's own 6,492 replays; multi-character conditioning.

## Verification summary

Every milestone has a gate above; the three load-bearing ones are the **golden-file loader test** (M2, vs their toy dataset), the **synthetic-delay/overfit tests** (M3/M4), and the **offline training-vs-inference logit match** (M6). End-to-end success = M7 model visibly plays Fox competently vs CPU lvl 9 in Dolphin, then M9 pushes past the imitation teacher in head-to-head win rate.
