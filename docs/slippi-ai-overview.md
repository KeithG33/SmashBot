# slippi-ai (Phillip II) — How It Works

Summary of a deep dive into [vladfi1/slippi-ai](https://github.com/vladfi1/slippi-ai) (July 2026, branch `main`, Python ≥3.12). This is the SOTA open-source Melee bot: **imitation learning on human Slippi replays, then RL fine-tuning**, played through Slippi Dolphin via libmelee. The codebase has three parallel backends: `slippi_ai/tf/` (TensorFlow+Sonnet, the mature path), `slippi_ai/jax/` (Flax NNX, actively developed), and shared framework-agnostic code in `slippi_ai/*.py`.

## Pipeline overview

```
.slp replays → slippi_db (peppi parser) → one zlib'd parquet per game + meta.json
            → imitation learning (behavior cloning, ~days on one GPU)
            → RL fine-tuning (PPO + KL-to-teacher, self-play in headless Dolphin fleet)
            → play via Dolphin + libmelee (human vs bot, netplay, twitch bot)
```

## 1. Data pipeline (`slippi_db/`)

- `parse_local.py` expects `Root/{Raw/ (zip/7z of .slp), raw.json, Parsed/, parsed.sqlite, meta.json}`.
- Offline parser: **peppi** (Rust, via `peppi-py-vladfi` fork). Online/inference parser: **libmelee** (`slippi_db/parse_libmelee.py`) — kept bit-compatible with the peppi parse (`preprocessing.assert_same_parse`) so live gamestate is embedded exactly like training data. This dual-parser consistency is a load-bearing design decision.
- Schema (`slippi_ai/types.py`): `Game(p0, p1: Player, stage, randall, fod_platforms, items[15])`; `Player(percent, facing, x, y, action u16, invulnerable, character, jumps_left, shield_strength, on_ground, controller, nana)`; `Controller(main_stick, c_stick, shoulder, buttons[8: A,B,X,Y,Z,L,R,D_UP])`. Stored as a single-column parquet StructArray, optionally zlib-compressed.
- Dataset filters (`make_local_dataset.py`): validity, ≥100 total damage, dedupe by match id, optional winner-only. **Minimum .slp version 3.18** (Randall/FoD/stadium features); `upgrade_slp.py` re-records older replays.

## 2. Training data (what Phillip actually trains on)

- **Fizzi's anonymized ranked dumps** (Platinum/Diamond/Master games; names become "Master Player" etc.) — shared via the Slippi Discord `#ai` channel. Community mirror on HuggingFace: [`erickfm/melee-ranked-replays`](https://huggingface.co/datasets/erickfm/melee-ranked-replays), sharded by character × rank-pair. Scale: order 10⁵–10⁶ games (an old public dump was 27GB compressed → 200GB of .slp).
- Plus large per-player dumps (Hax 85K games, iBDW 52K, Zain 13K, …) enabling **player-conditioned** imitation via a "name" input token.
- Typical run filters to one character (`allowed_characters=fox`, opponents=all); `swap=True` uses both players' perspectives (2× data); optional left/right `mirror` (applied after the train/test split).
- A toy dataset ships in-repo (`slippi_ai/data/toy_dataset/`) for smoke tests, and a pretrained 12-character model ("medium-v2") is downloadable from the README.

## 3. Model architecture (production config)

- **Embeddings** (`tf/embed.py`): composable typed embedding classes (Bool/Float/OneHot/Discrete/Struct). Player: percent×0.01, x,y×0.05, action one-hot 399 (clamped), character one-hot 33, jumps_left one-hot 7, shield×0.01, plus Nana (Ice Climbers). Stage one-hot 64, Randall/FoD platforms, 15 item slots each through a shared MLP(128,32). Input = `StateAction(state, previous_own_controller, name)` where `name` is a one-hot player-identity conditioning token (default eval name: `'Master Player'`).
- **Core network**: `tx_like` — *a transformer block layout with self-attention replaced by an LSTM*: Linear→512, then 3 × [Residual LSTM(512) → pre-LayerNorm ResBlock(ffw ×2, GELU, zero-init output)]. No attention → **O(1) inference per frame** at 60fps. (Registry also has mlp/lstm/gru/res_lstm.)
- **Controller head**: **autoregressive** over components in fixed order — buttons (8 Bernoulli), main stick x, y (17 bins each via uniform discretization), c-stick x, y, shoulder (5 bins) — each conditioned on previously sampled components through a residual stream (size 128), teacher-forced during training. Loss = summed per-component negative log-likelihood.
- **Value function**: separate 1-layer tx_like/512 net, squared TD error vs discounted returns, γ = 0.5^(1/(4·60)) (4-second reward half-life). Adam lr=1e-4, batch 512 × unroll 80 frames. Key metric: eval policy loss.
- An "observation filter" masks info the human couldn't know yet (e.g. tech animations are visually identical for the first few frames).
- Newer JAX-only work: a compact human-prior action space (`custom_v1`: polar stick clusters, fused button×c-stick categorical), Q-learning, and Nash-equilibrium policy extraction.

## 4. The frame-delay trick (the crux)

The bot plays with an intentional **18–21 frame (~300ms) reaction delay** — makes it human-like/beatable and absorbs netplay latency ("buffer donation").

- **Training**: with delay D and unroll U, each batch window has U+D+1 frames; states [0..U−1] predict actions [D+1..U+D] with previous-actions [D..U+D−1].
- **Inference**: `DelayedAgent` keeps a deque pre-filled with D no-op actions; each step pushes an observation, pops the action from D frames ago. Dolphin's own online delay is subtracted from the budget. The slack also allows batching several frames per NN call and letting the emulator run ahead of inference.
- This index math is notoriously off-by-one-prone; their test suite covers it and ours must too.

## 5. RL phase

- **PPO + KL-to-teacher**: loss = −pg·PPO-objective (log-space clip, ε=1e-2) + β·actor-KL + w·**forward KL(policy‖teacher)** − entropy bonus. The forward KL (rather than the usual reverse) deliberately lets the agent *refine* human play instead of imitating all of it, mistakes included. Teacher = the frozen imitation checkpoint; the trainable policy initializes from the same checkpoint.
- **Reward** (`reward.py`): zero-sum by construction — −1 per death, −0.01 × damage taken (Nana at 0.5×), optional penalties for bad ledge grabs and off-stage stalling, γ half-life 4s.
- **Self-play infra**: ~96 headless Dolphin instances with Fizzi's fast-forward gecko code (unlimited emulation speed, instant match restart — requires vladfi1's **ExiAI** Dolphin build), async batched GPU inference, opponents = level-9 CPU / self (periodically synced snapshot or both-sides training) / other checkpoints. Dolphin leaks memory → envs restarted periodically with a burn-in. RL is CPU/emulator-bound, not GPU-bound.

## 6. Infra & requirements

- **Single-GPU project**: author's reference box is an i7-11700K + RTX 3080Ti + 64GB RAM. Imitation ≈ a few days to a week; our RTX 3090 / 48-thread / 125GB machine is the same class or better.
- Key deps: `peppi-py-vladfi`, vladfi1's `libmelee` fork (ExiAI/ffw/EXI-inputs support), pyarrow, wandb, absl+fancyflags. TF extra pulls dm-sonnet/tf-nightly; JAX extra pulls flax NNX.
- Runtime needs: Melee **v1.02 NTSC ISO**, Slippi Dolphin (ExiAI build for headless/ffw), min .slp 3.18, frozen Pokémon Stadium.

## What our project reuses vs rebuilds

**Reuse** (via pinned submodule): `slippi_db/*` parsing, `slippi_ai/types.py` schema, `slippi_ai/dolphin.py` process management, `slippi_ai/reward.py`, `parse_libmelee` for inference-time encoding. **Rebuild in PyTorch**: data loader, embeddings, tx_like network, autoregressive head, delay logic, BC and RL training loops, agent/eval wrappers. See the project plan for milestones.
