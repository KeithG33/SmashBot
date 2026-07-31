# The Previous Attempt (2024 "SmashBot") — What It Was, Why It Stalled, What Survives

Post-mortem of the first-try project living in `/home/kage/smashbot_workspace/` (kept untouched for posterity). Activity ran May–Oct 2024; 196 wandb runs; none of the final months' work was ever committed or pushed (`SmashBot/` has substantial uncommitted changes on top of 14 commits, remote `github.com/KeithG33/SmashBot`).

## What it did

- **Game interface**: libmelee (a checkout of vladfi1's fork, with local uncommitted edits) driving the Slippi Dolphin AppImage; bot on port 1 vs human/CPU on port 2 (`SmashBot/libmelee_bot_test*.py`).
- **Approach**: PyTorch behavior cloning from Slippi replays, with a half-built PPO fine-tune.
  - Model: custom `SmashTransformer` — nn.TransformerEncoder (384-dim, 6 layers, 8 heads) over a 10-frame observation window, plus a second small encoder over previous actions fused via cross-attention, then a policy head.
  - Observation: hand-picked 52 features (10 prev actions, distance + stage, 20 features per player).
  - Action: 10-dim controller vector — BCE loss on 5 buttons + **MSE regression on 5 analog axes**.
  - RL: minimalRL's continuous PPO wrapped around the transformer; a 491-line Gymnasium env over live Dolphin. Barely exercised.
- **Data**: .slp → libmelee frame loop → numpy `(B, S, 21)` → hickle/HDF5 chunks (the derived datasets were later deleted; only raw .slp remain).

## Why it stalled (and what slippi-ai does differently)

1. **Stalled debugging sequence handling in BC** — the run history (imitation.py → imitation_simple.py → imitation_sequence.py → a final "seq_fix" checkpoint) shows the effort died fighting sequence/window bugs with no test harness to localize them. *New project: explicit unit-test gates (synthetic-delay test, overfit sanity, train-vs-inference logit match) before anything goes live.*
2. **Regressing analog sticks with MSE** — MSE on stick coordinates averages over multimodal choices (e.g. "up-left or up-right" averages to "up"). *slippi-ai discretizes sticks into bins and treats everything as autoregressive classification.*
3. **No reaction-delay modeling** — the bot predicted the current frame's action from the current frame's state; humans react ~15+ frames late, and netplay adds latency. *slippi-ai trains with an explicit 18-frame delay, which is central to its design.*
4. **Hand-picked 52 features** vs slippi-ai's comprehensive typed embedding of the full game state (399 action states, items, Nana, platforms…).
5. **Tiny dataset** — 118 self-recorded games vs the 10⁵–10⁶ anonymized ranked games Phillip trains on.
6. **~10-frame context window** vs a recurrent core carrying unbounded history at O(1) per frame.

None of this was unreasonable for a first solo attempt — each gap is exactly what the slippi-ai design solves, which is why the new project follows it faithfully before innovating.

## Assets that survive into the new project (referenced by path, not copied)

| Asset | Path |
|---|---|
| Melee v1.02 NTSC ISO | `/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso` |
| Slippi Online AppImage | `/home/kage/smashbot_workspace/Slippi_Online-x86_64.AppImage` |
| **ExiAI AppImage** (vladfi1's AI build: headless + fast-forward) | `/home/kage/smashbot_workspace/Slippi_Online-x86_64-ExiAI.AppImage` |
| Personal replay archive — 6,492 games through Jul 2026, recent files slp v3.19.1 | `/home/kage/slippi/Replays/` |
| Old debug replays — 118 games, **slp v3.16.0** (below slippi-ai's 3.18 min; needs upgrade or substitution) | `/home/kage/smashbot_workspace/dataset/SlippiGames/slp/` |
| Working Dolphin/Slippi configs | `~/.config/SlippiOnline/`, `~/.config/Slippi Launcher/` |
| Fast Rust .slp inspector CLI | `/home/kage/smashbot_workspace/slp-linux-x86_64/.../slp` |

Also noteworthy: the old `Untitled Diagram.drawio` sketches a never-built next-gen design (transformer encoder → policy head → xLSTM long-horizon memory) — the xLSTM memory idea is a candidate for the post-baseline "improvements" phase.

**Housekeeping flag**: the old `libmelee/` checkout and `SmashBot/` repo contain uncommitted local changes that exist nowhere else. They're preserved as-is; if the workspace is ever cleaned, diff/commit them first.
