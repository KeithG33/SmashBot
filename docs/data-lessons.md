# Data Lessons: Where the 2024 Attempt Went Wrong (Parsing & Embedding)

A post-mortem comparison of the old SmashBot data pipeline (`SmashBot/smashbot/data/extract_dataset_sequence.py`)
against slippi-ai's approach — written after replicating the latter in ShineBot (M0–M3).
Companion to [previous-attempt.md](previous-attempt.md). Ranked by how fatal each issue was.

## 1. Categorical IDs fed as raw numbers (the big one)

The old 20-dim player vector included `pstate.action.value`, `pstate.character.value`, and
`stage.value` as plain scalars. `action` is an animation-state ID from 0–386 — action 23
(falling) and action 24 (falling-aerial) are neighbors numerically but action 212 (grab) is
not "ten times more" than action 21. A network fed `action=212` as a float has to waste
enormous capacity discovering that this axis has no metric structure — and mostly it just
can't; it learns a blurry average response across nearby IDs.

This is *the* most informative feature in Melee — the animation state tells you almost
everything about what a player can and will do next — and it was handed to the network in a
nearly unlearnable encoding. slippi-ai one-hots it into 399 dimensions (and character into
33, stage into 64), so "opponent is in grab" is a clean, linearly-separable input feature.
Note the contrast in input width: the old observation was 52 floats; theirs is ~1000+ dims,
most of it one-hot structure. **Wide-and-sparse beats narrow-and-dense for this kind of state.**

## 2. MSE regression on stick positions

The old loss was BCE on 5 buttons + **MSE on 5 analog values**. Stick inputs are extremely
multimodal: at a decision point, "hold up-left" and "hold up-right" are both common, and
MSE's optimal prediction is their *average* — stick straight up, which is a different move
entirely (and often the worst of the three). Trained this way, the bot systematically
outputs mushy interpolated inputs that no human ever pressed. It also can't *sample* —
there's no distribution, just one deterministic blend.

slippi-ai never regresses: sticks are discretized into 17 bins per axis and everything
becomes classification, so the model represents genuine multimodal distributions and
sampling picks one mode cleanly. The autoregressive head then goes further: it factorizes
the *joint* distribution (buttons first, then stick conditioned on buttons), so it can't
emit incoherent combinations like "B pressed + stick position that makes no sense with B."

## 3. The causally-relevant state wasn't in the input window

Two compounding issues. First, no delay modeling: the model was trained to predict the
frame-t action from state up to frame t. But humans react ~15–20 frames late — the human's
frame-t button press was *caused* by what happened around frame t−18. Second, the context
was a 10-frame buffer — 166ms. Put together: the state that actually explains the target
action had usually **already scrolled out of the window** before the action arrived. The
network was being asked to predict effects whose causes it literally could not see. That
alone would make training feel "off" no matter what else was fixed — loss plateaus at a
mushy baseline and nothing seems to help.

slippi-ai makes the delay a first-class design element (train on state[t] → action[t+18])
and uses a recurrent core that carries state across the whole game, so context is unbounded
instead of 10 frames.

## 4. Materialized sliding windows in storage

The old extractor wrote `(10-frame window, action)` per player per frame — every frame
stored ~10× (once per window it appears in) × 2 players, before gzip. That's why the
derived datasets were multi-GB from 118 games and why iterating on "what features do I
extract?" was painful: any change meant a full re-extraction of everything. slippi-ai
stores each game **once**, losslessly, as a columnar parquet of the full typed game state,
and does windowing/feature-encoding *at load time*. Changing the embedding costs nothing;
changing the window or delay costs nothing. The parse is slow and done once; everything
downstream is cheap and revisable. This is probably the biggest "workflow" lesson:
**keep raw-ish data on disk, push interpretation to load time.**

## 5. No data hygiene or parity checks

The old filter set was: skip >2 players, skip Ice Climbers. No damage/length filters (games
where someone stood in place made it in), no dedup, no notion of who won or how good the
players were. slippi-ai filters for validity, ≥100 total damage, dedupes, can keep winners
only, and conditions on player identity so "Master Player" data is distinguishable from
"Platinum Player" data. And critically, it *proves* training and inference see identical
encodings (`assert_same_parse` between its two parsers) — whereas the old pipeline had no
way to detect an encoding drift between extraction and live play. To be fair, it shared one
`parse_game_state` function between both paths, which was the right instinct — but with
continuous regression outputs, the prev-action fed back at inference never matched anything
from training anyway (exposure bias), and nothing would have told you.

## What the old attempt got right

Worth remembering: using both players' perspectives (slippi-ai's `swap=True` does the
same), feeding previous actions back as input, merging L/R and X/Y buttons (slippi-ai's
newer `custom_v1` action space does exactly that merge), sharing one parse function between
training and inference, and multiprocessing the extraction. The mechanics were fine.

## The one-line summary

The parsing was fine mechanically — libmelee frame loop, multiprocessing, buffering all
worked. Where it went wrong is that **encoding decisions were treated as an afterthought**
(raw IDs, ad-hoc scales like `/300` and `/550`, MSE targets), while slippi-ai treats the
embedding as the core design artifact — a typed, composable schema where every field has a
deliberate representation (one-hot, scaled float, discretized bin), the controller is a
probability distribution rather than a regression target, and the delay/context structure
matches how humans actually generate the data. That, plus store-raw-interpret-late, is
really the whole gap.
