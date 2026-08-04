# From GameCube memory to the input tensor

How a frame of Melee becomes the 2431-dimensional float vector the network
consumes. Covers the full journey — parsing, numeric encoding, embedding —
and ends with a column-by-column map of the tensor, so you can point at any
offset and say what lives there, where it came from, and what values it can
take.

Everything here is generated from the real embedding tree in
`smashbot/embed.py` (sizes, offsets, scales are code-derived, not
hand-copied). The packed fast path (`PackedStructForward`) produces this
exact same layout — it changes how the vector is computed, never what's in
it.

---

## 1. The pipeline at a glance

```
                offline (training)                     live (play)
        .slp replay ──peppi──▶ parquet          Dolphin ──libmelee──▶ GameState
                     \                                     /
                      ▼                                   ▼
              typed Game struct  (identical by construction: assert_same_parse)
                              │
                              ▼   from_state()          [numpy, per-leaf]
              numpy struct: casts, buckets, validity policies
                              │
                              ▼   torch conversion      [data loader / agent]
              torch tensors, one per leaf field
                              │
                              ▼   embedding forward     [part of the model]
              float32 vector, shape [..., 2431]
                              │
                              ▼
              Linear 2431 → hidden  →  SGU/transformer core → heads
```

Two parse paths, one schema. Training data is parsed offline from `.slp`
replays by **peppi** into per-game parquet (via `slippi_db`); at play time
**libmelee** reads Dolphin's memory and `slippi_ai.parse_libmelee` builds
the *same* `Game` struct. `assert_same_parse` cross-validates the two
parsers frame-by-frame, which is what guarantees the model sees the same
world at play that it saw in training.

The model's actual input is a `StateAction` triple:

- **state** — the `Game` struct for the current frame (both players, stage,
  items…)
- **action** — *our own controller output from the previous frame* (under
  the 18-frame delay alignment: the network autoregresses on what it
  previously pressed)
- **name** — the identity id we condition on ("Master Player" etc., a rank
  proxy in the ranked dataset)

## 2. Stage one: `from_state` — raw values → typed numpy

Each leaf embedding owns a `from_state` that normalizes the raw parsed value
into its declared numpy dtype, **before** anything touches torch. This is
where validation and bucketing happen (it runs in the data-loader workers in
training, and in the agent thread at play):

| leaf kind | from_state behavior |
|---|---|
| `BoolEmbedding` | cast to `np.bool_` |
| `FloatEmbedding` | cast to `np.float32` (no scaling yet) |
| `OneHotEmbedding` | cast to declared int dtype, after its **policy**: `CLAMP` clips to `[0, size-1]`; `ERROR` raises on out-of-range (data bug guard); `EXTRA` maps invalid ids to a dedicated extra bucket; `EMPTY` passes through (handled at embed time) |
| `DiscreteEmbedding` | **buckets a float in [0,1]** to an integer bin: `(x * n + 0.5).astype(uint8)` — e.g. stick axes → 17 bins |

## 3. Stage two: embedding forward — typed values → float columns

At model input, each leaf expands into its slice of the 2431-wide vector:

| leaf kind | encoding | resulting column values |
|---|---|---|
| `BoolEmbedding` | `on` if true else `off` (1 column) | `{0, 1}` — except `facing`: `{-1, +1}` |
| `FloatEmbedding` | `(x + bias) * scale`, clamped to `[-10, 10]` (1 column) | small floats, see ranges below |
| `OneHotEmbedding` | one-hot over `size` columns | exactly one `1.0`, rest `0.0` (all-zero row for invalid ids under `EMPTY`) |
| `DiscreteEmbedding` | one-hot over the bins | one `1.0` among 17 (sticks) / 5 (shoulder) |
| items `MLPWrapper` | flat item encoding (254) → shared MLP(128, 32) with ReLU | 32 **learned** features per item slot, `>= 0` |

## 4. The column map

Total: **2431 columns**, float32. Section overview:

| columns | section | width |
|---|---|---|
| 0 – 893 | **player 0 (us)** incl. Nana sub-block | 893 |
| 893 – 1786 | **player 1 (opponent)**, identical layout | 893 |
| 1786 – 1850 | **stage** one-hot | 64 |
| 1850 – 1852 | **Randall** x, y | 2 |
| 1852 – 1854 | **FoD platform heights** left, right | 2 |
| 1854 – 2334 | **items**, 15 slots × 32 learned features | 480 |
| 2334 – 2415 | **previous controller** (our action, frame t−1) | 81 |
| 2415 – 2431 | **name / identity** one-hot | 16 |

### Player block (offsets relative to block start; p0 at 0, p1 at 893)

| rel. cols | field | type | values |
|---|---|---|---|
| 0 | percent | float | raw 0–999% × 0.01 → **0 … 9.99** |
| 1 | facing | bool | **−1** (left) / **+1** (right) |
| 2 | x | float | stage coords × 0.05, clamp ±10 (saturates past ±200 game units — deep blastzone only) |
| 3 | y | float | same scale/clamp as x |
| 4 – 403 | action state | one-hot 399 | animation id 0–398 (`CLAMP` — rare Kirby-copy ids clip to 398) |
| 403 – 436 | character | one-hot 33 | internal character id 0–32 |
| 436 | invulnerable | bool | 0 / 1 |
| 437 – 444 | jumps left | one-hot 7 | 0–6 (Puff/Kirby reach 6) |
| 444 | shield strength | float | raw 0–60 HP × 0.01 → **0 … 0.60** |
| 445 | on ground | bool | 0 / 1 |
| 446 – 892 | **Nana sub-block** | — | the same 10 fields again for the Ice Climbers partner, plus one final `exists` bool (col 892 rel.) — all-default when no Nana |

### Previous-controller block (absolute cols 2334–2415)

| cols | field | type | values |
|---|---|---|---|
| 2334 – 2342 | buttons A, B, X, Y, Z, L, R, D_UP | 8 bools | 0 / 1 each (digital presses; D_UP ≈ taunt) |
| 2342 – 2359 | main stick x | one-hot 17 | raw axis [0,1] bucketed to bins 0–16 (bin 8 = neutral) |
| 2359 – 2376 | main stick y | one-hot 17 | same |
| 2376 – 2393 | c-stick x | one-hot 17 | same |
| 2393 – 2410 | c-stick y | one-hot 17 | same |
| 2410 – 2415 | analog shoulder | one-hot 5 | raw [0,1] bucketed to bins 0–4 (0 = released, 4 = full press) |

This is also exactly the space the **controller head predicts** — the
autoregressive head samples buttons → main x → main y → c x → c y →
shoulder in struct order, and the sampled result becomes next frame's
2334–2415 block.

### Items detail (cols 1854–2334)

Each of the 15 engine item slots is first encoded flat (254 dims: `exists`
bool + item **type** one-hot 238 (`EXTRA`: unknown types → bucket 237) +
item **state** one-hot 13 (`EXTRA`) + x, y at the usual 0.05 scale), then
pushed through one **shared** MLP (254 → 128 → 32, ReLU). The 32 outputs
per slot are learned features — individually meaningless, non-negative,
zeros-ish but *not* exactly zero for empty slots. All 15 slots share the
same MLP weights; slot order carries no meaning.

### Name (cols 2415–2431)

One-hot over 16 identity ids, frequency-ranked at dataset build (in the
ranked dataset: `Platinum Player`=0, `Master Player`=1, `Diamond Player`=2).
Policy `EMPTY`: an out-of-range id embeds as **all zeros** (an
"unconditioned" row). At play we condition on `Master Player`.

## 5. Reading the tensor left to right (the summary you asked for)

> Columns **0–893** are *our* player: five scalar floats (percent scaled to
> ~0–10, position scaled by 0.05 and clamped to ±10, shield to 0–0.6), two
> ±1/0-1 booleans, and three one-hots — a big 399-wide animation-state
> one-hot, 33-wide character, 7-wide jumps — followed by the same layout
> repeated for Nana plus an `exists` bit. Columns **893–1786** are the
> opponent, byte-for-byte the same layout. Then a 64-wide **stage** one-hot,
> four scalar floats for **Randall and the FoD platforms**, and **480
> learned item features** (15 slots × 32, ReLU outputs of a shared MLP —
> the only columns without fixed meaning). Columns **2334–2415** are what
> *we pressed last frame*: 8 button bits, four 17-bin one-hot stick axes,
> and a 5-bin shoulder. The final **16 columns** are the identity/rank
> conditioning one-hot. Every column is float32; everything except the
> item features and the float scalars is exactly 0 or 1 (facing/Nana-facing
> being −1/+1), and every float scalar lives comfortably inside [−10, 10]
> by construction.

## 6. Things worth knowing when debugging

- **Sparsity**: of 2431 columns, a typical frame has ~30 non-zero ones
  outside the item block. The vector is dominated by empty one-hot space.
- **The clamp almost never binds**: ±10 after scaling corresponds to ±200
  game units (positions) or 1000% (percent) — reachable only in blastzone
  flight or absurd percent. It's a NaN/outlier guard, not a working range.
- **Dtype discipline**: the play path upcasts all int leaves to int64
  before the embed (dynamo guard uniformity); this does not change any
  encoded value (verified bitwise in `test_packed_embed.py`).
- **Where to look in code**: schema/order — `embed.py`
  (`make_game_embedding`, `get_controller_embedding`,
  `get_state_action_embedding`); numeric rules — each leaf class's
  `from_state`/`forward`; fast path — `PackedStructForward` (same file);
  live parsing — `slippi_ai/slippi_db/parse_libmelee.py` (vendored).
- **Regenerating the column map**: the table above was produced by walking
  the embedding tree; if the schema ever changes, re-run the walk (see
  `test_packed_embed.py::_rand_input` for the same traversal pattern) rather
  than hand-editing offsets.
