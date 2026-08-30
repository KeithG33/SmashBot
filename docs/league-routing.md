# League routing: per-match opponents on a static grid

AlphaStar draws the opponent **per match**. We used to draw per *generation*:
every 250 learner steps an auction assigned one member to each of 12 slots,
and the 12 envs hard-wired to a slot all switched brains — which needed
parked spares, outgoing seats, forced mid-game swaps, and made the opponent
mix swing on a 250-step clock. This design keeps the thing that makes the
league fast (one captured vmap forward over a fixed grid) and unfixes the
only relationship that caused all of that: **which env sits in which cell**.

## Pieces

- **`rl/agent.LeagueAgent` — the grid.** `S` weight *slices* × `N` *cells*.
  A slice holds one member's weights (a stacked copy; `load_slice(s, sd)`
  writes it in place, so captured replays see it). A cell is a seat with its
  own recurrent state, prev action and delay queue. One forward per frame:
  CUDA = the vmap captured into a manual CUDA graph; CPU = the same vmap run
  eagerly. No per-slice loop, no live modules, one throwaway template for
  `functional_call` (the policy has tied parameters; see the class docstring).
- **`rl/league.LeagueSeats` — the allocator.** Slices are a weight cache
  (LRU-reclaimed when empty, weights kept while empty); cells are seats.
  `place(env, member)` seats an env on a slice holding that member, loading
  the member into an *empty* slice if none does. Phillip's agent (his own
  architecture) is one more pool with fixed capacity. A slice's weights only
  ever change when it is empty: nobody is swapped mid-game.
- **`rl/league.League` — the protocol.** Per env: `member_now` (who it is
  fighting, the payoff label) and `member_next` (drawn one game ahead).
  At a game boundary: credit the ended game to `member_now`, release the
  seat, take a seat for `member_next`, draw the one after. If the drawn
  member has no free seat at that moment, fall back to a PFSP draw over
  members that *do* have room (counted: `rl/league/fallback_rate`).
- **`rl/pool.SnapshotPool.draw_member`** — one PFSP draw over the whole
  league (two-stage class weighting, f_hard/f_var/explore as before, with
  replacement). `boot_draws` covers every member once at startup.
- **`rl/rollouts.DolphinRolloutWorker`** — gathers every cell's env view
  into the grid by an index map, runs the grid (and Phillip), scatters
  controllers back to envs. Harvest groups are fixed: "ours" = all grid
  cells, "phillip" = his rows; eligibility per cell = occupied ∧ opponent
  char whitelisted.

## The env contract (`rl/env_process.py`)

One command key for league envs, sent every frame:
`opp_next = {"kind": "policy"|"cpu", "char_lock": CHAR|None}` — what the
opponent seat should be for the env's **next** game. The env applies it at
a game boundary *after* reading that frame's command, so a draw made when
game *g* starts shapes game *g+1*: the lock pins the next game's character
(imports play their main); a kind change makes the current game the
Dolphin's last (spare pre-booted now, recycle at the boundary adopts cpu or
returns to a policy seat). The env keeps reporting what it actually serves
(`opp_serving`, `result_serving`), so attribution follows reality.

## Grid slack

The grid is nearly always full (every league env sits somewhere), so an env
can only *move* to a member whose slice has a free cell. `N` is sized as
`ceil((league_envs + S) / S)` — one slice's worth of spare cells — so free
seats float to where demand is and a drained slice can be reloaded for a
hot member. Idle cells compute garbage on always-reset state; harmless.

## Knobs

`league_slices` (S; 107 MB of VRAM each — at most S distinct members
resident at once), `phillip_capacity` (rows in his agent; 0 = one slice's
worth), `snapshot_interval` (a new ghost joins the league; nothing else
happens on that clock any more).

## What this removed

Auctions and `apply_assignments`, slot policies (12 live modules),
`_SlotPool`/spares/parking, outgoing seats and evictions, pending adoption,
per-slot char locks and cpu wishes, the per-slot loop forward, and the
post-auction fps trough (parked ghosts stepping un-batched spares).

## v9: static shares

Fixed opponents left the draw entirely. Imports are DEDICATED envs served
by a static agent (one slice per member, cells permanently assigned at
boot via `import_dedicated_envs`; char-locked members pin their character,
`@ANY` members redraw per game). Phillip is dedicated via the classic
`ref_envs` machinery. The teacher is folded out: snapshot-0000000 (= the
BC teacher) rides the archive as an ordinary ghost. PFSP's whole world is
ghosts + cpu on the sampler grid; `league_slices` sizes that grid alone.
Fixed members keep their payoff rows (the worker credits results under
the same keys), so `I:`/`R:`/`rl/imports/*`/`rl/phillip/winrate` read
continuously across the v8->v9 seam (`metric_imports` surfaces dedicated
imports in `category_estimates`). The legacy league-member flags
(`league_teacher` / `league_phillip` / league-drawn imports) remain
supported for older configs.
