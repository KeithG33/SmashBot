"""Per-match league routing (AlphaStar draws the opponent per MATCH).

Three pieces, all Dolphin-free and unit-testable:

- MemberWeights: member key -> policy state_dict on CPU (teacher copy,
  imports, the snapshot archive through a small LRU).
- LeagueSeats: the seat allocator. The league grid (rl/agent.LeagueAgent)
  is S weight SLICES x N CELLS; a slice is a weight-cache entry holding one
  member, a cell is a seat. Phillip's agent (his own architecture) is one
  more pool with a fixed capacity. Envs sit wherever their drawn member is
  loaded; a slice's weights change only when it is EMPTY — nobody is ever
  swapped mid-game, so there are no parks, spares or outgoing seats.
- League: the per-env protocol. At the start of every game an env draws the
  opponent for its NEXT game (one game ahead, so the env can arm a
  char-locked import's character and pre-boot a recycle for cpu); at the
  boundary it credits the ended game to the member that played it and
  takes a seat for the drawn one. When the drawn member has no free seat
  at that moment (the grid is nearly full by design), the env falls back
  to a PFSP draw over members that DO have room — counted, so the bias is
  visible (fallback_rate).
"""

from __future__ import annotations

import collections
import random
import typing as tp

import torch

from smashbot.rl.pool import SnapshotPool, _is_import_key


class MemberWeights:
    """member key -> state_dict (CPU). Teacher and imports are held for the
    run; snapshots load from the archive through an LRU (107 MB each).
    warm(member) loads in a background thread — draws happen a game ahead,
    so by the time a seat needs the weights the disk read is long done and
    get() never stalls the frame loop."""

    def __init__(self, fixed: dict[str, dict], lru: int = 16):
        import threading

        self._fixed = dict(fixed)
        self._cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.lru = lru
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Thread] = {}

    def set(self, member: str, state_dict: dict) -> None:
        """Replace a fixed member's weights (teacher swap); resident slices
        refresh at their next load."""
        self._fixed[member] = state_dict

    def _put(self, member: str, sd: dict) -> None:
        with self._lock:
            self._cache[member] = sd
            self._cache.move_to_end(member)
            while len(self._cache) > self.lru:
                self._cache.popitem(last=False)

    def warm(self, member: str) -> None:
        """Start loading `member` from disk if it is not cached (no-op for
        fixed members and loads already in flight)."""
        import threading

        if member in self._fixed:
            return
        with self._lock:
            if member in self._cache or member in self._inflight:
                return
            t = threading.Thread(
                target=lambda: self._put(member, torch.load(member, map_location="cpu")),
                daemon=True,
            )
            self._inflight[member] = t
        t.start()

    def get(self, member: str) -> dict:
        if member in self._fixed:
            return self._fixed[member]
        t = self._inflight.pop(member, None)
        if t is not None:
            t.join()
        with self._lock:
            sd = self._cache.get(member)
            if sd is not None:
                self._cache.move_to_end(member)
                return sd
        sd = torch.load(member, map_location="cpu")
        self._put(member, sd)
        return sd


class _Pool:
    """One seat pool: a grid slice (loadable member) or Phillip (fixed)."""

    __slots__ = ("index", "capacity", "member", "loadable", "occupants", "last_used")

    def __init__(self, index: int, capacity: int, member: str | None, loadable: bool):
        self.index = index
        self.capacity = capacity
        self.member = member
        self.loadable = loadable
        self.occupants: dict[int, int] = {}  # env -> row
        self.last_used = 0

    def free_rows(self) -> list[int]:
        used = set(self.occupants.values())
        return [r for r in range(self.capacity) if r not in used]


Seat = tuple[int, int]  # (pool index, row); pool index < S = grid slice


class LeagueSeats:
    """Seat allocator over S loadable grid slices (+ Phillip's fixed pool).

    place(env, member) seats an env on a pool holding `member` with a free
    row, loading the member into an EMPTY slice (LRU-reclaimed) when none
    holds it; None when impossible. release(env) frees the seat. Slices
    keep their weights while empty (a cache), so a member that drains and
    comes back costs no reload."""

    PHILLIP = "phillip"

    def __init__(
        self, slices: int, cells: int, loader: tp.Callable[[int, str], None],
        phillip_capacity: int = 0,
        mover: tp.Callable[[Seat, Seat], None] | None = None,
    ):
        self.S, self.N = slices, cells
        self._load = loader
        self._move = mover  # cell state move for compaction (None = off)
        self.compactions = 0
        self.pools = [_Pool(s, cells, None, True) for s in range(slices)]
        if phillip_capacity > 0:
            self.pools.append(_Pool(slices, phillip_capacity, self.PHILLIP, False))
        self._seat_of: dict[int, Seat] = {}
        self._clock = 0
        self.loads = 0

    # ------------------------------------------------------------ queries

    def seat_of(self, env: int) -> Seat | None:
        return self._seat_of.get(env)

    def member_at(self, pool: int) -> str | None:
        return self.pools[pool].member

    def room(self, member: str) -> int:
        """Free rows on pools already holding `member` (+ one empty
        slice's worth if one could be loaded)."""
        n = sum(
            p.capacity - len(p.occupants) for p in self.pools if p.member == member
        )
        if member != self.PHILLIP and self._reclaimable():
            n += self.N
        return n

    def members_with_room(
        self, candidates: tp.Iterable[str], resident_only: bool = False,
    ) -> list[str]:
        """Candidates that can be seated now; resident_only = without
        loading anything (a free row on a pool already holding them)."""
        if resident_only:
            free = {
                p.member for p in self.pools
                if p.member is not None and len(p.occupants) < p.capacity
            }
            return [m for m in candidates if m in free]
        return [m for m in candidates if self.room(m) > 0]

    def env_of_rows(self, pool: int) -> list[int | None]:
        """Row -> env (None = idle) for one pool."""
        p = self.pools[pool]
        out: list[int | None] = [None] * p.capacity
        for env, row in p.occupants.items():
            out[row] = env
        return out

    def _reclaimable(self) -> list[_Pool]:
        """Empty slices, least recently used first (unloaded ones first)."""
        empties = [p for p in self.pools if p.loadable and not p.occupants]
        return sorted(empties, key=lambda p: (p.member is not None, p.last_used))

    # ------------------------------------------------------------ actions

    def prefetch(self, member: str) -> bool:
        """Load `member` into an empty slice if it is not resident (draws
        happen a game ahead: make the weights resident before the seat is
        needed). False if nothing could be done."""
        if member == self.PHILLIP or any(p.member == member for p in self.pools):
            return False
        return self._load_into_empty(member) is not None

    def _load_into_empty(self, member: str) -> _Pool | None:
        empties = self._reclaimable()
        if not empties and self._move is not None:
            self._compact()
            empties = self._reclaimable()
        if not empties:
            return None
        p = empties[0]
        self._load(p.index, member)
        p.member = member
        self.loads += 1
        return p

    def _compact(self) -> bool:
        """Free one slice without touching any brain: a member holding
        several slices packs the occupants of its emptiest slice into free
        rows of its other slices (same weights — the moves are exact)."""
        by_member: dict[str, list[_Pool]] = {}
        for p in self.pools:
            if p.loadable and p.member is not None:
                by_member.setdefault(p.member, []).append(p)
        best = None
        for member, pools in by_member.items():
            if len(pools) < 2:
                continue
            donor = min(pools, key=lambda p: len(p.occupants))
            free = sum(p.capacity - len(p.occupants) for p in pools if p is not donor)
            if len(donor.occupants) <= free and (
                best is None or len(donor.occupants) < len(best[0].occupants)
            ):
                best = (donor, [p for p in pools if p is not donor])
        if best is None:
            return False
        donor, others = best
        for env, row in list(donor.occupants.items()):
            dst = next(p for p in others if len(p.occupants) < p.capacity)
            dst_row = dst.free_rows()[0]
            self._move((donor.index, row), (dst.index, dst_row))
            del donor.occupants[env]
            dst.occupants[env] = dst_row
            self._seat_of[env] = (dst.index, dst_row)
        self.compactions += 1
        return True

    def place(self, env: int, member: str) -> Seat | None:
        assert env not in self._seat_of, f"env {env} already seated"
        self._clock += 1
        holders = [p for p in self.pools if p.member == member and len(p.occupants) < p.capacity]
        pool = holders[0] if holders else None
        if pool is None and member != self.PHILLIP:
            pool = self._load_into_empty(member)
        if pool is None:
            return None
        row = pool.free_rows()[0]
        pool.occupants[env] = row
        pool.last_used = self._clock
        self._seat_of[env] = (pool.index, row)
        return pool.index, row

    def release(self, env: int) -> None:
        seat = self._seat_of.pop(env, None)
        if seat is not None:
            self.pools[seat[0]].occupants.pop(env)


class League:
    """Per-env opponent protocol over the seats. The worker calls:
      boot(envs)                 first members (+ the draw for game 2)
      on_boundary(env, payload)  a game ended on env -> credit, adopt, draw
      next_command(env)          the "opp_next" dict to send every frame
    and reads member_now / seat_of for attribution and routing."""

    CPU = "cpu"

    def __init__(
        self,
        pool: SnapshotPool,
        seats: LeagueSeats,
        locks: dict[str, str],  # import key -> locked char
        rng: random.Random,
        on_result: tp.Callable[[str, bool], None] | None = None,
        cpu_enabled: bool = False,
        warm: tp.Callable[[str], None] | None = None,
    ):
        self.pool = pool
        self.seats = seats
        self.locks = locks
        self.rng = rng
        self.on_result = on_result
        self.cpu_enabled = cpu_enabled
        self.warm = warm  # e.g. MemberWeights.warm: disk read off the frame loop
        self.member_now: dict[int, str] = {}
        self.member_next: dict[int, str] = {}
        # reset flags to raise on the league grid this frame: ENVS whose
        # seat starts a new game. Envs, not coordinates: a compaction later
        # in the same frame can relocate a cell, which would leave a stale
        # (slice, row) pointing at someone else's seat.
        self.fresh_envs: list[int] = []
        self.draws = 0
        self.fallbacks = 0
        # protocol mismatches the env reported vs what was asked, by kind
        self.warn = {"cpu_not_adopted": 0, "still_cpu": 0, "lock_mismatch": 0}

    # ------------------------------------------------------------ draws

    def lock_of(self, member: str) -> str | None:
        return self.locks.get(member) if _is_import_key(member) else None

    def _draw(self, allowed: tp.Collection[str] | None = None) -> str | None:
        m = self.pool.draw_member(self.rng, allowed)
        if m == self.CPU and not self.cpu_enabled:
            return None
        return m

    def _draw_next(self, env: int) -> None:
        """Draw env's opponent for its NEXT game and make the weights
        resident early when possible."""
        m = self._draw()
        if m is None:
            m = self.member_now[env]
        self.member_next[env] = m
        self.draws += 1
        if m not in (self.CPU, LeagueSeats.PHILLIP):
            if self.warm is not None:
                self.warm(m)
            self.seats.prefetch(m)

    def next_command(self, env: int) -> dict:
        m = self.member_next[env]
        return {
            "kind": "cpu" if m == self.CPU else "policy",
            "char_lock": self.lock_of(m),
        }

    # ------------------------------------------------------------ lifecycle

    def boot(self, envs: tp.Sequence[int]) -> dict[int, str | None]:
        """First members for `envs` (every member once, then draws) and
        seats for them. Returns env -> char lock for the cold boot."""
        picks = self.pool.boot_draws(self.rng, len(envs))
        assert len(picks) == len(envs), (
            f"league has nothing to draw for {len(envs)} envs — seed the "
            f"snapshot archive before booting the worker"
        )
        locks: dict[int, str | None] = {}
        for env, m in zip(envs, picks):
            seat = self.seats.place(env, m)
            if seat is None:  # no room (more distinct members than slices)
                alt = self._draw(self.seats.members_with_room(self._league_keys()))
                m = alt if alt is not None else m
                seat = self.seats.place(env, m)
                assert seat is not None, "league grid cannot seat its boot envs"
            self.member_now[env] = m
            self.fresh_envs.append(env)
            locks[env] = self.lock_of(m)
            self._draw_next(env)
        return locks

    def _league_keys(self) -> list[str]:
        return [k for k in list(self.pool.league_members) + list(self.pool.archive)
                if k != self.CPU]

    def on_boundary(self, env: int, serving: str | None, opp_char: str | None,
                    won: bool | None) -> None:
        """A game just ended on `env` (the next one starts this frame).
        `serving` / `opp_char` describe the NEW game as the env reports it;
        `won` is the ended game's decided result (None = draw/undecided)."""
        if won is not None and self.on_result is not None:
            self.on_result(self.member_now[env], won)
        wanted = self.member_next[env]
        if wanted == self.CPU:
            if serving == "cpu":
                self.seats.release(env)
                self.member_now[env] = self.CPU
            else:  # the recycle into cpu did not happen: keep the old seat
                self.warn["cpu_not_adopted"] += 1
                self._keep_seat(env)
        else:
            if serving == "cpu":  # still engine-driven: nothing to seat yet
                self.warn["still_cpu"] += 1
            else:
                self._adopt(env, wanted, opp_char)
        self._draw_next(env)

    def _keep_seat(self, env: int) -> None:
        if self.seats.seat_of(env) is not None:
            self.fresh_envs.append(env)  # new game in the same cell

    def _adopt(self, env: int, wanted: str, opp_char: str | None) -> None:
        self.seats.release(env)
        m = wanted
        lock = self.lock_of(m)
        if lock is not None and opp_char is not None and opp_char != lock:
            # the env armed a different char (e.g. a menu-misselect recycle
            # redrew it): this game is NOT the import's
            self.warn["lock_mismatch"] += 1
            m = None
        seat = self.seats.place(env, m) if m is not None else None
        if seat is None:
            # fallback: a PFSP draw over members that have room NOW (resident
            # ones first — no load in the frame loop) and can play the
            # character the env already armed
            fits = lambda k: self.lock_of(k) is None or self.lock_of(k) == opp_char
            keys = self._league_keys()
            allowed = [
                k for k in self.seats.members_with_room(keys, resident_only=True)
                if fits(k)
            ] or [k for k in self.seats.members_with_room(keys) if fits(k)]
            m = self._draw(allowed) if allowed else None
            if m is None:  # nothing fits: sit back down where we were
                m = self.member_now[env]
            seat = self.seats.place(env, m)
            assert seat is not None, f"no seat for env {env} ({m})"
            self.fallbacks += 1
        self.member_now[env] = m
        self.fresh_envs.append(env)

    @property
    def warnings(self) -> int:
        return sum(self.warn.values())

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / max(1, self.draws)
