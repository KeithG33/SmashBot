"""Per-match league routing on the PFSP grid (v5 design, ported from the
Dolphin fleet — AlphaStar draws the opponent per MATCH).

- MemberWeights: member key -> state_dict (CPU) through an LRU; warm()
  loads in a background thread a game ahead, so seating never stalls the
  frame loop on a disk read.
- LeagueSeats: the seat allocator over the grid's S weight SLICES x N
  CELLS. A slice is a weight-cache entry holding one member, a cell is a
  seat. Envs sit wherever their drawn member is loaded; a slice's weights
  change only when it is EMPTY — nobody is ever swapped mid-game. When no
  slice is free, a member holding several slices donates its emptiest
  (compaction: exact cell-state moves between slices of the same weights).
- League: the per-env protocol. Each env draws its NEXT opponent a game
  ahead (prefetch), and at its own game boundary releases its seat and
  takes one for the drawn member; if that member has no room at that
  moment the env falls back to a PFSP draw over members that DO have room
  (counted: fallback_rate).
"""
from __future__ import annotations

import collections
import random
import threading
import typing as tp

import torch

from smashbot.rl.pool import SnapshotPool, _is_import_key


class MemberWeights:
    def __init__(self, path_of: tp.Callable[[str], str], lru: int = 16):
        self._path_of = path_of
        self._cache: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self.lru = lru
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Thread] = {}

    def _load(self, member: str) -> dict:
        return torch.load(self._path_of(member), map_location="cpu",
                          weights_only=True)

    def _put(self, member: str, sd: dict) -> None:
        with self._lock:
            self._cache[member] = sd
            self._cache.move_to_end(member)
            while len(self._cache) > self.lru:
                self._cache.popitem(last=False)

    def warm(self, member: str) -> None:
        with self._lock:
            if member in self._cache or member in self._inflight:
                return
            t = threading.Thread(
                target=lambda: self._put(member, self._load(member)), daemon=True)
            self._inflight[member] = t
        t.start()

    def ready(self, member: str) -> bool:
        with self._lock:
            return member in self._cache

    def get(self, member: str) -> dict:
        t = self._inflight.pop(member, None)
        if t is not None:
            t.join()
        with self._lock:
            sd = self._cache.get(member)
            if sd is not None:
                self._cache.move_to_end(member)
                return sd
        sd = self._load(member)
        self._put(member, sd)
        return sd


class _Slice:
    __slots__ = ("index", "capacity", "member", "occupants", "last_used")

    def __init__(self, index: int, capacity: int):
        self.index = index
        self.capacity = capacity
        self.member: str | None = None
        self.occupants: dict[int, int] = {}  # env -> cell row
        self.last_used = 0

    def free_rows(self) -> list[int]:
        used = set(self.occupants.values())
        return [r for r in range(self.capacity) if r not in used]


Seat = tuple[int, int]  # (slice, row)


class LeagueSeats:
    def __init__(self, slices: int, cells: int,
                 loader: tp.Callable[[int, str], None],
                 mover: tp.Callable[[Seat, Seat], None] | None = None,
                 drain_slices: int = 2):
        self.S, self.N = slices, cells
        self.drain_slices = drain_slices
        self._load = loader
        self._move = mover
        self.slices = [_Slice(s, cells) for s in range(slices)]
        self._seat_of: dict[int, Seat] = {}
        self._drain: set = set()
        self._clock = 0
        self.loads = 0
        self.compactions = 0

    def seat_of(self, env: int) -> Seat | None:
        return self._seat_of.get(env)

    def member_at(self, s: int) -> str | None:
        return self.slices[s].member

    def occupancy(self) -> int:
        return len(self._seat_of)

    def room(self, member: str) -> int:
        n = sum(p.capacity - len(p.occupants)
                for p in self.slices if p.member == member)
        if self._reclaimable():
            n += self.N
        return n

    def draining(self) -> set:
        """With no empty slice, the `drain_slices` least-occupied slices are
        DRAINING: closed to new seats, so they empty within about one game
        and reload for fresh draws. Sticky per slice until it empties.
        Without this, wide slices never empty and the resident set
        ossifies at the boot draw (v5's 36 thin slices drained by themselves)."""
        self._drain = {p for p in self._drain if p.occupants}
        if any(not p.occupants for p in self.slices):
            return self._drain
        while len(self._drain) < self.drain_slices:
            cands = [p for p in self.slices if p not in self._drain]
            self._drain.add(min(cands, key=lambda p: (len(p.occupants), p.last_used)))
        return self._drain

    def members_with_room(self, candidates: tp.Iterable[str],
                          resident_only: bool = False) -> list[str]:
        if resident_only:
            drain = self.draining()
            free = {p.member for p in self.slices
                    if p.member is not None and len(p.occupants) < p.capacity
                    and p not in drain}
            out = [m for m in candidates if m in free]
            if out or not drain:
                return out
            dm = {p.member for p in drain if len(p.occupants) < p.capacity}
            return [m for m in candidates if m in dm]
        return [m for m in candidates if self.room(m) > 0]

    def _reclaimable(self) -> list[_Slice]:
        empties = [p for p in self.slices if not p.occupants]
        return sorted(empties, key=lambda p: (p.member is not None, p.last_used))

    def resident(self, member: str) -> bool:
        return any(p.member == member for p in self.slices)

    def prefetch(self, member: str) -> bool:
        """Load `member` into an empty slice ahead of its seat (draws happen
        a game ahead). Never compacts; False if no empty slice."""
        if self.resident(member) or not self._reclaimable():
            return False
        return self._load_into_empty(member) is not None

    def _load_into_empty(self, member: str) -> _Slice | None:
        empties = self._reclaimable()
        if not empties and self._move is not None and self._compact():
            empties = self._reclaimable()
        if not empties:
            return None
        p = empties[0]
        self._load(p.index, member)
        p.member = member
        self.loads += 1
        return p

    def _compact(self) -> bool:
        by_member: dict[str, list[_Slice]] = {}
        for p in self.slices:
            if p.member is not None:
                by_member.setdefault(p.member, []).append(p)
        best = None
        for member, ps in by_member.items():
            if len(ps) < 2:
                continue
            donor = min(ps, key=lambda p: len(p.occupants))
            free = sum(p.capacity - len(p.occupants) for p in ps if p is not donor)
            if len(donor.occupants) <= free and (
                    best is None or len(donor.occupants) < len(best[0].occupants)):
                best = (donor, [p for p in ps if p is not donor])
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

    def place(self, env: int, member: str, allow_drain: bool = True) -> Seat | None:
        """Seat env on a slice holding `member`: the fullest non-draining
        holder (pack, so sparse slices drain and reload), else load into an
        empty slice. The draining slice only if allow_drain (boot / last
        resort) — refusing it lets callers with a fallback keep it draining."""
        assert env not in self._seat_of, f"env {env} already seated"
        self._clock += 1
        drain = self.draining()
        holders = [p for p in self.slices
                   if p.member == member and len(p.occupants) < p.capacity]
        pref = sorted((p for p in holders if p not in drain),
                      key=lambda p: -len(p.occupants))
        p = pref[0] if pref else self._load_into_empty(member)
        if p is None and holders and allow_drain:
            p = holders[0]
        if p is None:
            return None
        row = p.free_rows()[0]
        p.occupants[env] = row
        p.last_used = self._clock
        self._seat_of[env] = (p.index, row)
        return p.index, row

    def release(self, env: int) -> None:
        seat = self._seat_of.pop(env, None)
        if seat is not None:
            self.slices[seat[0]].occupants.pop(env)


class League:
    """boot(envs) seats the first members; on_boundary(env) at each env's
    game end re-seats it for the member drawn a game ahead. member_now[env]
    is the payoff label of the game being played; member_next[env] the one
    the next game will be configured for (char locks)."""

    def __init__(self, pool: SnapshotPool, seats: LeagueSeats,
                 locks: dict[str, str], rng: random.Random,
                 warm: tp.Callable[[str], None] | None = None,
                 ready: tp.Callable[[str], bool] | None = None):
        self.pool = pool
        self.seats = seats
        self.locks = locks
        self.rng = rng
        self.warm = warm            # start a background disk read
        self.ready = ready or (lambda m: True)   # weights cached (no stall)?
        self._prefetch: set[str] = set()   # drawn-ahead members not yet resident
        self.prefetches = 0
        self.member_now: dict[int, str] = {}
        self.member_next: dict[int, str] = {}
        self.draws = 0
        self.fallbacks = 0

    def lock_of(self, member: str) -> str | None:
        return self.locks.get(member) if _is_import_key(member) else None

    def _keys(self) -> list[str]:
        return [k for k in list(self.pool.league_members) + list(self.pool.archive)
                if k != "cpu"]

    def _draw(self, allowed=None) -> str | None:
        m = self.pool.draw_member(self.rng, allowed)
        return None if m == "cpu" else m

    def _draw_next(self, env: int) -> None:
        m = self._draw() or self.member_now[env]
        self.member_next[env] = m
        self.draws += 1
        if not self.seats.resident(m):
            if self.warm is not None:   # disk read off the frame loop
                self.warm(m)
            self._prefetch.add(m)

    def tick(self) -> None:
        """Once per frame: drawn-ahead members whose weights are cached go
        into empty slices now, so their seats need no fallback later."""
        if not self._prefetch:
            return
        for m in list(self._prefetch):
            if self.seats.resident(m):
                self._prefetch.discard(m)
            elif self.ready(m):
                if not self.seats.prefetch(m):
                    break               # no empty slice: try next frame
                self.prefetches += 1
                self._prefetch.discard(m)

    def boot(self, envs: tp.Sequence[int]) -> None:
        picks = self.pool.boot_draws(self.rng, len(envs))
        assert len(picks) == len(envs), "league has nothing to draw — seed the archive"
        for env, m in zip(envs, picks):
            if self.seats.place(env, m) is None:
                alt = self._draw(self.seats.members_with_room(self._keys()))
                m = alt if alt is not None else m
                assert self.seats.place(env, m) is not None, "cannot seat boot envs"
            self.member_now[env] = m
            self._draw_next(env)

    def on_boundary(self, env: int) -> Seat:
        """env's game just ended: release its seat, seat member_next (or a
        fallback with room), draw the one after. Returns the new seat."""
        self.seats.release(env)
        m = self.member_next[env]
        seat = self.seats.place(env, m, allow_drain=False)
        if seat is None:
            # fallback: PFSP draw over members with room now (resident
            # first, never the draining slice); the draining slice itself
            # only when nothing else can seat this env
            keys = self._keys()
            allowed = (self.seats.members_with_room(keys, resident_only=True)
                       or self.seats.members_with_room(keys))
            m = self._draw(allowed) if allowed else None
            seat = self.seats.place(env, m, allow_drain=False) if m else None
            if seat is None:
                m = m or self.member_now[env]
                seat = self.seats.place(env, m, allow_drain=True)
            if seat is None:
                m = self.member_now[env]
                seat = self.seats.place(env, m, allow_drain=True)
            assert seat is not None, f"no seat for env {env} ({m})"
            self.fallbacks += 1
        self.member_now[env] = m
        self._draw_next(env)
        return seat

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / max(1, self.draws)
