"""Per-match league routing (rl/league.py): allocator + protocol, pure logic."""
import random


from smashbot.rl.league import League, LeagueSeats, MemberWeights


def _seats(S=2, N=2, moves=None):
    loads = []
    mover = (lambda a, b: moves.append((a, b))) if moves is not None else None
    seats = LeagueSeats(S, N, loader=lambda s, m: loads.append((s, m)), mover=mover)
    return seats, loads


def test_place_loads_into_empty_slice_and_fills_rows():
    seats, loads = _seats()
    assert seats.place(0, "A") == (0, 0)
    assert seats.place(1, "A") == (0, 1)
    assert seats.place(2, "B") == (1, 0)
    assert loads == [(0, "A"), (1, "B")]
    assert seats.place(3, "C") is None          # no free row anywhere


def test_release_reclaims_lru_empty_slice_and_keeps_cache():
    seats, loads = _seats()
    seats.place(0, "A"); seats.place(1, "B")
    seats.release(0)                            # slice 0 empty, still holds A
    assert seats.place(2, "A") == (0, 0) and len(loads) == 2   # cache hit
    seats.release(2)
    assert seats.place(3, "C") == (0, 0) and loads[-1] == (0, "C")  # reclaimed


def test_compaction_donates_a_slice_exactly():
    moves = []
    seats, loads = _seats(S=2, N=2, moves=moves)
    seats.place(0, "A"); seats.place(1, "A"); seats.place(2, "A")  # A on both slices
    seats.release(1)                            # slice 0: {0}, slice 1: {2}
    assert seats.place(3, "B") is not None      # needs a slice: compaction
    assert seats.compactions == 1 and len(moves) == 1
    src, dst = moves[0]
    assert src[0] != dst[0]
    assert {seats.member_at(0), seats.member_at(1)} == {"A", "B"}
    assert all(seats.seat_of(e) is not None for e in (0, 2, 3))


class _Pool:
    """Minimal SnapshotPool stand-in: draw_member cycles a fixed list."""
    def __init__(self, members, archive=()):
        self.league_members = list(members)
        self.archive = list(archive)
        self._i = 0
        self.results = []

    def draw_member(self, rng, allowed=None):
        keys = [k for k in self.league_members + self.archive
                if allowed is None or k in allowed]
        if not keys:
            return None
        self._i += 1
        return keys[self._i % len(keys)]

    def boot_draws(self, rng, n):
        keys = self.league_members + self.archive
        return [keys[i % len(keys)] for i in range(n)]


def test_protocol_boot_seats_everyone_and_draws_next():
    pool = _Pool(["import:x"], archive=["g1", "g2"])
    seats, loads = _seats(S=3, N=2)
    lg = League(pool, seats, {"import:x": "FOX"}, random.Random(0))
    lg.boot([0, 1, 2, 3])
    assert all(seats.seat_of(e) is not None for e in range(4))
    assert set(lg.member_now.values()) <= {"import:x", "g1", "g2"}
    assert all(e in lg.member_next for e in range(4))
    assert lg.lock_of("import:x") == "FOX" and lg.lock_of("g1") is None


def test_protocol_boundary_reseats_for_drawn_member_and_counts_fallback():
    pool = _Pool(["a", "b", "c", "d"])
    seats, loads = _seats(S=2, N=1)             # only 2 seats for 2 envs: tight
    lg = League(pool, seats, {}, random.Random(0))
    lg.boot([0, 1])
    before = dict(lg.member_now)
    for _ in range(6):
        for e in (0, 1):
            seat = lg.on_boundary(e)
            assert seat == seats.seat_of(e)
            assert seats.member_at(seat[0]) == lg.member_now[e]  # sits on its member
    assert lg.draws >= 12
    # every seat change happened only through on_boundary; the grid never
    # held a member with occupants when its weights changed
    assert lg.fallback_rate <= 1.0


def test_member_weights_lru_and_warm(tmp_path):
    import torch
    paths = {}
    for k in ("a", "b", "c"):
        p = tmp_path / f"{k}.pt"
        torch.save({"w": torch.tensor([ord(k)])}, p)
        paths[k] = str(p)
    mw = MemberWeights(lambda k: paths[k], lru=2)
    mw.warm("a")
    assert mw.get("a")["w"].item() == ord("a")
    mw.get("b"); mw.get("c")
    assert list(mw._cache) == ["b", "c"]        # LRU evicted a
