"""Per-match league routing (rl/league.py + the worker's grid routing).

Dolphin-free: the allocator and protocol are exercised directly; the
worker tests drive DolphinRolloutWorker.collect() over fake env pipes that
honor the env-side contract (opp_next is applied one game ahead: the lock
arms the next game's character, a kind change recycles into it).
"""

import os
import random

import numpy as np
import pytest
import torch
import tree

from smashbot import encode
from smashbot.rl.agent import BatchedPolicyAgent, LeagueAgent
from smashbot.rl.league import League, LeagueSeats, MemberWeights
from smashbot.rl.pool import SnapshotPool
from smashbot.tests.test_opponent_league import (
    _FakeEnvs, _Stub, _make_worker,
)
from smashbot.tests.test_ppo import _tiny_policy
from smashbot.tests.test_rollouts import _rand_raw_game


# ------------------------------------------------------------ allocator


def _seats(S=3, N=2, phillip=0):
    loads = []
    seats = LeagueSeats(S, N, loader=lambda s, m: loads.append((s, m)),
                        phillip_capacity=phillip)
    return seats, loads


def test_seats_place_loads_into_empty_slice_and_fills_rows():
    seats, loads = _seats()
    assert seats.place(0, "a") == (0, 0) and loads == [(0, "a")]
    assert seats.place(1, "a") == (0, 1) and loads == [(0, "a")]  # same slice
    assert seats.place(2, "a") == (1, 0) and loads[-1] == (1, "a")  # overflow: 2nd slice
    assert seats.place(3, "b") == (2, 0) and loads[-1] == (2, "b")
    assert seats.place(4, "c") is None  # full: no empty slice for c
    assert seats.env_of_rows(0) == [0, 1] and seats.env_of_rows(1) == [2, None]
    assert seats.room("a") == 1 and seats.room("c") == 0


def test_seats_release_reclaims_lru_empty_slice_and_keeps_cache():
    seats, loads = _seats(S=2, N=1)
    seats.place(0, "a"); seats.place(1, "b")
    seats.release(0)  # slice 0 empty but still holds "a" (cache)
    assert seats.member_at(0) == "a"
    assert seats.place(2, "a") == (0, 0) and len(loads) == 2  # no reload
    seats.release(2); seats.release(1)
    # both empty: LRU reclaim takes the least recently used (slice 1: "b"
    # was used before "a" was re-seated)
    assert seats.place(3, "c") == (1, 0) and loads[-1] == (1, "c")


def test_seats_prefetch_and_members_with_room():
    seats, loads = _seats(S=2, N=2)
    assert seats.prefetch("a") and loads == [(0, "a")]
    assert not seats.prefetch("a")  # already resident
    seats.place(0, "a"); seats.place(1, "a")
    assert seats.members_with_room(["a", "b"]) == ["a", "b"]  # a: via slice 1
    seats.place(2, "b"); seats.place(3, "b")
    assert seats.members_with_room(["a", "b", "c"]) == []


def test_seats_phillip_pool_is_fixed_capacity_not_loadable():
    seats, loads = _seats(S=1, N=1, phillip=2)
    assert seats.place(0, "phillip") == (1, 0)
    assert seats.place(1, "phillip") == (1, 1)
    assert seats.place(2, "phillip") is None  # capacity 2, never "loaded"
    assert loads == []
    assert not seats.prefetch("phillip")


# ------------------------------------------------------------ protocol


class _ScriptedPool:
    """SnapshotPool stand-in: draws come from a queue (then repeat the last)."""

    def __init__(self, members, archive=(), draws=()):
        self.league_members = list(members)
        self.archive = list(archive)
        self.queue = list(draws)
        self.results = []

    def draw_member(self, rng, allowed=None):
        m = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if allowed is not None and m not in allowed:
            cands = [k for k in self.league_members + self.archive if k in allowed]
            return cands[0] if cands else None
        return m

    def boot_draws(self, rng, n):
        members = [m for m in self.league_members if m != "cpu"] + self.archive
        return (members * n)[:n]

    def record_result(self, key, won):
        self.results.append((key, won))


def _league(members, archive=(), draws=(), S=2, N=2, phillip=0, cpu=False, locks=None):
    pool = _ScriptedPool(members, archive, draws)
    seats, _ = _seats(S, N, phillip)
    lg = League(pool, seats, locks=locks or {}, rng=random.Random(0),
                on_result=pool.record_result, cpu_enabled=cpu)
    return lg, pool, seats


def test_protocol_boot_covers_members_then_draws_next():
    lg, pool, seats = _league(["teacher"], archive=["g1"], draws=["g1"])
    locks = lg.boot([0, 1])
    assert lg.member_now == {0: "teacher", 1: "g1"}
    assert locks == {0: None, 1: None}
    assert {seats.seat_of(0), seats.seat_of(1)} == {(0, 0), (1, 0)}
    assert lg.member_next == {0: "g1", 1: "g1"}  # drawn one game ahead
    assert lg.next_command(0) == {"kind": "policy", "char_lock": None}
    assert sorted(lg.fresh_seats) == [(0, 0), (1, 0)]


def test_protocol_boundary_credits_old_member_and_adopts_drawn():
    lg, pool, seats = _league(["teacher"], archive=["g1"], draws=["g1", "teacher"])
    lg.boot([0])
    lg.fresh_seats.clear()
    assert lg.member_now[0] == "teacher" and lg.member_next[0] == "g1"
    lg.on_boundary(0, serving="policy", opp_char="MARTH", won=True)
    assert pool.results == [("teacher", True)]  # credited to who PLAYED
    assert lg.member_now[0] == "g1"
    assert seats.member_at(seats.seat_of(0)[0]) == "g1"
    assert lg.fresh_seats == [seats.seat_of(0)]  # new game in that cell
    assert lg.member_next[0] == "teacher"  # the next draw already made
    # an undecided game is adopted but not credited
    lg.on_boundary(0, serving="policy", opp_char="FOX", won=None)
    assert pool.results == [("teacher", True)]
    assert lg.member_now[0] == "teacher"


def test_protocol_import_lock_rides_the_command_and_mismatch_falls_back():
    lg, pool, seats = _league(
        ["teacher", "import:v3"], draws=["import:v3", "teacher"],
        locks={"import:v3": "FOX"},
    )
    lg.boot([0])
    assert lg.next_command(0) == {"kind": "policy", "char_lock": "FOX"}
    # env armed FOX as told: the import is adopted
    lg.on_boundary(0, "policy", "FOX", won=False)
    assert lg.member_now[0] == "import:v3" and lg.fallbacks == 0
    # next draw is the teacher (unlocked) -> command clears the lock
    assert lg.next_command(0) == {"kind": "policy", "char_lock": None}
    # a second env whose armed char contradicts the lock it was sent:
    # NOT credited as the import — a fallback seat with a compatible member
    lg2, pool2, seats2 = _league(
        ["teacher", "import:v3"], draws=["import:v3", "teacher"],
        locks={"import:v3": "FOX"},
    )
    lg2.boot([0])
    lg2.on_boundary(0, "policy", "MARTH", won=True)
    assert lg2.member_now[0] == "teacher" and lg2.fallbacks == 1 and lg2.warnings == 1


def test_protocol_cpu_draw_needs_the_recycle_to_have_happened():
    lg, pool, seats = _league(["teacher", "cpu"], draws=["cpu", "teacher"], cpu=True)
    lg.boot([0])
    assert lg.next_command(0) == {"kind": "cpu", "char_lock": None}
    # recycle did not happen yet: keep the old seat, flag it
    lg.on_boundary(0, serving="policy", opp_char="FOX", won=True)
    assert lg.member_now[0] == "teacher" and lg.warnings == 1
    assert seats.seat_of(0) is not None
    # now the env reports cpu: seat released, member cpu, no cell
    lg.member_next[0] = "cpu"
    lg.on_boundary(0, serving="cpu", opp_char="FOX", won=False)
    assert lg.member_now[0] == "cpu" and seats.seat_of(0) is None
    # back to a policy member through another recycle
    lg.member_next[0] = "teacher"
    lg.on_boundary(0, serving="policy", opp_char="FOX", won=None)
    assert lg.member_now[0] == "teacher" and seats.seat_of(0) is not None
    # cpu draws are refused when league_cpu is off
    lg_off, _, _ = _league(["teacher", "cpu"], draws=["cpu"], cpu=False)
    lg_off.boot([0])
    assert lg_off.member_next[0] == "teacher"


def test_protocol_fallback_when_no_room_is_counted():
    # one slice, one cell: env 0 holds "g1"; env 1 draws "g2" but the only
    # slice is occupied -> falls back to a member with room (g1's own cell
    # is released first, so it sits back on g1)
    lg, pool, seats = _league([], archive=["g1", "g2"], draws=["g2"], S=1, N=1)
    lg.boot([0])
    assert lg.member_now[0] == "g1"
    lg.on_boundary(0, "policy", "FOX", won=True)
    # the single slice was empty after release -> g2 loaded, no fallback
    assert lg.member_now[0] == "g2" and lg.fallbacks == 0
    # 2x2 grid, boot g1,g2,g1,g2: slice 0 = {env0, env2} on g1, slice 1 =
    # {env1, env3} on g2. env 1 draws g1: g1's slice is full and env 1's
    # old slice still holds env 3 -> no empty slice -> fallback onto a
    # member with room (g2 itself), counted
    lg, pool, seats = _league([], archive=["g1", "g2"], draws=["g1"], S=2, N=2)
    lg.boot([0, 1, 2, 3])
    assert [lg.member_now[i] for i in range(4)] == ["g1", "g2", "g1", "g2"]
    lg.member_next[1] = "g1"
    lg.on_boundary(1, "policy", "FOX", won=None)
    assert lg.member_now[1] == "g2" and lg.fallbacks == 1
    assert lg.fallback_rate == pytest.approx(1 / lg.draws)


# ------------------------------------------------------------ grid agent


def test_grid_load_slice_changes_only_that_slice():
    grid = LeagueAgent(_tiny_policy(seed=0), 3, 2, name_code=1, device="cpu")
    donor = _tiny_policy(seed=7).state_dict()
    before = {k: v.clone() for k, v in grid._stacked_params.items()}
    grid.load_slice(1, donor)
    for k, v in grid._stacked_params.items():
        assert torch.equal(v[0], before[k][0]) and torch.equal(v[2], before[k][2])
        assert torch.equal(v[1], donor[k])


def test_grid_step_rows_and_record_cover_all_cells():
    from smashbot import embed as embed_lib

    S, N = 2, 3
    grid = LeagueAgent(_tiny_policy(seed=0), S, N, name_code=1, device="cpu")
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)
    raw = _rand_raw_game(game, (S, N), rng)
    views = tree.map_structure(
        lambda x: torch.from_numpy(np.ascontiguousarray(
            x.astype(np.int64) if x.dtype.kind in "iu" else x)),
        game.from_state(raw),
    )
    resets = torch.zeros(S, N, dtype=torch.bool)
    grid.reset_cell(1, 2)
    rows, rec = grid.step(views, resets)
    assert rows.shape == (S * N, 13)
    assert rec.name.shape == (S * N,)
    for leaf in tree.flatten(rec.state):
        assert leaf.shape[0] == S * N


# ------------------------------------------------------------ worker routing


class _ProtocolEnvs(_FakeEnvs):
    """Fake envs that honor opp_next one game ahead: at a scripted boundary
    the NEW game serves the kind and character the worker last sent."""

    def __init__(self, worker, seed=0, opp_chars=None):
        super().__init__(worker, seed, opp_chars)
        self.next = {}      # dolphin -> last opp_next
        self.armed = {}     # dolphin -> (serving, char) for the next game
        self.boundaries = {}  # dolphin -> (p1, p2) to deliver next frame

    def install(self, monkeypatch):
        super().install(monkeypatch)
        w = self.worker
        for i in w.league_idx:
            self.armed[i] = ("policy", self.opp_chars.get(i, "FOX"))
        orig_sends = [c.send for c in w._conns]

        def tap(i, send):
            def _send(cmd):
                nx = cmd.get("opp_next")
                if nx is not None:
                    self.next[i] = nx
                send(cmd)
            return _send

        for i, c in enumerate(w._conns):
            c.send = tap(i, c.send)

    def end_game(self, i, stocks=(4, 0)):
        """Script a boundary on dolphin i for the next frame; the game that
        starts then adopts the last opp_next (kind + lock, else a redraw)."""
        self.final_stocks[i] = stocks
        nx = self.next.get(i, {"kind": "policy", "char_lock": None})
        kind = "cpu" if nx["kind"] == "cpu" else "policy"
        char = nx["char_lock"] or ["MARTH", "FALCO", "PEACH"][self.t % 3]
        self.armed[i] = (kind, char)
        self.serving[i] = kind
        self.opp_chars[i] = char


def _league_worker(monkeypatch, tmp_path, **kw):
    worker, envs = _make_worker(
        monkeypatch, pool_dir=str(tmp_path), **kw,
    )
    # swap the plain fake for the protocol-aware one
    proto = _ProtocolEnvs(worker, seed=1, opp_chars=dict(envs.opp_chars))
    proto.install(monkeypatch)
    return worker, proto


def test_worker_routes_every_league_env_to_a_seat_and_sends_opp_next(monkeypatch, tmp_path):
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=6, teacher_envs=0, league_slices=2,
        league_teacher=True, members=[0, 100],
    )
    assert worker.league_idx == list(range(6))
    assert worker._grid_cells == 2 * 4  # (6 + 2 slack) / 2 slices
    lg = worker.league
    assert set(lg.member_now) == set(range(6))
    for i in range(6):
        assert lg.seats.seat_of(i) is not None
    worker.collect(1)
    for i, conn in enumerate(worker._conns):
        port = worker.specs[i].student_port
        for cmd in conn.sent:
            assert set(cmd) == {port, 3 - port, "opp_next"}
            assert set(cmd["opp_next"]) == {"kind", "char_lock"}
            encode.controller_from_flat(cmd[3 - port])


def test_worker_boundary_adopts_drawn_member_and_credits_payoff(monkeypatch, tmp_path):
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=4, teacher_envs=0, league_slices=2,
        league_teacher=True, members=[0],
    )
    lg = worker.league
    pool = lg.pool
    worker.collect(1)
    wanted = lg.member_next[1]
    was = lg.member_now[1]
    port = worker.specs[1].student_port
    envs.end_game(1, stocks=(4, 1) if port == 1 else (1, 4))  # student wins
    worker.collect(1)
    assert lg.member_now[1] == wanted or lg.fallbacks > 0
    assert pool.payoff[was]["games"] == 1 and pool.payoff[was]["wins"] == 1
    # the seat the env sits on holds the member it fights
    s, r = lg.seats.seat_of(1)
    assert lg.seats.member_at(s) == lg.member_now[1]


def test_worker_import_lock_pipeline_pins_the_char(monkeypatch, tmp_path):
    imp = str(tmp_path / "imp.pt")
    torch.save(_tiny_policy(seed=9).state_dict(), imp)
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=4, teacher_envs=0, league_slices=2,
        league_imports=[f"v3={imp}@FOX"], members=[0],
        pfsp_explore=1.0,  # uniform draws: the import gets drawn quickly
    )
    lg = worker.league
    worker.collect(1)
    # drive boundaries until some env's NEXT is the import, then verify the
    # command carried the lock and the adopted game is played as FOX
    seen = False
    for t in range(40):
        for i in range(4):
            if lg.member_next[i] == "import:v3":
                assert envs.next[i] == {"kind": "policy", "char_lock": "FOX"}
                envs.end_game(i)
                worker.collect(1)
                assert envs.opp_chars[i] == "FOX"
                assert lg.member_now[i] == "import:v3" or lg.fallbacks > 0
                seen = True
                break
        if seen:
            break
        envs.end_game(t % 4)
        worker.collect(1)
    assert seen


def test_worker_cpu_member_has_no_seat_and_engine_drives_the_port(monkeypatch, tmp_path):
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=4, teacher_envs=0, league_slices=2,
        league_cpu=True, members=[0], pfsp_explore=1.0,
    )
    lg = worker.league
    worker.collect(1)
    seen = False
    for t in range(60):
        for i in range(4):
            if lg.member_next[i] == "cpu":
                assert envs.next[i]["kind"] == "cpu"
                envs.end_game(i)
                worker.collect(1)
                assert lg.member_now[i] == "cpu"
                assert lg.seats.seat_of(i) is None
                port = worker.specs[i].student_port
                cmd = worker._conns[i].sent[-1]
                assert 3 - port not in cmd  # engine AI drives the seat
                seen = True
                break
        if seen:
            break
        envs.end_game(t % 4)
        worker.collect(1)
    assert seen


def test_worker_phillip_is_routed_to_his_agent_and_harvest_reencodes(monkeypatch, tmp_path):
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=4, teacher_envs=0, league_slices=1,
        league_phillip=True, members=[0], harvest=True, pfsp_explore=1.0,
        phillip_capacity=2, char_whitelist=["FOX", "MARTH", "FALCO", "PEACH"],
    )
    lg = worker.league
    ph = worker._runtime.phillip
    assert set(worker._harvest_groups) == {"ours", "phillip"}
    out = []
    for t in range(30):
        out += worker.collect(1)
        envs.end_game(t % 4)
    assert any(m == "phillip" for m in lg.member_now.values()) or lg.draws > 0
    imit = [tr for tr in out if tr.kind == "imitation"]
    assert imit  # whitelisted opponent seats were harvested
    assert ph.S == 1 and ph.N == 2  # Phillip = a 1-slice grid
    n_student_leaves = len(tree.flatten(worker.student._neutral_encoded))
    for tr in imit:
        # student schema everywhere (phillip rows re-encoded)
        assert len(tree.flatten(tr.actions.controller_state)) == n_student_leaves
        assert int(tr.name[0, 0]) == worker._student_name_code


def test_worker_self_play_composes_with_league(monkeypatch, tmp_path):
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=8, teacher_envs=0, self_envs=2,
        league_slices=2, league_teacher=True, members=[0],
    )
    assert worker.num_dolphins == 6 and worker.num_rows == 8
    assert worker.league_idx == [2, 3, 4, 5]
    trajs = worker.collect(2)
    assert [t.kind for t in trajs] == ["ppo", "ppo"]
    assert all(t.rewards.shape[0] == 8 for t in trajs)
    # self-play dolphins: both ports driven by the student, no opp_next
    for i in (0, 1):
        port = worker.specs[i].student_port
        assert all(set(cmd) == {port, 3 - port} for cmd in worker._conns[i].sent)


def test_worker_idle_cells_never_harvest(monkeypatch, tmp_path):
    # 2 league envs on a 1x3 grid: one idle cell every frame
    worker, envs = _league_worker(
        monkeypatch, tmp_path, num_envs=2, teacher_envs=0, league_slices=1,
        league_teacher=True, members=[0], harvest=True,
        char_whitelist=["FOX", "MARTH", "FALCO", "PEACH"],
    )
    g = worker._harvest_groups["ours"]
    assert g.rows == [0, 1, 2]
    out = []
    for _ in range(12):
        out += worker.collect(1)
    imit = [t for t in out if t.kind == "imitation"]
    assert imit
    for t in imit:
        assert t.rewards.shape[0] <= 2  # never the idle third cell


# ------------------------------------------------------------ weights


def test_member_weights_lru_and_fixed(tmp_path):
    paths = []
    for j in range(3):
        pth = str(tmp_path / f"snapshot-{j:07d}.pt")
        torch.save(_tiny_policy(seed=j).state_dict(), pth)
        paths.append(pth)
    fixed = {"teacher": _tiny_policy(seed=42).state_dict()}
    w = MemberWeights(fixed, lru=2)
    assert w.get("teacher") is fixed["teacher"]
    a = w.get(paths[0]); b = w.get(paths[1]); c = w.get(paths[2])
    assert list(w._cache) == [paths[1], paths[2]]  # LRU evicted the first
    assert w.get(paths[0]) is not a  # reloaded
    w.set("teacher", fixed["teacher"])


@pytest.mark.skipif(
    not os.environ.get("SMASHBOT_GPU_TESTS") or not torch.cuda.is_available(),
    reason="production capture path needs CUDA (SMASHBOT_GPU_TESTS=1)",
)
def test_grid_capture_matches_eager_on_gpu():
    """The captured CUDA-graph forward (production) == the same vmap run
    eagerly, frame after frame, including a mid-stream load_slice on
    slice 0 and a cell reset; controller rows bitwise, logits to fp eps."""
    from smashbot import embed as embed_lib

    torch.manual_seed(0)
    S, N = 3, 4
    mk = lambda: _tiny_policy(seed=0)
    cap = LeagueAgent(mk(), S, N, name_code=1, device="cuda", temperature=1e-6)
    eag = LeagueAgent(mk(), S, N, name_code=1, device="cuda", temperature=1e-6,
                      capture=False)
    for s in range(S):
        sd = _tiny_policy(seed=11 + s).state_dict()
        cap.load_slice(s, sd); eag.load_slice(s, sd)
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)
    for frame in range(6):
        raw = _rand_raw_game(game, (S, N), rng)
        views = tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x)).cuda(),
            game.from_state(raw),
        )
        resets = torch.zeros(S, N, dtype=torch.bool, device="cuda")
        if frame == 3:
            resets[0, 0] = True
            cap.reset_cell(0, 0); eag.reset_cell(0, 0)
        rows_c, rec_c = cap.step(views, resets)
        rows_e, rec_e = eag.step(views, resets)
        assert np.array_equal(rows_c, rows_e), f"frame {frame} rows"
        for x, y in zip(tree.flatten(rec_c._replace(logits=0)), tree.flatten(rec_e._replace(logits=0))):
            if isinstance(x, torch.Tensor):
                assert torch.equal(x, y)
        for x, y in zip(tree.flatten(rec_c.logits), tree.flatten(rec_e.logits)):
            assert torch.allclose(x, y, atol=1e-4, rtol=1e-4)
        if frame == 2:  # reload slice 0 mid-stream (the template's slice)
            donor = _tiny_policy(seed=99).state_dict()
            cap.load_slice(0, donor); eag.load_slice(0, donor)


def test_member_weights_warm_loads_in_background(tmp_path):
    pth = str(tmp_path / "snapshot-0000001.pt")
    torch.save(_tiny_policy(seed=1).state_dict(), pth)
    w = MemberWeights({})
    w.warm(pth)
    w.warm(pth)  # idempotent while in flight / cached
    sd = w.get(pth)  # joins the load; no second read
    assert pth in w._cache and w._inflight == {}
    assert w.get(pth) is sd


def test_fallback_prefers_resident_members():
    # 2x2 grid, boot g1,g2,g1,g2. env 1 draws g3 (not resident): its old
    # slice still holds env 3, g1's slice is full -> no empty slice -> the
    # fallback must sit on a RESIDENT member with a free row (g2) and load
    # nothing inside the frame loop
    lg, pool, seats = _league([], archive=["g1", "g2", "g3"], draws=["g3"], S=2, N=2)
    lg.boot([0, 1, 2, 3])
    assert [lg.member_now[i] for i in range(4)] == ["g1", "g2", "g1", "g2"]
    lg.member_next[1] = "g3"
    loads_before = seats.loads
    lg.on_boundary(1, "policy", "FOX", won=None)
    assert lg.member_now[1] == "g2" and lg.fallbacks == 1
    assert seats.loads == loads_before  # nothing loaded


def test_seats_compaction_donates_a_slice_exactly():
    moves = []
    loads = []
    seats = LeagueSeats(2, 3, loader=lambda s, m: loads.append((s, m)),
                        mover=lambda a, b: moves.append((a, b)))
    # "a" holds both slices: slice 0 = {0, 1}, slice 1 = {2}
    for env in (0, 1, 2):
        seats.place(env, "a")
    seats.release(2); seats.place(2, "a")
    assert seats.seat_of(2) == (0, 2)  # filled slice 0 first
    seats.place(3, "a")  # overflow -> slice 1
    assert seats.seat_of(3) == (1, 0)
    # "b" is drawn: no empty slice, but slice 1 (1 occupant) can be packed
    # into slice 0 only if it has room — release one env from slice 0 first
    seats.release(1)
    assert seats.place(4, "b") == (1, 0)  # compaction freed slice 1
    assert seats.compactions == 1 and moves == [((1, 0), (0, 1))]
    assert seats.seat_of(3) == (0, 1) and seats.member_at(1) == "b"
    assert loads[-1] == (1, "b")


def test_grid_move_cell_is_exact():
    from smashbot import embed as embed_lib

    S, N = 2, 2
    sd = _tiny_policy(seed=3).state_dict()
    ref = LeagueAgent(_tiny_policy(seed=0), S, N, name_code=1, device="cpu", temperature=1e-6)
    mv = LeagueAgent(_tiny_policy(seed=0), S, N, name_code=1, device="cpu", temperature=1e-6)
    for g in (ref, mv):
        g.load_slice(0, sd); g.load_slice(1, sd)  # same member on both slices
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(5):
        raw = _rand_raw_game(game, (S, N), rng)
        frames.append(tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x)),
            game.from_state(raw)))
    resets = torch.zeros(S, N, dtype=torch.bool)
    # env sits at (0, 0) in ref; in mv it is moved to (1, 1) after frame 2
    for t, v in enumerate(frames):
        rows_r, _ = ref.step(v, resets)
        if t == 3:
            mv.move_cell((0, 0), (1, 1))
        v_mv = v
        if t >= 3:  # feed the env's view at its new cell
            v_mv = tree.map_structure(lambda x: x.clone(), v)
            tree.map_structure(lambda a, b: a[1, 1].copy_(b[0, 0]), v_mv, v)
        rows_m, _ = mv.step(v_mv, resets)
        src = rows_r[0 * N + 0]
        dst = rows_m[(1 * N + 1) if t >= 3 else 0]
        assert np.array_equal(src, dst), f"frame {t}"


def test_single_slice_grid_matches_vmap_grid():
    """S=1 takes the no-vmap path; its rows/records must equal slice 0 of
    a 2-slice vmap grid loaded with the same weights (saturated sampling)."""
    from smashbot import embed as embed_lib

    sd = _tiny_policy(seed=5).state_dict()
    one = LeagueAgent(_tiny_policy(seed=0), 1, 3, name_code=1, device="cpu", temperature=1e-6)
    two = LeagueAgent(_tiny_policy(seed=0), 2, 3, name_code=1, device="cpu", temperature=1e-6)
    one.load_slice(0, sd); two.load_slice(0, sd); two.load_slice(1, sd)
    game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)
    for t in range(4):
        raw = _rand_raw_game(game, (2, 3), rng)
        v2 = tree.map_structure(lambda x: torch.from_numpy(np.ascontiguousarray(
            x.astype(np.int64) if x.dtype.kind in "iu" else x)), game.from_state(raw))
        v1 = tree.map_structure(lambda x: x[:1], v2)
        r2 = torch.zeros(2, 3, dtype=torch.bool); r1 = r2[:1]
        if t == 2:
            r2[0, 1] = True; one.reset_cell(0, 1); two.reset_cell(0, 1)
        rows1, rec1 = one.step(v1, r1)
        rows2, rec2 = two.step(v2, r2)
        assert np.array_equal(rows1, rows2[:3]), f"frame {t}"
        assert torch.equal(rec1.name, rec2.name[:3])
