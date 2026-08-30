"""Stream-parallel serving (rollouts._serve_submit + the parallel INFER
branch): the wiring must produce byte-identical results to the serial
path, and concurrent CUDA-graph replays on separate streams must match a
serial replay of the same agents."""

import numpy as np
import pytest
import torch
import tree

from smashbot import embed as embed_lib
from smashbot.rl.agent import LeagueAgent
from smashbot.rl.rollouts import DolphinRolloutWorker
from smashbot.tests.test_opponent_league import _make_worker
from smashbot.tests.test_ppo import _tiny_policy
from smashbot.tests.test_rollouts import _rand_raw_game


class _DeferredFuture:
    """Runs the task at result() time — after student+groups, i.e. at the
    exact position the serial path runs the grids, so the global-RNG draw
    order (and therefore every sample) matches the serial worker."""

    def __init__(self, fn):
        self._fn = fn

    def result(self):
        self._fn()


def _worker_pair(monkeypatch, tmp_path):
    kwargs = dict(
        num_envs=12, teacher_envs=0, league_slices=2,
        members=[0], harvest=True, pfsp_explore=1.0,
        import_dedicated_envs=2,
        league_imports=[
            "fx=/tmp/does-not-matter.pt@FOX",
            "free=/tmp/does-not-matter2.pt@ANY",
        ],
        char_whitelist=["FOX", "MARTH", "FALCO", "PEACH"],
    )
    a, _ = _make_worker(monkeypatch, pool_dir=tmp_path / "a", **kwargs)
    b, _ = _make_worker(monkeypatch, pool_dir=tmp_path / "b", **kwargs)
    return a, b


def _assert_same_trajectories(ta, tb):
    assert len(ta) == len(tb)
    for x, y in zip(ta, tb):
        def eq(u, v):
            if isinstance(u, torch.Tensor):
                assert torch.equal(u, v)
            elif isinstance(u, np.ndarray):
                assert np.array_equal(u, v)
            else:
                assert u == v
        tree.map_structure(eq, x, y)


def test_parallel_branch_matches_serial_wiring(monkeypatch, tmp_path):
    """The parallel INFER branch (mocked submit, no streams on CPU) must
    produce the same trajectories, harvest, and controller commands as
    the serial branch."""
    serial, par = _worker_pair(monkeypatch, tmp_path)

    submitted = []

    def fake_submit(name, fn):
        submitted.append(name)
        return _DeferredFuture(fn)

    par._serve_parallel = True
    par._serve_frames = 10 ** 6  # past the capture warm-up gate
    monkeypatch.setattr(par, "_serve_submit", fake_submit)

    torch.manual_seed(7)
    out_a = serial.collect(1)
    torch.manual_seed(7)
    out_b = par.collect(1)

    # ONE serving lane: imports are pinned slices inside the merged grid
    assert set(submitted) == {"grid"}
    _assert_same_trajectories(out_a, out_b)
    for ca, cb in zip(serial._conns, par._conns):
        assert len(ca.sent) == len(cb.sent)
        for cmd_a, cmd_b in zip(ca.sent, cb.sent):
            assert sorted(map(str, cmd_a)) == sorted(map(str, cmd_b))


def _cuda_ready(need_bytes=1_500_000_000):
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free >= need_bytes


@pytest.mark.skipif(
    not _cuda_ready(), reason="needs CUDA with >=1.5GB free"
)
def test_serve_submit_streams_match_serial_cuda():
    """Two captured LeagueAgents stepped concurrently through the real
    _serve_submit (threads + per-agent streams) must produce bitwise the
    same controller rows and logits as stepping identical twins serially
    on the default stream."""

    def make_pair():
        grid = LeagueAgent(
            _tiny_policy(seed=0), 3, 2, name_code=1, device="cuda",
            temperature=1e-6,
        )
        imp = LeagueAgent(
            _tiny_policy(seed=0), 2, 3, name_code=1, device="cuda",
            temperature=1e-6,
        )
        for s in range(3):
            grid.load_slice(s, _tiny_policy(seed=11 + s).state_dict())
        for s in range(2):
            imp.load_slice(s, _tiny_policy(seed=21 + s).state_dict())
        return grid, imp

    game = embed_lib.EmbedConfig().make_game_embedding()

    def views_for(shape, rng):
        raw = _rand_raw_game(game, shape, rng)
        return tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x)).cuda(),
            game.from_state(raw),
        )

    rng = np.random.default_rng(0)
    frames = [
        (views_for((3, 2), rng), views_for((2, 3), rng)) for _ in range(6)
    ]
    rz_g = torch.zeros(3, 2, dtype=torch.bool, device="cuda")
    rz_i = torch.zeros(2, 3, dtype=torch.bool, device="cuda")

    # threaded pair through the production helper vs a serial twin pair.
    # _serve_submit only touches these attributes of the worker:
    class _Host:
        def __init__(self):
            self._serve_pool = None
            self._serve_streams = {}
            self._prof = None

    host = _Host()
    grid_p, imp_p = make_pair()
    # captures must happen single-threaded (as the warm-up gate does in
    # the worker): run one serial frame's worth of capture first
    grid_p.infer(frames[0][0], rz_g)
    grid_p.execute()  # drain the extra queue entry from the capture frame
    imp_p.infer(frames[0][1], rz_i)
    imp_p.execute()
    grid_s2, imp_s2 = make_pair()  # fresh twins so states start equal
    grid_s2.infer(frames[0][0], rz_g)
    grid_s2.execute()
    imp_s2.infer(frames[0][1], rz_i)
    imp_s2.execute()

    for vg, vi in frames[1:]:
        out = {}

        def g_task(vg=vg):
            out["g"] = grid_p.step(vg, rz_g)

        def i_task(vi=vi):
            out["i"] = imp_p.step(vi, rz_i)

        futs = [
            DolphinRolloutWorker._serve_submit(host, "grid", g_task),
            DolphinRolloutWorker._serve_submit(host, "imports", i_task),
        ]
        for f in futs:
            f.result()
        cur = torch.cuda.current_stream()
        for s in host._serve_streams.values():
            cur.wait_stream(s)
        rows_g2, rec_g2 = grid_s2.step(vg, rz_g)
        rows_i2, rec_i2 = imp_s2.step(vi, rz_i)
        assert np.array_equal(out["g"][0], rows_g2)
        assert np.array_equal(out["i"][0], rows_i2)
        for a, b in zip(
            tree.flatten(out["g"][1].logits), tree.flatten(rec_g2.logits)
        ):
            assert torch.equal(a, b)
        for a, b in zip(
            tree.flatten(out["i"][1].logits), tree.flatten(rec_i2.logits)
        ):
            assert torch.equal(a, b)
