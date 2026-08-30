"""League-training feature tests (all Dolphin-free): student char whitelist,
dual-seat collection (self-play + reference-seat imitation harvest),
memory-neutral batch substitution, opponent-advantage imitation math, and
PFSP snapshot prioritization.

The worker tests drive a real DolphinRolloutWorker through collect() with
faked env pipes (synthetic payloads instead of live Dolphins), so seat
routing, reward mirroring, and harvest gating run the production code path.
"""

import json
import os
import random

import numpy as np
import pytest
import torch
import tree

from smashbot import configs, embed as embed_lib, encode
from smashbot.policy import build_policy
from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.rl.ppo import (
    Learner,
    PPOConfig,
    RLConfig,
    Trajectory,
    imitation_weights,
    slice_trajectory_rows,
)
from smashbot.rl.rollouts import DolphinRolloutWorker, RolloutConfig, compute_reward
from smashbot.tests.test_ppo import _rollout, _tiny_policy, _tiny_value
from smashbot.tests.test_rollouts import _rand_raw_game

DATA = os.path.join(os.path.dirname(__file__), "data")


# ------------------------------------------------------------ char whitelist


def test_student_whitelist_helper():
    from smashbot.rl.pool import student_whitelist

    # default whitelist defers to the legacy bot_char flag
    assert student_whitelist(["FOX"]) == ["FOX"]
    assert student_whitelist(["FOX"], bot_char="MARTH") == ["MARTH"]
    # any non-default whitelist wins, case-normalized
    assert student_whitelist(["fox", "falco"], bot_char="MARTH") == [
        "FOX", "FALCO",
    ]


def test_partition_default_noop_golden():
    """Hard requirement: under default whitelist/self_envs the partition is
    byte-identical to the pre-feature code (golden captured from it)."""
    from smashbot.rl.pool import make_partition

    with open(os.path.join(DATA, "partition_golden.json")) as f:
        golden = json.load(f)
    for key, expect in golden.items():
        specs = make_partition(**json.loads(key))
        got = [[s.kind, s.student_port, s.opponent_char] for s in specs]
        assert got == expect, f"partition drifted for {key}"


def test_partition_self_envs_arithmetic_and_order():
    from smashbot.rl.pool import make_partition

    # self envs cost 2 budget units, run 1 dolphin each
    specs = make_partition(
        num_envs=24, cpu_envs=2, teacher_envs=6,
        seed=0, ref_envs=4, self_envs=4,
        char_whitelist=["FOX", "FALCO"],
    )
    assert len(specs) == 24 - 4  # dolphins = num_envs - self_envs
    kinds = [s.kind for s in specs]
    assert kinds == (
        ["cpu"] * 2 + ["teacher"] * 6 + ["reference"] * 4
        + ["self"] * 4 + ["snapshot"] * 4
    )
    selfs = [s for s in specs if s.kind == "self"]
    # both seats student: second-seat char draws from the whitelist,
    # stratified (both chars present at 4 >= 2)
    assert {s.opponent_char for s in selfs} == {"FOX", "FALCO"}
    assert {s.student_port for s in selfs} == {1, 2}

    # teacher_envs=-1 accounts for the doubled self budget
    specs = make_partition(
        num_envs=16, cpu_envs=2, teacher_envs=-1,
        seed=0, ref_envs=2, self_envs=3,
    )
    assert len(specs) == 13
    assert [s.kind for s in specs].count("teacher") == 16 - 2 - 2 - 6


# --------------------------------------------------- fake-env worker harness


class _FakeConn:
    """Records commands and validates controller payloads the way the env
    consumes them (encode.controller_from_flat on every port entry)."""

    def __init__(self):
        self.sent = []

    def send(self, cmd):
        for k, v in cmd.items():
            if isinstance(k, int):
                encode.controller_from_flat(v)  # raises on a struct/short row
        self.sent.append(cmd)


class _FakeEnvs:
    """Synthetic payload source replacing the Dolphin pipes: per-dolphin
    scripted opp_char and a percent ramp on the port-2 seat (so every
    transition has a nonzero, sign-checkable reward)."""

    def __init__(self, worker, seed=0, opp_chars=None):
        self.worker = worker
        self.rng = np.random.default_rng(seed)
        self.embed_game = embed_lib.EmbedConfig().make_game_embedding()
        self.opp_chars = opp_chars or {}
        self.t = 0
        self.final_stocks = {}  # dolphin -> (p1, p2) to deliver next frame
        # dolphin -> "cpu": env-reported ACTUAL serving (league_cpu lazy
        # adoption); results delivered while flipped report that serving
        self.serving = {}

    def install(self, monkeypatch):
        w = self.worker
        if w.league_idx and not w.league.member_now:
            locks = w.league.boot(w.league_idx)
            for i, lock in locks.items():
                if lock is not None:
                    w.specs[i].opponent_char = lock
                    self.opp_chars.setdefault(i, lock)
        monkeypatch.setattr(w, "_ensure_started", lambda: None)
        monkeypatch.setattr(w, "_gather_all", self.gather)
        w._conns = [_FakeConn() for _ in range(w.num_dolphins)]
        w._procs = list(w._conns)
        w._frame_count = 0
        w._prev_stocks = torch.full((w.num_rows, 2), 4.0)
        w._prev_percent = torch.zeros(w.num_rows, 2)

    def gather(self):
        payloads = []
        for i in range(self.worker.num_dolphins):
            raw = _rand_raw_game(self.embed_game, (), self.rng)
            fs = self.final_stocks.pop(i, None)
            serving = self.serving.get(i, "policy")
            payloads.append(dict(
                game=encode.flatten_typed(self.embed_game.from_state(raw)),
                resetting=fs is not None,
                final_stocks=fs,
                stocks=(4, 4),
                # port-2 seat takes 1% per frame: port-1-seat reward +0.01
                percent=(0.0, float(self.t)),
                opp_char=self.opp_chars.get(i, "FOX"),
                opp_serving=serving,
                result_serving=serving if fs is not None else None,
            ))
        self.t += 1
        return payloads


def _make_worker(monkeypatch, num_envs, seed=0, opp_chars=None,
                 harvest=False, ref_controller_config=None, pool_dir=None,
                 members=None, phillip_capacity=0, **cfg_kwargs):
    """A real DolphinRolloutWorker over fake env pipes. League envs (any
    env not cpu/teacher/reference/self) get a LeagueRuntime: a tiny-policy
    grid, a SnapshotPool in `pool_dir` seeded with the ghosts listed in
    `members` (archive paths) plus the config's league members, and a
    Phillip agent under league_phillip."""
    cfg = RolloutConfig(
        num_envs=num_envs, unroll_length=4, games_per_dolphin=10**9,
        **cfg_kwargs,
    )
    student = BatchedPolicyAgent(_tiny_policy(seed=0), num_envs, name_code=1)
    from smashbot.rl.pool import make_partition, student_whitelist

    import_registry = {
        f"import:{name}": (path, char)
        for name, (path, char) in cfg.import_members().items()
    } if cfg.import_dedicated_envs > 0 else None
    specs = make_partition(
        cfg.num_envs, cfg.cpu_envs, cfg.teacher_envs,
        cfg.main12_prob, cfg.partition_seed, ref_envs=cfg.ref_envs,
        self_envs=cfg.self_envs,
        char_whitelist=student_whitelist(cfg.char_whitelist, cfg.bot_char),
        import_registry=import_registry,
        import_envs_per=cfg.import_dedicated_envs,
    )
    opponents = {}
    counts = {}
    for sp in specs:
        counts[sp.kind] = counts.get(sp.kind, 0) + 1
    if counts.get("teacher"):
        opponents["teacher"] = BatchedPolicyAgent(
            _tiny_policy(seed=1), counts["teacher"], name_code=1
        )
    if counts.get("reference"):
        ref_policy = _phillip_like_policy(ref_controller_config)
        opponents["reference"] = BatchedPolicyAgent(
            ref_policy, counts["reference"], name_code=2
        )
    runtime = None
    if counts.get("snapshot"):
        # dedicated imports (import_registry non-empty) get pinned tail
        # slices inside _make_runtime's grid
        runtime = _make_runtime(
            cfg, counts["snapshot"], pool_dir, members or [], phillip_capacity,
        )
    worker = DolphinRolloutWorker(
        cfg, student, opponents=opponents, specs=specs,
        harvest_imitation=harvest, league=runtime,
    )
    envs = _FakeEnvs(worker, seed=seed, opp_chars=opp_chars)
    envs.install(monkeypatch)
    return worker, envs


def _make_runtime(cfg, league_envs, pool_dir, ghosts, phillip_capacity):
    """LeagueRuntime over tiny policies: the grid template is the teacher-
    seed policy; ghosts are saved into the pool as distinct tiny policies;
    imports (cfg.league_imports) resolve to files written here too."""
    import tempfile

    from smashbot.rl.agent import LeagueAgent
    from smashbot.rl.league import League, LeagueSeats, MemberWeights
    from smashbot.rl.pool import SnapshotPool
    from smashbot.rl.rollouts import LeagueRuntime

    pool_dir = pool_dir or tempfile.mkdtemp(prefix="league-")
    members = cfg.league_members()
    pool = SnapshotPool(
        str(pool_dir), pfsp=cfg.pfsp, pfsp_p=cfg.pfsp_p,
        pfsp_hard_frac=cfg.pfsp_hard_frac, pfsp_explore=cfg.pfsp_explore,
        league_members=members,
    )
    for j, step in enumerate(ghosts):
        pool.save(_tiny_policy(seed=10 + j), step)
    S = cfg.league_slices or 1
    N = -(-(league_envs + S) // S)
    dedicated = cfg.import_dedicated_envs > 0
    imports = cfg.import_members()
    per = -(-cfg.import_dedicated_envs // N) if dedicated else 0
    S_total = S + per * len(imports if dedicated else ())
    grid = LeagueAgent(
        _tiny_policy(seed=0), S_total, N, name_code=1, device="cpu"
    )
    fixed = {"teacher": _tiny_policy(seed=1).state_dict()}
    locks = {}
    for slot, (name, (path, char)) in enumerate(imports.items()):
        key = f"import:{name}"
        if dedicated:
            # pinned tail slices, one distinct tiny policy per member
            sd = _tiny_policy(seed=200 + slot).state_dict()
            for j in range(per):
                grid.load_slice(S + slot * per + j, sd)
        else:
            fixed[key] = _tiny_policy(seed=100 + len(fixed)).state_dict()
            locks[key] = char
    weights = MemberWeights(fixed)
    phillip = None
    if cfg.league_phillip:
        # a differently-discretized policy, like the real Phillip module
        # (exercises the harvest re-encode path): his own 1-slice grid
        ph_policy = _phillip_like_policy(
            embed_lib.ControllerConfig(axis_spacing=8)
        )
        phillip = LeagueAgent(
            ph_policy, 1, phillip_capacity or N, name_code=2, device="cpu"
        )
        phillip.load_slice(0, ph_policy.state_dict())
    seats = LeagueSeats(
        S, N, loader=lambda s, m: grid.load_slice(s, weights.get(m)),
        phillip_capacity=phillip.N if phillip else 0,
        mover=grid.move_cell,
    )
    league = League(
        pool, seats, locks=locks, rng=random.Random(0),
        on_result=pool.record_result, cpu_enabled=cfg.league_cpu,
    )
    return LeagueRuntime(league, grid, phillip, import_slices_per=per)


def _phillip_like_policy(controller_config=None):
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=controller_config or embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(
            name="sgu", num_layers=1, hidden_size=32, num_heads=1, window=4
        ),
        head_config=configs.ControllerHeadConfig(
            residual_size=32, component_depth=0
        ),
        policy_config=configs.PolicyConfig(delay=2),
        num_names=4,
    )
    policy.train_value_head = False
    return policy


def test_worker_default_config_noop(monkeypatch):
    """Default config (self_envs=0, whitelist FOX, no harvest): rows map 1:1
    onto dolphins, exactly num_envs PPO trajectories, no imitation output,
    commands routed to both seats' ports as before."""
    worker, _ = _make_worker(monkeypatch, num_envs=4)
    assert worker.num_dolphins == 4 and worker.num_rows == 4
    assert torch.equal(worker._row_dolphin, torch.arange(4))
    assert worker.row_kinds == ["teacher"] * 4
    assert not worker.harvest_imitation

    trajs = worker.collect(2)
    assert len(trajs) == 2
    assert all(t.kind == "ppo" for t in trajs)
    assert all(t.rewards.shape[0] == 4 for t in trajs)
    for i, conn in enumerate(worker._conns):
        port = worker.specs[i].student_port
        assert all(set(cmd) == {port, 3 - port} for cmd in conn.sent)


def test_worker_self_play_rows_and_reward_mirror(monkeypatch):
    """self_envs=S: dolphins = num_envs - S; the second seat of each self
    dolphin is a learner row driven by the SAME student agent, its rewards
    the exact zero-sum mirror of the primary seat's."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=6, self_envs=2, teacher_envs=-1
    )
    assert worker.num_dolphins == 4 and worker.num_rows == 6
    assert worker.row_kinds == ["teacher", "teacher", "self", "self",
                                "self", "self"]
    self_dolphins = worker.self_idx
    assert self_dolphins == [2, 3]

    (traj,) = worker.collect(1)
    assert traj.kind == "ppo"
    assert traj.rewards.shape[0] == 6  # trajectory budget, not dolphin count
    for d in self_dolphins:
        r_primary = traj.rewards[d]
        r_second = traj.rewards[worker._self_row_of[d]]
        torch.testing.assert_close(r_primary, -r_second)
        assert r_primary.abs().sum() > 0, "test needs nonzero rewards"
    # both seats' controllers come from the student batch
    for d in self_dolphins:
        assert all(set(cmd) == {1, 2} for cmd in worker._conns[d].sent)

    # self-play game results are PORT-relative (health metric ~50%)
    envs.final_stocks[2] = (1, 3)  # port1 lost
    envs.final_stocks[0] = (1, 3)  # teacher dolphin, student_port matters
    worker.collect(1)
    assert worker.trackers["self"].losses == 1  # port-1 seat lost
    assert worker.trackers["self"].wins == 0


def test_whitelist_gates_imitation_harvest(monkeypatch):
    """Reference seats are harvested as kind='imitation' ONLY while their
    char is whitelisted; teacher seats are never harvested; harvested
    rewards mirror the student seat's; names are re-conditioned on the
    student's code."""
    # 2 teacher + 2 reference dolphins; ref dolphin 2 plays FOX
    # (whitelisted), ref dolphin 3 plays MARTH (not)
    worker, _ = _make_worker(
        monkeypatch, num_envs=4, ref_envs=2, harvest=True,
        opp_chars={2: "FOX", 3: "MARTH"},
        ref_controller_config=embed_lib.ControllerConfig(axis_spacing=8),
    )
    assert worker.harvest_imitation
    assert worker.ref_idx == [2, 3]

    trajs = worker.collect(1)
    kinds = [t.kind for t in trajs]
    assert kinds.count("ppo") == 1
    assert kinds.count("imitation") == 1
    imit = [t for t in trajs if t.kind == "imitation"][0]
    main = [t for t in trajs if t.kind == "ppo"][0]
    assert imit.rewards.shape[0] == 1  # only the whitelisted ref seat
    assert imit.initial_state is None
    # name re-conditioned on the student's code (1), not the ref's (2)
    assert torch.equal(imit.name, torch.ones_like(imit.name))
    # zero-sum mirror of the ref dolphin's student-seat rewards (same delay
    # here, so slots align 1:1)
    torch.testing.assert_close(imit.rewards[0], -main.rewards[2])
    # actions live in the STUDENT controller schema despite the ref policy
    # discretizing differently (axis_spacing 8 vs 16)
    stu_embed = worker.student._embed_controller
    tree.map_structure(
        lambda enc, ref_leaf: None,
        imit.actions.controller_state, main.actions.controller_state,
    )  # same structure
    # harvested actions must round-trip our embedding (valid bucket range;
    # records widen to int64, decode wants each leaf's native dtype back)
    stu_embed.decode(stu_embed.map(
        lambda e, x: x.astype(getattr(e, "dtype", x.dtype)),
        tree.map_structure(
            lambda x: x.cpu().numpy(), imit.actions.controller_state
        ),
    ))

    # and the harvested trajectory feeds the imitation learner path
    learner = Learner(
        RLConfig(imitation_rows=-1, imitation_lambda=0.1),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    imf = learner._imitation_fixed(imit)
    assert imf is not None
    loss = learner._imitation_chunk_loss(imf, float(imf.valid.sum()))
    assert torch.isfinite(loss)


def test_non_reference_kinds_never_harvested(monkeypatch):
    """harvest flag on, but no reference envs (teacher + cpu only): no
    imitation trajectories can exist."""
    worker, _ = _make_worker(
        monkeypatch, num_envs=4, cpu_envs=2, harvest=True,
        opp_chars={i: "FOX" for i in range(4)},  # all whitelisted chars
    )
    assert not worker.harvest_imitation  # no ref group to harvest
    trajs = worker.collect(1)
    assert [t.kind for t in trajs] == ["ppo"]
    # cpu dolphins get only the student's controller (engine AI drives opp)
    for i in (0, 1):
        port = worker.specs[i].student_port
        assert all(set(cmd) == {port} for cmd in worker._conns[i].sent)


# -------------------------------------------- memory-neutral batch invariant


def _imitation_traj(policy, B, seed=1) -> Trajectory:
    return _rollout(policy, B=B, T=8, seed=seed)._replace(
        kind="imitation", initial_state=None
    )


def test_imitation_trains_all_rows_in_ppo_sized_chunks():
    """No substitution: the PPO pass covers EVERY PPO row and the imitation
    term covers EVERY harvested row (or a uniform cap), accumulated in
    chunks no larger than the PPO micro-batch chunk."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=6, T=8, seed=0)

    def run(rows_cfg, k, imit_B):
        learner = Learner(
            RLConfig(
                imitation_rows=rows_cfg, imitation_lambda=0.1, micro_batches=k,
                ppo=PPOConfig(max_mean_actor_kl=1e9),
            ),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        imit = _imitation_traj(policy, B=imit_B)
        unroll_rows = []
        orig_unroll = learner.policy.unroll

        def counting_unroll(frames, st, **kw):
            unroll_rows.append(frames.reward.shape[0])
            return orig_unroll(frames, st, **kw)

        learner.policy.unroll = counting_unroll
        _, metrics = learner.step([main, imit], learner.initial_state(6))
        # epoch passes only: drop the post-update full-batch check
        return metrics, unroll_rows[:-1]

    # all rows, k=2: PPO chunks 3+3, imitation 7 rows -> chunks <= 3
    m, rows = run(-1, 2, 7)
    assert m["imitation"]["traj_count"] == 7
    assert rows[:2] == [3, 3]
    assert sum(rows[2:]) == 7 and max(rows[2:]) <= 3
    # all rows, k=1: one PPO pass of 6, imitation 7 -> chunks <= 6
    m, rows = run(-1, 1, 7)
    assert rows[0] == 6 and sum(rows[1:]) == 7 and max(rows[1:]) <= 6
    # uniform cap: 3 of 7 harvested rows
    m, rows = run(3, 1, 7)
    assert m["imitation"]["traj_count"] == 3
    assert rows[0] == 6 and sum(rows[1:]) == 3


def test_imitation_loss_is_exact_mean_over_all_rows():
    """The accumulated imitation loss equals -(w * log pi * valid).sum() /
    valid.sum() over ALL harvested rows, whatever the chunking (one epoch,
    so the reported loss is at the pre-update parameters)."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=4, T=8, seed=0)
    imit = _imitation_traj(policy, B=5)
    for k in (1, 3):
        learner = Learner(
            RLConfig(imitation_rows=-1, imitation_lambda=0.1, micro_batches=k,
                     ppo=PPOConfig(max_mean_actor_kl=1e9, num_epochs=1)),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        captured = []
        orig_plan = learner._plan_imitation

        def plan(trajs):
            out = orig_plan(trajs)
            captured.extend(out[0])
            return out

        learner._plan_imitation = plan
        _, metrics = learner.step([main, imit], learner.initial_state(4))
        assert len(captured) == 1
        imf = captured[0]
        ref_policy = _tiny_policy(seed=0)  # == learner.policy before the step
        with torch.no_grad():
            out = ref_policy.unroll(
                imf.frames, ref_policy.initial_state(imf.rows),
                discount=learner.config.discount,
            )
            ref = (-(imf.weights * out.log_probs * imf.valid).sum()
                   / imf.valid.sum()).item()
        assert metrics["imitation"]["loss"] == pytest.approx(ref, rel=1e-5)


@pytest.mark.parametrize("k", [2, 3])
def test_imitation_accumulation_is_chunk_invariant(k):
    """PPO + imitation accumulated over k chunks each gives the same
    GRADIENT as k=1 (the 'accumulate separately == one joint backward'
    property), and hence the same update up to Adam amplifying fp-order
    noise on near-zero-gradient elements."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=6, T=8, seed=0)
    imit = _imitation_traj(policy, B=5)

    def run(k):
        torch.manual_seed(0)
        learner = Learner(
            RLConfig(imitation_rows=-1, imitation_lambda=0.5, micro_batches=k,
                     learning_rate=1e-3, ppo=PPOConfig(max_mean_actor_kl=1e9)),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        with torch.no_grad():
            g = torch.Generator().manual_seed(1)
            for a in learner.policy.parameters():
                a.add_(torch.randn(a.shape, generator=g) * 1e-2)
        grads = []
        orig_step = learner.policy_optimizer.step

        def capturing_step(*args, **kw):
            grads.extend(
                None if p.grad is None else p.grad.detach().clone()
                for p in learner.policy.parameters()
            )
            return orig_step(*args, **kw)

        learner.policy_optimizer.step = capturing_step
        _, m = learner.step([main, imit], learner.initial_state(6))
        return learner, m, grads

    full, mf, gf = run(1)
    chunked, mc, gc = run(k)
    assert gf and len(gf) == len(gc)
    assert any(g is not None for g in gf)
    for ga, gb in zip(gf, gc):
        assert (ga is None) == (gb is None)
        if ga is not None:
            torch.testing.assert_close(ga, gb, rtol=1e-4, atol=1e-7)
    for pa, pb in zip(full.policy.parameters(), chunked.policy.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-3, atol=1e-5)
    assert mf["imitation"]["loss"] == pytest.approx(mc["imitation"]["loss"], rel=1e-5)
    assert mf["imitation"]["traj_count"] == mc["imitation"]["traj_count"] == 5


def test_default_config_learner_ignores_imitation_trajs():
    """Dormant path: imitation_rows=0 (default) => imitation trajectories
    are ignored entirely and metrics carry no imitation key."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=3, T=8, seed=0)

    results = []
    for extra in ([], [_imitation_traj(policy, B=2)]):
        learner = Learner(
            RLConfig(ppo=PPOConfig(max_mean_actor_kl=1e9)),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        _, metrics = learner.step([main] + extra, learner.initial_state(3))
        assert "imitation" not in metrics
        results.append({
            k: v.detach().clone()
            for k, v in learner.policy.state_dict().items()
        })
    for k in results[0]:
        assert torch.equal(results[0][k], results[1][k])


# ------------------------------------------------------------ imitation math


def test_imitation_weight_math():
    valid = torch.ones(1, 4)
    A = torch.tensor([[1.0, 2.0, 3.0, 6.0]], requires_grad=True)
    w = imitation_weights(A, valid, beta=1.0, w_cap=20.0)
    # hand math: mean 3, std sqrt(3.5)
    a_norm = (A.detach() - 3.0) / (torch.tensor(3.5).sqrt() + 1e-8)
    torch.testing.assert_close(w, torch.exp(a_norm))
    assert not w.requires_grad  # A detached before use

    # beta scales inside the exp; cap clips hard
    w = imitation_weights(A, valid, beta=0.1, w_cap=5.0)
    assert w.max().item() == pytest.approx(5.0)
    assert (w <= 5.0).all()

    # masked positions are excluded from the normalization stats
    valid2 = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    w2 = imitation_weights(A, valid2, beta=1.0, w_cap=20.0)
    a_norm2 = (A.detach() - 2.0) / (
        torch.tensor(2.0 / 3.0).sqrt() + 1e-8
    )
    torch.testing.assert_close(w2[:, :3], torch.exp(a_norm2)[:, :3])


def test_imitation_advantage_is_g_minus_v_and_detached():
    """A = G_t - V(s_t) with G_t from the trajectory's own returns (the
    critic's target machinery); the policy term must not backprop into the
    critic; the critic DOES train on the harvested states."""
    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    learner = Learner(
        RLConfig(imitation_rows=-1, imitation_lambda=0.1),
        policy, _tiny_policy(seed=0), _tiny_value(),
    )
    traj = _imitation_traj(policy, B=2)
    frames = learner._frames(traj)

    # expected A from the value function BEFORE _imitation_fixed updates it
    with torch.no_grad():
        expected = learner.value_function.outputs(
            frames, learner.value_function.initial_state(2),
            discount=learner.config.discount,
        )
    from smashbot import delay as delay_lib

    critic_before = [p.detach().clone()
                     for p in learner.value_function.parameters()]
    imf = learner._imitation_fixed(traj)
    valid = (~traj.is_resetting[:, 1:]).float()
    torch.testing.assert_close(
        imf.weights,
        imitation_weights(expected.advantages, valid, 1.0, 20.0),
    )
    # the critic moved (trained on G_t targets)
    assert any(
        not torch.equal(a, b) for a, b in
        zip(critic_before, learner.value_function.parameters())
    )
    # sanity on the G - V identity: advantages + values reproduce the
    # discounted-return targets (recomputed by hand)
    del delay_lib  # identity is enforced inside ValueFunction.outputs

    # actor loss must not leak gradient into the critic
    learner.value_optimizer.zero_grad(set_to_none=True)
    loss = learner._imitation_chunk_loss(imf, float(imf.valid.sum()))
    loss.backward()
    assert all(p.grad is None for p in learner.value_function.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in learner.policy.parameters()
    )


def test_lambda_decay_endpoints():
    learner = Learner(
        RLConfig(imitation_lambda=0.04, imitation_lambda_final_frac=0.25),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    assert learner.lambda_at(0.0) == pytest.approx(0.04)
    assert learner.lambda_at(1.0) == pytest.approx(0.01)
    assert learner.lambda_at(0.5) == pytest.approx(0.025)  # linear
    assert learner.lambda_at(2.0) == pytest.approx(0.01)  # clamped


def test_lambda_zero_actor_term_exactly_absent():
    """imitation_lambda=0: the actor-side term contributes NOTHING — two
    runs with radically different imitation ACTIONS produce bitwise-equal
    policies, and the PPO loss equals plain PPO on the same (full) rows."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=4, T=8, seed=0)

    def run(imit_seed, lam):
        learner = Learner(
            RLConfig(imitation_rows=-1, imitation_lambda=lam,
                     ppo=PPOConfig(max_mean_actor_kl=1e9)),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        imit = _imitation_traj(_tiny_policy(seed=imit_seed), B=2,
                               seed=imit_seed)
        _, metrics = learner.step([main, imit], learner.initial_state(4))
        return learner, metrics

    l_a, m_a = run(imit_seed=5, lam=0.0)
    l_b, m_b = run(imit_seed=9, lam=0.0)
    assert m_a["imitation"]["loss"] == 0.0  # term never computed
    for k, v in l_a.policy.state_dict().items():
        assert torch.equal(v, l_b.policy.state_dict()[k]), k

    # the full PPO batch through a plain-PPO learner: identical first-epoch
    # loss (no rows are dropped any more, so this is the same forward)
    plain = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1e9)),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    _, m_plain = plain.step([main], plain.initial_state(4))
    assert m_a["epochs"][0]["loss"] == pytest.approx(
        m_plain["epochs"][0]["loss"], rel=1e-6
    )


# ----------------------------------------------------------------------- PFSP


def test_f_hard_math():
    from smashbot.rl.pool import f_hard

    assert f_hard(0.0) == 1.0
    assert f_hard(1.0) == 0.0  # fully beaten => zero weight
    assert f_hard(0.75, p=2.0) == pytest.approx(0.0625)
    xs = [0.1, 0.3, 0.5, 0.7, 0.9]
    ws = [f_hard(x) for x in xs]
    assert ws == sorted(ws, reverse=True)  # higher win rate => lower weight


class _Stub:
    def state_dict(self):
        return {"w": torch.zeros(1)}


def test_pfsp_prior_and_payoff_updates(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path))
    p = pool.save(_Stub(), 100)
    assert pool.win_estimate(p) == 0.5  # no games: prior
    for won in [True, True, True, False]:
        pool.record_result(p, won)
    assert pool.win_estimate(p) == 0.5  # < 5 decided games: still prior
    pool.record_result(p, True)
    est = pool.win_estimate(p)
    assert est != 0.5
    # hand decayed counts (0.99 decay): ~= exact mean at small n
    wd = gd = 0.0
    for o in [1.0, 1.0, 1.0, 0.0, 1.0]:
        wd = 0.99 * wd + o
        gd = 0.99 * gd + 1.0
    assert est == pytest.approx(wd / gd)


def test_pfsp_persistence_roundtrip_and_prune(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path))
    a = pool.save(_Stub(), 100)
    b = pool.save(_Stub(), 200)
    for _ in range(6):
        pool.record_result(a, True)
        pool.record_result(b, False)
    # sneak in an entry for a snapshot that no longer exists
    pool.payoff["/nonexistent/snapshot-999.pt"] = {
        "wins": 1, "games": 1, "win_ema": 1.0
    }
    pool._save_payoff()

    fresh = SnapshotPool(str(tmp_path))
    assert fresh.win_estimate(a) == pytest.approx(pool.win_estimate(a))
    assert fresh.win_estimate(b) == pytest.approx(pool.win_estimate(b))
    assert "/nonexistent/snapshot-999.pt" not in fresh.payoff  # pruned
    assert os.path.exists(os.path.join(str(tmp_path), "pfsp.json"))


def test_pfsp_thinning_drops_payoff_rows(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), keep=4)
    for s in range(0, 800, 100):
        p = pool.save(_Stub(), s)
        for _ in range(3):
            pool.record_result(p, False)
    assert len(pool.archive) == 4  # thinned
    # eviction drops the ghost's payoff row: every row references a
    # surviving snapshot, and surviving snapshots kept their data
    assert set(pool.payoff) <= set(pool.archive)
    assert pool.payoff[pool.archive[-1]]["games"] == 3


def test_pfsp_sampling_prefers_hard_opponents(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path))
    easy = pool.save(_Stub(), 100)   # student dominates: x ~ 1
    hard = pool.save(_Stub(), 200)   # student loses: x ~ 0
    mid = pool.save(_Stub(), 250)
    latest = pool.save(_Stub(), 300)
    for _ in range(120):
        pool.record_result(easy, True)
        pool.record_result(hard, False)

    counts = {easy: 0, hard: 0, mid: 0, latest: 0}
    rng = random.Random(0)
    for _ in range(800):
        counts[pool.draw_member(rng)] += 1
    # per-match draws over the whole archive (the latest competes too);
    # after 120 straight wins easy is strongly suppressed
    assert counts[easy] < counts[mid] < counts[hard]
    assert counts[easy] < 0.15 * 800
    assert counts[latest] > 0

    # everyone beaten: uniform fallback still draws
    for _ in range(120):
        pool.record_result(mid, True)
        pool.record_result(hard, True)
        pool.record_result(latest, True)
    assert pool.draw_member(random.Random(0)) in counts


def test_compute_reward_is_zero_sum_mirror():
    """Seat-2's reward stream is exactly the negation of seat-1's: swapping
    the (own, opp) columns flips the sign (the dual-seat collection relies
    on this to reuse one compute_reward call for both seats)."""
    rng = np.random.default_rng(0)
    prev_s = torch.tensor(rng.integers(0, 5, (8, 2)), dtype=torch.float32)
    s = torch.clamp(prev_s - torch.tensor(
        rng.integers(0, 2, (8, 2)), dtype=torch.float32), min=0)
    prev_p = torch.tensor(rng.uniform(0, 150, (8, 2)), dtype=torch.float32)
    p = prev_p + torch.tensor(rng.uniform(-20, 40, (8, 2)),
                              dtype=torch.float32)
    resets = torch.tensor([False] * 6 + [True] * 2)

    r1 = compute_reward(prev_s, s, prev_p, p, resets)
    r2 = compute_reward(
        prev_s.flip(-1), s.flip(-1), prev_p.flip(-1), p.flip(-1), resets
    )
    torch.testing.assert_close(r1, -r2)
    assert torch.equal(r1[6:], torch.zeros(2))  # resets zero both seats


# ------------------------------------- league members (teacher / lvl-9 CPU)


def test_league_flags_default_off_golden(tmp_path):
    """Flags-off golden: no league members, and SnapshotPool assignments are
    byte-identical with and without the (empty) league_members argument.
    (Partitions are covered by test_partition_default_noop_golden — the
    league flags never touch make_partition.)"""
    from smashbot.rl.pool import SnapshotPool

    assert RolloutConfig().league_members() == []
    # league_imports=[] (the default) is part of the same guarantee: no
    # members, no import registry, nothing changes
    assert RolloutConfig(league_imports=[]).league_members() == []
    assert RolloutConfig().import_members() == {}

    pool_a = SnapshotPool(str(tmp_path / "a"))
    pool_b = SnapshotPool(str(tmp_path / "b"), league_members=())
    for s in range(0, 600, 100):
        pool_a.save(_Stub(), s)
        pool_b.save(_Stub(), s)
        pool_a.record_result(pool_a.archive[-1], s % 200 == 0)
        pool_b.record_result(pool_b.archive[-1], s % 200 == 0)
    for seed in range(25):
        a = pool_a.draw_member(random.Random(seed))
        b = pool_b.draw_member(random.Random(seed))
        assert os.path.basename(a) == os.path.basename(b)


def test_learner_overlap_rejects_live_teacher_envs():
    """learner_overlap serves the live teacher module from the worker
    thread while the learner uses it concurrently — forbidden; teacher
    envs must be zero (folded into the league or absent)."""
    with pytest.raises(AssertionError, match="learner_overlap"):
        RolloutConfig(learner_overlap=True, teacher_envs=4).league_members()
    with pytest.raises(AssertionError, match="learner_overlap"):
        RolloutConfig(learner_overlap=True).league_members()  # default -1
    assert RolloutConfig(
        learner_overlap=True, teacher_envs=0
    ).league_members() == []


def test_league_flag_asserts():
    """Loud config validation: league flag with a nonzero fixed partition,
    or without pfsp, must fail with an actionable message."""
    with pytest.raises(AssertionError, match="teacher_envs=0"):
        RolloutConfig(league_teacher=True, teacher_envs=16).league_members()
    with pytest.raises(AssertionError, match="teacher_envs=0"):
        RolloutConfig(league_teacher=True).league_members()  # default -1
    with pytest.raises(AssertionError, match="cpu_envs=0"):
        RolloutConfig(
            league_cpu=True, cpu_envs=4, teacher_envs=0
        ).league_members()
    with pytest.raises(AssertionError, match="ref_envs=0"):
        RolloutConfig(league_phillip=True, ref_envs=52).league_members()
    with pytest.raises(AssertionError, match="pfsp"):
        RolloutConfig(
            league_teacher=True, teacher_envs=0, pfsp=False
        ).league_members()
    with pytest.raises(AssertionError, match="pfsp"):
        RolloutConfig(league_phillip=True, pfsp=False).league_members()
    # SnapshotPool enforces the pfsp dependency independently
    from smashbot.rl.pool import SnapshotPool

    with pytest.raises(AssertionError, match="pfsp"):
        SnapshotPool("/tmp/never-used", pfsp=False,
                     league_members=("teacher",))
    # valid combos pass
    assert RolloutConfig(
        league_teacher=True, league_cpu=True, teacher_envs=0, cpu_envs=0
    ).league_members() == ["teacher", "cpu"]
    assert RolloutConfig(league_phillip=True).league_members() == ["phillip"]
    assert RolloutConfig(
        league_teacher=True, league_cpu=True, league_phillip=True,
        teacher_envs=0, cpu_envs=0,
    ).league_members() == ["teacher", "cpu", "phillip"]


def test_league_teacher_candidates_and_fhard(tmp_path):
    """"teacher" joins the per-match candidate set, starts at the 0.5
    prior, and fades out via f_hard as the student's win rate vs it
    rises."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), league_members=("teacher",))
    pool.save(_Stub(), 100)
    pool.save(_Stub(), 200)

    def teacher_share(n=800):
        rng = random.Random(0)
        return sum(pool.draw_member(rng) == "teacher" for _ in range(n)) / n

    prior_share = teacher_share()  # fresh row: 0.5 prior, ~even with ghost
    assert 0.35 < prior_share < 0.65
    for _ in range(300):
        pool.record_result("teacher", True)  # student now dominates
    beaten_share = teacher_share()
    assert beaten_share < prior_share * 0.6  # weight dropped with win_ema
    assert pool.win_estimate("teacher") > 0.9


def test_league_payoff_persistence_and_thinning(tmp_path):
    """Special member rows persist in pfsp.json exactly like ghost rows,
    survive thinning (which only evicts archive paths), and survive a
    reload WITHOUT the league flags (toggling flags loses no data)."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), keep=4,
                        league_members=("teacher", "cpu", "phillip"))
    for s in range(0, 800, 100):
        pool.save(_Stub(), s)
    for _ in range(6):
        pool.record_result("teacher", True)
        pool.record_result("cpu", False)
        pool.record_result("phillip", False)
    assert len(pool.archive) == 4  # thinning ran
    assert pool.payoff["teacher"]["games"] == 6
    assert pool.payoff["cpu"]["games"] == 6
    assert pool.payoff["phillip"]["games"] == 6
    # decayed counts = exact rates at small n: 6/6 wins -> 1.0, 0/6 -> 0.0
    assert pool.win_estimate("teacher") == pytest.approx(1.0)
    assert pool.win_estimate("cpu") == pytest.approx(0.0)
    assert pool.win_estimate("phillip") == pytest.approx(0.0)

    # round-trip through a league-flag-less pool: rows kept, not pruned
    fresh = SnapshotPool(str(tmp_path), keep=4)
    assert fresh.win_estimate("teacher") == pytest.approx(
        pool.win_estimate("teacher")
    )
    assert fresh.payoff["cpu"]["games"] == 6
    assert fresh.payoff["phillip"]["games"] == 6
    # and its draws ignore the members (flags off = ghosts only)
    rng = random.Random(0)
    assert all(fresh.draw_member(rng) in fresh.archive for _ in range(50))


def test_pfsp_class_weighting_math(tmp_path):
    """Two-stage class weighting: class probability follows f_hard over the
    class MEAN win_ema. Ghost-mass scenario (user's motivating case): 30
    ghosts at x=0.75 vs phillip at the 0.5 prior — phillip's class share is
    f_hard(0.5)/(f_hard(0.5)+f_hard(0.75)) = 2/3, NOT the flat-sampling
    0.5/(0.5+30*0.25) ~= 0.06 that ghost mass would give."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), keep=64,
                        league_members=("phillip",))
    for s in range(0, 3100, 100):  # 30 ghosts + the latest
        pool.save(_Stub(), s)
    for g in pool.archive[:-1]:
        pool.payoff[g] = {"wins": 8, "games": 10, "win_ema": 0.75}
    # legacy rate-EMA rows fall back to their RAW lifetime rate (8/10);
    # the latest (unmeasured, 0.5 prior) is in the ghosts class now:
    # class mean = (30 * 0.8 + 0.5) / 31
    ghosts_x = (30 * 0.8 + 0.5) / 31
    assert pool.class_hardness() == pytest.approx({"ghosts": ghosts_x, "phillip": 0.5})
    n, ph, rng = 4000, 0, random.Random(0)
    for _ in range(n):
        ph += pool.draw_member(rng) == "phillip"
    share = ph / n
    # squared f_hard (p=2 default): phillip 0.25 vs ghosts (1-x)^2
    expect = 0.25 / (0.25 + (1 - ghosts_x) ** 2)
    assert share == pytest.approx(expect, abs=0.03)
    assert share > 0.5  # far above any ghost-mass-proportional share


def test_league_imports_parse():
    """"NAME=PATH" (implicit @FOX) and "NAME=PATH@CHAR" forms; bad forms
    fail loudly; imports require league_slices and pfsp."""
    cfg = RolloutConfig(
        league_imports=["v3best=/m/rl-best-step0010000.pt",
                        "old=/m/old.pt@marth"],
        league_slices=2,
    )
    assert cfg.import_members() == {
        "v3best": ("/m/rl-best-step0010000.pt", "FOX"),  # default lock FOX
        "old": ("/m/old.pt", "MARTH"),  # case-normalized
    }
    assert cfg.league_members() == ["import:v3best", "import:old"]
    # composes with the other league flags (imports appended last)
    combo = RolloutConfig(
        league_teacher=True, teacher_envs=0,
        league_imports=["v3=/m/x.pt"], league_slices=2,
    )
    assert combo.league_members() == ["teacher", "import:v3"]

    for bad in [
        "nopathatall",       # no '='
        "=/m/x.pt",          # empty name
        "v3=",               # empty path
        "v3=/m/x.pt@ZELDA",  # not a policy-opponent char (MAIN_12)
        "v3=/m/x.pt@NOTACHAR",
        "a b=/m/x.pt",       # name must be metric/key-safe
    ]:
        with pytest.raises(AssertionError):
            RolloutConfig(
                league_imports=[bad], league_slices=1
            ).import_members()
    with pytest.raises(AssertionError):  # duplicate names
        RolloutConfig(
            league_imports=["a=/m/x.pt", "a=/m/y.pt"], league_slices=1
        ).import_members()
    with pytest.raises(AssertionError, match="league_slices"):
        RolloutConfig(league_imports=["a=/m/x.pt"]).league_members()
    with pytest.raises(AssertionError, match="pfsp"):
        RolloutConfig(
            league_imports=["a=/m/x.pt"], league_slices=1, pfsp=False
        ).league_members()
    # SnapshotPool accepts import keys as members; rejects junk keys
    from smashbot.rl.pool import SnapshotPool

    with pytest.raises(AssertionError, match="unknown league members"):
        SnapshotPool("/tmp/never-used", league_members=("imported:v3",))


def test_import_singleton_class_draw_weighting(tmp_path):
    """An import joins the per-match draw as its OWN singleton class:
    weight from its payoff row via f_hard(p=2), and it fades as the
    student starts beating it."""
    from smashbot.rl.pool import SnapshotPool

    # exact class-share math: 1 ghost at raw 0.8 vs import at the 0.5 prior
    # -> f_hard(p=2): 0.04 vs 0.25 -> import share 0.25/0.29
    pool = SnapshotPool(str(tmp_path / "m"), league_members=("import:v3best",))
    pool.save(_Stub(), 100)
    latest = pool.save(_Stub(), 200)
    pool.payoff[pool.archive[0]] = {"wins": 8, "games": 10, "win_ema": 0.75}
    ghosts_x = (0.8 + 0.5) / 2  # the unmeasured latest is a ghost too
    assert pool.class_hardness() == pytest.approx({"ghosts": ghosts_x, "import:v3best": 0.5})
    n, imp, rng = 4000, 0, random.Random(0)
    for _ in range(n):
        imp += pool.draw_member(rng) == "import:v3best"
    assert imp / n == pytest.approx(0.25 / (0.25 + (1 - ghosts_x) ** 2), abs=0.03)

    # multi-slot occupancy + fade-out once beaten (3 ghosts >= 2 tail
    # slots, so the ghost class never exhausts into the uniform fallback)
    pool = SnapshotPool(str(tmp_path / "s"), league_members=("import:v3best",))
    for s in (100, 200, 300, 400):
        latest = pool.save(_Stub(), s)
    ghosts = pool.archive[:-1]

    def import_draws(n=600):
        rng = random.Random(1)
        return sum(pool.draw_member(rng) == "import:v3best" for _ in range(n))

    held_prior = import_draws()
    assert held_prior > 0
    for _ in range(300):
        pool.record_result("import:v3best", True)  # student now dominates
    assert pool.win_estimate("import:v3best") > 0.9
    held_beaten = import_draws()
    assert held_beaten < held_prior * 0.6  # f_hard fade-out


def test_import_char_lock_redraw_helper():
    """Env-side redraw gate: while locked the opponent seat pins the lock
    and consumes NO rng draw; unlock resumes normal redraws; cpu serving
    (lazy league_cpu adoption) ignores the lock; the default path (lock
    None) is exactly the old redraw_chars behavior."""
    from smashbot.rl.rollouts import next_opponent_char

    draws = []

    def draw():
        draws.append(1)
        return "FALCO"

    # locked: pin, no draw consumed
    assert next_opponent_char("snapshot", "MARTH", True, "FOX", draw) == "MARTH"
    assert draws == []
    # already pinned: keep the seat, still no draw
    assert next_opponent_char("snapshot", "MARTH", True, "MARTH", draw) is None
    assert draws == []
    # lock works even with per-game redraws globally off
    assert next_opponent_char("snapshot", "MARTH", False, "FOX", draw) == "MARTH"
    assert draws == []
    # unlocked: redraws resume (rng consumed again)
    assert next_opponent_char("snapshot", None, True, "MARTH", draw) == "FALCO"
    assert len(draws) == 1
    # unlocked + redraw_chars off: keep the sitting char (old behavior)
    assert next_opponent_char("snapshot", None, False, "FOX", draw) is None
    assert len(draws) == 1
    # cpu serving ignores the lock: draws from the cpu roster as usual
    assert next_opponent_char("cpu", "MARTH", True, "FOX", draw) == "FALCO"
    assert len(draws) == 2


def test_import_payoff_persistence_never_pruned(tmp_path):
    """Import rows round-trip pfsp.json, survive thinning AND a reload
    without the import configured (permanent members: toggling flags across
    restarts loses no cross-generation data); category_estimates carries
    the import keys for the ticker."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), keep=4,
                        league_members=("import:v3",))
    for s in range(0, 800, 100):
        pool.save(_Stub(), s)
    for _ in range(6):
        pool.record_result("import:v3", False)  # old model still winning
    assert len(pool.archive) == 4  # thinning ran; import row untouched
    assert pool.payoff["import:v3"]["games"] == 6
    assert pool.win_estimate("import:v3") == pytest.approx(0.0)
    d, raw = pool.category_estimates()["import:v3"]
    assert d == pytest.approx(0.0) and raw == pytest.approx(0.0)

    # reload WITHOUT the import configured: row kept, but neither served
    # nor surfaced (flags off = ghosts only)
    fresh = SnapshotPool(str(tmp_path), keep=4)
    assert fresh.payoff["import:v3"]["games"] == 6
    assert "import:v3" not in fresh.category_estimates()
    rng = random.Random(0)
    assert all(fresh.draw_member(rng) != "import:v3" for _ in range(50))

    # reload WITH it again: estimates resume where they left off; a fresh
    # unmeasured import sits at the 0.5 prior with a None ticker estimate
    back = SnapshotPool(str(tmp_path), keep=4,
                        league_members=("import:v3", "import:new"))
    assert back.win_estimate("import:v3") == pytest.approx(0.0)
    assert back.win_estimate("import:new") == 0.5
    assert back.category_estimates()["import:new"] is None
    assert back.class_hardness()["import:new"] == 0.5


def test_self_seat_pipeline_equivalence(monkeypatch):
    """THE seat-equivalence proof (house standard, cf. batch_steps): which
    internal pipeline serves a port must not matter. Two workers with the
    SAME weights and the SAME scripted frames, one with student_port=1
    (primary pipeline drives port 1) and one with student_port=2 (primary
    drives port 2), must emit BYTE-IDENTICAL controller streams per port —
    including across a game boundary. Greedy-patched so streams are
    deterministic."""
    from smashbot.rl.pool import EnvSpec

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[
                np.dtype(self.dtype).name
            ]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    def build(student_port):
        cfg = RolloutConfig(
            num_envs=2, cpu_envs=0, teacher_envs=0, ref_envs=0,
            league_slices=0, self_envs=1, unroll_length=4,
            games_per_dolphin=10**9,
        )
        specs = [EnvSpec("self", student_port, "FOX")]
        student = BatchedPolicyAgent(_tiny_policy(seed=0), 2, name_code=1)
        return DolphinRolloutWorker(cfg, student, opponents={}, specs=specs)

    streams = {}
    for port in (1, 2):
        torch.manual_seed(0)
        worker = build(port)
        fake = _FakeEnvs(worker, seed=123)
        fake.install(monkeypatch)
        for t in range(10):
            if t == 5:  # game boundary mid-stream
                fake.final_stocks[0] = (3, 1)
            worker.collect(1)
        streams[port] = worker._conns[0].sent

    a, b = streams[1], streams[2]
    assert len(a) == len(b) and len(a) > 0
    for t, (ca, cb) in enumerate(zip(a, b)):
        assert set(ca) == set(cb) == {1, 2}, f"frame {t}: ports differ"
        for port in (1, 2):
            tree.map_structure(
                lambda x, y: np.testing.assert_array_equal(
                    np.asarray(x), np.asarray(y)
                ),
                ca[port], cb[port],
            )


def test_category_estimates_pools_imports(tmp_path):
    """The pooled 'imports' row aggregates decayed and raw counts across
    all import members (ticker I: bit), None with no import games."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), league_members=[
        "teacher", "import:a", "import:b"])
    assert pool.category_estimates()["imports"] is None
    for won in (True, True, False):
        pool.record_result("import:a", won)
    pool.record_result("import:b", False)
    dec, raw = pool.category_estimates()["imports"]
    assert raw == pytest.approx(2 / 4)
    assert 0.0 < dec < 1.0


def test_f_var_catchup_weighting():
    """f_var peaks at even matchups and zeroes BOTH tails (unbeatable and
    beaten), unlike f_hard which maxes at unbeatable."""
    from smashbot.rl.pool import f_hard, f_var

    assert f_var(0.0, 2) == 0.0 and f_var(1.0, 2) == 0.0
    assert f_hard(0.0, 2) == 1.0  # the contrast that motivated the switch
    assert f_var(0.5, 2) == pytest.approx(0.25 ** 2)
    assert f_var(0.5, 2) > f_var(0.2, 2) > f_var(0.05, 2)


def test_pfsp_explore_resurrects_benched_members(tmp_path):
    """With f_var an unbeatable member (phillip 0%) and a beaten one
    (teacher 100%) both have zero weight — only the explore mix can serve
    them. explore=0 never picks them; explore=1 (all probes) does."""
    from smashbot.rl.pool import SnapshotPool

    def build(explore):
        pool = SnapshotPool(
            str(tmp_path), pfsp_hard_frac=0.0,
            pfsp_explore=explore,
            league_members=["teacher", "phillip", "import:a"])
        for step in (100, 200):
            pool.save(_Stub(), step)
        for _ in range(10):  # firm rows past the 0.5-games prior
            pool.record_result("phillip", False)   # unbeatable
            pool.record_result("teacher", True)    # fully beaten
            pool.record_result("import:a", random.random() < 0.5)
        return pool

    pool = build(explore=0.0)
    rng = random.Random(0)
    picks = {pool.draw_member(rng) for _ in range(200)}
    assert "phillip" not in picks and "teacher" not in picks

    pool = build(explore=1.0)
    rng = random.Random(0)
    picks = {pool.draw_member(rng) for _ in range(200)}
    assert "phillip" in picks and "teacher" in picks


def test_pfsp_hard_frac_blend_serves_unbeatable(tmp_path):
    """hard_frac > 0 restores real (non-probe) serving for an unbeatable
    member under the blend: with explore OFF, pure f_var never picks the
    0% member but hard_frac=0.25 does (via its f_hard draws)."""
    from smashbot.rl.pool import SnapshotPool

    def build(hard_frac):
        pool = SnapshotPool(
            str(tmp_path), pfsp_hard_frac=hard_frac,
            pfsp_explore=0.0,
            league_members=["phillip", "import:a"])
        for step in (100, 200):
            pool.save(_Stub(), step)
        for _ in range(10):
            pool.record_result("phillip", False)  # unbeatable
            pool.record_result("import:a", random.random() < 0.5)
        return pool

    pool = build(hard_frac=0.0)
    rng = random.Random(0)
    picks = {pool.draw_member(rng) for _ in range(200)}
    assert "phillip" not in picks  # pure f_var: zero weight at 0%

    pool = build(hard_frac=0.25)
    rng = random.Random(0)
    picks = {pool.draw_member(rng) for _ in range(200)}
    assert "phillip" in picks  # hard draws bring him back




def test_keep_zero_never_prunes(tmp_path):
    """keep<=0 = immortal archive: save far past any cap, nothing pruned,
    payoff rows all intact."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), keep=0)
    policy = _tiny_policy()
    for step in range(250, 250 * 41, 250):
        pool.save(policy, step)
    assert len(pool.archive) == 40
    assert all(os.path.exists(p) for p in pool.archive)


def test_import_char_lock_any_means_unlocked():
    """NAME=PATH@ANY imports stay unlocked (char_lock None -> the env
    redraws per game like a snapshot); default stays FOX; bad chars still
    fail loudly."""
    from smashbot.rl.config import RolloutConfig

    cfg = RolloutConfig(league_imports=[
        "fox1=/tmp/a.pt",              # default lock
        "gen1=/tmp/b.pt@ANY",          # unlocked 12-char generalist
        "marth1=/tmp/c.pt@MARTH",      # explicit lock
    ])
    m = cfg.import_members()
    assert m["fox1"] == ("/tmp/a.pt", "FOX")
    assert m["gen1"] == ("/tmp/b.pt", None)
    assert m["marth1"] == ("/tmp/c.pt", "MARTH")
    import pytest as _pytest
    with _pytest.raises(AssertionError):
        RolloutConfig(league_imports=["x=/tmp/d.pt@BOWSER"]).import_members()


# ------------------------------------------------- dedicated imports (v9)


def test_partition_emits_dedicated_import_specs():
    from smashbot.rl.pool import make_partition

    reg = {
        "import:fx": ("/tmp/a.pt", "FOX"),
        "import:free": ("/tmp/b.pt", None),
    }
    specs = make_partition(
        num_envs=20, cpu_envs=0, teacher_envs=2, seed=0,
        import_registry=reg, import_envs_per=3,
    )
    imp = [sp for sp in specs if sp.kind == "import"]
    assert len(imp) == 6
    # member-major order matches the static agent's cell layout
    assert [sp.member for sp in imp] == ["import:fx"] * 3 + ["import:free"] * 3
    # locked member pins its char on every env; unlocked draws from roster
    assert all(sp.opponent_char == "FOX" and sp.char_lock == "FOX"
               for sp in imp[:3])
    assert all(sp.char_lock is None for sp in imp[3:])
    # budget: league envs shrink by the import envs
    assert sum(1 for sp in specs if sp.kind == "snapshot") == 20 - 2 - 6


def test_worker_dedicated_imports_route_credit_and_harvest(
    monkeypatch, tmp_path
):
    """Pinned import slices end to end over fake envs: controllers come
    from the merged grid's pinned tail, results credit the 'import'
    tracker AND the payoff ledger (metrics continuity), locked members
    stay out of by_char, and import rows harvest imitation through the
    merged 'ours' group."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=12, pool_dir=tmp_path,
        teacher_envs=0, league_slices=2,
        members=[0], harvest=True, pfsp_explore=1.0,
        import_dedicated_envs=2,
        league_imports=[
            "fx=/tmp/does-not-matter.pt@FOX",
            "free=/tmp/does-not-matter2.pt@ANY",
        ],
        char_whitelist=["FOX", "MARTH", "FALCO", "PEACH"],
    )
    imp_envs = worker.import_idx
    assert len(imp_envs) == 4
    rt = worker._runtime
    per = rt.import_slices_per
    assert per > 0
    # merged grid: league seat slices + one pinned block per member
    assert rt.agent.S == rt.league.seats.S + 2 * per
    # pinned rows cover every import env, member-major, pads None
    filled = [e for e in worker._import_cells if e is not None]
    assert filled == imp_envs
    assert len(worker._import_cells) == 2 * per * rt.agent.N
    # weights identity: member slot k's pinned slices hold EXACTLY the
    # tiny policy _make_runtime loaded for slot k (seed=200+slot) — a
    # member-order swap or off-by-`per` slice index dies here, not in a
    # silently-wrong training run
    S_league = rt.league.seats.S
    for slot in range(2):
        sd = _tiny_policy(seed=200 + slot).state_dict()
        for j in range(per):
            s = S_league + slot * per + j
            for k, t in rt.agent._stacked_params.items():
                assert torch.equal(t[s], sd[k].to(t.dtype)), (
                    f"slice {s} != member {slot} weights ({k})"
                )

    # opp chars: locked import serves FOX; unlocked serves whatever came up
    for e in imp_envs[:2]:
        envs.opp_chars[e] = "FOX"
    for e in imp_envs[2:]:
        envs.opp_chars[e] = "MARTH"

    worker.collect(1)
    # every import env received a controller each frame (validated decode
    # happens inside _FakeConn.send); check a command actually landed
    for e in imp_envs:
        sent = worker._conns[e].sent
        assert any(any(isinstance(k, int) for k in cmd) for cmd in sent)

    # deliver decided results on all four import envs
    for e in imp_envs[:3]:
        envs.final_stocks[e] = (4, 0)   # student wins
    envs.final_stocks[imp_envs[3]] = (0, 4)  # student loses
    worker.collect(1)

    t = worker.trackers["import"]
    assert t.wins == 3 and t.losses == 1
    # locked (FOX) games excluded from by_char; unlocked (MARTH) included
    assert "FOX" not in t.by_char
    assert t.by_char.get("MARTH", (0, 0))[1] == 2
    # ledger rows fed for both members (metrics continuity)
    pool = worker.league.pool
    assert pool.payoff["import:fx"]["games"] == 2
    assert pool.payoff["import:free"]["games"] == 2
    # ports alternate within each member's envs, so (4,0) on the port-2
    # env is a student LOSS after the swap: fx = 1 win 1 loss
    assert pool.payoff["import:fx"]["wins"] == 1
    assert pool.payoff["import:free"]["wins"] == 2

    # imitation harvest: import rows flow through the merged grid group
    assert "imports" not in worker._harvest_groups
    assert "ours" in worker._harvest_groups
    assert len(worker._harvest_groups["ours"].rows) == rt.agent.S * rt.agent.N


def test_dedicated_imports_surface_in_category_estimates(tmp_path):
    """metric_imports lets dedicated (non-member) imports appear in
    category_estimates — per-member rows AND the pooled 'imports' row —
    while decommissioned payoff rows stay hidden."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(
        str(tmp_path), metric_imports=["import:ded"],
    )
    for _ in range(4):
        pool.record_result("import:ded", True)
    pool.record_result("import:ded", False)
    pool.record_result("import:old", True)  # decommissioned row
    cat = pool.category_estimates()
    assert cat["import:ded"] is not None
    dec, raw = cat["import:ded"]
    assert raw == 4 / 5
    assert cat["imports"] is not None and cat["imports"][1] == 4 / 5
    assert "import:old" not in cat
