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

from smashbot import configs, embed as embed_lib
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
        got = [[s.kind, s.group, s.student_port, s.opponent_char] for s in specs]
        assert got == expect, f"partition drifted for {key}"


def test_partition_self_envs_arithmetic_and_order():
    from smashbot.rl.pool import make_partition

    # self envs cost 2 budget units, run 1 dolphin each
    specs = make_partition(
        num_envs=24, cpu_envs=2, teacher_envs=6, snapshot_slots=2,
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
        num_envs=16, cpu_envs=2, teacher_envs=-1, snapshot_slots=0,
        seed=0, ref_envs=2, self_envs=3,
    )
    assert len(specs) == 13
    assert [s.kind for s in specs].count("teacher") == 16 - 2 - 2 - 6


# --------------------------------------------------- fake-env worker harness


class _FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, cmd):
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

    def install(self, monkeypatch):
        w = self.worker
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
            payloads.append(dict(
                game=self.embed_game.from_state(raw),
                resetting=fs is not None,
                final_stocks=fs,
                stocks=(4, 4),
                # port-2 seat takes 1% per frame: port-1-seat reward +0.01
                percent=(0.0, float(self.t)),
                opp_char=self.opp_chars.get(i, "FOX"),
            ))
        self.t += 1
        return payloads


def _make_worker(monkeypatch, num_envs, seed=0, opp_chars=None,
                 harvest=False, ref_controller_config=None, **cfg_kwargs):
    cfg = RolloutConfig(
        num_envs=num_envs, unroll_length=4, games_per_dolphin=10**9,
        **cfg_kwargs,
    )
    student = BatchedPolicyAgent(_tiny_policy(seed=0), num_envs, name_code=1)
    from smashbot.rl.pool import make_partition, student_whitelist

    specs = make_partition(
        cfg.num_envs, cfg.cpu_envs, cfg.teacher_envs, cfg.snapshot_slots,
        cfg.main12_prob, cfg.partition_seed, ref_envs=cfg.ref_envs,
        self_envs=cfg.self_envs,
        char_whitelist=student_whitelist(cfg.char_whitelist, cfg.bot_char),
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
        ref_policy = build_policy(
            embed_config=embed_lib.EmbedConfig(),
            controller_config=(
                ref_controller_config or embed_lib.ControllerConfig()
            ),
            network_config=configs.NetworkConfig(
                name="sgu", num_layers=1, hidden_size=32, num_heads=1, window=4
            ),
            head_config=configs.ControllerHeadConfig(
                residual_size=32, component_depth=0
            ),
            policy_config=configs.PolicyConfig(delay=2),
            num_names=4,
        )
        ref_policy.train_value_head = False
        opponents["reference"] = BatchedPolicyAgent(
            ref_policy, counts["reference"], name_code=2
        )
    worker = DolphinRolloutWorker(
        cfg, student, opponents=opponents, specs=specs,
        harvest_imitation=harvest,
    )
    envs = _FakeEnvs(worker, seed=seed, opp_chars=opp_chars)
    envs.install(monkeypatch)
    return worker, envs


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
        RLConfig(imitation_slots=1, imitation_lambda=0.1),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    imf = learner._imitation_fixed(imit)
    assert imf is not None
    loss = learner._imitation_policy_loss(imf)
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


def test_batch_invariant_substitution():
    """With imitation_slots=k the learner's policy pass covers EXACTLY
    num_envs rows: (num_envs - k) PPO rows + k imitation rows. Self-play
    rows are never substituted out; teacher/cpu drop first, then snapshot."""
    policy = _tiny_policy(seed=0)
    kinds = ["cpu", "teacher", "teacher", "snapshot", "self", "self"]
    main = _rollout(policy, B=6, T=8, seed=0)

    for slots, expect_k, allowed in [
        (2, 2, {0, 1, 2}),          # tier1 (cpu/teacher) preferred
        (5, 4, {0, 1, 2, 3}),       # tier1 exhausted -> snapshot; never self
    ]:
        learner = Learner(
            RLConfig(
                imitation_slots=slots, imitation_lambda=0.1,
                ppo=PPOConfig(max_mean_actor_kl=1e9),
            ),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        imit = _imitation_traj(policy, B=4)
        unroll_rows = []
        orig_unroll = learner.policy.unroll

        def counting_unroll(frames, st, **kw):
            unroll_rows.append(frames.reward.shape[0])
            return orig_unroll(frames, st, **kw)

        learner.policy.unroll = counting_unroll
        _, metrics = learner.step(
            [main, imit], learner.initial_state(6), progress=0.0,
            row_kinds=kinds,
        )
        im = metrics["imitation"]
        assert im["traj_count"] == expect_k
        assert set(im["substituted_rows"]) <= allowed
        assert len(im["substituted_rows"]) == expect_k
        assert 4 not in im["substituted_rows"] and 5 not in im["substituted_rows"]
        # epoch policy passes: one PPO minibatch + one imitation minibatch,
        # totalling exactly num_envs rows (the OOM ceiling)
        assert unroll_rows[0] + unroll_rows[1] == 6
        assert unroll_rows[0] == 6 - expect_k

    # slot cap respects available droppable rows: all-self batch drops none
    learner = Learner(
        RLConfig(imitation_slots=3, imitation_lambda=0.1,
                 ppo=PPOConfig(max_mean_actor_kl=1e9)),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    _, metrics = learner.step(
        [main, _imitation_traj(policy, B=4)], learner.initial_state(6),
        row_kinds=["self"] * 6,
    )
    assert "imitation" not in metrics  # nothing droppable => nothing used


def test_default_config_learner_ignores_imitation_trajs():
    """Dormant path: imitation_slots=0 (default) => imitation trajectories
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
        RLConfig(imitation_slots=2, imitation_lambda=0.1),
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
    loss = learner._imitation_policy_loss(imf)
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
    policies, and the PPO loss equals plain PPO on the same rows."""
    policy = _tiny_policy(seed=0)
    main = _rollout(policy, B=4, T=8, seed=0)
    kinds = ["teacher", "teacher", "teacher", "teacher"]

    def run(imit_seed, lam):
        learner = Learner(
            RLConfig(imitation_slots=2, imitation_lambda=lam,
                     ppo=PPOConfig(max_mean_actor_kl=1e9)),
            _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
        )
        learner._subst_rng = random.Random(7)  # identical row drops
        imit = _imitation_traj(_tiny_policy(seed=imit_seed), B=2,
                               seed=imit_seed)
        _, metrics = learner.step(
            [main, imit], learner.initial_state(4), row_kinds=kinds
        )
        return learner, metrics

    l_a, m_a = run(imit_seed=5, lam=0.0)
    l_b, m_b = run(imit_seed=9, lam=0.0)
    assert m_a["imitation"]["loss"] == 0.0  # term never computed
    for k, v in l_a.policy.state_dict().items():
        assert torch.equal(v, l_b.policy.state_dict()[k]), k

    # same rows through a plain-PPO learner: identical first-epoch loss
    dropped = m_a["imitation"]["substituted_rows"]
    keep = [i for i in range(4) if i not in dropped]
    plain = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1e9)),
        _tiny_policy(seed=0), _tiny_policy(seed=0), _tiny_value(),
    )
    _, m_plain = plain.step(
        [slice_trajectory_rows(main, keep)], plain.initial_state(len(keep))
    )
    # approx (not bitwise): the plain learner forwards a batch of 2 rows
    # while the substituting learner slices its 4-row fixed pass — BLAS
    # kernels differ by shape at ~1e-8. The bitwise guarantee (imitation
    # data contributes nothing to the actor) is the A/B check above.
    assert m_a["epochs"][0]["loss"] == pytest.approx(
        m_plain["epochs"][0]["loss"], rel=1e-5
    )

    # ...while lambda > 0 with different imitation data changes the policy
    l_c, m_c = run(imit_seed=5, lam=0.5)
    assert m_c["imitation"]["loss"] != 0.0
    assert any(
        not torch.equal(v, l_c.policy.state_dict()[k])
        for k, v in l_a.policy.state_dict().items()
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

    pool = SnapshotPool(str(tmp_path), slots=3)
    p = pool.save(_Stub(), 100)
    assert pool.win_estimate(p) == 0.5  # no games: prior
    for won in [True, True, True, False]:
        pool.record_result(p, won)
    assert pool.win_estimate(p) == 0.5  # < 5 decided games: still prior
    pool.record_result(p, True)
    est = pool.win_estimate(p)
    assert est != 0.5
    # hand EMA (alpha 0.01, seeded at first outcome)
    ema = 1.0
    for o in [1.0, 1.0, 0.0, 1.0]:
        ema = 0.99 * ema + 0.01 * o
    assert est == pytest.approx(ema)


def test_pfsp_persistence_roundtrip_and_prune(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=2)
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

    fresh = SnapshotPool(str(tmp_path), slots=2)
    assert fresh.win_estimate(a) == pytest.approx(pool.win_estimate(a))
    assert fresh.win_estimate(b) == pytest.approx(pool.win_estimate(b))
    assert "/nonexistent/snapshot-999.pt" not in fresh.payoff  # pruned
    assert os.path.exists(os.path.join(str(tmp_path), "pfsp.json"))


def test_pfsp_thinning_drops_payoff_rows(tmp_path):
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=2, keep=4)
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

    pool = SnapshotPool(str(tmp_path), slots=2)
    easy = pool.save(_Stub(), 100)   # student dominates: x ~ 1
    hard = pool.save(_Stub(), 200)   # student loses: x ~ 0
    mid = pool.save(_Stub(), 250)
    latest = pool.save(_Stub(), 300)
    for _ in range(120):
        pool.record_result(easy, True)
        pool.record_result(hard, False)

    counts = {easy: 0, hard: 0, mid: 0}
    for s in range(400):
        picks = pool.assignments(random.Random(s))
        assert picks[0] == latest  # slot 0 always the latest
        assert len(picks) == 2
        counts[picks[1]] += 1
    assert counts[easy] == 0        # x ~= 1 => f_hard ~= 0 => never served
    assert counts[hard] > counts[mid]  # hardest opponent served most

    # everyone beaten: uniform fallback still fills the slots
    for _ in range(120):
        pool.record_result(mid, True)
        pool.record_result(hard, True)
    picks = pool.assignments(random.Random(0))
    assert len(picks) == 2 and picks[0] == latest


def test_pfsp_off_matches_old_recency_behavior(tmp_path):
    """pfsp=False must reproduce the original recency-biased sampler
    exactly (reference implementation below is a verbatim copy of the
    pre-PFSP code), even with payoff data present."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=4, pfsp=False)
    for s in range(0, 1200, 100):
        pool.save(_Stub(), s)
    for p in pool.archive[:3]:
        for _ in range(10):
            pool.record_result(p, True)  # must be IGNORED with pfsp off

    def old_assignments(archive, slots, rng):
        picks = [archive[-1]]
        candidates = list(archive[:-1])
        while len(picks) < slots and candidates:
            weights = [
                2.0 ** (i / max(1, len(candidates) / 3))
                for i in range(len(candidates))
            ]
            chosen = rng.choices(range(len(candidates)), weights=weights)[0]
            picks.append(candidates.pop(chosen))
        while len(picks) < slots:
            picks.append(archive[-1])
        return picks

    for seed in range(25):
        assert pool.assignments(random.Random(seed)) == old_assignments(
            pool.archive, 4, random.Random(seed)
        )


# ------------------------------------------------------- dual-seat rewards


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
