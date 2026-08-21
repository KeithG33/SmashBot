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
        # dolphin -> "cpu": env-reported ACTUAL serving (league_cpu lazy
        # adoption); results delivered while flipped report that serving
        self.serving = {}

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
            serving = self.serving.get(i, "policy")
            payloads.append(dict(
                game=self.embed_game.from_state(raw),
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
    slot_counts = {}
    for sp in specs:
        counts[sp.kind] = counts.get(sp.kind, 0) + 1
        if sp.kind == "snapshot":
            slot_counts[sp.group] = slot_counts.get(sp.group, 0) + 1
    for g, n in slot_counts.items():
        opponents[("slot", g)] = BatchedPolicyAgent(
            _tiny_policy(seed=3 + g), n, name_code=1
        )
    if counts.get("teacher"):
        opponents["teacher"] = BatchedPolicyAgent(
            _tiny_policy(seed=1), counts["teacher"], name_code=1
        )
    if counts.get("reference"):
        ref_policy = _phillip_like_policy(ref_controller_config)
        opponents["reference"] = BatchedPolicyAgent(
            ref_policy, counts["reference"], name_code=2
        )
    worker = DolphinRolloutWorker(
        cfg, student, opponents=opponents, specs=specs,
        harvest_imitation=harvest,
    )
    if cfg.league_phillip:
        # a differently-discretized policy, like the real Phillip module
        # (exercises the harvest re-encode path); ONE module, agent rebuilt
        # per occupancy — mirrors train_rl's factory
        ph_policy = _phillip_like_policy(
            embed_lib.ControllerConfig(axis_spacing=8)
        )
        worker.phillip_factory = lambda n: BatchedPolicyAgent(
            ph_policy, n, name_code=2
        )
    envs = _FakeEnvs(worker, seed=seed, opp_chars=opp_chars)
    envs.install(monkeypatch)
    return worker, envs


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
    # hand decayed counts (0.99 decay): ~= exact mean at small n
    wd = gd = 0.0
    for o in [1.0, 1.0, 1.0, 0.0, 1.0]:
        wd = 0.99 * wd + o
        gd = 0.99 * gd + 1.0
    assert est == pytest.approx(wd / gd)


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
    # prior-seeded EMA after 120 straight wins: x ~ 0.85, so easy is
    # strongly suppressed but no longer EXACTLY zero-weight
    assert counts[easy] < counts[mid] < counts[hard]
    assert counts[easy] < 0.15 * 400
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

    pool_a = SnapshotPool(str(tmp_path / "a"), slots=3)
    pool_b = SnapshotPool(str(tmp_path / "b"), slots=3, league_members=())
    for s in range(0, 600, 100):
        pool_a.save(_Stub(), s)
        pool_b.save(_Stub(), s)
        pool_a.record_result(pool_a.archive[-1], s % 200 == 0)
        pool_b.record_result(pool_b.archive[-1], s % 200 == 0)
    for seed in range(25):
        a = pool_a.assignments(random.Random(seed))
        b = pool_b.assignments(random.Random(seed))
        assert [os.path.basename(p) for p in a] == [
            os.path.basename(p) for p in b
        ]


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
        SnapshotPool("/tmp/never-used", slots=2, pfsp=False,
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
    """"teacher" joins the candidate set for non-latest slots, starts at the
    0.5 prior, and fades out via f_hard as the student's win_ema vs it
    rises. Slot 0 stays the latest snapshot always."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=2, league_members=("teacher",))
    latest = pool.save(_Stub(), 100)
    # archive of one: the only non-latest candidate is the teacher
    assert pool.assignments(random.Random(0)) == [latest, "teacher"]

    pool.save(_Stub(), 200)  # a second snapshot: teacher vs ghost
    latest = pool.archive[-1]

    def teacher_share(n=400):
        c = 0
        for s in range(n):
            picks = pool.assignments(random.Random(s))
            assert picks[0] == latest  # slot 0 ALWAYS the latest
            c += picks[1] == "teacher"
        return c / n

    prior_share = teacher_share()  # fresh row: 0.5 prior, ~even with ghost
    assert 0.35 < prior_share < 0.65
    for _ in range(300):
        pool.record_result("teacher", True)  # student now dominates
    beaten_share = teacher_share()
    assert beaten_share < prior_share * 0.6  # weight dropped with win_ema
    assert pool.win_estimate("teacher") > 0.9


def test_league_singletons_repeat_ghosts_dont(tmp_path):
    """Class-weighted sampling semantics: a singleton member class CAN hold
    multiple slots per epoch; a ghost serves at most one (stage-2 without
    replacement); slot 0 stays the (single) latest snapshot."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(
        str(tmp_path), slots=5, league_members=("teacher", "cpu")
    )
    latest = None
    for s in (100, 200):
        latest = pool.save(_Stub(), s)
    ghost = pool.archive[0]
    saw_multi_singleton = False
    for seed in range(100):
        picks = pool.assignments(random.Random(seed))
        assert len(picks) == 5 and picks[0] == latest
        assert picks.count(latest) == 1  # classes fill every slot: no pads
        assert picks.count(ghost) <= 1  # without replacement within ghosts
        saw_multi_singleton |= (
            picks.count("teacher") > 1 or picks.count("cpu") > 1
        )
    assert saw_multi_singleton  # singletons may hold several slots at once


def test_apply_assignments_teacher_copy_and_cpu_lazy(tmp_path):
    """Slot refresh routing: a "teacher" assignment copies the LIVE teacher
    module's weights (state_dict copy — later teacher mutations must NOT
    propagate until the next refresh); "cpu" only records the desired kind
    and leaves attribution on the previous member (lazy adoption)."""
    from smashbot.rl.pool import apply_assignments

    class _Worker:
        """Records begin_transition announcements (the worker's deferred
        adoption bookkeeping is exercised by the collect-loop tests)."""
        def __init__(self):
            self.slot_desired = {}
            self.slot_char_lock = {}
            self.announced = []

        def begin_transition(self, slot, key, lock, slot_policy=None):
            # the announcement must see the PREVIOUS weights: snapshot them
            self.announced.append((
                slot, key, lock,
                None if slot_policy is None
                else slot_policy.weight.detach().clone(),
            ))

    torch.manual_seed(0)
    teacher = torch.nn.Linear(3, 2)
    slot0, slot1 = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    init1 = slot1.weight.detach().clone()
    ghost = torch.nn.Linear(3, 2)
    snap = str(tmp_path / "snapshot-0000100.pt")
    torch.save(ghost.state_dict(), snap)

    w, keys = _Worker(), {}
    apply_assignments([snap, "teacher"], [(0, slot0), (1, slot1)],
                      teacher, w, keys)
    torch.testing.assert_close(slot0.weight, ghost.weight)
    torch.testing.assert_close(slot1.weight, teacher.weight)
    assert keys == {0: snap, 1: "teacher"}
    assert w.slot_desired == {0: "policy", 1: "policy"}
    # non-import members never set a char lock
    assert w.slot_char_lock == {0: None, 1: None}
    # announced BEFORE the load: slot1 still held its init weights
    assert [(a[0], a[1], a[2]) for a in w.announced] == [
        (0, snap, None), (1, "teacher", None)]
    torch.testing.assert_close(w.announced[1][3], init1)

    # the copy is a snapshot of the live module, not a reference: a teacher
    # hot-swap mid-epoch leaves the serving slot on its copy until refresh
    with torch.no_grad():
        teacher.weight.add_(1.0)
    assert not torch.equal(slot1.weight, teacher.weight)

    # "cpu": desired flips; the slot policy is left alone (envs adopt cpu
    # at their recycle; until then they play the parked weights)
    frozen = {k: v.detach().clone() for k, v in slot1.state_dict().items()}
    apply_assignments([snap, "cpu"], [(1, slot1)], teacher, w, keys)
    assert w.slot_desired[1] == "cpu" and keys[1] == "cpu"
    for k, v in slot1.state_dict().items():
        assert torch.equal(v, frozen[k])

    # "phillip": routing only — the slot policy module is NEVER touched
    # (his architecture differs); desired returns to "policy"
    apply_assignments([snap, "phillip"], [(1, slot1)], teacher, w, keys)
    assert keys[1] == "phillip"
    assert w.slot_desired[1] == "policy"
    for k, v in slot1.state_dict().items():
        assert torch.equal(v, frozen[k])

    # short assignment list (early training): out-of-range slots untouched
    before = (w.slot_desired[1], keys[1], len(w.announced))
    apply_assignments([snap], [(0, slot0), (1, slot1)], teacher, w, keys)
    assert (w.slot_desired[1], keys[1], len(w.announced) - 1) == before


def test_league_payoff_persistence_and_thinning(tmp_path):
    """Special member rows persist in pfsp.json exactly like ghost rows,
    survive thinning (which only evicts archive paths), and survive a
    reload WITHOUT the league flags (toggling flags loses no data)."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=2, keep=4,
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
    fresh = SnapshotPool(str(tmp_path), slots=2, keep=4)
    assert fresh.win_estimate("teacher") == pytest.approx(
        pool.win_estimate("teacher")
    )
    assert fresh.payoff["cpu"]["games"] == 6
    assert fresh.payoff["phillip"]["games"] == 6
    # and its assignments ignore the members (flags off = ghosts only)
    for seed in range(25):
        picks = fresh.assignments(random.Random(seed))
        assert "teacher" not in picks and "phillip" not in picks


def test_league_cpu_lazy_adoption_worker(monkeypatch):
    """Worker-level league_cpu mechanics: the desired kind is piggybacked on
    the command dicts, envs adopt only when THEY report cpu serving,
    attribution (trackers + on_snapshot_game) follows actual serving before
    and after, and a fully-cpu slot is excluded from opponent inference
    without breaking the collect() row bookkeeping."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=1,
        cpu_envs=0, league_cpu=True,
    )
    slot_envs = [i for i, sp in enumerate(worker.specs)
                 if sp.kind == "snapshot"]
    assert slot_envs == [2, 3]
    calls = []
    worker.on_snapshot_game = lambda key, w: calls.append((key, w))
    slot_policy = worker.opponents[("slot", 0)].policy

    def cmds(i):
        got = worker._conns[i].sent[-1]
        return got

    # phase 1: boot assignment (adopted immediately — nothing in flight);
    # desired=policy: slot envs get both seats' inputs plus the opp_kind
    # marker; teacher envs are untouched
    worker.begin_transition(0, "ghostA", None, slot_policy)
    assert {worker.env_member[i] for i in slot_envs} == {"ghostA"}
    envs.final_stocks[2] = (4, 0)  # port1 (student) wins on a slot env
    worker.collect(1)
    for i in slot_envs:
        port = worker.specs[i].student_port
        assert set(cmds(i)) == {port, 3 - port, "opp_kind"}
        assert cmds(i)["opp_kind"] == "policy"
    for i in (0, 1):
        port = worker.specs[i].student_port
        assert set(cmds(i)) == {port, 3 - port}
    assert worker.trackers["snapshot"].wins == 1
    assert calls == [("ghostA", True)]

    # phase 1b: the slot moves to live-teacher weights. No spare-brain
    # factory in this harness -> legacy instant swap; games log under the
    # teacher kind for ticker/wandb continuity and credit "teacher"
    worker.begin_transition(0, "teacher", None, slot_policy)
    assert {worker.env_member[i] for i in slot_envs} == {"teacher"}
    envs.final_stocks[3] = (0, 4)
    worker.collect(1)
    tracked = (worker.trackers["teacher"].wins,
               worker.trackers["teacher"].losses)
    assert tracked == ((1, 0) if worker.specs[3].student_port == 2
                       else (0, 1))
    assert calls[-1][0] == "teacher"
    worker.begin_transition(0, "ghostA", None, slot_policy)

    # phase 2: refresh desires cpu — envs have NOT adopted yet: inputs still
    # flow to the opponent seat FROM THE SLOT POLICY IN PLACE (cpu never
    # overwrites it), results still attribute to the snapshot
    worker.begin_transition(0, "cpu", None, slot_policy)
    worker.slot_desired[0] = "cpu"
    assert worker.slot_inplace_key[0] == "ghostA"
    assert worker.slot_pending[0] == set(slot_envs)
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    for i in slot_envs:
        port = worker.specs[i].student_port
        assert set(cmds(i)) == {port, 3 - port, "opp_kind"}
        assert cmds(i)["opp_kind"] == "cpu"
    assert worker.trackers["snapshot"].wins == 2
    assert worker.trackers["cpu"].wins == 0
    assert calls[-1] == ("ghostA", True)

    # phase 3: env 2 adopts at its recycle; env 3 hasn't — mixed slot still
    # runs the (full-batch) policy, but the cpu env gets no opponent input
    envs.serving[2] = "cpu"
    envs.final_stocks[2] = (4, 0)  # port-1 student wins
    envs.final_stocks[3] = (0, 4)  # env 3 seats the student on port 2
    worker.collect(1)
    p2 = worker.specs[2].student_port
    assert set(cmds(2)) == {p2, "opp_kind"}
    p3 = worker.specs[3].student_port
    assert set(cmds(3)) == {p3, 3 - p3, "opp_kind"}
    assert worker.trackers["cpu"].wins == 1  # env 2's game: actual cpu
    assert worker.trackers["snapshot"].wins == 3  # env 3: still policy
    assert {calls[-1], calls[-2]} == {("cpu", True), ("ghostA", True)}
    # env 2 adopted cpu at its boundary; env 3 is still pending on ghostA
    assert worker.env_member[2] == "cpu" and worker.env_member[3] == "ghostA"
    assert worker.slot_pending[0] == {3}

    # phase 4: whole slot serving cpu: opponent inference skipped for the
    # slot, and collect() still yields well-formed full-budget trajectories
    envs.serving[3] = "cpu"
    step_calls = []
    slot_agent = worker.opponents[("slot", 0)]
    orig_step = slot_agent.step
    slot_agent.step = lambda *a, **k: (
        step_calls.append(1) or orig_step(*a, **k)
    )
    trajs = worker.collect(2)
    assert not step_calls  # no brain to run
    assert len(trajs) == 2
    for t in trajs:
        assert t.rewards.shape[0] == 4  # full learner-row budget
        assert torch.isfinite(t.rewards).all()
        for leaf in tree.flatten(t.actions.logits):
            if leaf.is_floating_point():
                assert torch.isfinite(leaf).all()
    for i in slot_envs:
        port = worker.specs[i].student_port
        assert set(cmds(i)) == {port, "opp_kind"}
    # teacher group still ran and got inputs
    for i in (0, 1):
        port = worker.specs[i].student_port
        assert set(cmds(i)) == {port, 3 - port}


def test_league_composed_with_self_play(monkeypatch):
    """league_teacher + league_cpu + self_envs + ref_envs compose: partition
    arithmetic holds (rows == num_envs, dolphins == num_envs - self_envs)
    and collect() runs clean with slots serving cpu."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=12, cpu_envs=0, teacher_envs=0,
        snapshot_slots=2, ref_envs=2, self_envs=2,
        league_teacher=True, league_cpu=True,
        char_whitelist=["FOX", "FALCO"],
    )
    assert worker.num_dolphins == 10 and worker.num_rows == 12
    kinds = [sp.kind for sp in worker.specs]
    assert kinds.count("snapshot") == 6 and kinds.count("teacher") == 0
    assert kinds.count("cpu") == 0
    assert worker.row_kinds.count("self") == 4  # both seats of 2 dolphins

    # one slot flips to cpu mid-run; self-play rows keep their mirror
    worker.slot_desired[0] = "cpu"
    for i, sp in enumerate(worker.specs):
        if sp.kind == "snapshot" and sp.group == 0:
            envs.serving[i] = "cpu"
    (traj,) = worker.collect(1)
    assert traj.rewards.shape[0] == 12
    assert torch.isfinite(traj.rewards).all()
    for d in worker.self_idx:
        torch.testing.assert_close(
            traj.rewards[d], -traj.rewards[worker._self_row_of[d]]
        )


# ------------------------- phillip league member + class-weighted sampling


def test_pfsp_class_weighting_math(tmp_path):
    """Two-stage class weighting: class probability follows f_hard over the
    class MEAN win_ema. Ghost-mass scenario (user's motivating case): 30
    ghosts at x=0.75 vs phillip at the 0.5 prior — phillip's class share is
    f_hard(0.5)/(f_hard(0.5)+f_hard(0.75)) = 2/3, NOT the flat-sampling
    0.5/(0.5+30*0.25) ~= 0.06 that ghost mass would give."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=2, keep=64,
                        league_members=("phillip",))
    for s in range(0, 3100, 100):  # 30 ghosts + the latest
        pool.save(_Stub(), s)
    for g in pool.archive[:-1]:
        pool.payoff[g] = {"wins": 8, "games": 10, "win_ema": 0.75}
    # legacy rate-EMA rows fall back to their RAW lifetime rate (8/10)
    assert pool.class_hardness() == {"ghosts": 0.8, "phillip": 0.5}

    latest = pool.archive[-1]
    n, ph = 4000, 0
    for s in range(n):
        picks = pool.assignments(random.Random(s))
        assert picks[0] == latest and len(picks) == 2
        ph += picks[1] == "phillip"
    share = ph / n
    # squared f_hard (p=2 default): phillip 0.5^2=0.25 vs ghosts-at-RAW-0.8
    # 0.2^2=0.04 -> share 0.25/0.29
    assert share == pytest.approx(0.25 / 0.29, abs=0.03)
    assert share > 0.5  # far above any ghost-mass-proportional share


def test_pfsp_class_sampler_ghost_stage2(tmp_path):
    """Stage 2 within the ghosts class keeps the existing per-ghost f_hard
    (harder ghosts serve more) and without-replacement across slots (a
    ghost holds at most one slot; singletons may repeat)."""
    from smashbot.rl.pool import SnapshotPool

    pool = SnapshotPool(str(tmp_path), slots=4, keep=10,
                        league_members=("teacher",))
    hard = pool.save(_Stub(), 100)
    easy = pool.save(_Stub(), 200)
    latest = pool.save(_Stub(), 300)
    pool.payoff[hard] = {"wins": 1, "games": 10, "win_ema": 0.1}
    pool.payoff[easy] = {"wins": 9, "games": 10, "win_ema": 0.9}

    hard_epochs = easy_epochs = 0
    teacher_multi = False
    for s in range(500):
        picks = pool.assignments(random.Random(s))
        assert picks[0] == latest
        tail = picks[1:]
        assert tail.count(hard) <= 1 and tail.count(easy) <= 1
        teacher_multi |= tail.count("teacher") > 1
        hard_epochs += hard in tail
        easy_epochs += easy in tail
    assert teacher_multi  # singleton class held several slots at once
    assert hard_epochs > easy_epochs  # per-ghost f_hard preserved


def test_league_phillip_routing_and_multislot(monkeypatch):
    """A slot assigned phillip routes its rows to Phillip's own agent (the
    slot policy idles); occupancy can span multiple slots (agent rebuilt to
    the summed row count); rows still yield full-budget student-seat PPO
    trajectories; games log under tracker kind "reference" and pay off to
    the "phillip" key."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=6, teacher_envs=2, snapshot_slots=2,
        league_phillip=True,
    )
    slot_of = {i: sp.group for i, sp in enumerate(worker.specs)
               if sp.kind == "snapshot"}
    assert slot_of == {2: 0, 3: 0, 4: 1, 5: 1}
    calls = []
    worker.on_snapshot_game = lambda key, w: calls.append((key, w))
    pol = {g: worker.opponents[("slot", g)].policy for g in (0, 1)}
    steps = {0: 0, 1: 0}

    def wrap(g):
        agent = worker.opponents[("slot", g)]
        orig = agent.step

        def stepped(*a, **k):
            steps[g] += 1
            return orig(*a, **k)

        agent.step = stepped

    wrap(0)
    wrap(1)

    # boot assignments; no phillip serving: both slot agents run, no
    # phillip agent exists
    worker.begin_transition(0, "ghostA", None, pol[0])
    worker.begin_transition(1, "ghostB", None, pol[1])
    worker.collect(1)
    assert steps[0] > 0 and steps[1] > 0
    assert worker._phillip_agent is None

    # slot 0 -> phillip (deferred): rows keep ghostA until THEIR game ends.
    # Deliver both boundaries on the first frame: the ended games still
    # credit ghostA; from that frame on phillip covers the rows and the
    # slot-0 policy idles
    worker.begin_transition(0, "phillip", None, pol[0])
    assert worker.slot_pending[0] == {2, 3}
    assert worker.slot_inplace_key[0] == "ghostA"
    s0 = steps[0]
    envs.final_stocks[2] = (4, 0)  # port-1 student wins on a ghostA game
    envs.final_stocks[3] = (4, 0)
    (traj,) = worker.collect(1)
    assert steps[0] == s0  # adopted on the boundary frame: policy idle
    assert worker._phillip_agent is not None
    assert worker._phillip_agent.num_envs == 2
    assert worker.trackers["snapshot"].wins >= 1
    assert worker.trackers["reference"].wins == 0
    assert {c[0] for c in calls[-2:]} == {"ghostA"}  # both ended games
    assert 0 not in worker.slot_pending and 0 not in worker.slot_inplace_key
    assert worker.env_member[2] == worker.env_member[3] == "phillip"
    for i in (2, 3):
        port = worker.specs[i].student_port
        assert all(set(cmd) == {port, 3 - port}
                   for cmd in worker._conns[i].sent)
    assert traj.rewards.shape[0] == 6  # full learner-row budget
    assert torch.isfinite(traj.rewards).all()

    # a game played by phillip logs under "reference" and pays "phillip"
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    assert worker.trackers["reference"].wins == 1
    assert calls[-1] == ("phillip", True)

    # multi-slot occupancy: agent rebuilt over both slots' rows once slot
    # 1's envs cross their boundaries
    worker.begin_transition(1, "phillip", None, pol[1])
    s1 = steps[1]
    envs.final_stocks[4] = (4, 0)
    envs.final_stocks[5] = (4, 0)
    worker.collect(1)
    assert steps[1] == s1
    assert worker._phillip_agent.num_envs == 4
    assert worker._phillip_rows_built == [2, 3, 4, 5]

    # back to snapshots: rows leave phillip at their boundaries and the
    # slot policies (now holding the new ghosts) resume stepping
    worker.begin_transition(0, "ghostC", None, pol[0])
    worker.begin_transition(1, "ghostD", None, pol[1])
    for i in (2, 3, 4, 5):
        envs.final_stocks[i] = (4, 0)
    worker.collect(1)
    assert steps[0] > s0 and steps[1] > s1
    assert {worker.env_member[i] for i in (2, 3)} == {"ghostC"}
    assert {worker.env_member[i] for i in (4, 5)} == {"ghostD"}


def test_league_phillip_imitation_follows_serving(monkeypatch):
    """The (dormant) imitation harvest keys off phillip-SERVING rows: no
    output while he serves nothing; once routed, exactly his rows are
    harvested, whitelist-gated, name-reconditioned, with the opponent-seat
    reward mirror; occupancy changes reset the partial chunk cleanly."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=1,
        league_phillip=True, harvest=True,
        opp_chars={2: "FOX", 3: "MARTH"},  # slot rows: one whitelisted
    )
    assert worker.harvest_imitation
    assert worker.ref_idx == []  # no fixed reference group

    trajs = worker.collect(1)
    assert [t.kind for t in trajs] == ["ppo"]  # phillip serving nothing

    # slot 0 -> phillip; rows adopt at their boundaries (delivered now)
    slot_envs = [i for i, sp in enumerate(worker.specs)
                 if sp.kind == "snapshot"]
    worker.begin_transition(0, "ghostA", None)
    worker.begin_transition(0, "phillip", None)
    for i in slot_envs:
        envs.final_stocks[i] = (4, 0)
    trajs = worker.collect(3)
    imits = [t for t in trajs if t.kind == "imitation"]
    assert imits  # harvested once his chunks fill
    for imit in imits:
        assert imit.rewards.shape[0] == 1  # only the FOX (whitelisted) row
        assert imit.initial_state is None
        # name re-conditioned on the student's code (1), not phillip's (2)
        assert torch.equal(imit.name, torch.ones_like(imit.name))
        # opponent-seat mirror of the fake envs' +0.01/frame student reward
        torch.testing.assert_close(
            imit.rewards, torch.full_like(imit.rewards, -0.01)
        )
        # actions round-trip the STUDENT embedding despite phillip's
        # different discretization (axis_spacing 8 vs 16)
        stu_embed = worker.student._embed_controller
        stu_embed.decode(stu_embed.map(
            lambda e, x: x.astype(getattr(e, "dtype", x.dtype)),
            tree.map_structure(
                lambda x: x.cpu().numpy(), imit.actions.controller_state
            ),
        ))

    # occupancy change mid-stream: partial chunk dropped, no output, no
    # crash; serving again restarts a fresh chunk
    worker.begin_transition(0, "ghostB", None)
    for i in slot_envs:
        envs.final_stocks[i] = (4, 0)
    trajs = worker.collect(1)
    assert [t.kind for t in trajs] == ["ppo"]
    assert worker._imit_rows == []


# ------------------------------- imported league members (previous-run bots)


def test_league_imports_parse():
    """"NAME=PATH" (implicit @FOX) and "NAME=PATH@CHAR" forms; bad forms
    fail loudly; imports require snapshot_slots and pfsp."""
    cfg = RolloutConfig(
        league_imports=["v3best=/m/rl-best-step0010000.pt",
                        "old=/m/old.pt@marth"],
        snapshot_slots=2,
    )
    assert cfg.import_members() == {
        "v3best": ("/m/rl-best-step0010000.pt", "FOX"),  # default lock FOX
        "old": ("/m/old.pt", "MARTH"),  # case-normalized
    }
    assert cfg.league_members() == ["import:v3best", "import:old"]
    # composes with the other league flags (imports appended last)
    combo = RolloutConfig(
        league_teacher=True, teacher_envs=0,
        league_imports=["v3=/m/x.pt"], snapshot_slots=2,
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
                league_imports=[bad], snapshot_slots=1
            ).import_members()
    with pytest.raises(AssertionError):  # duplicate names
        RolloutConfig(
            league_imports=["a=/m/x.pt", "a=/m/y.pt"], snapshot_slots=1
        ).import_members()
    with pytest.raises(AssertionError, match="snapshot_slots"):
        RolloutConfig(league_imports=["a=/m/x.pt"]).league_members()
    with pytest.raises(AssertionError, match="pfsp"):
        RolloutConfig(
            league_imports=["a=/m/x.pt"], snapshot_slots=1, pfsp=False
        ).league_members()
    # SnapshotPool accepts import keys as members; rejects junk keys
    from smashbot.rl.pool import SnapshotPool

    with pytest.raises(AssertionError, match="unknown league members"):
        SnapshotPool("/tmp/never-used", slots=2,
                     league_members=("imported:v3",))


def test_import_default_noop_worker(monkeypatch):
    """league_imports=[] leaves the worker byte-identical: no char-lock
    state, and the opp_char_lock command key is NEVER sent (env-side lock
    stays None forever = today's redraw behavior)."""
    worker, _ = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=2,
    )
    assert worker.slot_char_lock == {}
    assert not worker._has_imports
    worker.collect(1)
    for i, conn in enumerate(worker._conns):
        port = worker.specs[i].student_port
        expect = {port, 3 - port}
        assert all(set(cmd) == expect for cmd in conn.sent)


def test_import_auction_singleton_class(tmp_path):
    """An import joins the auction as its OWN singleton class: weight from
    its payoff row via f_hard(p=2), may hold multiple non-latest slots at
    once (with-replacement class draws), slot 0 stays the latest snapshot,
    and it fades as the student starts beating it."""
    from smashbot.rl.pool import SnapshotPool

    # exact class-share math: 1 ghost at raw 0.8 vs import at the 0.5 prior
    # -> f_hard(p=2): 0.04 vs 0.25 -> import share 0.25/0.29
    pool = SnapshotPool(str(tmp_path / "m"), slots=2,
                        league_members=("import:v3best",))
    pool.save(_Stub(), 100)
    latest = pool.save(_Stub(), 200)
    pool.payoff[pool.archive[0]] = {"wins": 8, "games": 10, "win_ema": 0.75}
    assert pool.class_hardness() == {"ghosts": 0.8, "import:v3best": 0.5}
    n, imp = 4000, 0
    for s in range(n):
        picks = pool.assignments(random.Random(s))
        assert picks[0] == latest and len(picks) == 2
        imp += picks[1] == "import:v3best"
    assert imp / n == pytest.approx(0.25 / 0.29, abs=0.03)

    # multi-slot occupancy + fade-out once beaten (3 ghosts >= 2 tail
    # slots, so the ghost class never exhausts into the uniform fallback)
    pool = SnapshotPool(str(tmp_path / "s"), slots=3,
                        league_members=("import:v3best",))
    for s in (100, 200, 300, 400):
        latest = pool.save(_Stub(), s)
    ghosts = pool.archive[:-1]

    def import_slots(n=300):
        held, multi = 0, False
        for s in range(n):
            picks = pool.assignments(random.Random(s))
            assert picks[0] == latest
            tail = picks[1:]
            for g in ghosts:
                assert tail.count(g) <= 1  # ghosts: without replacement
            held += tail.count("import:v3best")
            multi |= tail.count("import:v3best") > 1
        return held, multi

    held_prior, saw_multi = import_slots()
    assert saw_multi  # singleton class held several slots at once
    assert held_prior > 0
    for _ in range(300):
        pool.record_result("import:v3best", True)  # student now dominates
    assert pool.win_estimate("import:v3best") > 0.9
    held_beaten, _ = import_slots()
    assert held_beaten < held_prior * 0.6  # f_hard fade-out


def test_import_serving_char_lock_and_attribution(monkeypatch, tmp_path):
    """Fake-env collect() run: a slot assigned an import loads the given
    state_dict into its slot policy, the char lock reaches the slot's env
    conns (opp_char_lock command key), results attribute to the
    "import:NAME" payoff row via slot_keys, and reassignment clears the
    lock so redraws resume."""
    from smashbot.rl.pool import SnapshotPool, apply_assignments

    donor = _tiny_policy(seed=9)  # the "previous run's battery best"
    w_path = str(tmp_path / "rl-best-step0010000.pt")
    torch.save(donor.state_dict(), w_path)
    ghost = _tiny_policy(seed=7)
    snap = str(tmp_path / "snapshot-0000100.pt")
    torch.save(ghost.state_dict(), snap)

    worker, envs = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=2,
        league_imports=[f"v3={w_path}@MARTH"],
    )
    assert worker._has_imports
    # specs: 2 teacher + slot-0 env (2) + slot-1 env (3)
    assert [sp.kind for sp in worker.specs] == (
        ["teacher"] * 2 + ["snapshot"] * 2
    )
    m0 = worker.opponents[("slot", 0)].policy
    m1 = worker.opponents[("slot", 1)].policy
    teacher_module = worker.opponents["teacher"].policy
    imports = {"import:v3": (w_path, "MARTH")}
    keys = {}
    apply_assignments(
        [snap, "import:v3"], [(0, m0), (1, m1)], teacher_module, worker,
        keys, imports=imports,
    )
    # import state_dict loaded into the slot policy (stub-weight check);
    # slot 0 loaded the plain ghost
    for k, v in donor.state_dict().items():
        assert torch.equal(v, m1.state_dict()[k]), k
    for k, v in ghost.state_dict().items():
        assert torch.equal(v, m0.state_dict()[k]), k
    assert keys == {0: snap, 1: "import:v3"}
    assert worker.slot_char_lock == {0: None, 1: "MARTH"}
    # boot assignment: envs adopt immediately and the locked import's env
    # spec already pins the character for its very first game
    assert worker.env_member == {2: snap, 3: "import:v3"}
    assert worker.specs[3].opponent_char == "MARTH"
    assert not worker.slot_pending

    # an import assignment without the registry fails loudly
    with pytest.raises(AssertionError, match="import registry"):
        apply_assignments(
            [snap, "import:v3"], [(1, m1)], teacher_module, worker, {},
        )

    # payoff attribution exactly as train_rl wires it
    pool = SnapshotPool(str(tmp_path / "pool"), slots=2,
                        league_members=("import:v3",))
    worker.on_snapshot_game = lambda key, won: pool.record_result(key, won)
    envs.final_stocks[3] = (4, 0)  # student (port 1) beats the import
    worker.collect(1)
    assert pool.payoff["import:v3"] == pytest.approx(
        {"wins": 1, "games": 1, "wins_d": 1.0, "games_d": 1.0}
    )
    assert worker.trackers["snapshot"].wins == 1  # imports are policy ghosts

    # char-lock command channel: slot envs get the lock (import slot MARTH,
    # ghost slot None); teacher envs get no such key
    for i, conn in enumerate(worker._conns):
        port = worker.specs[i].student_port
        if worker.specs[i].kind == "snapshot":
            assert all(set(c) == {port, 3 - port, "opp_char_lock"}
                       for c in conn.sent)
            lock = "MARTH" if worker.specs[i].group == 1 else None
            assert all(c["opp_char_lock"] == lock for c in conn.sent)
        else:
            assert all(set(c) == {port, 3 - port} for c in conn.sent)

    # slot moves off the import: lock clears, envs are told to unlock —
    # but env 3 keeps FIGHTING the import until its game ends (deferred
    # adoption): no spare-brain factory here, so the harness falls back to
    # an instant brain swap, while the label still follows the brain
    apply_assignments(
        [snap, "teacher"], [(0, m0), (1, m1)], teacher_module, worker,
        keys, imports=imports,
    )
    assert worker.slot_char_lock == {0: None, 1: None}
    assert keys[1] == "teacher"
    assert worker.env_member[3] == "teacher"  # legacy instant swap
    assert not worker.slot_pending
    worker.collect(1)
    assert worker._conns[3].sent[-1]["opp_char_lock"] is None


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

    pool = SnapshotPool(str(tmp_path), slots=2, keep=4,
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
    fresh = SnapshotPool(str(tmp_path), slots=2, keep=4)
    assert fresh.payoff["import:v3"]["games"] == 6
    assert "import:v3" not in fresh.category_estimates()
    for seed in range(25):
        assert "import:v3" not in fresh.assignments(random.Random(seed))

    # reload WITH it again: estimates resume where they left off; a fresh
    # unmeasured import sits at the 0.5 prior with a None ticker estimate
    back = SnapshotPool(str(tmp_path), slots=2, keep=4,
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
            snapshot_slots=0, self_envs=1, unroll_length=4,
            games_per_dolphin=10**9,
        )
        specs = [EnvSpec("self", -1, student_port, "FOX")]
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

    pool = SnapshotPool(str(tmp_path), slots=3, league_members=[
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
            str(tmp_path), slots=6, pfsp_hard_frac=0.0,
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
    picks = set()
    for seed in range(30):
        picks.update(pool.assignments(random.Random(seed)))
    assert "phillip" not in picks and "teacher" not in picks

    pool = build(explore=1.0)
    picks = set()
    for seed in range(30):
        picks.update(pool.assignments(random.Random(seed)))
    assert "phillip" in picks and "teacher" in picks


def test_pfsp_hard_frac_blend_serves_unbeatable(tmp_path):
    """hard_frac > 0 restores real (non-probe) serving for an unbeatable
    member under the blend: with explore OFF, pure f_var never picks the
    0% member but hard_frac=0.25 does (via its f_hard draws)."""
    from smashbot.rl.pool import SnapshotPool

    def build(hard_frac):
        pool = SnapshotPool(
            str(tmp_path), slots=6, pfsp_hard_frac=hard_frac,
            pfsp_explore=0.0,
            league_members=["phillip", "import:a"])
        for step in (100, 200):
            pool.save(_Stub(), step)
        for _ in range(10):
            pool.record_result("phillip", False)  # unbeatable
            pool.record_result("import:a", random.random() < 0.5)
        return pool

    picks = set()
    pool = build(hard_frac=0.0)
    for seed in range(30):
        picks.update(pool.assignments(random.Random(seed)))
    assert "phillip" not in picks  # pure f_var: zero weight at 0%

    picks = set()
    pool = build(hard_frac=0.25)
    for seed in range(30):
        picks.update(pool.assignments(random.Random(seed)))
    assert "phillip" in picks  # hard draws bring him back


def _wrap_steps(agent, counter, key):
    orig = agent.step

    def stepped(*a, **k):
        counter[key] += 1
        return orig(*a, **k)

    agent.step = stepped


def test_deferred_adoption_parks_old_brain_until_boundary(monkeypatch):
    """Policy->policy reassignment with a spare-brain factory: the OLD
    weights are parked in the spare module and keep driving the slot's envs
    (brain + payoff label) until each env's own game boundary; rows flip
    one by one; the slot policy (new weights) only steps once a row has
    adopted; the spare is released when the last row crosses."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=1,
    )
    slot_envs = [i for i, sp in enumerate(worker.specs)
                 if sp.kind == "snapshot"]
    assert slot_envs == [2, 3]
    worker.outgoing_factory = lambda n: BatchedPolicyAgent(
        _tiny_policy(seed=50), n, name_code=1
    )
    calls = []
    worker.on_snapshot_game = lambda key, w: calls.append((key, w))
    slot_agent = worker.opponents[("slot", 0)]
    pol = slot_agent.policy
    steps = {"slot": 0, "spare": 0}
    _wrap_steps(slot_agent, steps, "slot")

    worker.begin_transition(0, "ghostA", None, pol)
    worker.collect(1)
    assert steps["slot"] > 0 and not worker._outgoing_agents

    # auction: ghostA -> ghostB. Announce, THEN overwrite the slot policy
    # (exactly apply_assignments' order)
    old_w = {k: v.detach().clone() for k, v in pol.state_dict().items()}
    worker.begin_transition(0, "ghostB", None, pol)
    pol.load_state_dict(_tiny_policy(seed=8).state_dict())
    spare = worker._outgoing_agents[0]
    _wrap_steps(spare, steps, "spare")
    for k, v in spare.policy.state_dict().items():
        assert torch.equal(v, old_w[k]), k  # parked = the OLD brain
    assert worker.slot_spare_key[0] == "ghostA"
    assert worker.slot_pending[0] == {2, 3}
    assert {worker.env_member[i] for i in slot_envs} == {"ghostA"}

    # no boundary yet: only the spare drives the rows; every env still
    # receives an opponent-seat controller
    s_slot, s_spare = steps["slot"], steps["spare"]
    worker.collect(1)
    assert steps["spare"] > s_spare and steps["slot"] == s_slot
    for i in slot_envs:
        port = worker.specs[i].student_port
        assert all({port, 3 - port} <= set(c) for c in worker._conns[i].sent)

    # env 2's game ends: that game is ghostA's; env 2 adopts ghostB, env 3
    # is still on ghostA -> both brains step, routed per row
    envs.final_stocks[2] = (4, 0)
    s_slot, s_spare = steps["slot"], steps["spare"]
    worker.collect(1)
    assert calls[-1][0] == "ghostA"
    assert worker.env_member[2] == "ghostB" and worker.env_member[3] == "ghostA"
    assert worker.slot_pending[0] == {3}
    assert steps["slot"] > s_slot and steps["spare"] > s_spare

    # env 3 crosses: transition complete, spare released
    envs.final_stocks[3] = (4, 0)
    worker.collect(1)
    assert {worker.env_member[i] for i in slot_envs} == {"ghostB"}
    assert 0 not in worker.slot_pending and 0 not in worker.slot_spare_key
    s_spare = steps["spare"]
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    assert steps["spare"] == s_spare  # idle
    assert calls[-1][0] == "ghostB"  # ghostB's game, ghostB's row


def test_deferred_adoption_waits_for_char_lock(monkeypatch):
    """A char-locked incoming import is adopted only at a boundary whose
    new game actually plays the locked character (the CSS pick lags the
    arming by one game): a boundary with the wrong char keeps the env on
    the old brain + label; the matching one flips it."""
    worker, envs = _make_worker(
        monkeypatch, num_envs=4, teacher_envs=2, snapshot_slots=1,
        league_imports=["x=/dev/null@MARTH"],
    )
    worker.outgoing_factory = lambda n: BatchedPolicyAgent(
        _tiny_policy(seed=50), n, name_code=1
    )
    calls = []
    worker.on_snapshot_game = lambda key, w: calls.append((key, w))
    pol = worker.opponents[("slot", 0)].policy
    worker.begin_transition(0, "ghostA", None, pol)
    worker.collect(1)

    worker.begin_transition(0, "import:x", "MARTH", pol)
    worker.slot_char_lock[0] = "MARTH"  # as apply_assignments does
    # boundary on env 2 but the new game is still FOX (lock not yet at
    # the CSS): stays on ghostA
    envs.opp_chars[2] = "FOX"
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    assert calls[-1][0] == "ghostA"
    assert worker.env_member[2] == "ghostA"
    assert 2 in worker.slot_pending[0]
    # next boundary: the game starting now IS Marth -> adopt
    envs.opp_chars[2] = "MARTH"
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    assert calls[-1][0] == "ghostA"  # the ended (FOX) game was ghostA's
    assert worker.env_member[2] == "import:x"
    envs.final_stocks[2] = (4, 0)
    worker.collect(1)
    assert calls[-1][0] == "import:x"  # first Marth game credits the import
