"""Rollout machinery tests (Dolphin-free): chunk assembly alignment, reward
delay-shifting, batched-agent equivalence with the single-env DelayedAgent
pathway, and end-to-end assembled-trajectory -> learner compatibility."""

import numpy as np
import pytest
import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent, FrameRecord
from smashbot.rl.ppo import Learner, PPOConfig, RLConfig
from smashbot.rl.rollouts import ChunkAssembler, compute_reward
from smashbot.tests.test_ppo import _tiny_policy, _tiny_value, _rand_states


def _fake_record(n_envs, t):
    """Distinguishable scalar payloads so alignment is checkable."""
    x = torch.full((n_envs,), float(t))
    return FrameRecord(state=x, prev_action=x + 0.5, logits=x + 0.25, name=x.long())


def test_assembler_chunking_and_reward_shift():
    T, D, N = 4, 2, 3
    asm = ChunkAssembler(unroll_length=T, delay=D)
    emitted = []
    for t in range(20):
        if t > 0:
            asm.push_reward(torch.full((N,), 100.0 + t))  # transition t-1 -> t
        snap = {"h": torch.tensor(float(t))} if t % T == 0 else None
        asm.push_frame(_fake_record(N, t), torch.zeros(N, dtype=torch.bool), snap)
        if asm.ready():
            emitted.append(asm.emit())

    first = emitted[0]
    # frames 0..T stacked
    assert torch.equal(first.states[:, 0], torch.zeros(3))
    assert torch.equal(first.states[:, T], torch.full((3,), float(T)))
    assert first.initial_state["h"].item() == 0.0
    # reward slot t = real transition t+D (pushed value 100 + (t+D+1))
    assert first.rewards.shape == (N, T)
    assert first.rewards[0, 0].item() == pytest.approx(100.0 + D + 1)
    assert first.rewards[0, T - 1].item() == pytest.approx(100.0 + D + T)

    second = emitted[1]
    # overlap: second chunk starts at frame T with the snapshot taken there
    assert torch.equal(second.states[:, 0], torch.full((3,), float(T)))
    assert second.initial_state["h"].item() == float(T)
    assert second.rewards[0, 0].item() == pytest.approx(100.0 + T + D + 1)


def test_compute_reward():
    prev_s = torch.tensor([[4.0, 4.0], [2.0, 3.0]])
    s = torch.tensor([[4.0, 3.0], [1.0, 3.0]])  # env0: opp died; env1: we died
    prev_p = torch.tensor([[10.0, 50.0], [80.0, 0.0]])
    p = torch.tensor([[10.0, 0.0], [95.0, 12.0]])  # opp % resets on death
    r = compute_reward(prev_s, s, prev_p, p, torch.zeros(2, dtype=torch.bool))
    assert r[0].item() == pytest.approx(1.0)  # kill, no damage exchanged
    # we died (-1), took 15 (-0.15), dealt 12 (+0.12)
    assert r[1].item() == pytest.approx(-1.0 - 0.15 + 0.12)
    # resets zero the reward
    r = compute_reward(prev_s, s, prev_p, p, torch.ones(2, dtype=torch.bool))
    assert torch.equal(r, torch.zeros(2))


def test_batched_agent_matches_independent_runs(monkeypatch):
    """N=2 envs fed identical states must behave like two synced single runs:
    env-batching cannot leak information across the batch. Sampling is
    patched to greedy so the streams are deterministic (independent per-row
    sampling would legitimately diverge them)."""
    from smashbot import embed as embed_lib

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[np.dtype(self.dtype).name]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    rng = np.random.default_rng(7)
    sae = policy.network.embed_state_action
    game_embed = dict(sae.embedding)["state"]

    agent = BatchedPolicyAgent(policy, num_envs=2, name_code=1)
    for t in range(6):
        state1 = _rand_states(game_embed, (1,), rng)
        both = tree.map_structure(lambda x: torch.cat([x, x], dim=0), state1)
        controllers, records, _ = agent.step(both)
        (record,) = records  # batch_steps=1: one record per step
        # ...but the two identical envs see identical logits every frame
        tree.map_structure(
            lambda t_: torch.testing.assert_close(t_[0], t_[1]),
            record.logits,
        )
        # and identical prev-action inputs
        tree.map_structure(
            lambda t_: torch.testing.assert_close(t_[0].float(), t_[1].float()),
            record.prev_action,
        )

    # resetting env 0 must not disturb env 1's recurrent state
    before = tree.map_structure(
        lambda t_: t_.clone() if isinstance(t_, torch.Tensor) else t_,
        agent.hidden,
    )
    agent.reset_env(0)
    tree.map_structure(
        lambda a, b: torch.testing.assert_close(a[1], b[1])
        if isinstance(a, torch.Tensor) and a.shape and a.shape[0] == 2
        else None,
        before,
        agent.hidden,
    )


def test_assembled_trajectory_feeds_learner():
    """Full loop sans Dolphin: batched agent rollout -> assembler -> learner,
    with the fresh-learner invariants (ratio == 1, actor KL == 0)."""
    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    value = _tiny_value()
    learner = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1.0)), policy, teacher, value
    )

    N, T, D = 2, 5, policy.delay
    agent = BatchedPolicyAgent(policy, num_envs=N)
    asm = ChunkAssembler(unroll_length=T, delay=D)
    rng = np.random.default_rng(3)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]

    t = 0
    traj = None
    while traj is None:
        if t > 0:
            asm.push_reward(torch.zeros(N))
        snap = agent.hidden_snapshot() if t % T == 0 else None
        states = _rand_states(game_embed, (N,), rng)
        _, records, _ = agent.step(states)
        (record,) = records
        asm.push_frame(record, torch.zeros(N, dtype=torch.bool), snap)
        if asm.ready():
            traj = asm.emit()
        t += 1

    fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(N))
    with torch.no_grad():
        _, metrics = learner._policy_loss(fixed)
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["actor_kl_mean"] == pytest.approx(0.0, abs=1e-5)


def test_batch_steps_equivalence(monkeypatch):
    """batch_steps=4 must produce the same executed controllers, records, and
    recurrent states as batch_steps=1 (greedy-patched so streams are
    deterministic), including across a mid-buffer reset."""
    from smashbot import embed as embed_lib
    from smashbot.tests.test_ppo import _tiny_policy, _rand_states

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[np.dtype(self.dtype).name]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)  # delay=2... need delay >= batch_steps
    policy.delay = 4
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]
    N, S = 2, 4
    rng = np.random.default_rng(11)
    frames = [_rand_states(game_embed, (N,), rng) for _ in range(8)]
    reset_seq = [torch.zeros(N, dtype=torch.bool) for _ in range(8)]
    reset_seq[5][1] = True  # mid-buffer reset for env 1

    a1 = BatchedPolicyAgent(policy, N, batch_steps=1)
    a4 = BatchedPolicyAgent(policy, N, batch_steps=S)

    recs1, recs4, exec1, exec4 = [], [], [], []
    for t in range(8):
        c1, r1, _ = a1.step(frames[t], reset_seq[t])
        c4, r4, _ = a4.step(frames[t], reset_seq[t])
        recs1 += r1
        recs4 += r4
        exec1.append(c1)
        exec4.append(c4)

    assert len(recs1) == 8 and len(recs4) == 8
    for r1, r4 in zip(recs1, recs4):
        tree.map_structure(
            lambda a, b: torch.testing.assert_close(a, b), r1.logits, r4.logits
        )
        tree.map_structure(
            lambda a, b: torch.testing.assert_close(a.float(), b.float()),
            r1.prev_action, r4.prev_action,
        )
    for c1, c4 in zip(exec1, exec4):
        for e1, e4 in zip(c1, c4):
            tree.map_structure(
                lambda a, b: np.testing.assert_allclose(
                    np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
                ),
                e1, e4,
            )
    tree.map_structure(
        lambda a, b: torch.testing.assert_close(a, b)
        if isinstance(a, torch.Tensor) else None,
        a1.hidden, a4.hidden,
    )


def _rand_raw_game(embed_game, shape, rng):
    """Random RAW (pre-from_state) game struct with valid leaf ranges."""
    from smashbot import embed as embed_lib

    def gen(e):
        if isinstance(e, embed_lib.StructEmbedding):
            return e.builder({k: gen(sub) for k, sub in e.embedding})
        if isinstance(e, embed_lib.MLPWrapper):
            return gen(e._embed)
        if isinstance(e, embed_lib.BoolEmbedding):
            return rng.integers(0, 2, shape).astype(bool)
        if isinstance(e, embed_lib.OneHotEmbedding):
            return rng.integers(0, e.input_size, shape).astype(np.int64)
        if isinstance(e, embed_lib.FloatEmbedding):
            return rng.uniform(-100, 100, shape).astype(np.float32)
        raise TypeError(e)

    return gen(embed_game)


def test_encoded_perspective_swap_commutes():
    """swap(encode(game)) == encode(swap(game)) — the opponent view can be
    built from the already-encoded struct (kills the second encode pass)."""
    from smashbot import embed as embed_lib

    embed_game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(0)
    game = _rand_raw_game(embed_game, (5,), rng)

    enc_then_swap = embed_game.from_state(game)
    enc_then_swap = enc_then_swap._replace(
        p0=enc_then_swap.p1, p1=enc_then_swap.p0
    )
    swap_then_enc = embed_game.from_state(
        game._replace(p0=game.p1, p1=game.p0)
    )
    tree.map_structure(
        lambda a, b: np.testing.assert_array_equal(a, b),
        enc_then_swap, swap_then_enc,
    )


def test_worker_side_encode_matches_policy_encode():
    """An independently-built embed tree (env process) must encode identically
    to the policy's own (same EmbedConfig -> same schema)."""
    from smashbot import embed as embed_lib
    from smashbot.tests.test_ppo import _tiny_policy

    policy = _tiny_policy(seed=0)
    worker_embed = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(1)
    game = _rand_raw_game(worker_embed, (3,), rng)
    tree.map_structure(
        lambda a, b: np.testing.assert_array_equal(a, b),
        policy.network.encode_game(game),
        worker_embed.from_state(game),
    )
    # parser output has PYTHON-scalar leaves (bool/int/float), not arrays:
    # the worker asarray-wraps before encoding — verify that shape works too
    scalar_game = tree.map_structure(
        lambda x: np.asarray(x.flat[0].item() if hasattr(x, "flat") else x),
        _rand_raw_game(worker_embed, (), rng),
    )
    worker_embed.from_state(scalar_game)  # must not raise


def test_teacher_watcher_and_hot_swap(tmp_path):
    """Watcher ignores unchanged/torn files, loads complete updates; swapping
    the learner's teacher in place changes teacher_kl and survives a step."""
    import os
    import time as time_lib

    from smashbot.rl.teacher_watch import TeacherWatcher
    from smashbot.tests.test_ppo import _make_learner

    learner, traj = _make_learner(ppo=PPOConfig(max_mean_actor_kl=1.0))
    path = tmp_path / "best.pt"

    def write_ckpt(policy):
        tmp = str(path) + ".tmp"
        torch.save({"state": {"policy": policy.state_dict()}}, tmp)
        os.replace(tmp, path)

    write_ckpt(learner.teacher)
    watcher = TeacherWatcher(str(path), settle_seconds=0.05)
    assert watcher.poll() is None  # unchanged since construction

    # a DIFFERENT teacher lands (atomic replace)
    from smashbot.tests.test_ppo import _tiny_policy
    other = _tiny_policy(seed=123)
    time_lib.sleep(0.02)
    write_ckpt(other)
    sd = watcher.poll()
    assert sd is not None
    assert watcher.poll() is None  # consumed; no re-trigger

    # torn write: partial garbage without atomic replace -> skipped, no raise
    with open(path, "wb") as f:
        f.write(b"partial garbage")
    assert watcher.poll() is None

    # hot swap: teacher_kl was ~0 (teacher == policy init); after swapping in
    # a different teacher it must be > 0, and a learner step still runs
    state = learner.initial_state(3)
    fixed, _, _ = learner._fixed_pass(traj, state)
    with torch.no_grad():
        _, before = learner._policy_loss(fixed)
    learner.teacher.load_state_dict(sd)
    state = state._replace(teacher=learner.teacher.initial_state(3))
    fixed, state, _ = learner._fixed_pass(traj, learner.initial_state(3))
    with torch.no_grad():
        _, after = learner._policy_loss(fixed)
    assert before["teacher_kl"] == pytest.approx(0.0, abs=1e-5)
    assert after["teacher_kl"] > 0.01
    _, metrics = learner.step([traj], learner.initial_state(3))
    assert np.isfinite(metrics["post_update"]["loss"])


def test_game_tracker():
    from smashbot.rl.rollouts import GameTracker

    gt = GameTracker(window=4)
    for fs in [(3, 0), (2, 0), (0, 1), (0, 0), (4, 0)]:  # oldest rolls out
        gt.add_game(fs)
    gt.add_kill(45.0)
    gt.add_kill(95.0)
    gt.add_death(120.0)
    st = gt.stats()
    assert st["games_played"] == 5
    # window: (2,0)W (0,1)L (0,0)D (4,0)W -> decided 3, wins 2
    assert st["win_rate_recent"] == pytest.approx(2 / 3)
    assert st["avg_stock_diff"] == pytest.approx((2 - 1 + 0 + 4) / 4)
    assert st["avg_percent_at_kill"] == pytest.approx(70.0)
    assert st["avg_percent_at_death"] == pytest.approx(120.0)


def test_pool_partition_and_snapshots(tmp_path):
    from smashbot.rl.pool import EnvSpec, SnapshotPool, make_partition, MAIN_12

    specs = make_partition(
        64, cpu_envs=8, teacher_envs=16, snapshot_slots=5, seed=1
    )
    assert len(specs) == 64
    kinds = [s.kind for s in specs]
    assert kinds.count("cpu") == 8
    assert kinds.count("teacher") == 16
    assert kinds.count("snapshot") == 40
    # slots evenly filled; policy opponents main-12 only; seats balanced
    from collections import Counter

    slots = Counter(s.group for s in specs if s.kind == "snapshot")
    assert all(v == 8 for v in slots.values()) and len(slots) == 5
    from smashbot.rl.pool import CPU_CHARS, OFF_ROSTER, OPPONENT_CHARS

    for s_ in specs:
        if s_.kind != "cpu":
            assert s_.opponent_char in OPPONENT_CHARS  # Sheik allowed here
        else:
            assert s_.opponent_char in CPU_CHARS + OFF_ROSTER
            # CPUs cannot be Sheik: proven live, 362/362 spawned Zelda
            assert s_.opponent_char != "SHEIK"
        assert s_.opponent_char != "ZELDA"  # unpickable on netplay CSS
        assert s_.student_port in (1, 2)
    seats = Counter(s.student_port for s in specs)
    assert abs(seats[1] - seats[2]) <= 2

    import torch as t

    class P(t.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.w = t.nn.Parameter(t.tensor([v]))

    pool = SnapshotPool(str(tmp_path), slots=3, keep=4)
    assert pool.assignments() == []
    for step, v in enumerate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        pool.save(P(v), step)
    assert len(pool.archive) == 4  # keep=4 pruned oldest
    import random as r

    picks = pool.assignments(r.Random(0))
    assert len(picks) == 3
    assert picks[0] == pool.archive[-1]  # slot 0 = latest
    assert len(set(picks)) == 3  # without replacement when possible


def test_reset_target_positions_masked():
    """Regression (rl-ab-base2 NaN): at t = reset-1 the stream's next-action
    is the reset-substituted neutral — a fictional target the actor never
    sampled, whose AR teacher-forcing chain diverges from the recorded
    logits. Those positions must be masked out: fresh learner over a rollout
    CONTAINING resets keeps ratio ~= 1 with zero anomalous samples."""
    torch.manual_seed(0)
    policy = _tiny_policy(seed=0)
    teacher = _tiny_policy(seed=0)
    learner = Learner(
        RLConfig(ppo=PPOConfig(max_mean_actor_kl=1.0)), policy, teacher,
        _tiny_value(),
    )
    N, T = 2, 6
    agent = BatchedPolicyAgent(policy, num_envs=N)
    asm = ChunkAssembler(unroll_length=T, delay=policy.delay)
    rng = np.random.default_rng(5)
    game_embed = dict(policy.network.embed_state_action.embedding)["state"]

    t = 0
    traj = None
    while traj is None:
        resets = torch.zeros(N, dtype=torch.bool)
        if t in (2, 4):
            resets[t % N] = True  # mid-chunk game boundaries
        if t > 0:
            asm.push_reward(torch.zeros(N))
        snap = agent.hidden_snapshot() if t % T == 0 else None
        _, records, _ = agent.step(_rand_states(game_embed, (N,), rng), resets)
        (record,) = records
        asm.push_frame(record, resets, snap)
        if asm.ready():
            traj = asm.emit()
        t += 1

    assert traj.is_resetting.any(), "test must exercise resets"
    fixed, _, _ = learner._fixed_pass(traj, learner.initial_state(N))
    # the masked positions exist
    assert fixed.valid.min().item() == 0.0
    with torch.no_grad():
        _, metrics = learner._policy_loss(fixed)
    assert metrics["anomalous_samples"] == 0
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["actor_kl_mean"] == pytest.approx(0.0, abs=1e-4)


def test_pool_partition_reference_envs():
    from smashbot.rl.pool import MAIN_12, make_partition

    specs = make_partition(
        num_envs=16, cpu_envs=4, teacher_envs=-1, snapshot_slots=0,
        seed=3, ref_envs=4,
    )
    kinds = [s.kind for s in specs]
    assert kinds.count("cpu") == 4
    assert kinds.count("reference") == 4
    assert kinds.count("teacher") == 8
    refs = [s for s in specs if s.kind == "reference"]
    # medium-v2 plays exactly the main 12 (verified from its checkpoint)
    assert all(s.opponent_char in MAIN_12 for s in refs)
    # both seats represented so the student isn't port-biased vs the ref
    assert {s.student_port for s in refs} == {1, 2}


def test_snapshot_pool_exponential_thinning(tmp_path):
    import torch as _torch

    from smashbot.rl.pool import SnapshotPool

    class Stub:
        def state_dict(self):
            return {"w": _torch.zeros(1)}

    pool = SnapshotPool(str(tmp_path), slots=3, keep=12)
    for step in range(0, 6000, 100):
        pool.save(Stub(), step)
    steps = [SnapshotPool._step_of(p) for p in pool.archive]
    assert len(steps) == 12
    assert steps == sorted(steps)
    assert steps[0] == 0  # log-spacing anchor (interior-only eviction)
    assert steps[-1] == 5900  # latest always present
    recents = steps[-8:]
    assert recents == list(range(5200, 6000, 100))  # dense recent window
    old_gaps = [b - a for a, b in zip(steps[:-8], steps[1:-7])]
    # old region thinned: strictly sparser than the recent window's spacing
    assert min(old_gaps) > 100
    # NOT FIFO: old region spans deep history, not just the recent stretch
    assert steps[3] - steps[0] > 2000


def test_async_delayed_agent_matches_sync(monkeypatch):
    """AsyncDelayedAgent must emit the IDENTICAL controller sequence as the
    sync DelayedAgent — same policy, same frames, greedy-patched sampling.
    (The async agent overlaps compute with emulation; behavior must not
    change by even one frame.)"""
    from smashbot import embed as embed_lib
    from smashbot.eval.agent import AsyncDelayedAgent, DelayedAgent
    from smashbot.tests.test_ppo import _tiny_policy

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[np.dtype(self.dtype).name]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    def make(cls):
        torch.manual_seed(0)
        policy = _tiny_policy(seed=0)
        policy.delay = 4
        agent = cls(policy, own_port=1, opponent_port=2, device="cpu")
        return agent

    sync, awa = make(DelayedAgent), make(AsyncDelayedAgent)

    # bypass the libmelee Parser: feed identical raw games directly
    class StubParser:
        def __init__(self, seq):
            self.seq = seq
            self.i = 0

        def get_game(self, _gs):
            g = self.seq[self.i % len(self.seq)]
            self.i += 1
            return g

    import sys
    sys.path.insert(0, "/tmp/claude-1000/-home-kage-smashbot-workspace/622f61f0-32b7-4321-b892-7040871af8a8/scratchpad")
    from test_ref_bridge import rand_raw_game

    embed_game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(5)
    frames = [rand_raw_game(embed_game, rng) for _ in range(12)]
    sync.parser = StubParser(frames)
    awa.parser = StubParser(frames)

    outs_sync = [sync.step(None) for _ in range(12)]
    outs_async = [awa.step(None) for _ in range(12)]
    awa.drain()

    for a, b in zip(outs_sync, outs_async):
        tree.map_structure(
            lambda x, y: np.testing.assert_array_equal(np.asarray(x), np.asarray(y)),
            a, b,
        )


def test_async_agent_absorbs_slow_samples(monkeypatch):
    """Inference spikes must neither block step() nor change the emitted
    sequence: the pipeline lags and catches up inside the delay-queue slack."""
    import time as time_lib

    from smashbot import embed as embed_lib
    from smashbot.eval.agent import AsyncDelayedAgent, DelayedAgent
    from smashbot.tests.test_ppo import _tiny_policy

    monkeypatch.setattr(
        embed_lib.OneHotEmbedding, "sample",
        lambda self, logits, temperature=None: logits.argmax(-1).to(
            {"uint8": torch.uint8, "int32": torch.int32}[np.dtype(self.dtype).name]
        ),
    )
    monkeypatch.setattr(
        embed_lib.BoolEmbedding, "sample",
        lambda self, logits, temperature=None: logits.squeeze(-1) > 0,
    )

    def make(cls, spike):
        torch.manual_seed(0)
        policy = _tiny_policy(seed=0)
        policy.delay = 6
        if spike:
            orig = policy.sample

            calls = {"n": 0}

            def slow_sample(*a, **k):
                calls["n"] += 1
                if calls["n"] % 3 == 0:
                    time_lib.sleep(0.03)  # 30ms spike, every 3rd frame
                return orig(*a, **k)

            policy.sample = slow_sample
        return cls(policy, own_port=1, opponent_port=2, device="cpu")

    import sys
    sys.path.insert(0, "/tmp/claude-1000/-home-kage-smashbot-workspace/622f61f0-32b7-4321-b892-7040871af8a8/scratchpad")
    from test_ref_bridge import rand_raw_game

    class StubParser:
        def __init__(self, seq):
            self.seq, self.i = seq, 0

        def get_game(self, _gs):
            g = self.seq[self.i % len(self.seq)]
            self.i += 1
            return g

    embed_game = embed_lib.EmbedConfig().make_game_embedding()
    rng = np.random.default_rng(9)
    frames = [rand_raw_game(embed_game, rng) for _ in range(10)]

    sync = make(DelayedAgent, spike=False)
    awa = make(AsyncDelayedAgent, spike=True)  # spikes ONLY on async side
    sync.parser = StubParser(frames)
    awa.parser = StubParser(frames)

    outs_sync = [sync.step(None) for _ in range(10)]
    step_times = []
    outs_async = []
    for _ in range(10):
        t0 = time_lib.perf_counter()
        outs_async.append(awa.step(None))
        step_times.append(time_lib.perf_counter() - t0)
        time_lib.sleep(0.0167)  # real frame cadence: compute catches up here
    awa.drain()

    # identical sequence despite 30ms spikes on every 3rd sample
    for a, b in zip(outs_sync, outs_async):
        tree.map_structure(
            lambda x, y: np.testing.assert_array_equal(
                np.asarray(x), np.asarray(y)
            ),
            a, b,
        )
    # and step() never blocked on a spike (queue slack absorbed the lag)
    assert max(step_times) < 0.02, f"step blocked: {max(step_times)*1e3:.1f}ms"
