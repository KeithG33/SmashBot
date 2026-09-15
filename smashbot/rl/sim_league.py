"""Multi-opponent sim rollout: a pool of opponents on player-1, grouped by
weights, with imitation harvest of every non-self seat.

Builds on sim_rollout's single-opponent loop. Each env is assigned an opponent
group (self / a phillip tier / a PFSP-pool member); player-1 inference runs one
forward per group on the slot-swapped view, and every harvested group assembles
its seat into a kind="imitation" Trajectory (advantage-weighted imitation in the
learner). The student seat is the kind="ppo" Trajectory.

This first version uses a STATIC split assignment (partition envs by the pool
shares) with no mid-run re-draw. PFSP selection + payoff updates + reset re-roll
layer on top next.

Reuses unchanged: BatchedPolicyAgent (sample+delay), ChunkAssembler (Trajectory
assembly), compute_reward, EnvBatch.step_and_reset.
"""
from __future__ import annotations

import numpy as np
import torch

from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.rl.rollouts import ChunkAssembler, compute_reward
from smashbot.rl import sim_env
from smashbot.rl.sim_rollout import _states_to_torch, _seat_stats


class _Group:
    """One opponent identity serving a fixed subset of env rows."""

    def __init__(self, gid, policy, env_idx, harvest, unroll, device, name_code):
        self.gid = gid
        self.env_idx = np.asarray(env_idx, dtype=np.int64)   # env rows this opp plays
        self.harvest = harvest
        self.n = len(self.env_idx)
        self.agent = BatchedPolicyAgent(policy, self.n, name_code=name_code, device=device)
        self.assembler = ChunkAssembler(unroll, policy.delay) if harvest else None
        self._pushed = 0
        self._prev = None
        self._reset = np.ones(self.n, dtype=bool)   # its envs start fresh


class MultiOpponentSimWorker:
    def __init__(self, student_policy, opponents, batch_size, unroll_length, data_dir,
                 stage, char_pairs, name_code=1, device="cpu"):
        """opponents: list of (gid, policy, env_idx, harvest, name_code).
        char_pairs: [(p0_char, p1_char)] per env (len == batch_size)."""
        import melee_sim as msl
        self.msl = msl
        self.N = batch_size
        self.unroll = unroll_length
        self.device = device
        self.student = BatchedPolicyAgent(student_policy, batch_size, name_code=name_code, device=device)
        self.assembler = ChunkAssembler(unroll_length, student_policy.delay)
        self._pushed = 0
        self._prev = None
        self._reset_mask = np.ones(batch_size, dtype=bool)
        self.groups = [
            _Group(gid, pol, idx, harv, unroll_length, device, nc)
            for (gid, pol, idx, harv, nc) in opponents
        ]
        # env_idx must partition [0, N)
        covered = np.concatenate([g.env_idx for g in self.groups]) if self.groups else np.array([], int)
        assert sorted(covered.tolist()) == list(range(batch_size)), "opponent env_idx must partition all envs"
        self.env = msl.EnvBatch(batch_size=batch_size, length=max(64, unroll_length + 1), data_dir=data_dir)
        cfgs = [msl.MatchConfig(stage=stage, players=(msl.PlayerConfig(a), msl.PlayerConfig(b)))
                for (a, b) in char_pairs]
        self.env.configure_matches(cfgs)
        self.env.reset_all()

    def collect(self, num_frames):
        ppo_out, imit_out = [], []
        env, dev, T = self.env, self.device, self.unroll
        for _ in range(num_frames):
            if env.t >= env.length:
                env.reset_cursor()
            obs = env.current_frame
            reset_np = self._reset_mask
            reset_t = torch.as_tensor(reset_np, device=dev)

            # ---- student (player 0) ----
            states = _states_to_torch(sim_env.encode_obs(obs), dev)
            want = (self._pushed % T == 0)
            records, hidden_before = self.student.infer(states, reset_t, want_snapshot=want)
            to_exec = self.student.execute(np.nonzero(reset_np)[0].tolist())
            sim_env.write_controllers(env, to_exec, player=0)

            # ---- opponents (player 1), grouped ----
            p1 = [None] * self.N
            for g in self.groups:
                gobs = obs[g.env_idx]
                gstates = _states_to_torch(sim_env.encode_obs(gobs, self_slot=1, opp_slot=0), dev)
                greset = torch.as_tensor(g._reset, device=dev)
                gwant = (g._pushed % T == 0)
                grecords, ghidden = g.agent.infer(gstates, greset, want_snapshot=gwant)
                gexec = g.agent.execute(np.nonzero(g._reset)[0].tolist())
                for k, i in enumerate(g.env_idx):
                    p1[i] = gexec[k]
                if g.harvest:
                    for rec in grecords:
                        snap = ghidden if g._pushed % T == 0 else None
                        g.assembler.push_frame(rec, greset, snap)
                        g._pushed += 1
            sim_env.write_controllers(env, p1, player=1)

            # ---- rewards ----
            stocks, percent = _seat_stats(obs)         # [N,2] self,opp (player-0 view)
            if self._prev is not None:
                reward = compute_reward(
                    torch.as_tensor(self._prev[0]), torch.as_tensor(stocks),
                    torch.as_tensor(self._prev[1]), torch.as_tensor(percent), reset_t.cpu()).to(dev)
                self.assembler.push_reward(reward)
                for g in self.groups:
                    if g.harvest:  # opponent seat reward = zero-sum mirror
                        g.assembler.push_reward((-reward[g.env_idx]).clone())
            self._prev = (stocks, percent)

            # ---- student records ----
            for rec in records:
                snap = hidden_before if self._pushed % T == 0 else None
                self.assembler.push_frame(rec, reset_t, snap)
                self._pushed += 1

            is_resetting, _term = env.step_and_reset()
            self._reset_mask = np.asarray(is_resetting, dtype=bool)
            for g in self.groups:
                g._reset = self._reset_mask[g.env_idx]

            if self.assembler.ready():
                ppo_out.append(self.assembler.emit())
            for g in self.groups:
                if g.harvest and g.assembler.ready():
                    imit_out.append(g.assembler.emit()._replace(kind="imitation"))
        return ppo_out, imit_out

    def close(self):
        self.env.close()
