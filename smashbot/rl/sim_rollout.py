"""Sim-backed rollout worker: produces the same Trajectory the learner already
consumes, from a batched melee-sim-light EnvBatch instead of the Dolphin fleet.

Reuses, unchanged:
  * BatchedPolicyAgent  -- sampling, the delay queue, FrameRecord emission.
  * ChunkAssembler      -- FrameRecord/reward -> Trajectory (delay shift, T+1 overlap).
  * compute_reward      -- stock/percent -> reward.
  * EnvBatch.step_and_reset -- steps, resets finished games in place, wraps the
    ring cursor, and returns is_resetting/terminal.
New here is only the orchestration: encode sim frames, feed the agent, write its
controllers back, step, compute rewards.

First version drives player-0 with the student and leaves player-1 neutral
(mechanics/Trajectory validation). Opponent policies on player-1 come next.
"""
from __future__ import annotations

import numpy as np
import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.rl.rollouts import ChunkAssembler, compute_reward
from smashbot.rl import sim_env


def _states_to_torch(encoded, device):
    def cvt(x):
        a = np.asarray(x)
        if a.dtype == np.bool_:
            return torch.as_tensor(a, device=device)
        if a.dtype.kind in "iu":
            return torch.as_tensor(a.astype(np.int64), device=device)
        return torch.as_tensor(a.astype(np.float32), device=device)
    return tree.map_structure(cvt, encoded)


def _seat_stats(obs):
    """(stocks[N,2], percent[N,2]) with col0=self (slot0), col1=opp (slot1)."""
    s = obs["slots"]
    stocks = np.stack([s[:, 0]["stocks"], s[:, 1]["stocks"]], axis=1).astype(np.float32)
    percent = np.stack([s[:, 0]["percent"], s[:, 1]["percent"]], axis=1).astype(np.float32)
    return stocks, percent


class SimRolloutWorker:
    def __init__(self, policy, batch_size, unroll_length, data_dir,
                 stage, chars, name_code=1, device="cpu",
                 opponent_policy=None, opp_name_code=1):
        import melee_sim as msl
        self.N = batch_size
        self.unroll = unroll_length
        self.device = device
        self.student = BatchedPolicyAgent(policy, batch_size, name_code=name_code, device=device)
        # player-1 opponent (None -> neutral). Sees the slot-swapped view
        # (self_slot=1) so its own perspective is p0. Its own delay queue and
        # recurrent state; not harvested for training yet.
        self.opponent = (
            None if opponent_policy is None
            else BatchedPolicyAgent(opponent_policy, batch_size, name_code=opp_name_code, device=device)
        )
        self.assembler = ChunkAssembler(unroll_length, policy.delay)
        # ring buffer a bit longer than the unroll so cursor wraps are rare.
        self.env = msl.EnvBatch(batch_size=batch_size, length=max(64, unroll_length + 1), data_dir=data_dir)
        self.env.configure_match(stage=stage, players=[msl.PlayerConfig(chars[0]), msl.PlayerConfig(chars[1])])
        self.env.reset_all()
        self._records_pushed = 0
        self._prev = None                                   # (stocks, percent)
        self._reset_mask = np.ones(batch_size, dtype=bool)  # fresh games = all reset

    def collect(self, num_frames):
        out = []
        env, dev, T = self.env, self.device, self.unroll
        for _ in range(num_frames):
            if env.t >= env.length:      # wrap the ring before reading/writing
                env.reset_cursor()       # carries current obs to slot 0, no game reset
            obs = env.current_frame
            reset_np = self._reset_mask
            reset_t = torch.as_tensor(reset_np, device=dev)
            reset_idx = np.nonzero(reset_np)[0].tolist()

            states = _states_to_torch(sim_env.encode_obs(obs), dev)
            want_snap = (self._records_pushed % T == 0)
            records, hidden_before = self.student.infer(states, reset_t, want_snapshot=want_snap)
            to_execute = self.student.execute(reset_idx)          # delayed controllers
            sim_env.write_controllers(env, to_execute, player=0)

            if self.opponent is not None:                          # player-1 opponent
                opp_states = _states_to_torch(
                    sim_env.encode_obs(obs, self_slot=1, opp_slot=0), dev)
                self.opponent.infer(opp_states, reset_t, want_snapshot=False)
                opp_exec = self.opponent.execute(reset_idx)
                sim_env.write_controllers(env, opp_exec, player=1)

            stocks, percent = _seat_stats(obs)
            if self._prev is not None:
                reward = compute_reward(
                    torch.as_tensor(self._prev[0]), torch.as_tensor(stocks),
                    torch.as_tensor(self._prev[1]), torch.as_tensor(percent),
                    reset_t.cpu(),
                ).to(dev)
                self.assembler.push_reward(reward)
            self._prev = (stocks, percent)

            for rec in records:
                snap = hidden_before if self._records_pushed % T == 0 else None
                self.assembler.push_frame(rec, reset_t, snap)
                self._records_pushed += 1

            is_resetting, _terminal = env.step_and_reset()
            self._reset_mask = np.asarray(is_resetting, dtype=bool)

            if self.assembler.ready():
                out.append(self.assembler.emit())
        return out

    def close(self):
        self.env.close()
