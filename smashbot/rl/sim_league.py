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


def make_reencoder(opp_embed, stu_embed, student_name_code, device):
    """Harvested-chunk fixup for an opponent with its OWN config (the
    phillips): re-encode the controller stream through the student's
    embedding and recondition the name — the sim twin of
    DolphinRolloutWorker._traj_reencoder. Logits stay opponent-schema
    (unused by the imitation loss)."""
    import tree as _tree

    def reencode(traj):
        encoded_np = opp_embed.map(
            lambda e, x: x.astype(getattr(e, "dtype", x.dtype)),
            _tree.map_structure(
                lambda x: x.cpu().numpy(), traj.actions.controller_state
            ),
        )
        raw = opp_embed.decode(encoded_np)
        prev = _tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(
                    x.astype(np.int64) if x.dtype.kind in "iu" else x
                )
            ).to(device),
            stu_embed.from_state(raw),
        )
        return traj._replace(
            actions=traj.actions._replace(controller_state=prev),
            name=torch.full_like(traj.name, student_name_code),
        )

    return reencode


class _Group:
    """One opponent identity serving a fixed subset of env rows."""

    def __init__(self, gid, policy, env_idx, harvest, unroll, device, name_code,
                 reencode=None, precision="fp32"):
        self.gid = gid
        self.env_idx = np.asarray(env_idx, dtype=np.int64)   # env rows this opp plays
        self.harvest = harvest
        self.n = len(self.env_idx)
        self.agent = BatchedPolicyAgent(policy, self.n, name_code=name_code,
                                        device=device, precision=precision)
        self.assembler = ChunkAssembler(unroll, policy.delay) if harvest else None
        self.reencode = reencode
        self._pushed = 0
        self._prev = None
        self._reset = np.ones(self.n, dtype=bool)   # its envs start fresh


class MultiOpponentSimWorker:
    def __init__(self, student_policy, opponents, batch_size, unroll_length, data_dir,
                 stage, char_pairs, name_code=1, device="cpu", record_fn=None,
                 precision="fp32"):
        """opponents: list of (gid, policy, env_idx, harvest, name_code).
        char_pairs: [(p0_char, p1_char)] per env (len == batch_size).
        stage: one msl.Stage for all envs, or a per-env list (len == batch_size).
        record_fn(env_i, opponent_gid, student_stocks, opp_stocks): called on
        each decided game (terminal-frame stocks, student = player 0).
        precision: BatchedPolicyAgent precision for every seat ("fp16" is the
        probe-validated rollout setting from v10)."""
        import melee_sim as msl
        self.msl = msl
        self.N = batch_size
        self.unroll = unroll_length
        self.device = device
        self.record_fn = record_fn
        self.student = BatchedPolicyAgent(student_policy, batch_size, name_code=name_code,
                                          device=device, precision=precision)
        self.assembler = ChunkAssembler(unroll_length, student_policy.delay)
        self._pushed = 0
        self._prev = None
        self._reset_mask = np.ones(batch_size, dtype=bool)
        stu_embed = student_policy.controller_head.controller_embedding
        self.groups = []
        for (gid, pol, idx, harv, nc) in opponents:
            re = None
            if harv:
                # student-schema fixup (see make_reencoder): identity for
                # matching configs (snapshots/imports), a real re-encode +
                # name recondition for the phillips
                re = make_reencoder(
                    pol.controller_head.controller_embedding, stu_embed,
                    name_code, device)
                if pol.delay != student_policy.delay:
                    print(f"NOTE: imitation harvest delay mismatch — {gid} "
                          f"{pol.delay} vs student {student_policy.delay} "
                          f"({pol.delay - student_policy.delay:+d} frames)",
                          flush=True)
            self.groups.append(
                _Group(gid, pol, idx, harv, unroll_length, device, nc, re,
                       precision=precision))
        # env -> opponent gid, for outcome recording
        self.env_opp = np.empty(batch_size, dtype=object)
        for g in self.groups:
            self.env_opp[g.env_idx] = g.gid
        # env_idx must partition [0, N)
        covered = np.concatenate([g.env_idx for g in self.groups]) if self.groups else np.array([], int)
        assert sorted(covered.tolist()) == list(range(batch_size)), "opponent env_idx must partition all envs"
        self.env = msl.EnvBatch(batch_size=batch_size, length=max(64, unroll_length + 1), data_dir=data_dir)
        stages = stage if isinstance(stage, (list, tuple)) else [stage] * batch_size
        cfgs = [msl.MatchConfig(stage=s, players=(msl.PlayerConfig(a), msl.PlayerConfig(b)))
                for s, (a, b) in zip(stages, char_pairs)]
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

            is_resetting, term = env.step_and_reset()
            self._reset_mask = np.asarray(is_resetting, dtype=bool)
            if self.record_fn is not None:
                done = np.asarray(term["done"], dtype=bool)
                if done.any():
                    fs, _ = _seat_stats(env.current_frame)  # terminal frame stocks
                    for i in np.nonzero(done)[0]:
                        self.record_fn(int(i), self.env_opp[i],
                                       int(fs[i, 0]), int(fs[i, 1]))
            for g in self.groups:
                g._reset = self._reset_mask[g.env_idx]

            if self.assembler.ready():
                ppo_out.append(self.assembler.emit())
            for g in self.groups:
                if g.harvest and g.assembler.ready():
                    traj = g.assembler.emit()._replace(kind="imitation")
                    if g.reencode is not None:
                        traj = g.reencode(traj)
                    imit_out.append(traj)
        return ppo_out, imit_out

    def close(self):
        self.env.close()


# ---------------------------------------------------------------- opponent pool

import random as _random

from smashbot.rl.pool import SnapshotPool


class SimLeague:
    """Opponent pool + per-env assignment for the sim league.

    Three fixed shares (self / phillips / PFSP pool) per the design:
      * self       -- the current policy (mirror), no harvest.
      * phillips    -- 5 fixed tiers, per-tier env fraction, harvested.
      * PFSP pool   -- snapshots (League.archive) + fox imports (import:NAME),
                       drawn by League.draw_member, harvested.
    Reuses League for the PFSP draw + payoff ledger (pfsp.json), so a resume
    loads v10's archive + payoff table straight from the snapshot dir.

    Assignment is coarse (re-partitioned every `repartition_every` frames), so
    within a period an env keeps its opponent and a game reset just resets
    hidden -- no cross-group state migration.
    """

    def __init__(self, current_policy, snapshot_dir, phillips, fox_imports,
                 self_frac=0.30, device="cpu", pfsp_hard_frac=0.25, pfsp_explore=0.15,
                 name_resolver=None, config_from=None):
        # phillips: {tier_id: (policy, frac, name_code)}; fox_imports: {"import:NAME": path}
        # config_from: full checkpoint whose config builds the bare-state members
        #   (snapshots + fox imports are saved as bare state_dicts, no config).
        self.current = current_policy
        self.device = device
        self.self_frac = self_frac
        self.phillips = phillips
        self.fox_paths = dict(fox_imports)
        self.league = SnapshotPool(
            snapshot_dir, keep=0, pfsp=True,
            pfsp_hard_frac=pfsp_hard_frac, pfsp_explore=pfsp_explore,
            league_members=list(fox_imports.keys()),
        )
        self._cache = {}          # member_key -> (policy, name_code)
        self._resolve = name_resolver or (lambda nm: 1)
        self._cfg = None
        if config_from is not None:
            from smashbot import saving
            self._cfg = saving.load_checkpoint(config_from)["config"]

    # ---- policy resolution (lazy load + cache) ----
    def get(self, key):
        """(policy, name_code) for an assignment key: 'self', 'phillip:<id>',
        a snapshot path, or 'import:<name>'."""
        if key == "self":
            return self.current, self._sc
        if key.startswith("phillip:"):
            pol, _frac, nc = self.phillips[key.split(":", 1)[1]]
            return pol, nc
        if key in self._cache:
            return self._cache[key]
        path = self.fox_paths[key] if key.startswith("import:") else key
        pol, _nm = self._load_bare(path)
        pol.train_value_head = False
        pol.requires_grad_(False)
        pol.eval()
        # snapshots share the student's name_map exactly; fox imports lack their
        # own map -> self name_code is the correct/defensible conditioning.
        val = (pol, self._sc)
        self._cache[key] = val
        return val

    def _load_bare(self, path):
        """Build a policy skeleton from config_from and load a bare state_dict
        (snapshots + fox imports carry no config of their own)."""
        import torch as _torch
        from smashbot import configs, embed as embed_lib
        from smashbot.policy import build_policy
        if self._cfg is None:
            raise RuntimeError("SimLeague needs config_from to load bare-state members")
        cfg = self._cfg
        pol = build_policy(
            embed_config=embed_lib.EmbedConfig(),
            controller_config=embed_lib.ControllerConfig(
                axis_spacing=cfg["head"]["axis_spacing"],
                shoulder_spacing=cfg["head"]["shoulder_spacing"],
            ),
            network_config=configs.NetworkConfig(**cfg["network"]),
            head_config=configs.ControllerHeadConfig(**cfg["head"]),
            policy_config=configs.PolicyConfig(**cfg["policy"]),
            num_names=cfg["data"]["max_names"],
        ).to(self.device)
        state = _torch.load(path, map_location=self.device, weights_only=True)
        pol.load_state_dict(state)
        pol.eval()
        return pol, {}

    def set_self_name_code(self, nc):
        self._sc = nc

    # ---- per-env assignment ----
    def partition(self, N, rng=None, max_pfsp_members=None):
        """{member_key: np.ndarray(env indices)} covering all N envs.

        max_pfsp_members bounds the number of DISTINCT PFSP opponents drawn
        this period (each is a full resident policy — memory discipline, like
        the Dolphin league's fixed weight slots). None = draw per env (one
        group per distinct draw, unbounded residency)."""
        rng = rng or _random.Random()
        idx = list(range(N))
        rng.shuffle(idx)
        cur = 0
        groups: dict[str, list] = {}

        def take(n):
            nonlocal cur
            g = idx[cur:cur + n]
            cur += n
            return g

        groups["self"] = take(round(self.self_frac * N))
        for tier, (_pol, frac, _nc) in self.phillips.items():
            groups[f"phillip:{tier}"] = take(round(frac * N))
        pfsp_envs = idx[cur:]
        if max_pfsp_members is None:
            for i in pfsp_envs:                       # one group per distinct draw
                m = self.league.draw_member(rng) or "self"
                groups.setdefault(m, []).append(i)
        elif pfsp_envs:
            # draw up to K distinct members, spread the pfsp envs across them
            members, seen = [], set()
            for _ in range(max_pfsp_members * 8):
                if len(members) >= max_pfsp_members:
                    break
                m = self.league.draw_member(rng) or "self"
                if m not in seen:
                    seen.add(m)
                    members.append(m)
            if not members:
                members = ["self"]
            for j, i in enumerate(pfsp_envs):
                groups.setdefault(members[j % len(members)], []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if len(v)}

    def record(self, key, won: bool):
        """Record a decided game outcome for a PFSP-pool member (snapshots +
        fox imports). Phillips/self aren't in the PFSP ledger."""
        if key == "self" or key.startswith("phillip:"):
            return
        self.league.record_result(key, won)
