"""Multi-opponent sim rollout: a pool of opponents on player-1, grouped by
weights, with imitation harvest of every non-self seat.

Builds on sim_rollout's single-opponent loop. Each env is assigned an opponent
group (self / a phillip tier / a PFSP-pool member); player-1 inference runs one
forward per group on the slot-swapped view, and every harvested group assembles
its seat into a kind="imitation" Trajectory (advantage-weighted imitation in the
learner). The student seat is the kind="ppo" Trajectory.

SimLeague owns the pool: shares partitioned per period (worker rebuilt on the
period boundary by train_sim.SimLeagueWorker — within a period an env keeps
its opponent), PFSP draw + payoff via SnapshotPool (pfsp.json), decided games
recorded through record_fn.

Reuses unchanged: BatchedPolicyAgent (sample+delay), ChunkAssembler (Trajectory
assembly), compute_reward, EnvBatch.step_and_reset.
"""
from __future__ import annotations

import random as _random

import numpy as np
import torch
import tree

from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.rl.pool import SnapshotPool
from smashbot.rl.rollouts import ChunkAssembler, compute_reward
from smashbot.rl.ppo import slice_trajectory_rows
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


class PfspGrid:
    """A set of opponent slots on ONE LeagueAgent: S slices (one member
    each) x Nc cells, stepped as a single captured vmap forward per frame —
    dolphin's league serving transplanted to the sim. fp16 stacked weights
    and carried state; a member swap is an in-place load_slice (visible to
    the captured graph), never a rebuild. Persists across re-partition
    periods: assign() reloads changed members and remaps cells to this
    period's env rows. Slices may serve FEWER env rows than Nc (padding):
    pad cells forward garbage against env 0 with reset always high, and
    their rows are sliced out of every emitted chunk.

    Harvest: one ChunkAssembler over all S*Nc cells; initial_state=None
    (the dolphin grid harvest convention — the learner's imitation path
    never unrolls from a carried state). `reencode` (optional) fixes up
    emitted chunks for members with their own schema (the phillips)."""

    def __init__(self, template, slices, cells, name_code, unroll, device,
                 reencode=None):
        from smashbot.rl.agent import LeagueAgent
        self.S, self.Nc = slices, cells
        cuda = torch.device(device).type == "cuda"
        dt = torch.float16 if cuda else torch.float32   # fp16 needs autocast
        self.agent = LeagueAgent(template, slices, cells, name_code, device,
                                 weights_dtype=dt,
                                 state_dtype=torch.float16 if cuda else None)
        self.assembler = ChunkAssembler(unroll, template.delay)
        self.reencode = reencode
        self.members = [None] * slices        # member key per slice
        self.env_idx = None                   # [S][rows] env rows (per period)
        self.cell_env = None                  # [S*Nc] env row per cell (pad -> 0)
        self.valid = None                     # [S*Nc] real (non-pad) cells
        self.idx_t = None                     # gather index on device
        self.device = device

    def assign(self, assignments, get_state):
        """assignments: [(key, env_rows)] one per slice, this period's map
        (len(env_rows) <= Nc; the rest of the slice is padding). Loads
        changed members in place and resets every cell (a re-partition
        resets all envs)."""
        assert len(assignments) == self.S, (len(assignments), self.S)
        self.env_idx = []
        cell_env = np.zeros(self.S * self.Nc, dtype=np.int64)
        valid = np.zeros(self.S * self.Nc, dtype=bool)
        for s, (key, rows) in enumerate(assignments):
            rows = np.asarray(rows, dtype=np.int64)
            assert len(rows) <= self.Nc, (key, len(rows), self.Nc)
            if self.members[s] != key:
                self.agent.load_slice(s, get_state(key))
                self.members[s] = key
            self.env_idx.append(rows)
            cell_env[s * self.Nc:s * self.Nc + len(rows)] = rows
            valid[s * self.Nc:s * self.Nc + len(rows)] = True
        self.cell_env = cell_env
        self.valid = valid
        self.valid_t = torch.as_tensor(np.nonzero(valid)[0], device=self.device)
        self.idx_t = torch.as_tensor(cell_env, device=self.device)
        for s in range(self.S):
            for n in range(self.Nc):
                self.agent.reset_cell(s, n)



class _Group:
    """One opponent identity serving a fixed subset of env rows."""

    def __init__(self, gid, policy, env_idx, harvest, unroll, device, name_code,
                 reencode=None, precision="fp32", state_dtype=None):
        self.gid = gid
        self.env_idx = np.asarray(env_idx, dtype=np.int64)   # env rows this opp plays
        self.idx_t = torch.as_tensor(self.env_idx, device=device)  # GPU gather index
        self.harvest = harvest
        self.n = len(self.env_idx)
        self.agent = BatchedPolicyAgent(policy, self.n, name_code=name_code,
                                        device=device, precision=precision,
                                        state_dtype=state_dtype)
        self.assembler = ChunkAssembler(unroll, policy.delay) if harvest else None
        self.reencode = reencode
        self._pushed = 0
        self._reset = np.ones(self.n, dtype=bool)   # its envs start fresh


class MultiOpponentSimWorker:
    def __init__(self, student_policy, opponents, batch_size, unroll_length, data_dir,
                 stage, char_pairs, name_code=1, device="cpu", record_fn=None,
                 precision="fp32", grid: PfspGrid | None = None, grids=()):
        """opponents: list of (gid, policy, env_idx, harvest, name_code) —
        the per-policy groups (self + phillips). PFSP members ride in `grid`
        (a PfspGrid already assign()ed for this period) as one merged forward.
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
        # grids: any number of LeagueAgent-backed opponent grids (PFSP
        # slots, the phillip grid); `grid` kept as the single-grid alias
        self.grids = list(grids) + ([grid] if grid is not None else [])
        self.grid = self.grids[0] if self.grids else None  # legacy alias
        self.student = BatchedPolicyAgent(student_policy, batch_size, name_code=name_code,
                                          device=device, precision=precision)
        self.student.set_flat_controllers(True)
        self.ff = sim_env.FlatFrames(device)
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
            # seat-B fp16 state: the self-play mirror is an OPPONENT seat
            # (never harvested, no loss path) — same argument/verification
            # as the grid's fp16 state. Diagnostic: rl/self/win_rate_ema
            # must hold ~0.5 (a degraded seat B shows as seat A winning).
            sd = (torch.float16
                  if gid == "self" and precision == "fp16" else None)
            self.groups.append(
                _Group(gid, pol, idx, harv, unroll_length, device, nc, re,
                       precision=precision, state_dtype=sd))
        for g in self.groups:
            g.agent.set_flat_controllers(True)
        # env -> opponent gid, for outcome recording
        self.env_opp = np.empty(batch_size, dtype=object)
        for g in self.groups:
            self.env_opp[g.env_idx] = g.gid
        for gr in self.grids:
            for s in range(gr.S):
                self.env_opp[gr.env_idx[s]] = gr.members[s]
        # groups + grids must partition [0, N)
        covered = [g.env_idx for g in self.groups] + [
            gr.cell_env[gr.valid] for gr in self.grids]
        covered = np.concatenate(covered) if covered else np.array([], int)
        assert sorted(covered.tolist()) == list(range(batch_size)), "opponent env_idx must partition all envs"
        for gr in self.grids:
            # a FULL worker rebuild resets _prev, so the first frame pushes
            # a record with no matching reward — the grids' PERSISTENT
            # assemblers would desync by one (and leak a FrameRecord) per
            # rebuild. Fresh assemblers on construction; reassign() keeps them.
            gr.assembler = ChunkAssembler(unroll_length, gr.agent.delay)
        self.env = msl.EnvBatch(batch_size=batch_size, length=max(64, unroll_length + 1), data_dir=data_dir)
        stages = stage if isinstance(stage, (list, tuple)) else [stage] * batch_size
        cfgs = [msl.MatchConfig(stage=s, players=(msl.PlayerConfig(a), msl.PlayerConfig(b)))
                for s, (a, b) in zip(stages, char_pairs)]
        self.env.configure_matches(cfgs)
        self.env.reset_all()

    def reassign(self, new_env_idx: dict, char_pairs, stage) -> None:
        """New period WITHOUT rebuilding agents: rebuilding makes the
        cudagraph trees re-record (changed liveness between replays), which
        grew the private pools +0.6 GiB per period and OOM'd the launch.
        Group sizes are constant by construction, so a period is just:
        remap env rows, reset every seat (the existing reset path clears
        queues/hidden/prev), reconfigure matches. The grid must already be
        assign()ed for this period. Assemblers keep their buffered tail —
        the learner sees the boundary as a normal all-envs game reset
        (is_resetting masks state and zeroes the boundary reward)."""
        import melee_sim as msl
        for g in self.groups:
            idx = np.asarray(new_env_idx[g.gid], dtype=np.int64)
            assert len(idx) == g.n, (g.gid, len(idx), g.n)
            g.env_idx = idx
            g.idx_t = torch.as_tensor(idx, device=self.device)
            g._reset = np.ones(g.n, dtype=bool)
        self._reset_mask = np.ones(self.N, dtype=bool)
        self.env_opp = np.empty(self.N, dtype=object)
        for g in self.groups:
            self.env_opp[g.env_idx] = g.gid
        for gr in self.grids:
            for sl in range(gr.S):
                self.env_opp[gr.env_idx[sl]] = gr.members[sl]
        covered = [g.env_idx for g in self.groups] + [
            gr.cell_env[gr.valid] for gr in self.grids]
        covered = np.concatenate(covered)
        assert sorted(covered.tolist()) == list(range(self.N)), "reassign must partition all envs"
        stages = stage if isinstance(stage, (list, tuple)) else [stage] * self.N
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
            # EXECUTE (pop the delay queue) BEFORE infer: on reset frames
            # execute() rebuilds the queue, and an infer-first order lets
            # that rebuild discard the fresh sample — after which the queue
            # runs at delay-1 forever (review finding #1: every seat was
            # acting one frame ahead of its BC conditioning).
            reset_idx = np.nonzero(reset_np)[0].tolist()
            p0_rows = np.stack(self.student.execute(reset_idx))
            sim_env.write_controller_rows(env, p0_rows, player=0)
            # flat encoding: ONE numpy encode, THREE H2D copies; the swap
            # is a column permutation and every view below is 3 gathers +
            # struct views (was ~120 per-leaf launches per view)
            flats = self.ff.to_device(sim_env.encode_flats(obs))
            states = self.ff.view(flats)
            want = (self._pushed % T == 0)
            records, hidden_before = self.student.infer(states, reset_t, want_snapshot=want)

            # ---- opponents (player 1) ----
            opp_flats = self.ff.swap(flats)
            p1_rows = np.empty((self.N, 13), dtype=np.float32)
            for g in self.groups:
                p1_rows[g.env_idx] = np.stack(
                    g.agent.execute(np.nonzero(g._reset)[0].tolist()))
                gstates = self.ff.view(opp_flats, rows=g.idx_t)
                greset = torch.as_tensor(g._reset, device=dev)
                # want_snapshot only for harvested groups: self's _pushed
                # never advances, so the old expression cloned the self
                # seat's full KV state EVERY frame (review finding #3)
                gwant = g.harvest and (g._pushed % T == 0)
                grecords, _gh = g.agent.infer(gstates, greset, want_snapshot=gwant)
                if g.harvest:
                    for rec in grecords:
                        # initial_state=None: the imitation learner starts
                        # from zeros and never reads it (grid convention)
                        g.assembler.push_frame(rec, greset, None)
                        g._pushed += 1
            for gr in self.grids:                # each grid: ONE forward
                gr_reset = reset_np[gr.cell_env]                  # [S*Nc]
                gr_reset[~gr.valid] = True       # pads: perpetual reset
                # queue rebuild only for REAL resetting cells (pads' rows
                # are discarded; rebuilding their queues every frame would
                # be pure python churn)
                for cell in np.nonzero(gr_reset & gr.valid)[0]:
                    gr.agent.reset_cell(cell // gr.Nc, cell % gr.Nc)
                rows_all = gr.agent.execute()
                p1_rows[gr.cell_env[gr.valid]] = rows_all[gr.valid]
                gviews = self.ff.view(opp_flats, rows=gr.idx_t,
                                      lead=(gr.S, gr.Nc))
                grec = gr.agent.infer(
                    gviews, torch.as_tensor(
                        gr_reset.reshape(gr.S, gr.Nc), device=dev))
                gr.assembler.push_frame(
                    grec, torch.as_tensor(gr_reset, device=dev), None)
            sim_env.write_controller_rows(env, p1_rows, player=1)

            # ---- rewards ----
            stocks, percent = _seat_stats(obs)         # [N,2] self,opp (player-0 view)
            if self._prev is not None:
                reward = compute_reward(
                    torch.as_tensor(self._prev[0]), torch.as_tensor(stocks),
                    torch.as_tensor(self._prev[1]), torch.as_tensor(percent),
                    torch.as_tensor(reset_np)).to(dev)
                self.assembler.push_reward(reward)
                for g in self.groups:
                    if g.harvest:  # opponent seat reward = zero-sum mirror
                        g.assembler.push_reward((-reward[g.idx_t]).clone())
                for gr in self.grids:  # pads carry env0's mirror, sliced at emit
                    gr.assembler.push_reward((-reward[gr.idx_t]).clone())
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
            for gr in self.grids:
                if gr.assembler.ready():
                    traj = gr.assembler.emit()._replace(kind="imitation")
                    if not gr.valid.all():   # drop pad rows
                        traj = slice_trajectory_rows(
                            traj, np.nonzero(gr.valid)[0].tolist())
                    if gr.reencode is not None:   # phillip schema fixup
                        traj = gr.reencode(traj)
                    imit_out.append(traj)
        return ppo_out, imit_out

    def close(self):
        self.env.close()


# ---------------------------------------------------------------- opponent pool


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
                 self_frac=0.30, device="cpu", pfsp_hard_frac=0.25, pfsp_explore=0.075,
                 config_from=None, self_name_code=1, compile_fn=None):
        # phillips: {tier_id: (policy, frac, name_code)}; fox_imports: {"import:NAME": path}
        # config_from: full checkpoint whose config builds the bare-state members
        #   (snapshots + fox imports are saved as bare state_dicts, no config).
        # self_name_code: the student's conditioning code — also used for bare
        #   members (snapshots share the student's name_map; imports lack one).
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
        self._slots = []          # [skeleton, loaded_key] per PFSP slot
        self.compile_fn = compile_fn  # sample -> compiled sample, for skeletons
        self._sc = self_name_code
        self._cfg = None
        if config_from is not None:
            from smashbot import saving
            self._cfg = saving.load_checkpoint(config_from)["config"]

    # ---- policy resolution ----
    def get(self, key, slot=None):
        """(policy, name_code) for an assignment key: 'self', 'phillip:<id>',
        a snapshot path, or 'import:<name>'.

        PFSP members (snapshots + imports) are served from persistent SLOT
        skeletons: pass `slot` (0..K-1) and the member's weights are copied
        in place into that slot's policy. The skeleton object — and its
        compiled sample graph — persists across periods, so a member swap is
        a weight copy, never a recompile (cudagraph replays read weights by
        pointer)."""
        if key == "self":
            return self.current, self._sc
        if key.startswith("phillip:"):
            pol, _frac, nc = self.phillips[key.split(":", 1)[1]]
            return pol, nc
        path = self.fox_paths[key] if key.startswith("import:") else key
        assert slot is not None, "bare-state members are served from slots"
        while len(self._slots) <= slot:
            self._slots.append([None, None])  # [skeleton, loaded_key]
        sk = self._slots[slot]
        if sk[0] is None:
            sk[0] = self._make_skeleton()
        if sk[1] != key:
            self._load_into(sk[0], path)
            sk[1] = key
        # snapshots share the student's name_map exactly; fox imports lack
        # their own map -> self name_code is the correct conditioning.
        return sk[0], self._sc

    def _make_skeleton(self):
        """Empty policy built from config_from (snapshots + fox imports are
        bare state_dicts, no config of their own). compile_fn, if set, wraps
        .sample once at construction — the wrapper then serves every member
        loaded into this skeleton."""
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
        pol.train_value_head = False
        pol.requires_grad_(False)
        pol.eval()
        if self.compile_fn is not None:
            pol.sample = self.compile_fn(pol.sample)
        return pol

    def _load_into(self, skeleton, path):
        """Copy a member's weights into a skeleton IN PLACE (compiled graphs
        read parameters by pointer, so replays see the new member)."""
        import torch as _torch
        state = _torch.load(path, map_location=self.device, weights_only=True)
        with _torch.no_grad():
            res = skeleton.load_state_dict(state, strict=True)
        return res

    def get_state(self, key):
        """A PFSP member's raw state_dict (for PfspGrid.load_slice)."""
        import torch as _torch
        path = self.fox_paths[key] if key.startswith("import:") else key
        return _torch.load(path, map_location=self.device, weights_only=True)

    def make_grid_template(self):
        """Uncompiled skeleton for PfspGrid/LeagueAgent (it deep-copies the
        template and builds its own captured vmap forward)."""
        fn, self.compile_fn = self.compile_fn, None
        try:
            return self._make_skeleton()
        finally:
            self.compile_fn = fn

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
            # K distinct members, EQUAL-size slots (remainder envs fold into
            # self): every group's row count is then constant across periods,
            # so per-slot compiled graphs never see a new shape.
            members, seen = [], set()
            for _ in range(max_pfsp_members * 8):
                if len(members) >= max_pfsp_members:
                    break
                m = self.league.draw_member(rng) or "self"
                if m not in seen and m != "self":
                    seen.add(m)
                    members.append(m)
            per = len(pfsp_envs) // len(members) if members else 0
            if per == 0:
                groups["self"] += pfsp_envs
            else:
                for j, m in enumerate(members):
                    groups[m] = pfsp_envs[j * per:(j + 1) * per]
                groups["self"] += pfsp_envs[len(members) * per:]
        return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items() if len(v)}

    def record(self, key, won: bool):
        """Record a decided game outcome for a PFSP-pool member (snapshots +
        fox imports). Phillips/self aren't in the PFSP ledger."""
        if key == "self" or key.startswith("phillip:"):
            return
        self.league.record_result(key, won)
