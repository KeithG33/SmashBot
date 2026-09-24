"""Multi-opponent sim rollout (v10's league design on melee-sim-light).

Env layout is STATIC: self envs / phillip-tier envs / PFSP envs, fixed for
the run — no periods, no re-partition. Every game runs to its natural end
(stock-out or the 8-minute timer); at each env's own game boundary the
worker records the result, re-draws the match (chars, stage, ports, seed)
and, for PFSP envs, re-seats the env on the grid for the opponent drawn a
game ahead (rl/league.py: weights only ever load into EMPTY slices, so no
seat is swapped mid-game).

Learner rows = every env's student seat + the second seat of each self
env, all served by ONE student forward (v10's row layout: rows are the
VRAM budget, envs are cheap). Opponent seats: the phillip grid (static
cells) and the PFSP grid (dynamic cells), one stacked forward each; every
grid seat is harvested as a replay (rollouts.HarvestAssembler).
"""
from __future__ import annotations

import random as _random

import numpy as np
import torch

from smashbot.rl.agent import BatchedPolicyAgent
from smashbot.networks import check_loadable
from smashbot.rl.league import League, LeagueSeats, MemberWeights
from smashbot.rl.pool import SnapshotPool
from smashbot.rl.rollouts import ChunkAssembler, HarvestAssembler, compute_reward
from smashbot.rl import sim_env
from smashbot.rl.sim_env import seat_stats, states_to_torch  # noqa: F401


class PfspGrid:
    """S weight slices x Nc cells on ONE LeagueAgent (one captured vmap
    forward per frame). Cells are seats: `seat`/`unseat`/`move` keep the
    cell->env map that the per-frame gather uses; idle cells forward
    garbage against env 0 with reset held high and their rows are sliced
    out of every emitted chunk. kept() tells the harvest which cells held
    one occupant since the last frame (v10: a cell's rows enter a chunk only
    if it was occupied for the WHOLE chunk).

    Static use (the phillip tiers): assign_static() seats contiguous env
    rows once. Dynamic use (PFSP): rl/league.League drives seat changes at
    game boundaries; load() is the LeagueSeats loader."""

    def __init__(self, template, slices, cells, name_code, device):
        from smashbot.rl.agent import LeagueAgent
        self.S, self.Nc = slices, cells
        cuda = torch.device(device).type == "cuda"
        dt = torch.float16 if cuda else torch.float32
        self.agent = LeagueAgent(template, slices, cells, name_code, device,
                                 weights_dtype=dt,
                                 state_dtype=torch.float16 if cuda else None)
        self.device = device
        self.members = [None] * slices
        n = slices * cells
        self.cell_env = np.zeros(n, dtype=np.int64)
        self.valid = np.zeros(n, dtype=bool)
        self.changed = np.zeros(n, dtype=bool)
        self._cell_of: dict[int, int] = {}
        self._dirty = True
        self.idx_t = self.valid_t = None
        self.sync()

    # ---- weights ----
    def load(self, s, key, get_state):
        self.agent.load_slice(s, get_state(key))
        self.members[s] = key

    # ---- seating ----
    def assign_static(self, rows_per_slice):
        for s, rows in enumerate(rows_per_slice):
            assert len(rows) <= self.Nc, (s, len(rows), self.Nc)
            for n, e in enumerate(rows):
                self.seat(int(e), s, n)

    def seat(self, env, s, n):
        cell = s * self.Nc + n
        assert not self.valid[cell], f"cell {cell} occupied"
        self.cell_env[cell] = env
        self.valid[cell] = True
        self._cell_of[env] = cell
        self.agent.reset_cell(s, n)
        self._dirty = True

    def unseat(self, env):
        cell = self._cell_of.pop(env)
        self.valid[cell] = False
        self.changed[cell] = True
        self._dirty = True

    def move(self, src, dst):
        """LeagueSeats compaction: exact cell-state move between slices of
        the same member. Both cells drop out of the current harvest chunk."""
        self.agent.move_cell(src, dst)
        a = src[0] * self.Nc + src[1]
        b = dst[0] * self.Nc + dst[1]
        env = int(self.cell_env[a])
        self.cell_env[b] = env
        self.valid[b], self.valid[a] = True, False
        self.changed[a] = self.changed[b] = True
        self._cell_of[env] = b
        self._dirty = True

    def kept(self) -> np.ndarray:
        """Cells whose occupant is the previous frame's; read once per frame."""
        kept = self.valid & ~self.changed
        self.changed[:] = False
        return kept

    def sync(self):
        if self._dirty:
            self.idx_t = torch.as_tensor(self.cell_env, device=self.device)
            self.valid_t = torch.as_tensor(np.nonzero(self.valid)[0],
                                           device=self.device)
            self._dirty = False


class _Group:
    """One opponent identity serving a fixed subset of env rows (the eval
    arena's single opponent; not used by training)."""

    def __init__(self, gid, policy, env_idx, device, name_code,
                 precision="fp32", state_dtype=None, capture=False):
        self.gid = gid
        self.env_idx = np.asarray(env_idx, dtype=np.int64)
        self.idx_t = torch.as_tensor(self.env_idx, device=device)
        self.n = len(self.env_idx)
        self.agent = BatchedPolicyAgent(policy, self.n, name_code=name_code,
                                        device=device, precision=precision,
                                        state_dtype=state_dtype, capture=capture)
        self._reset = np.ones(self.n, dtype=bool)


class MultiOpponentSimWorker:
    def __init__(self, student_policy, opponents, batch_size, unroll_length, data_dir,
                 stage, char_pairs, name_code=1, device="cpu", record_fn=None,
                 precision="fp32", grids=(), event_fn=None, self_idx=(),
                 league=None, pfsp_grid=None, match_fn=None, max_frame=28800,
                 seed=0, capture=False):
        """opponents: [(gid, policy, env_idx, name_code)] fixed groups (eval
        arena; not harvested). grids: static PfspGrids (phillip tiers), env ->
        gid via gid_of_env below. self_idx: envs whose player-1 seat is the
        student too; those seats are learner rows N.. (v10 layout).
        league/pfsp_grid: per-match PFSP routing (rl/league.League) over a
        dynamic PfspGrid. match_fn(env, member) -> (MatchConfig, info): the
        match for env's NEXT game (called at boot and every game end);
        default = the fixed stage/char_pairs given, fresh seed per game.
        record_fn(env, gid, s0, s1) / event_fn(env, gid, kind, pct) are
        called with the gid of the game the frames belong to; game_info[env]
        holds match_fn's info for that game."""
        import melee_sim as msl
        self.msl = msl
        self.N = batch_size
        self.unroll = unroll_length
        self.device = device
        self.record_fn = record_fn
        self.event_fn = event_fn
        self.grids = list(grids)
        self.pfsp_grid = pfsp_grid
        self.league = league
        self.self_idx = np.asarray(list(self_idx), dtype=np.int64)
        self.self_idx_t = torch.as_tensor(self.self_idx, device=device)
        self.rows = batch_size + len(self.self_idx)
        self.student = BatchedPolicyAgent(
            student_policy, self.rows, name_code=name_code, device=device,
            precision=precision, capture=capture,
            # capture's static state buffers hold fp16-computed values; fp32
            # storage costs an up/down cast per layer per frame (12.9 -> 9.4 ms
            # at 400 rows) and the snapshot values are identical
            state_dtype=torch.float16 if capture and precision == "fp16" else None)
        self.student.set_flat_controllers(True)
        self.ff = sim_env.FlatFrames(device)
        self.item_slots = sim_env.ItemSlots(batch_size)
        self.student.set_flat_inputs(self.ff.view)
        for gr in self._all_grids():   # phillip grid AND the PFSP grid step through flats
            gr.agent.set_flat_inputs(self.ff.view)
        self.assembler = ChunkAssembler(unroll_length, student_policy.delay)
        self._pushed = 0
        self._prev = None
        self._reset_mask = np.ones(batch_size, dtype=bool)
        self.groups = [_Group(gid, pol, idx, device, nc, precision=precision, capture=capture)
                       for (gid, pol, idx, nc) in opponents]
        self.harvests = [
            HarvestAssembler(unroll_length, student_policy.delay,
                             student_policy.controller_head.controller_embedding, name_code)
            for _ in self._all_grids()]
        for g in self.groups:
            g.agent.set_flat_controllers(True)
            g.agent.set_flat_inputs(self.ff.view)
        # env -> gid of the game its frames belong to (committed at the
        # entry frame; pending between a game's end and its successor's
        # first frame so terminal-transition events credit the right game)
        self.env_opp = np.empty(batch_size, dtype=object)
        self.game_info = [None] * batch_size
        self._pending: dict[int, tuple] = {}
        self.env_opp[self.self_idx] = "self"
        for g in self.groups:
            self.env_opp[g.env_idx] = g.gid
        for gr in self.grids:
            for s in range(gr.S):
                for n in range(gr.Nc):
                    c = s * gr.Nc + n
                    if gr.valid[c]:
                        self.env_opp[gr.cell_env[c]] = gr.members[s]
        self._pfsp_envs = set()
        if league is not None:
            for e, m in league.member_now.items():
                self.env_opp[e] = m
                self._pfsp_envs.add(e)
        covered = np.concatenate(
            [self.self_idx] + [g.env_idx for g in self.groups]
            + [gr.cell_env[gr.valid] for gr in self.grids]
            + ([pfsp_grid.cell_env[pfsp_grid.valid]] if pfsp_grid is not None else []))
        assert sorted(covered.tolist()) == list(range(batch_size)), \
            "seats must partition all envs"
        self.max_frame = max_frame
        self._rng = _random.Random(seed)
        if match_fn is None:
            stages = stage if isinstance(stage, (list, tuple)) else [stage] * batch_size

            def match_fn(e, member):
                a, b = char_pairs[e]
                cfg = msl.MatchConfig(
                    stage=stages[e], players=(msl.PlayerConfig(a), msl.PlayerConfig(b)),
                    seed=self._rng.getrandbits(31), max_frame=self.max_frame)
                return cfg, None
        self.match_fn = match_fn
        self.env = msl.EnvBatch(batch_size=batch_size, length=max(64, unroll_length + 1),
                                data_dir=data_dir)
        cfgs = []
        for e in range(batch_size):
            cfg, info = self.match_fn(e, self.env_opp[e])
            self.game_info[e] = info
            cfgs.append(cfg)
        self.env.configure_matches(cfgs)
        self.env.reset_all()

    # ---- per-frame loop ----
    def _all_grids(self):
        return self.grids + ([self.pfsp_grid] if self.pfsp_grid is not None else [])

    def collect(self, num_frames):
        ppo_out, imit_out = [], []
        env, dev, T = self.env, self.device, self.unroll
        N = self.N
        for _ in range(num_frames):
            if env.t >= env.length:
                env.reset_cursor()
            if self.league is not None:
                self.league.tick()          # drawn-ahead members -> empty slices
            obs = env.current_frame
            reset_np = self._reset_mask
            for e in np.nonzero(reset_np)[0]:      # entry frames: commit
                pend = self._pending.pop(int(e), None)
                if pend is not None:
                    self.env_opp[e], self.game_info[e] = pend
            reset_t = torch.as_tensor(reset_np, device=dev)
            reset_rows_np = np.concatenate([reset_np, reset_np[self.self_idx]])
            reset_rows = torch.as_tensor(reset_rows_np, device=dev)

            # ---- student rows: every env's player 0 + self envs' player 1 ----
            # execute (pop) BEFORE infer: a reset frame's queue rebuild would
            # otherwise discard the fresh sample (delay-1 forever)
            rows = np.stack(self.student.execute(np.nonzero(reset_rows_np)[0].tolist()))
            sim_env.write_controller_rows(env, rows[:N], player=0)
            items = self.item_slots.place(obs["items"], reset_np)
            flats = self.ff.to_device(sim_env.encode_flats(obs, items))
            opp_flats = self.ff.swap(flats)
            if len(self.self_idx):
                row_flats = tuple(torch.cat([a, b.index_select(0, self.self_idx_t)])
                                  for a, b in zip(flats, opp_flats))
            else:
                row_flats = flats
            states = self.ff.view(row_flats)
            want = (self._pushed % T == 0)
            records, hidden_before = self.student.infer(states, reset_rows, want_snapshot=want,
                                                        flats=row_flats)

            # ---- opponent seats (player 1) ----
            p1_rows = np.empty((N, 13), dtype=np.float32)
            if len(self.self_idx):
                p1_rows[self.self_idx] = rows[N:]
            for g in self.groups:
                p1_rows[g.env_idx] = np.stack(
                    g.agent.execute(np.nonzero(g._reset)[0].tolist()))
                gflats = tuple(t.index_select(0, g.idx_t) for t in opp_flats)
                gstates = self.ff.view(gflats)
                g.agent.infer(gstates, torch.as_tensor(g._reset, device=dev),
                              want_snapshot=False, flats=gflats)
            for gr, harvest in zip(self._all_grids(), self.harvests):   # each grid: ONE forward
                gr.sync()
                gr_reset = reset_np[gr.cell_env]
                gr_reset[~gr.valid] = True         # idle cells: perpetual reset
                for cell in np.nonzero(gr_reset & gr.valid)[0]:
                    gr.agent.reset_cell(cell // gr.Nc, cell % gr.Nc)
                rows_all = gr.agent.execute()
                p1_rows[gr.cell_env[gr.valid]] = rows_all[gr.valid]
                gflats = tuple(t.index_select(0, gr.idx_t).view(gr.S, gr.Nc, t.shape[-1])
                               for t in opp_flats)
                gviews = self.ff.view(gflats)
                grec = gr.agent.infer(
                    gviews, torch.as_tensor(gr_reset.reshape(gr.S, gr.Nc), device=dev),
                    flats=gflats)
                harvest.push_frame(grec.state, rows_all, torch.as_tensor(gr_reset, device=dev),
                                   gr.kept())
            sim_env.write_controller_rows(env, p1_rows, player=1)

            # ---- rewards ----
            stocks, percent = seat_stats(obs)
            if self._prev is not None and self.event_fn is not None:
                ps, pp = self._prev
                live = ~reset_np
                for i in np.nonzero(live & (stocks[:, 1] < ps[:, 1]))[0]:
                    self.event_fn(int(i), self.env_opp[i], "kill", float(pp[i, 1]))
                for i in np.nonzero(live & (stocks[:, 0] < ps[:, 0]))[0]:
                    self.event_fn(int(i), self.env_opp[i], "death", float(pp[i, 0]))
            if self._prev is not None:
                reward = compute_reward(
                    torch.as_tensor(self._prev[0]), torch.as_tensor(stocks),
                    torch.as_tensor(self._prev[1]), torch.as_tensor(percent),
                    torch.as_tensor(reset_np)).to(dev)
                if len(self.self_idx):
                    self.assembler.push_reward(
                        torch.cat([reward, -reward.index_select(0, self.self_idx_t)]))
                else:
                    self.assembler.push_reward(reward)
                for gr, harvest in zip(self._all_grids(), self.harvests):
                    harvest.push_reward((-reward[gr.idx_t]).clone())
            self._prev = (stocks, percent)

            for rec in records:
                snap = hidden_before if self._pushed % T == 0 else None
                self.assembler.push_frame(rec, reset_rows, snap)
                self._pushed += 1

            is_resetting, term = env.step_and_reset()
            self._reset_mask = np.asarray(is_resetting, dtype=bool)
            done = np.asarray(term["done"], dtype=bool)
            if done.any():
                fs, _ = seat_stats(env.current_frame)   # terminal frame
                for i in np.nonzero(done)[0]:
                    self._on_done(int(i), int(fs[i, 0]), int(fs[i, 1]))
            for g in self.groups:
                g._reset = self._reset_mask[g.env_idx]

            if self.assembler.ready():
                ppo_out.append(self.assembler.emit())
            for harvest in self.harvests:
                if harvest.ready():
                    traj = harvest.emit()
                    if traj is not None:
                        imit_out.append(traj)
        return ppo_out, imit_out

    def _on_done(self, e, s0, s1):
        """env e's game ended this step (terminal frame published; the sim
        resets it on the next step). Record, re-seat (PFSP), re-draw the
        match. The new gid/info commit on the entry frame."""
        if self.record_fn is not None:
            self.record_fn(e, self.env_opp[e], s0, s1)
        if e in self._pfsp_envs:
            self.pfsp_grid.unseat(e)
            s, n = self.league.on_boundary(e)
            self.pfsp_grid.seat(e, s, n)
            member = self.league.member_now[e]
        else:
            member = self.env_opp[e]
        cfg, info = self.match_fn(e, member)
        self.env.configure_matches([cfg], env_ids=[e])
        self._pending[e] = (member, info)

    def close(self):
        self.env.close()


# ---------------------------------------------------------------- opponent pool


class SimLeague:
    """Opponent pool for the sim league: self / phillip tiers / PFSP pool
    (snapshots through SnapshotPool's draw + payoff ledger, pfsp.json — a
    resume loads the archive and table straight from the snapshot dir).
    layout() fixes the env rows once per run."""

    def __init__(self, snapshot_dir, phillips,
                 self_frac=0.30, device="cpu", pfsp_hard_frac=0.25, pfsp_explore=0.075,
                 config_from=None):
        # phillips: {tier: (policy, frac, name_code)}
        self.device = device
        self.self_frac = self_frac
        self.phillips = phillips
        self.league = SnapshotPool(
            snapshot_dir, keep=0,
            pfsp_hard_frac=pfsp_hard_frac, pfsp_explore=pfsp_explore,
        )
        self.weights = MemberWeights(lambda path: path)   # members are snapshot paths
        self._cfg = None
        if config_from is not None:
            from smashbot import saving
            self._cfg = saving.load_checkpoint(config_from)["config"]

    def _make_skeleton(self):
        """Empty policy built from config_from (snapshots are bare
        state_dicts, no config of their own)."""
        from smashbot.policy import build_policy_from_config
        if self._cfg is None:
            raise RuntimeError("SimLeague needs config_from to load bare-state members")
        pol = build_policy_from_config(self._cfg).to(self.device)
        pol.requires_grad_(False)
        pol.eval()
        return pol

    def _load_into(self, skeleton, path):
        import torch as _torch
        state = _torch.load(path, map_location=self.device, weights_only=True)
        check_loadable({}, state)   # bare weights carry no config: only today's names pass
        with _torch.no_grad():
            return skeleton.load_state_dict(state, strict=True)

    def get_state(self, key):
        state = self.weights.get(key)
        check_loadable({}, state)   # bare weights carry no config: only today's names pass
        return state

    def make_grid_template(self):
        return self._make_skeleton()

    def layout(self, N):
        """Static env rows: {'self': rows, 'phillip:<tier>': rows, 'pfsp': rows}."""
        cur = 0
        out = {}

        def take(n):
            nonlocal cur
            rows = np.arange(cur, cur + n, dtype=np.int64)
            cur += n
            return rows

        out["self"] = take(round(self.self_frac * N))
        for tier, (_pol, frac, _nc) in self.phillips.items():
            out[f"phillip:{tier}"] = take(round(frac * N))
        out["pfsp"] = np.arange(cur, N, dtype=np.int64)
        return out

    def make_league(self, seats, rng):
        return League(self.league, seats, locks={}, rng=rng,
                      warm=self.weights.warm, ready=self.weights.ready)

    def record(self, key, won: bool):
        """Decided game outcome for a PFSP member; phillips/self aren't in
        the ledger."""
        if key == "self" or key.startswith("phillip:"):
            return
        self.league.record_result(key, won)
