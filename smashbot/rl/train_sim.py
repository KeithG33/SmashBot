"""Sim-backend RL training driver: melee-sim-light rollouts + the existing
PPO learner. Reached via `train_rl --backend sim`; reuses train_rl's learner,
checkpoint schema, teacher watcher, and overlap pipeline, but replaces the
Dolphin fleet with SimLeague + MultiOpponentSimWorker.

Pool design (locked): fixed self-play (no harvest) / 5 fixed phillip tiers
(harvest, medium 4% < plat 6% < diamond 7% < master 8% < gm 10%) / PFSP pool
(v10 snapshots + top-3 fox imports, harvest). Coarse re-partition every
`repartition_interval` learner steps — within a period each env keeps its
opponent, so a game reset only resets hidden state (no cross-group state
migration).

Logging: `rl/phillip/{tier}/*` per fixed tier, fox imports under
`rl/snapshots/{name}` (they sit in the PFSP ledger next to the ghosts),
`rl/self/*`, per-ghost `rl/snapshots/s{step}` win estimates.
"""
from __future__ import annotations

import dataclasses
import os
import random
import time

MODELS = "/home/kage/drive2/ShineBot/models"

# MAIN_12 -> melee_sim Character names
_MSL_CHAR = {
    "FOX": "FOX", "FALCO": "FALCO", "MARTH": "MARTH", "SHEIK": "SHEIK",
    "JIGGLYPUFF": "JIGGLYPUFF", "CPTFALCON": "FALCON", "PEACH": "PEACH",
    "YOSHI": "YOSHI", "POPO": "ICE_CLIMBERS", "LUIGI": "LUIGI",
    "PIKACHU": "PIKACHU", "SAMUS": "SAMUS",
}


@dataclasses.dataclass
class SimRolloutConfig:
    num_envs: int = 320          # measured 3090 ceiling w/ headroom (mb 6)
    unroll_length: int = 240
    data_dir: str = "/home/kage/drive2/ShineBot/msl-data"
    rollout_precision: str = "fp16"
    # --- pool shares (fractions of num_envs) ---
    self_frac: float = 0.30
    phillip_tiers: tuple[str, ...] = ("medium", "plat", "diamond", "master", "gm")
    phillip_fracs: tuple[float, ...] = (0.04, 0.06, 0.07, 0.08, 0.10)
    # everything left after self+phillips (~35%) is the PFSP pool
    # names match v10's ledger keys (import:imp9000/imp10000) so their payoff
    # rows carry over on a seeded resume; s9500 is new to the ledger
    fox_imports: tuple[str, ...] = (
        f"imp9000:{MODELS}/rl-v3-tournament1st-step0009000.pt",
        f"imp10000:{MODELS}/rl-best-step0010000-phillip56.pt",
        "s9500:/home/kage/drive2/ShineBot/runs/rl-pool-v3/snapshots/snapshot-0009500.pt",
    )
    max_pfsp_members: int = 8    # distinct resident PFSP policies per period
    repartition_interval: int = 25  # learner steps between coarse re-partitions
    # --- PFSP / snapshots (v10 values) ---
    pfsp_hard_frac: float = 0.25
    pfsp_explore: float = 0.075
    snapshot_interval: int = 1500  # snapshots are kept forever (SimLeague keep=0)
    # seed a fresh run's snapshot dir from a previous run (symlinks + pfsp.json)
    seed_snapshots_from: str = ""
    # --- matches ---
    char_whitelist: tuple[str, ...] = tuple(_MSL_CHAR)  # uniform MAIN_12
    partition_seed: int = 0


def _msl():
    import melee_sim as msl
    return msl


def _seed_snapshots(dst_dir: str, src_dir: str) -> None:
    """Symlink a previous run's snapshots + carry its pfsp.json into a fresh
    snapshot dir, so SnapshotPool adopts the archive and payoff on boot.
    Ghost payoff keys are absolute snapshot paths — rewritten to the new dir
    (else the resumed pool drops every ghost's winrate)."""
    import json
    os.makedirs(dst_dir, exist_ok=True)
    if any(f.endswith(".pt") for f in os.listdir(dst_dir)):
        return  # already populated (resumed run)
    n = 0
    src_dir = os.path.abspath(src_dir)
    for f in sorted(os.listdir(src_dir)):
        if f.endswith(".pt"):
            os.symlink(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            n += 1
    pfsp = os.path.join(src_dir, "pfsp.json")
    k = 0
    if os.path.exists(pfsp):
        with open(pfsp) as fh:
            payoff = json.load(fh)
        remapped = {
            (os.path.join(dst_dir, os.path.basename(key))
             if os.path.dirname(key) == src_dir else key): v
            for key, v in payoff.items()
        }
        k = len(remapped)
        with open(os.path.join(dst_dir, "pfsp.json"), "w") as fh:
            json.dump(remapped, fh)
    print(f"seeded {n} snapshots + {k} payoff entries from {src_dir}", flush=True)


class SimLeagueWorker:
    """Owns the SimLeague + the current-period MultiOpponentSimWorker.

    collect(n) matches DolphinRolloutWorker's contract: run until n PPO
    chunks are assembled, return them with harvested imitation chunks
    appended. maybe_repartition(step) rebuilds the worker on period
    boundaries (envs reset; opponents re-drawn; PFSP re-sampled)."""

    def __init__(self, cfg: SimRolloutConfig, league, serving_policy,
                 name_code: int, device: str):
        self.cfg = cfg
        self.lg = league
        self.policy = serving_policy
        self.name_code = name_code
        self.device = device
        self.rng = random.Random(cfg.partition_seed)
        self._worker = None
        self._grid = None                # persistent PFSP grid (captured graph)
        self._phillip_grid = None        # persistent phillip grid (5 tiers)
        self._env_char = None            # per-env opponent char name (for trackers)
        self.part: dict = {}
        from smashbot.rl.rollouts import GameTracker
        # tracker per logging class; phillip tiers get their own
        self.trackers = {
            "self": GameTracker(),
            "snapshots": GameTracker(),
            "imports": GameTracker(),
            **{f"phillip:{t}": GameTracker() for t in cfg.phillip_tiers},
        }
        self._build()

    # ---- period lifecycle ----
    def _tracker_of(self, gid: str):
        if gid == "self" or gid.startswith("phillip:"):
            return self.trackers[gid]
        if gid.startswith("import:"):
            return self.trackers["imports"]
        return self.trackers["snapshots"]

    def _on_event(self, env_i: int, gid: str, kind: str, percent: float) -> None:
        tr = self._tracker_of(gid)
        (tr.add_kill if kind == "kill" else tr.add_death)(percent)

    def _on_game(self, env_i: int, gid: str, s0: int, s1: int) -> None:
        if s0 != s1:  # ties never enter the PFSP ledger (dolphin's rule)
            self.lg.record(gid, s0 > s1)
        # char-LOCKED members (fox imports) are excluded from by_char, as in
        # the dolphin worker: a locked member ties its character's column to
        # its own strength
        char = None if gid.startswith("import:") else self._env_char[env_i]
        self._tracker_of(gid).add_game((s0, s1), char)

    def _build(self) -> None:
        msl = _msl()
        cfg = self.cfg
        N = cfg.num_envs
        self.part = self.lg.partition(N, self.rng,
                                      max_pfsp_members=cfg.max_pfsp_members)
        opponents = []
        pfsp = []                        # [(key, env_rows)] for the PFSP grid
        phillip = []                     # [(key, env_rows)] for the phillip grid
        for key, idx in self.part.items():
            if key == "self":
                pol, nc = self.lg.get(key)
                opponents.append((key, pol, idx, False, nc))
            elif key.startswith("phillip:"):
                phillip.append((key, idx))
            else:
                pfsp.append((key, idx))
        from smashbot.rl.sim_league import PfspGrid, make_reencoder
        grids = []
        # --- phillip grid: all 5 tiers on ONE stacked forward (their LSTM
        # steps via the hand-rolled cell — cuDNN has no vmap rule). Slices
        # are padded to the largest tier; members never change.
        if phillip:
            phillip.sort(key=lambda kv: kv[0])          # stable slice order
            if self._phillip_grid is None:
                tiers = [k.split(":", 1)[1] for k, _ in phillip]
                tmpl = self.lg.phillips[tiers[0]][0]
                for m in tmpl.modules():
                    if type(m).__name__ == "RecurrentWrapper":
                        m.manual_step = True
                stu_embed = self.policy.controller_head.controller_embedding
                self._phillip_grid = PfspGrid(
                    tmpl, len(phillip), max(len(r) for _, r in phillip),
                    self.name_code, cfg.unroll_length, self.device,
                    reencode=make_reencoder(
                        tmpl.controller_head.controller_embedding,
                        stu_embed, self.name_code, self.device),
                )
                for s, t in enumerate(tiers):           # per-slice name codes
                    self._phillip_grid.agent._name[s] = self.lg.phillips[t][2]
                print(f"phillip grid: {len(phillip)} tiers x "
                      f"{self._phillip_grid.Nc} cells (delay {tmpl.delay})",
                      flush=True)
            self._phillip_grid.assign(
                phillip,
                lambda key: self.lg.phillips[key.split(':', 1)[1]][0].state_dict())
            grids.append(self._phillip_grid)
        K = cfg.max_pfsp_members
        if len(pfsp) == K and len({len(r) for _, r in pfsp}) == 1:
            # full house of equal slots -> ONE captured vmap forward for all
            # PFSP members (member swaps are in-place load_slice)
            if self._grid is None:
                self._grid = PfspGrid(
                    self.lg.make_grid_template(), K, len(pfsp[0][1]),
                    self.name_code, cfg.unroll_length, self.device)
            self._grid.assign(pfsp, self.lg.get_state)
            grids.append(self._grid)
        else:
            # league too small to fill the slots (fresh run boot): fall back
            # to per-slot compiled skeletons until it grows
            for slot, (key, idx) in enumerate(pfsp):
                pol, nc = self.lg.get(key, slot=slot)
                opponents.append((key, pol, idx, True, nc))
        # per-env characters: student uniform MAIN_12; opponent uniform
        # MAIN_12 except fox imports (char-locked FOX, as in v10)
        chars = [getattr(msl.Character, _MSL_CHAR[c.upper()])
                 for c in cfg.char_whitelist]
        env_opp = {}
        for key, idx in self.part.items():
            for i in idx:
                env_opp[int(i)] = key
        char_pairs, self._env_char = [], []
        for i in range(N):
            student_c = self.rng.choice(chars)
            if env_opp[i].startswith("import:"):
                opp_c = msl.Character.FOX
            else:
                opp_c = self.rng.choice(chars)
            char_pairs.append((student_c, opp_c))
            self._env_char.append(opp_c.name)
        stages = [self.rng.choice(list(msl.Stage)) for _ in range(N)]
        if self._worker is not None:
            # new period, same agents: rebuilding re-records the cudagraph
            # trees (pool ratchet). Remap in place; full rebuild only if
            # the group structure changed (fresh-run league growth).
            new_map = {gid: idx for (gid, _p, idx, _h, _nc) in opponents}
            same = (grids == self._worker.grids
                    and {g.gid for g in self._worker.groups} == set(new_map)
                    and all(len(new_map[g.gid]) == g.n
                            for g in self._worker.groups))
            if same:
                self._worker.reassign(new_map, char_pairs, stages)
                return
            print("re-partition: group structure changed — full rebuild",
                  flush=True)
            self._worker.close()
            self._worker = None
            import gc
            gc.collect()
        from smashbot.rl.sim_league import MultiOpponentSimWorker
        self._worker = MultiOpponentSimWorker(
            self.policy, opponents, N, cfg.unroll_length, cfg.data_dir,
            stages, char_pairs, name_code=self.name_code, device=self.device,
            record_fn=self._on_game, precision=cfg.rollout_precision,
            grids=grids, event_fn=self._on_event,
        )

    def maybe_repartition(self, step: int) -> bool:
        if step > 0 and step % self.cfg.repartition_interval == 0:
            self._build()
            return True
        return False

    def env_share(self) -> dict:
        """{logging_class: env count} for the current period."""
        out: dict = {}
        for key, idx in self.part.items():
            if key == "self" or key.startswith("phillip:"):
                c = key
            elif key.startswith("import:"):
                c = "imports"
            else:
                c = "snapshots"
            out[c] = out.get(c, 0) + len(idx)
        return out

    def collect(self, num_trajectories: int) -> list:
        ppo, imit = [], []
        while len(ppo) < num_trajectories:
            p, i = self._worker.collect(self.cfg.unroll_length)
            ppo += p
            imit += i
        return ppo + imit

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()


def run(args) -> None:
    """Sim-backend main loop. `args` is train_rl.Config (with args.sim)."""
    import torch

    from smashbot import saving
    from smashbot.eval.game import load_policy, resolve_name_code
    from smashbot.rl.ppo import Learner
    from smashbot.rl.sim_league import SimLeague
    from smashbot.rl.train_rl import build_value_function, _save_rl_checkpoint

    scfg: SimRolloutConfig = args.sim
    device = args.runtime.device
    assert device == "cuda", "sim backend is a GPU training path"
    if os.environ.get("SMASHBOT_MEMDEBUG"):
        # allocator history with python stacks; dumped on OOM (below) for
        # torch.cuda.memory._snapshot analysis
        torch.cuda.memory._record_memory_history(max_entries=200000)

    policy, name_map, step = load_policy(args.ckpt, device)
    policy.train_value_head = False
    teacher, _, _ = load_policy(args.ckpt, device)
    teacher.train_value_head = False
    ckpt = saving.load_checkpoint(args.ckpt)
    value_fn = build_value_function(ckpt["config"], device)
    value_fn.load_state_dict(ckpt["state"]["value"])
    name_code = resolve_name_code(name_map, args.runtime.name)
    print(f"teacher/init: {args.ckpt} (BC step {step}); conditioning code {name_code}")

    learner = Learner(args.learner, policy, teacher, value_fn)
    print(f"learner precision: {learner.precision}")

    # ---- restore (same schema as the dolphin path) ----
    start_step = 0
    restored_trackers = None
    run_dir = f"{args.runtime.run_dir}/{args.runtime.tag}"
    if args.runtime.restore:
        rpath = args.runtime.restore
        if rpath == "auto":
            rpath = f"{run_dir}/latest.pt"
            if not os.path.exists(rpath):
                rpath = ""
                print("restore auto: no checkpoint yet, starting fresh")
        if rpath:
            rl_ckpt = saving.load_checkpoint(rpath)
            policy.load_state_dict(rl_ckpt["state"]["policy"])
            value_fn.load_state_dict(rl_ckpt["state"]["value"])
            if "policy_opt" in rl_ckpt["state"]:
                learner.policy_optimizer.load_state_dict(rl_ckpt["state"]["policy_opt"])
                learner.value_optimizer.load_state_dict(rl_ckpt["state"]["value_opt"])
            start_step = rl_ckpt["state"]["step"] + 1
            restored_trackers = rl_ckpt["state"].get("trackers")
            print(f"restored RL run from {rpath} at step {start_step}")
    _save_rl_checkpoint.policy_opt = learner.policy_optimizer
    _save_rl_checkpoint.value_opt = learner.value_optimizer

    # ---- serving copy + compile (train_rl's overlap pattern) ----
    import copy as _copy
    serving_policy = _copy.deepcopy(policy)
    serving_policy.requires_grad_(False).eval()
    serving_policy.train_value_head = False
    print("learner overlap: ON — student serves a published weight copy; "
          "rollouts are one update stale", flush=True)
    if args.runtime.compile:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 128
        serving_policy.sample = torch.compile(serving_policy.sample, mode="reduce-overhead")

    # ---- league ----
    snap_dir = f"{run_dir}/snapshots"
    if scfg.seed_snapshots_from:
        _seed_snapshots(snap_dir, scfg.seed_snapshots_from)
    phillips = {}
    for tier, frac in zip(scfg.phillip_tiers, scfg.phillip_fracs):
        fname = "medium-v2-torch.pt" if tier == "medium" else f"{tier}-torch.pt"
        path = f"{MODELS}/{fname}"
        pol, pnm, _ = load_policy(path, device)
        pol.train_value_head = False
        pol.requires_grad_(False)
        pol.eval()
        # all tiers serve from the phillip grid (one stacked forward)
        phillips[tier] = (pol, frac, resolve_name_code(pnm, "Master Player"))
        print(f"phillip:{tier} <- {fname} ({frac:.0%} of envs)")
    fox = {}
    for spec in scfg.fox_imports:
        name, path = spec.split(":", 1)
        assert os.path.exists(path), f"fox import {name}: {path} missing"
        fox[f"import:{name}"] = path
        print(f"import:{name} <- {path} (FOX lock)")
    league = SimLeague(
        serving_policy, snap_dir, phillips=phillips, fox_imports=fox,
        self_frac=scfg.self_frac, device=device,
        pfsp_hard_frac=scfg.pfsp_hard_frac, pfsp_explore=scfg.pfsp_explore,
        config_from=args.ckpt, self_name_code=name_code,
        # PFSP slot skeletons compile once per slot shape; member swaps are
        # in-place weight copies visible to the captured graphs
        compile_fn=(
            (lambda s: torch.compile(s, mode="reduce-overhead"))
            if args.runtime.compile else None
        ),
    )
    if not league.league.archive:
        league.league.save(policy, start_step)
        print(f"boot snapshot: seeded empty archive at step {start_step}", flush=True)
    worker = SimLeagueWorker(scfg, league, serving_policy, name_code, device)
    print(f"sim league: {len(worker.part)} groups over {scfg.num_envs} envs "
          f"(re-partition every {scfg.repartition_interval} steps)", flush=True)
    if restored_trackers:
        # dolphin-run kinds -> sim tracker keys ("reference" was the
        # dedicated medium-v2 phillip in v10)
        remap = {"snapshot": "snapshots", "import": "imports",
                 "reference": "phillip:medium"}
        loaded = []
        for kind, st in restored_trackers.items():
            key = remap.get(kind, kind)
            if key in worker.trackers:
                worker.trackers[key].load_state(st)
                loaded.append(key)
        print(f"tracker EMAs restored for {sorted(loaded)}")
    _save_rl_checkpoint.tracker_states = lambda: {
        k: t.state() for k, t in worker.trackers.items()
    }

    import wandb
    wandb.init(
        project="shinebot", id=args.runtime.wandb_id or args.runtime.tag,
        name=args.runtime.tag, mode=args.runtime.wandb_mode,
        config=dataclasses.asdict(args), resume="allow",
    )

    state = learner.initial_state(scfg.num_envs, device)
    torch.cuda.synchronize()
    print(f"[vram] boot complete: alloc {torch.cuda.memory_allocated()/2**30:.2f} "
          f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
    t0 = time.time()

    league.league.payoff_autosave = False  # flushed below, not per game

    def _pre_step(i):
        if (i + 1) % args.runtime.checkpoint_interval == 0:
            league.league._save_payoff()   # debounced ledger flush
        if i > 0 and i % scfg.snapshot_interval == 0:
            path = league.league.save(policy, i)
            print(f"[{i}] snapshot saved: {os.path.basename(path)} joins the league",
                  flush=True)
        if i != start_step and worker.maybe_repartition(i):
            # no learner-state reset needed: the fresh worker's first
            # trajectories carry is_resetting=True on frame 0, which zeroes
            # the learner-side carried state per env
            print(f"[{i}] re-partitioned: "
                  + " ".join(f"{k}={v}" for k, v in sorted(worker.env_share().items())),
                  flush=True)

    def _post_step(i, metrics):
        if (i + 1) % args.runtime.checkpoint_interval == 0:
            _save_rl_checkpoint(f"{run_dir}/latest.pt", ckpt["config"], policy,
                                value_fn, name_map, i, args.ckpt)
        if i % args.runtime.log_interval != 0:
            return
        frames = ((i + 1 - start_step) * args.runtime.trajectories_per_step
                  * scfg.num_envs * scfg.unroll_length)

        def _quiet(k, v):
            return v == 0 and (k.startswith("nf_") or k.endswith("_nonfinite")
                               or k == "anomalous_samples")

        log = {"rl/" + k: v for k, v in metrics["post_update"].items()
               if not _quiet(k, v)}
        log.update({"rl/value_" + k: v for k, v in metrics["value"].items()
                    if not _quiet(k, v)})
        log["rl/reverted"] = float(metrics["reverted"])
        if learner.grad_scaler is not None:
            log["rl/grad_scaler_scale"] = learner.grad_scaler.get_scale()
        im = metrics.get("imitation")
        if im:
            for k in ("loss", "w_mean", "w_max", "traj_count", "lambda"):
                log[f"rl/imitation/{k}"] = im[k]

        # ---- league panels ----
        share = worker.env_share()
        pool = league.league
        for cls, cnt in share.items():
            log[f"rl/{cls.replace(':', '/')}/envs"] = cnt
        for tier in scfg.phillip_tiers:            # per-tier winrate panels
            tr = worker.trackers[f"phillip:{tier}"]
            if tr.wins + tr.losses + tr.draws:
                for k, v in tr.stats().items():
                    log[f"rl/phillip/{tier}/{k}"] = v
        for cls in ("self", "snapshots", "imports"):
            tr = worker.trackers[cls]
            if tr.wins + tr.losses + tr.draws:
                for k, v in tr.stats().items():
                    log[f"rl/{cls}/{k}"] = v
        for g_path in pool.archive:                # per-ghost win estimates
            log[f"rl/snapshots/s{pool._step_of(g_path):07d}"] = (
                pool.win_estimate(g_path))
        for key in fox:                            # fox imports live w/ ghosts
            row = pool.payoff.get(key)
            if row and row.get("games"):
                log[f"rl/snapshots/{key.split(':', 1)[1]}"] = pool.win_estimate(key)
        log["rl/frames_per_sec"] = frames / max(1e-9, time.time() - t0)
        log["rl/frames"] = frames
        wandb.log(log, step=i)

        if i % 5 == 0:
            games = sum(t.wins + t.losses + t.draws for t in worker.trackers.values())
            phil_bits = " ".join(
                f"{t[:2]}{worker.trackers[f'phillip:{t}'].win_ema:.2f}"
                for t in scfg.phillip_tiers
                if worker.trackers[f"phillip:{t}"].wins + worker.trackers[f"phillip:{t}"].losses)
            print(f"[{i}] {phil_bits} ({games:.0f}g) | "
                  f"tKL {log.get('rl/teacher_kl', float('nan')):.4f} "
                  f"aKL {log.get('rl/actor_kl_mean', float('nan')):.5f} "
                  f"{'REVERTED ' if log['rl/reverted'] else ''}| "
                  f"{log['rl/frames_per_sec']:.0f} fps", flush=True)

    # ---- publish + overlap pipeline (train_rl's pattern) ----
    import concurrent.futures
    overlap_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="learner")
    overlap_stream = torch.cuda.Stream()
    _s_sd = serving_policy.state_dict()
    _p_sd = policy.state_dict()
    _pub_pairs = [(_s_sd[k], _p_sd[k]) for k in _s_sd]

    def _publish():
        with torch.no_grad():
            for dst, src in _pub_pairs:
                dst.copy_(src)

    try:
        fut = None
        fut_i = None
        for i in range(start_step, args.runtime.steps):
            trajectories = worker.collect(args.runtime.trajectories_per_step)
            if i < start_step + 5 or os.environ.get("SMASHBOT_PROFILE"):
                print(f"[vram] post-collect {i}: "
                      f"alloc {torch.cuda.memory_allocated()/2**30:.2f} "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} "
                      f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
            if fut is not None:
                state, metrics = fut.result()
                _post_step(fut_i, metrics)
            _pre_step(i)  # θ final: previous update joined
            _publish()
            if i == start_step:
                # FIRST learner step runs sequentially: its one-time
                # allocations (cuBLAS/cuDNN workspaces, autocast caches)
                # must not co-peak with a concurrent collect.
                state, metrics = learner.step(
                    trajectories, state,
                    progress=i / max(1, args.runtime.steps))
                _post_step(i, metrics)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"[vram] post-first-step: "
                      f"alloc {torch.cuda.memory_allocated()/2**30:.2f} "
                      f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB",
                      flush=True)
                continue
            ready = torch.cuda.Event()
            ready.record()

            def _run(traj=trajectories, st=state, ev=ready, step_i=i,
                     prog=i / max(1, args.runtime.steps)):
                _t0 = time.perf_counter()
                overlap_stream.wait_event(ev)
                with torch.cuda.stream(overlap_stream):
                    out = learner.step(traj, st, progress=prog)
                overlap_stream.synchronize()
                if os.environ.get("SMASHBOT_PROFILE"):
                    print(f"[learner] step {step_i}: "
                          f"{time.perf_counter() - _t0:.1f}s | "
                          f"alloc {torch.cuda.memory_allocated() / 2**30:.2f} "
                          f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB",
                          flush=True)
                return out

            fut, fut_i = overlap_pool.submit(_run), i
            # drop the loop's redundant reference: the closure keeps the
            # trajectories alive for the learner; without this they also
            # survive the whole NEXT collect (~0.5-1 GiB of dead co-peak)
            trajectories = None
        if fut is not None:
            state, metrics = fut.result()
            _post_step(fut_i, metrics)
    except torch.OutOfMemoryError:
        if os.environ.get("SMASHBOT_MEMDEBUG"):
            snap = f"{run_dir}/oom_snapshot.pickle"
            torch.cuda.memory._dump_snapshot(snap)
            print(f"[memdebug] OOM snapshot dumped to {snap}", flush=True)
        raise
    finally:
        worker.close()
        overlap_pool.shutdown(wait=False)
