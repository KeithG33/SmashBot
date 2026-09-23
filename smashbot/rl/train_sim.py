"""Sim-backend RL training driver: melee-sim-light rollouts + the existing
PPO learner. Reached via `train_rl --backend sim`; reuses train_rl's learner,
checkpoint schema and overlap pipeline over SimLeague + MultiOpponentSimWorker.

Pool design (locked): self-play (both seats are learner rows, no harvest) /
5 fixed phillip tiers (harvest, medium 4% < plat 6% < diamond 7% < master 8%
< gm 10%) / PFSP pool (v10 snapshots + top-3 fox imports, harvest, drawn PER
MATCH with replacement and routed on the PFSP grid at each env's own game
boundary — rl/league.py). Env layout is static; every game runs to its end.

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
    # learner rows = num_envs + self envs (each self env feeds BOTH seats,
    # v10's layout); rows are the VRAM budget: 345 envs @ self_frac .30
    # = 449 rows. Shares are of ENVS: self 30 / phillips 35 / pfsp 35 ->
    # rows self 46% / phillips 27% / pfsp 27%; 120 pfsp envs over 60
    # slices = 2.0 games per loaded brain (fewer games per slice is what
    # keeps the per-match draw honest)
    num_envs: int = 345
    unroll_length: int = 240
    data_dir: str = "/home/kage/drive2/ShineBot/msl-data"
    rollout_precision: str = "fp16"
    # Manual static-buffer CUDA graph for the student forward. Its state
    # buffers are fp16 (sim_league): with fp32 statics the graph paid an
    # up/down cast per layer per frame and lost to cudagraph trees at fp16
    # (12.9 vs 12.0 ms @400 rows); with fp16 statics it wins (9.4 ms).
    capture_serving: bool = True
    # --- pool shares (fractions of num_envs) ---
    self_frac: float = 0.30       # of envs (row share 2s/(1+s))
    phillip_tiers: tuple[str, ...] = ("medium", "plat", "diamond", "master", "gm")
    phillip_fracs: tuple[float, ...] = (0.04, 0.06, 0.07, 0.08, 0.10)  # 35% of envs
    # everything left after self+phillips (~35%) is the PFSP pool
    # names match v10's ledger keys (import:imp9000/imp10000/imp9500) so
    # their payoff rows carry over on a seeded run
    fox_imports: tuple[str, ...] = (
        f"imp9000:{MODELS}/rl-v3-tournament1st-step0009000.pt",
        f"imp10000:{MODELS}/rl-best-step0010000-phillip56.pt",
        "imp9500:/home/kage/drive2/ShineBot/runs/rl-pool-v3/snapshots/snapshot-0009500.pt",
    )
    # PFSP grid weight slices = resident members. v10 ran 36 slices x 4
    # cells so a per-match draw usually found its member resident; each
    # fp16 slice is ~54 MB; 60 slices for 120 pfsp envs = 2.0 envs/slice,
    # past the knee where per-match draws find an empty slice (v10: 2.6) —
    # simulated 0% fallback on the curated league; watch rl/league/*
    pfsp_slices: int = 60
    max_game_frames: int = 28800  # Melee's 8-minute timer (60 fps)
    # --- PFSP / snapshots (v10 values) ---
    pfsp_hard_frac: float = 0.25
    pfsp_explore: float = 0.075
    snapshot_interval: int = 1500  # kept forever (keep=0): ~40 new ghosts by 100k
    # seed a fresh run's snapshot dir from a previous run (symlinks + pfsp.json),
    # curated: ghosts from seed_min_step on, the seed_keep_best HARDEST by
    # ledger winrate (Keith: early ghosts are weak; fewer members keep the
    # PFSP grid's fallback rate low)
    seed_snapshots_from: str = ""
    seed_min_step: int = 17000
    seed_keep_best: int = 30
    # --- matches ---
    char_whitelist: tuple[str, ...] = tuple(_MSL_CHAR)  # uniform MAIN_12
    seed: int = 0                 # match draws (chars/stage/ports/engine seed)


def _msl():
    import melee_sim as msl
    return msl


def _seed_snapshots(dst_dir: str, src_dir: str, min_step: int = 0,
                    keep_best: int = 0) -> None:
    """Symlink a previous run's snapshots + carry its pfsp.json into a fresh
    snapshot dir (SnapshotPool adopts archive + payoff on boot): ghosts from
    min_step on, the keep_best hardest by ledger winrate (0 = all). Ghost
    payoff keys are absolute snapshot paths — rewritten to the new dir
    (else every ghost's winrate is dropped)."""
    import json
    from smashbot.rl.pool import SnapshotPool
    os.makedirs(dst_dir, exist_ok=True)
    if any(f.endswith(".pt") for f in os.listdir(dst_dir)):
        return  # already populated (resumed run)
    src_dir = os.path.abspath(src_dir)
    src = SnapshotPool(src_dir, keep=0, pfsp=True)
    src.payoff_autosave = False
    cands = [g for g in src.archive if src._step_of(g) >= min_step]
    cands.sort(key=src.win_estimate)          # hardest (lowest student winrate) first
    kept = sorted(cands[:keep_best] if keep_best else cands)
    for g in kept:
        os.symlink(g, os.path.join(dst_dir, os.path.basename(g)))
    pfsp = os.path.join(src_dir, "pfsp.json")
    k = 0
    if os.path.exists(pfsp):
        with open(pfsp) as fh:
            payoff = json.load(fh)
        keep_paths = set(kept)
        remapped = {}
        for key, v in payoff.items():
            if os.path.dirname(key) == src_dir:
                if key not in keep_paths:
                    continue
                key = os.path.join(dst_dir, os.path.basename(key))
            remapped[key] = v
        k = len(remapped)
        with open(os.path.join(dst_dir, "pfsp.json"), "w") as fh:
            json.dump(remapped, fh)
    print(f"seeded {len(kept)} snapshots (steps >= {min_step}, {keep_best or 'all'} "
          f"hardest of {len(cands)}; {len(src.archive) - len(kept)} dropped) "
          f"+ {k} payoff entries from {src_dir}", flush=True)


class SimLeagueWorker:
    """Owns the SimLeague, the static env layout, both opponent grids and
    the per-match PFSP routing (rl/league.League). collect(n) runs until n
    PPO chunks are assembled, imitation chunks appended."""

    def __init__(self, cfg: SimRolloutConfig, league, serving_policy,
                 name_code: int, device: str):
        self.cfg = cfg
        self.lg = league
        self.policy = serving_policy
        self.name_code = name_code
        self.device = device
        self.rng = random.Random(cfg.seed)
        from smashbot.rl.rollouts import GameTracker
        self.trackers = {
            "self": GameTracker(),
            "snapshots": GameTracker(),
            "imports": GameTracker(),
            **{f"phillip:{t}": GameTracker() for t in cfg.phillip_tiers},
        }
        self._build()

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
        if s0 != s1:  # ties never enter the PFSP ledger
            self.lg.record(gid, s0 > s1)
        # char-locked members (fox imports) stay out of by_char: a locked
        # member ties its character's column to its own strength
        char = None if gid.startswith("import:") else self._worker.game_info[env_i]
        self._tracker_of(gid).add_game((s0, s1), char)

    def _match(self, env_i: int, member: str):
        """The next match for env_i vs `member`: uniform chars (FOX lock for
        imports), uniform stage, student on port 1 or 2 at random (v10:
        cancels port priority in aggregate), fresh engine seed, 8-min timer."""
        msl = _msl()
        student_c = self.rng.choice(self._chars)
        opp_c = msl.Character.FOX if member.startswith("import:") else self.rng.choice(self._chars)
        stage = self.rng.choice(self._stages)
        sp = self.rng.randrange(2)
        cfg = msl.MatchConfig(
            stage=stage,
            players=(msl.PlayerConfig(student_c, controller_port=sp),
                     msl.PlayerConfig(opp_c, controller_port=1 - sp)),
            seed=self.rng.getrandbits(31), max_frame=self.cfg.max_game_frames)
        return cfg, opp_c.name

    def _build(self) -> None:
        msl = _msl()
        cfg = self.cfg
        N = cfg.num_envs
        self._chars = [getattr(msl.Character, _MSL_CHAR[c.upper()]) for c in cfg.char_whitelist]
        self._stages = list(msl.Stage)
        self.part = self.lg.layout(N)
        from smashbot.rl.league import LeagueSeats
        from smashbot.rl.sim_league import (MultiOpponentSimWorker, PfspGrid,
                                            make_reencoder)
        stu_embed = self.policy.controller_head.controller_embedding
        # --- phillip grid: all tiers on ONE stacked forward (their LSTM
        # steps via the hand-rolled cell — cuDNN has no vmap rule); static
        # cells, slices padded to the largest tier
        tiers = list(cfg.phillip_tiers)
        rows = [self.part[f"phillip:{t}"] for t in tiers]
        tmpl = self.lg.phillips[tiers[0]][0]
        self._phillip_grid = PfspGrid(
            tmpl, len(tiers), max(len(r) for r in rows), self.name_code,
            cfg.unroll_length, self.device,
            reencode=make_reencoder(tmpl.controller_head.controller_embedding,
                                    stu_embed, self.name_code, self.device))
        for s, t in enumerate(tiers):
            self._phillip_grid.load(s, f"phillip:{t}",
                                    lambda k: self.lg.phillips[k.split(':', 1)[1]][0].state_dict())
            self._phillip_grid.agent._name[s] = self.lg.phillips[t][2]
        self._phillip_grid.assign_static(rows)
        for t in tiers:   # weights live in the grid stack now; free the GPU copies
            self.lg.phillips[t][0].to("cpu")
        print(f"phillip grid: {len(tiers)} tiers x {self._phillip_grid.Nc} cells "
              f"(delay {tmpl.delay})", flush=True)
        # --- PFSP grid: S slices x Nc cells with one slice's worth of slack
        # (v5 sizing) so seats float to demand; League routes per match
        pfsp_envs = self.part["pfsp"]
        S = cfg.pfsp_slices
        Nc = -(-(len(pfsp_envs) + S) // S)
        self._grid = PfspGrid(self.lg.make_grid_template(), S, Nc,
                              self.name_code, cfg.unroll_length, self.device)
        seats = LeagueSeats(S, Nc,
                            loader=lambda s, k: self._grid.load(s, k, self.lg.get_state),
                            mover=self._grid.move, drain_slices=max(2, S // 12))
        self.league = self.lg.make_league(seats, self.rng)
        self.league.boot([int(e) for e in pfsp_envs])
        for e in pfsp_envs:
            s, n = seats.seat_of(int(e))
            self._grid.seat(int(e), s, n)
        print(f"pfsp grid: {S} slices x {Nc} cells for {len(pfsp_envs)} envs; "
              f"boot seated {seats.occupancy()} ({seats.loads} slice loads)", flush=True)
        self._worker = MultiOpponentSimWorker(
            self.policy, [], N, cfg.unroll_length, cfg.data_dir, None, None,
            name_code=self.name_code, device=self.device,
            record_fn=self._on_game, precision=cfg.rollout_precision,
            grids=[self._phillip_grid], event_fn=self._on_event,
            self_idx=self.part["self"], league=self.league, pfsp_grid=self._grid,
            match_fn=self._match, max_frame=cfg.max_game_frames, seed=cfg.seed,
            capture=cfg.capture_serving)
        self.rows = self._worker.rows

    def env_share(self) -> dict:
        """{logging_class: env count} (pfsp split by current seating)."""
        out = {k: len(v) for k, v in self.part.items() if k != "pfsp"}
        for e in self.part["pfsp"]:
            m = self.league.member_now[int(e)]
            c = "imports" if m.startswith("import:") else "snapshots"
            out[c] = out.get(c, 0) + 1
        return out

    def league_stats(self) -> dict:
        seats = self.league.seats
        return {"fallback_rate": self.league.fallback_rate,
                "draws": self.league.draws, "slice_loads": seats.loads,
                "compactions": seats.compactions,
                "prefetches": self.league.prefetches,
                "resident_members": len({p.member for p in seats.slices if p.member})}

    def by_char(self) -> dict:
        """Pooled opponent-char winrates over every unlocked class."""
        agg: dict = {}
        for k, tr in self.trackers.items():
            if k == "imports":
                continue
            for c, (w, g) in tr.by_char.items():
                pw, pg = agg.get(c, (0, 0))
                agg[c] = (pw + w, pg + g)
        return agg

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
    from smashbot.training import compile_cores
    from smashbot.networks import use_manual_recurrent_step

    scfg: SimRolloutConfig = args.sim
    device = args.runtime.device
    assert device == "cuda", "sim backend is a GPU training path"
    if os.environ.get("SMASHBOT_MEMDEBUG"):
        # allocator history with python stacks; dumped on OOM (below) for
        # torch.cuda.memory._snapshot analysis
        torch.cuda.memory._record_memory_history(max_entries=200000)

    policy, name_map, step = load_policy(args.ckpt, device)
    teacher, _, _ = load_policy(args.ckpt, device)
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
            learner.policy_clipper.history = list(
                (rl_ckpt["state"].get("clip_history") or {}).get("policy", []))
            print(f"restored RL run from {rpath} at step {start_step}")
    _save_rl_checkpoint.policy_opt = learner.policy_optimizer
    _save_rl_checkpoint.value_opt = learner.value_optimizer
    _save_rl_checkpoint.clip_history = lambda: learner.policy_clipper.history

    # ---- serving copy + compile (train_rl's overlap pattern) ----
    import copy as _copy
    serving_policy = _copy.deepcopy(policy)
    serving_policy.requires_grad_(False).eval()
    use_manual_recurrent_step(serving_policy)   # capturable and fp16-faithful one-frame cells
    print("learner overlap: ON — student serves a published weight copy; "
          "rollouts are one update stale", flush=True)
    if args.runtime.compile:
        import torch._dynamo
        torch._dynamo.config.recompile_limit = 128
        # capture mode wraps this in our own CUDA graph, so compile for
        # kernels only (cudagraph trees cannot nest inside a manual capture)
        serving_policy.sample = torch.compile(
            serving_policy.sample,
            mode=None if scfg.capture_serving else "reduce-overhead")
        compile_cores(policy, value_fn)   # the learner's copies, after the serving deepcopy

    # ---- league ----
    snap_dir = f"{run_dir}/snapshots"
    if scfg.seed_snapshots_from:
        _seed_snapshots(snap_dir, scfg.seed_snapshots_from, scfg.seed_min_step,
                        scfg.seed_keep_best)
    phillips = {}
    for tier, frac in zip(scfg.phillip_tiers, scfg.phillip_fracs):
        fname = "medium-v2-torch.pt" if tier == "medium" else f"{tier}-torch.pt"
        path = f"{MODELS}/{fname}"
        pol, pnm, _ = load_policy(path, device)
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
    )
    if not league.league.archive:
        league.league.save(policy, start_step)
        print(f"boot snapshot: seeded empty archive at step {start_step}", flush=True)
    worker = SimLeagueWorker(scfg, league, serving_policy, name_code, device)
    print(f"sim league: {scfg.num_envs} envs -> {worker.rows} learner rows "
          f"(self {len(worker.part['self'])} x2, phillips "
          f"{sum(len(v) for k, v in worker.part.items() if k.startswith('phillip'))}, "
          f"pfsp {len(worker.part['pfsp'])})", flush=True)
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

    state = learner.initial_state(worker.rows, device)
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

    def _post_step(i, metrics):
        if (i + 1) % args.runtime.checkpoint_interval == 0:
            _save_rl_checkpoint(f"{run_dir}/latest.pt", ckpt["config"], policy,
                                value_fn, name_map, i, args.ckpt)
        if i % args.runtime.log_interval != 0:
            return
        frames = ((i + 1 - start_step) * args.runtime.trajectories_per_step
                  * worker.rows * scfg.unroll_length)

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
        for k, v in worker.league_stats().items():   # routing health
            log[f"rl/league/{k}"] = v
        for c, (w, g) in worker.by_char().items():   # per-matchup weakness
            if g >= 20:
                log[f"bychar/{c}"] = w / g
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
                if fut_i < start_step + 6:
                    # the footprint measurement IS the real run: launch with
                    # --runtime.steps <start+6> --runtime.wandb-mode disabled
                    print(f"[vram] overlapped step {fut_i}: "
                          f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} "
                          f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB",
                          flush=True)
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
                torch.cuda.synchronize()
                print(f"[vram] first learner step: "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} "
                      f"reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB",
                      flush=True)
                torch.cuda.empty_cache()
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
