"""RL fine-tuning entry point: PPO + KL-to-teacher over live Dolphin rollouts.

Usage:
  python -m smashbot.rl.train_rl --ckpt /path/to/mega-best.pt \
      --rollouts.num-envs 8 --runtime.steps 10000

The checkpoint provides everything: policy init, frozen teacher, critic init,
and the config/name_map (RL checkpoints stay play.py-compatible).
"""

from __future__ import annotations

import dataclasses
import os
import time

import tyro

# Module level stays torch-free on purpose: every env process re-imports
# this __main__ module (multiprocessing prepare()), and torch would cost
# each of them ~0.26 GB. Heavy imports live inside the functions below.
from smashbot.rl.config import RLConfig, RolloutConfig


@dataclasses.dataclass
class RuntimeConfig:
    tag: str = "rl-dev"
    steps: int = 1000
    trajectories_per_step: int = 1
    run_dir: str = "/home/kage/drive2/ShineBot/runs"
    checkpoint_interval: int = 50
    log_interval: int = 1
    wandb_mode: str = "online"
    name: str = "Master Player"
    compile: bool = True  # compile sample_n (the batched flush)
    # Hot-swappable teacher: poll this path (default: the --ckpt file) every
    # teacher_check_interval learner steps; on change, safely reload the
    # frozen teacher in place (see rl/teacher_watch.py).
    teacher_watch: str = ""
    teacher_check_interval: int = 100  # ~20 min at 64 envs (one step ~15s)
    restore: str = ""  # RL checkpoint path, or "auto" for <run_dir>/<tag>/latest.pt
    device: str = "cpu"  # rollouts are CPU-bound; learner device


@dataclasses.dataclass
class Config:
    ckpt: str = "/home/kage/drive2/ShineBot/models/mega-best-epoch1.8.pt"
    learner: RLConfig = dataclasses.field(default_factory=RLConfig)
    rollouts: RolloutConfig = dataclasses.field(default_factory=RolloutConfig)
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)


def build_value_function(cfg: dict, device: str):
    from smashbot import configs, embed as embed_lib
    from smashbot.networks import build_embed_network
    from smashbot.value import ValueFunction

    value_name = cfg["value"].get("name", "match")
    if value_name == "match":
        value_name = cfg["network"]["name"]
    net_cfg = configs.NetworkConfig(
        name=value_name,
        hidden_size=cfg["value"]["hidden_size"],
        num_layers=cfg["value"]["num_layers"],
        num_heads=cfg["network"]["num_heads"],
        window=cfg["value"].get("window", 0) or cfg["network"]["window"],
    )
    return ValueFunction(
        build_embed_network(
            embed_config=embed_lib.EmbedConfig(),
            controller_embedding=embed_lib.ControllerConfig(
                axis_spacing=cfg["head"]["axis_spacing"],
                shoulder_spacing=cfg["head"]["shoulder_spacing"],
            ).make_embedding(),
            num_names=cfg["data"]["max_names"],
            network_config=net_cfg,
        )
    ).to(device)


def _save_rl_checkpoint(
    path: str, config: dict, policy, value_fn, name_map, step: int, teacher: str
) -> None:
    """Same schema as BC checkpoints (config already a dict), so play.py and
    the eval harness load RL checkpoints unchanged."""
    import os

    import torch

    from smashbot import saving

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "config": config,
            "state": {
                "policy": policy.state_dict(),
                "value": value_fn.state_dict(),
                "policy_opt": _save_rl_checkpoint.policy_opt.state_dict(),
                "value_opt": _save_rl_checkpoint.value_opt.state_dict(),
                "name_map": name_map,
                "step": step,
                "teacher_ckpt": teacher,
                "trackers": _save_rl_checkpoint.tracker_states(),
            },
            "best_eval_loss": None,
            "version": saving.VERSION,
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    import torch

    from smashbot import saving
    from smashbot.eval.game import load_policy, resolve_name_code
    from smashbot.rl.agent import BatchedPolicyAgent
    from smashbot.rl.ppo import Learner
    from smashbot.rl.rollouts import DolphinRolloutWorker
    from smashbot.rl.teacher_watch import TeacherWatcher

    args = tyro.cli(Config)
    device = args.runtime.device

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
    # learner.precision is post-fallback (fp16 on cpu loudly reverts to fp32
    # inside Learner.__init__)
    print(
        f"learner precision: {learner.precision}"
        + (
            " (fp16 autocast on policy paths, value net fp32, GradScaler "
            "on the policy optimizer)"
            if learner.precision == "fp16"
            else ""
        )
    )

    start_step = 0
    restored_trackers = None
    if args.runtime.restore:
        import os as os_lib

        rpath = args.runtime.restore
        if rpath == "auto":
            rpath = f"{args.runtime.run_dir}/{args.runtime.tag}/latest.pt"
            if not os_lib.path.exists(rpath):
                rpath = ""  # supervisor-friendly: no checkpoint = fresh start
                print("restore auto: no checkpoint yet, starting fresh")
        if rpath:
            rl_ckpt = saving.load_checkpoint(rpath)
            policy.load_state_dict(rl_ckpt["state"]["policy"])
            value_fn.load_state_dict(rl_ckpt["state"]["value"])
            if "policy_opt" in rl_ckpt["state"]:
                learner.policy_optimizer.load_state_dict(
                    rl_ckpt["state"]["policy_opt"]
                )
                learner.value_optimizer.load_state_dict(
                    rl_ckpt["state"]["value_opt"]
                )
            start_step = rl_ckpt["state"]["step"] + 1
            restored_trackers = rl_ckpt["state"].get("trackers")
            print(f"restored RL run from {rpath} at step {start_step}")
    _save_rl_checkpoint.policy_opt = learner.policy_optimizer
    _save_rl_checkpoint.value_opt = learner.value_optimizer

    if args.runtime.compile:
        mode = "reduce-overhead" if device == "cuda" else "default"
        policy.sample = torch.compile(policy.sample, mode=mode)
        teacher.sample = torch.compile(teacher.sample, mode=mode)
        policy.sample_n = torch.compile(policy.sample_n, mode=mode)
        teacher.sample_n = torch.compile(teacher.sample_n, mode=mode)
    student_agent = BatchedPolicyAgent(
        policy, args.rollouts.num_envs, name_code=name_code, device=device,
        batch_steps=args.rollouts.batch_steps,
        precision=args.rollouts.rollout_precision,
    )

    from smashbot.rl.pool import (
        SnapshotPool, make_partition, student_whitelist,
    )

    rcfg = args.rollouts
    if not rcfg.log_tag:
        rcfg.log_tag = args.runtime.tag
    # league flags (teacher / lvl-9 CPU as PFSP members): validate up front —
    # loud assert beats 120 Dolphins booting into a mispartitioned run
    league = rcfg.league_members()
    if league:
        print(f"league members (per-match PFSP draws): {league}")
    # Imported league members (frozen checkpoints from a previous run):
    # {"import:NAME": (path, char_lock)}. Validate paths up front: loud
    # assert beats 120 Dolphins booting into a run whose benchmark opponent
    # can never serve.
    import_registry = {
        f"import:{name}": (path, char)
        for name, (path, char) in rcfg.import_members().items()
    }
    for key, (path, char) in import_registry.items():
        assert os.path.exists(path), (
            f"league import {key}: state_dict not found at {path}"
        )
        print(f"league import: {key} <- {path} @ {char} (char lock)")
    specs = make_partition(
        rcfg.num_envs, rcfg.cpu_envs, rcfg.teacher_envs,
        rcfg.main12_prob, rcfg.partition_seed,
        ref_envs=rcfg.ref_envs, self_envs=rcfg.self_envs,
        char_whitelist=student_whitelist(rcfg.char_whitelist, rcfg.bot_char),
    )
    opponents = {}
    counts = {}
    for spec in specs:
        if spec.kind in ("teacher", "reference", "snapshot"):
            counts[spec.kind] = counts.get(spec.kind, 0) + 1
    if "teacher" in counts:
        opponents["teacher"] = BatchedPolicyAgent(
            teacher, counts["teacher"], name_code=name_code, device=device,
            batch_steps=rcfg.batch_steps,
        )
    if "reference" in counts:
        # the ported medium-v2 (see scripts/port_ref_model.py): verified
        # 6.3e-13 vs TF at fp64. Delay 21 and its own name_map ride along
        # in the checkpoint; condition on ITS "Master Player" code.
        ref_policy, ref_names, _ = load_policy(rcfg.ref_ckpt, device)
        ref_policy.train_value_head = False
        ref_policy.requires_grad_(False)
        ref_policy.eval()
        if args.runtime.compile:
            # reduce-overhead restored: the 120-env config frees the learner
            # peak (batch-proportional), so opponents get CUDA graphs back
            # (default-mode cost ~130fps; measured)
            mode = "reduce-overhead" if device == "cuda" else "default"
            ref_policy.sample = torch.compile(ref_policy.sample, mode=mode)
        ref_code = resolve_name_code(ref_names, "Master Player")
        opponents["reference"] = BatchedPolicyAgent(
            ref_policy, counts["reference"], name_code=ref_code,
            device=device, batch_steps=rcfg.batch_steps,
        )
        print(f"reference: {rcfg.ref_ckpt} (delay {ref_policy.delay}, "
              f"name code {ref_code})")
    snapshot_pool = SnapshotPool(
        f"{args.runtime.run_dir}/{args.runtime.tag}/snapshots",
        pfsp=rcfg.pfsp, pfsp_p=rcfg.pfsp_p,
        pfsp_hard_frac=rcfg.pfsp_hard_frac, pfsp_explore=rcfg.pfsp_explore,
        league_members=league,
    )
    runtime = None
    league_envs = counts.get("snapshot", 0)
    if league_envs:
        from smashbot.rl.agent import LeagueAgent
        from smashbot.rl.league import League, LeagueSeats, MemberWeights

        assert rcfg.league_slices > 0, (
            f"{league_envs} league envs but league_slices=0"
        )
        # the grid: S slices x N cells, one slice's worth of slack cells so
        # free seats can float to where PFSP demand is (see league.py)
        S = rcfg.league_slices
        N = -(-(league_envs + S) // S)
        template, _, _ = load_policy(args.ckpt, "cpu")
        template.train_value_head = False
        grid = LeagueAgent(
            template, S, N, name_code=name_code, device=device,
            temperature=None,
            weights_dtype=getattr(torch, rcfg.league_weights_dtype),
        )
        # member weights: teacher (frozen copy), imports, snapshots (LRU)
        fixed = {"teacher": {k: v.detach().cpu() for k, v in teacher.state_dict().items()}}
        for key, (path, _char) in import_registry.items():
            fixed[key] = torch.load(path, map_location="cpu")
        weights = MemberWeights(fixed)
        phillip_agent = None
        if rcfg.league_phillip:
            # Phillip: his own architecture, so his own 1-slice grid (same
            # captured-vmap path as the league; fixed capacity of cells)
            ph_policy, ph_names, _ = load_policy(rcfg.ref_ckpt, "cpu")
            ph_policy.train_value_head = False
            ph_code = resolve_name_code(ph_names, "Master Player")
            cap = rcfg.phillip_capacity or 3 * N
            phillip_agent = LeagueAgent(
                ph_policy, 1, cap, name_code=ph_code, device=device,
                temperature=None,
                weights_dtype=getattr(torch, rcfg.league_weights_dtype),
            )
            print(f"phillip (league member): {rcfg.ref_ckpt} "
                  f"(delay {ph_policy.delay}, name code {ph_code}, "
                  f"capacity {cap})")
        seats = LeagueSeats(
            S, N, loader=lambda s, m: grid.load_slice(s, weights.get(m)),
            phillip_capacity=phillip_agent.N if phillip_agent else 0,
            mover=grid.move_cell,
        )
        import random as _random

        league_proto = League(
            snapshot_pool, seats,
            locks={k: char for k, (_p, char) in import_registry.items()},
            rng=_random.Random(rcfg.partition_seed ^ 0xA11A),
            on_result=snapshot_pool.record_result,
            cpu_enabled=rcfg.league_cpu, warm=weights.warm,
        )
        from smashbot.rl.rollouts import LeagueRuntime

        runtime = LeagueRuntime(league_proto, grid, phillip_agent)
        print(f"league grid: {S} slices x {N} cells for {league_envs} envs",
              flush=True)
    worker = DolphinRolloutWorker(
        args.rollouts, student_agent, opponents=opponents, specs=specs,
        harvest_imitation=args.learner.imitation_rows != 0, league=runtime,
    )
    if restored_trackers:
        for kind, st in restored_trackers.items():
            if kind in worker.trackers:
                worker.trackers[kind].load_state(st)
        print(f"tracker EMAs restored for {sorted(restored_trackers)}")
    _save_rl_checkpoint.tracker_states = lambda: {
        k: t.state() for k, t in worker.trackers.items()
    }

    import wandb

    wandb.init(
        project="shinebot", id=args.runtime.tag, name=args.runtime.tag,
        mode=args.runtime.wandb_mode, config=dataclasses.asdict(args),
    )

    run_dir = f"{args.runtime.run_dir}/{args.runtime.tag}"
    # a fresh run seeds its archive with the init weights so the league
    # has a ghost to draw from the first match on
    if league_envs and not snapshot_pool.archive:
        snapshot_pool.save(policy, start_step)
        print(f"boot snapshot: seeded empty archive at step {start_step}",
              flush=True)
    state = learner.initial_state(args.rollouts.num_envs, device)
    watcher = TeacherWatcher(args.runtime.teacher_watch or args.ckpt)
    teacher_swaps = 0
    t0 = time.time()
    try:
        for i in range(start_step, args.runtime.steps):
            if league_envs and i > 0 and i % rcfg.snapshot_interval == 0:
                # a new ghost joins the league; envs draw it per match
                # from now on (no auction, no swaps)
                path = snapshot_pool.save(policy, i)
                print(f"[{i}] snapshot saved: {os.path.basename(path)} "
                      f"joins the league", flush=True)
            if i > 0 and i % args.runtime.teacher_check_interval == 0:
                new_teacher = watcher.poll()
                if new_teacher is not None:
                    teacher.load_state_dict(new_teacher)  # in-place copy
                    if runtime is not None:
                        weights.set("teacher", {
                            k: v.detach().cpu() for k, v in teacher.state_dict().items()
                        })
                    state = state._replace(
                        teacher=teacher.initial_state(
                            args.rollouts.num_envs, device
                        )
                    )
                    teacher_swaps += 1
                    print(f"[{i}] TEACHER SWAPPED (#{teacher_swaps})")
            trajectories = worker.collect(args.runtime.trajectories_per_step)
            if os.environ.get("SMASHBOT_PROFILE") and device == "cuda":
                torch.cuda.synchronize()
                _base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
            state, metrics = learner.step(
                trajectories, state,
                progress=i / max(1, args.runtime.steps),
            )
            if os.environ.get("SMASHBOT_PROFILE") and device == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                print(f"[vram] step {i}: baseline {_base / 2**30:.2f} GiB "
                      f"(inference residency: weights + graph pools) | learner peak "
                      f"{peak / 2**30:.2f} GiB | activations {(peak - _base) / 2**30:.2f} GiB | "
                      f"reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)

            if i % args.runtime.log_interval == 0:
                # frames THIS BOOT only: after a restore, i includes the
                # restored steps but t0 is boot time — crediting them made
                # fps read ~80k until the ghost frames washed out.
                frames = (
                    (i + 1 - start_step) * args.runtime.trajectories_per_step
                    * args.rollouts.num_envs * args.rollouts.unroll_length
                )
                log = {
                    "rl/" + k: v
                    for k, v in metrics["post_update"].items()
                }
                log.update({
                    "rl/value_" + k: v for k, v in metrics["value"].items()
                })
                log["rl/reverted"] = float(metrics["reverted"])
                log["rl/teacher_swaps"] = teacher_swaps
                if learner.grad_scaler is not None:
                    # fp16 health gauge: collapsing scale = repeated overflow
                    # skips; a steady 2^15..2^17 is the healthy regime
                    log["rl/grad_scaler_scale"] = (
                        learner.grad_scaler.get_scale()
                    )
                im = metrics.get("imitation")
                if im:
                    log["rl/imitation/loss"] = im["loss"]
                    log["rl/imitation/w_mean"] = im["w_mean"]
                    log["rl/imitation/w_max"] = im["w_max"]
                    log["rl/imitation/traj_count"] = im["traj_count"]
                    log["rl/imitation/lambda"] = im["lambda"]
                if league_envs:
                    # class view of the two-stage PFSP: per-class hardness
                    # and how many league envs each class is fighting NOW.
                    # Imports log as rl/pfsp/import_{NAME}_* — the
                    # import_..._winrate row IS the cross-generation
                    # progress bar (are we beating the old model yet?)
                    hard = snapshot_pool.class_hardness()
                    held = {c: 0 for c in ("phillip", "teacher", "cpu",
                                           "ghosts", *hard)}
                    for m in worker.league.member_now.values():
                        held[m if m in held else "ghosts"] += 1
                    for cname, h in hard.items():
                        tag = (
                            f"import_{cname[len('import:'):]}"
                            if cname.startswith("import:")
                            else f"class_{cname}"
                        )
                        log[f"rl/pfsp/{tag}_winrate"] = h
                        log[f"rl/pfsp/{tag}_envs"] = held[cname]
                    # per-ghost winrates: the sawtooth Keith reads (each
                    # new ghost joins near parity and pulls the class mean
                    # down; old ghosts drift toward beaten)
                    for g_path in snapshot_pool.archive:
                        g_step = snapshot_pool._step_of(g_path)
                        log[f"rl/pfsp/ghost_{g_step:07d}_winrate"] = (
                            snapshot_pool.win_estimate(g_path)
                        )
                    lg = worker.league
                    log["rl/league/fallback_rate"] = lg.fallback_rate
                    log["rl/league/draws"] = lg.draws
                    for k, v in lg.warn.items():
                        log[f"rl/league/warn_{k}"] = v
                    log["rl/league/slice_loads"] = lg.seats.loads
                    log["rl/league/compactions"] = lg.seats.compactions
                for kind, tracker in worker.trackers.items():
                    for k, v in tracker.stats().items():
                        # ledger (rl/pfsp/class_*) already covers these;
                        # self keeps its EMA (no payoff row)
                        if (league_envs and kind != "self"
                                and k in ("win_rate", "win_rate_ema",
                                          "win_rate_recent")):
                            continue
                        log[f"rl/{kind}/{k}"] = v
                log["rl/frames_per_sec"] = frames / (time.time() - t0)
                wandb.log(log, step=i)
                games = sum(
                    log.get(f"rl/{k}/games_played", 0)
                    for k in ("cpu", "teacher", "snapshot", "reference", "self")
                )
                # Ticker categories come from the SAME ledger the draw
                # uses (pfsp decayed counts — user: "log what is in our
                # json"), so a member's ticker % IS its draw basis.
                # '--' = no league-era games yet. SP (self-play) has no
                # payoff row; it stays the tracker's ~50% health gauge.
                # kill@/die@ + per-kind EMAs remain in wandb only.
                cat = snapshot_pool.category_estimates() if league_envs else {}

                def _pct(x):  # "decayed/raw%" from the payoff ledger
                    if x is None:
                        return "--"
                    return f"{100 * x[0]:.0f}/{100 * x[1]:.0f}%"

                ref_bit = (
                    f"R:{_pct(cat.get('phillip'))} "
                    if rcfg.league_phillip else (
                        f"R:{log.get('rl/reference/win_rate_ema', 0.5):.0%} "
                        if worker.ref_idx else ""
                    )
                )
                sp_bit = (
                    f"SP:{log.get('rl/self/win_rate_ema', 0.5):.0%} "
                    if worker.self_idx else ""
                )
                imp_bit = (
                    f"I:{_pct(cat.get('imports'))} "
                    if rcfg.league_imports else ""
                )
                # league routing health: fallback share of draws, protocol
                # warnings (recycle/lock mismatches), slice loads
                lg_bit = (
                    f"lg:{worker.league.fallback_rate:.0%}fb/"
                    + "/".join(f"{v}{k[:4]}" for k, v in worker.league.warn.items())
                    + f"/{worker.league.seats.loads}ld "
                    if league_envs else ""
                )
                print(
                    f"[{i:4d}/{args.runtime.steps}] "
                    f"T:{_pct(cat.get('teacher'))} "
                    f"S:{_pct(cat.get('ghosts'))} "
                    f"C:{_pct(cat.get('cpu'))} "
                    f"{ref_bit}"
                    f"{imp_bit}"
                    f"{sp_bit}"
                    f"{lg_bit}"
                    f"({games:.0f}g) | "
                    f"tKL {log['rl/teacher_kl']:.4f} "
                    f"aKL {log['rl/actor_kl_mean']:.5f} "
                    f"{'REVERTED ' if log['rl/reverted'] else ''}| "
                    f"{log['rl/frames_per_sec']:.0f} fps",
                    flush=True,
                )

            if (i + 1) % args.runtime.checkpoint_interval == 0:
                _save_rl_checkpoint(
                    f"{run_dir}/latest.pt", ckpt["config"], policy, value_fn,
                    name_map, i, args.ckpt,
                )
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
