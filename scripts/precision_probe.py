"""Learner-precision probes: fp32 vs bf16 vs fp16 (docs/experiments-queue.md,
"Learner precision" — the budget-compressed Tier 1 + Tier 2).

Two subcommands:

fidelity — Tier 1, minutes, offline. Loads a saved REAL trajectory batch
  (scripts/capture_batch.py) and the matching learner (policy/teacher/value
  from the checkpoint), then runs the identical batch through the learner's
  fixed pass + policy loss under (a) fp32, (b) autocast bf16, (c) autocast
  fp16 — forward AND backward, no optimizer steps. Diffs each half-precision
  arm against fp32: logprob deltas, ratio-deviation-from-1 (the fresh-learner
  rollout/learner mismatch detector), ratio deltas, advantage deltas, aKL
  floor, loss-term deltas, grad-norm deltas, nonfinite counts. Safe beside
  the live run on cpu; use --device cuda only in a training gap.

memcurve — Tier 2, minutes, REQUIRES AN IDLE GPU (--i-have-the-gpu). Builds
  synthetic trajectory batches at increasing row counts and runs full
  learner steps under the chosen precision, reporting the measured
  peak-memory curve (rows unlocked) until OOM.

Autocast discipline (per the doc's full design): only the network unrolls
run under autocast — every sensitive quantity is already fp32 without any
ppo.py change, because the sensitive math bottoms out in ops autocast pins
to fp32 (log_softmax / binary_cross_entropy_with_logits produce the
log-probs, KLs and entropies) or in an explicit fp32 island (value.py wraps
its return recursion/advantages in autocast(enabled=False)). The ratio/
surrogate arithmetic then inherits fp32 from those inputs. Each arm records
the observed dtypes as a receipt ("dtypes" in the JSON).

fp16 note: fidelity runs no optimizer step, so no GradScaler is needed
here; a REAL fp16 training arm would need loss scaling (gradients can
underflow without it). memcurve likewise steps without a scaler — it
measures memory, not training health.
"""

from __future__ import annotations

import os

# Modest CPU footprint next to a live training run; must be set before torch
# initializes its thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import contextlib
import dataclasses
import json
import math
import sys
import time

import torch
import tree

ARMS = ("fp32", "bf16", "fp16")
_ARM_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

DEFAULT_CONFIG_FROM = "/home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt"


# --------------------------------------------------------------- primitives


def autocast_ctx(device_type: str, arm: str):
    """The probe's ONLY precision lever: fp32 = plain context, half arms =
    torch.autocast around the learner calls. Loss/ratio math stays fp32 by
    construction (see module docstring)."""
    if arm == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=_ARM_DTYPE[arm])


def to_device(struct, device):
    """Move every tensor leaf of a (possibly nested) structure to device."""
    return tree.map_structure(
        lambda t: t.to(device) if isinstance(t, torch.Tensor) else t, struct
    )


def _grad_stats(params) -> tuple[float, int]:
    """(global grad 2-norm, nonfinite grad element count), grads-as-they-are
    (no clipping, no mutation)."""
    sq, bad = 0.0, 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        bad += int((~torch.isfinite(g)).sum().item())
        finite = torch.nan_to_num(g)
        sq += float(torch.linalg.vector_norm(finite).item()) ** 2
    return math.sqrt(sq), bad


def _mean_dicts(dicts: list[dict]) -> dict:
    if not dicts:
        return {}
    return {
        k: sum(d[k] for d in dicts) / len(dicts)
        for k in dicts[0]
        if isinstance(dicts[0][k], (int, float))
    }


def freeze_value_updates(learner) -> None:
    """Probe hygiene: the learner's fixed pass trains the critic in place
    (value backward + optimizer step); no-op the step so probe passes
    mutate NOTHING and every arm sees identical weights."""
    learner.value_optimizer.step = lambda *a, **k: None


# ------------------------------------------------------------ learner build


def build_learner(
    ckpt: str = "",
    snapshot: str = "",
    config_from: str = "",
    device: str = "cpu",
    freeze: bool = True,
):
    """Real learner from a checkpoint, mirroring train_rl's construction:
    policy (+ optional bare-snapshot weights), frozen teacher (from the RL
    checkpoint's recorded teacher_ckpt when it exists, else the checkpoint's
    own weights), value net from the saved state, default RLConfig (the
    learner hyperparameters are CLI-only in train_rl and don't ride in the
    checkpoint; precision deltas don't depend on them).

    Probe hygiene (freeze=True, the fidelity default): the value optimizer's
    step() is replaced with a no-op so the learner's fixed pass (which
    normally trains the critic in place) mutates NOTHING — every arm sees
    identical weights. memcurve passes freeze=False: it wants the real
    allocation profile, critic Adam moments included."""
    from smashbot import saving
    from smashbot.eval.game import load_policy
    from smashbot.rl.ppo import Learner, RLConfig
    from smashbot.rl.train_rl import build_value_function

    src = ckpt or config_from
    assert src, "need --ckpt or --config-from"
    policy, _, step = load_policy(src, device)
    if snapshot:
        state = torch.load(snapshot, map_location=device, weights_only=True)
        policy.load_state_dict(state)
        step = int(
            os.path.basename(snapshot).split("-")[1].split(".")[0]
        )
    policy.train_value_head = False

    full = saving.load_checkpoint(src)
    teacher_path = full["state"].get("teacher_ckpt", "")
    if teacher_path and os.path.exists(teacher_path):
        teacher, _, _ = load_policy(teacher_path, device)
    else:
        if teacher_path:
            print(f"warning: recorded teacher {teacher_path} missing; "
                  f"using {src} as the teacher", flush=True)
        teacher, _, _ = load_policy(src, device)
    teacher.train_value_head = False

    value_fn = build_value_function(full["config"], device)
    if "value" in full["state"]:
        value_fn.load_state_dict(full["state"]["value"])

    learner = Learner(RLConfig(), policy, teacher, value_fn)
    if freeze:
        freeze_value_updates(learner)
    return learner, step


# ------------------------------------------------------------ fidelity arm


def run_arm(learner, trajectories, arm: str, device_type: str) -> dict:
    """One precision arm over the batch: the learner's real _fixed_pass +
    _policy_loss per trajectory (network unrolls under autocast for half
    arms), then loss.backward(). No optimizer steps; grads are measured and
    zeroed. Returns per-position tensors (cpu fp32) + scalar metrics."""
    learner.policy_optimizer.zero_grad(set_to_none=True)
    learner.value_optimizer.zero_grad(set_to_none=True)

    ppo_trajs = [
        t for t in trajectories if getattr(t, "kind", "ppo") != "imitation"
    ]
    assert ppo_trajs, "batch contains no PPO trajectories"
    batch_size = ppo_trajs[0].rewards.shape[0]
    param_device = next(learner.policy.parameters()).device
    state = learner.initial_state(batch_size, param_device)

    # Capture the policy's UnrollOutputs from inside _policy_loss (instance-
    # attribute shadow; removed in the finally) so per-position log-probs and
    # logits come from the exact tensors the loss used — no second unroll.
    captured: dict = {}
    orig_unroll = learner.policy.unroll

    def capturing_unroll(*a, **k):
        out = orig_unroll(*a, **k)
        captured["out"] = out
        return out

    learner.policy.unroll = capturing_unroll
    per: list[dict] = []
    metrics_list: list[dict] = []
    vmetrics_list: list[dict] = []
    dtypes: dict = {}
    try:
        for traj in ppo_trajs:
            ctx = autocast_ctx(device_type, arm)
            with ctx:
                fixed, state, vmetrics = learner._fixed_pass(traj, state)
                loss, metrics = learner._policy_loss(fixed)
                with torch.no_grad():
                    # per-position aKL under the SAME autocast the loss saw
                    akl = learner._ops.kl(
                        fixed.actor_logits, captured["out"].logits
                    ).detach().float().cpu()
            # backward outside the autocast region (recommended amp pattern;
            # the value net's backward runs inside _fixed_pass by design —
            # gradient math replays recorded ops, unaffected by the context)
            loss.backward()
            out = captured["out"]
            if not dtypes:  # dtype receipt: sensitive path must read fp32
                dtypes = {
                    "unroll_logits_leaf": str(
                        tree.flatten(out.logits)[0].dtype
                    ),
                    "log_probs": str(out.log_probs.dtype),
                    "advantages": str(fixed.advantages.dtype),
                    "actor_log_probs": str(fixed.actor_log_probs.dtype),
                }
            valid = fixed.valid.detach().float().cpu()
            log_probs = out.log_probs.detach().float().cpu()
            log_rhos = log_probs - fixed.actor_log_probs.detach().float().cpu()
            # invalid positions pinned to ratio 1, matching ppo's ratio_mean
            ratios = log_rhos.exp() * valid + (1 - valid)
            per.append({
                "log_probs": log_probs,
                "ratios": ratios,
                "advantages": fixed.advantages.detach().float().cpu(),
                "akl": akl,
                "valid": valid,
            })
            metrics_list.append(metrics)
            vmetrics_list.append(vmetrics)
    finally:
        learner.policy.__dict__.pop("unroll", None)

    policy_grad_norm, policy_grad_bad = _grad_stats(
        learner.policy.parameters()
    )
    value_grad_norm, value_grad_bad = _grad_stats(
        learner.value_function.parameters()
    )
    learner.policy_optimizer.zero_grad(set_to_none=True)
    learner.value_optimizer.zero_grad(set_to_none=True)

    cat = lambda key: torch.cat([p[key] for p in per], dim=0)
    tensors = {k: cat(k) for k in per[0]}
    nonfinite = {
        "log_probs": int((~torch.isfinite(tensors["log_probs"])).sum()),
        "ratios": int((~torch.isfinite(tensors["ratios"])).sum()),
        "advantages": int((~torch.isfinite(tensors["advantages"])).sum()),
        "akl": int((~torch.isfinite(tensors["akl"])).sum()),
        "policy_grads": policy_grad_bad,
        "value_grads": value_grad_bad,
    }
    from smashbot.rl import ppo as ppo_lib

    return {
        "arm": arm,
        "tensors": tensors,
        "metrics": ppo_lib._mean_dicts(metrics_list),
        "value_metrics": _mean_dicts(vmetrics_list),
        "policy_grad_norm": policy_grad_norm,
        "value_grad_norm": value_grad_norm,
        "nonfinite": nonfinite,
        "dtypes": dtypes,
    }


def _masked_mean(t: torch.Tensor, valid: torch.Tensor) -> float:
    return float((t * valid).sum() / valid.sum().clamp(min=1.0))


def compare_arms(base: dict, arm: dict, ratio_tol: float) -> dict:
    """JSON-ready comparison of one arm against the fp32 baseline. For the
    baseline itself this returns all-zero deltas (self-comparison)."""
    bt, at = base["tensors"], arm["tensors"]
    valid = bt["valid"]
    dmax = lambda k: float(((at[k] - bt[k]).abs() * valid).max())
    ratio_mean = arm["metrics"]["ratio_mean"]
    akl_mean = _masked_mean(at["akl"], valid)
    return {
        "max_abs_dlogprob": dmax("log_probs"),
        "max_abs_dratio": dmax("ratios"),
        "max_abs_dadvantage": dmax("advantages"),
        "ratio_mean": ratio_mean,
        # fresh-learner invariant: the learner recomputation of the rollout
        # policy's own actions must yield ratio == 1 (the rollout/learner
        # mismatch detector from the experiments doc)
        "ratio_mean_dev_from_1": abs(ratio_mean - 1.0),
        "ratio_invariant_ok": abs(ratio_mean - 1.0) <= ratio_tol,
        "akl_mean": akl_mean,
        "akl_max": float((at["akl"] * valid).max()),
        "d_akl_mean": akl_mean - _masked_mean(bt["akl"], valid),
        "loss": arm["metrics"]["loss"],
        "d_loss": arm["metrics"]["loss"] - base["metrics"]["loss"],
        "d_surrogate": (
            arm["metrics"]["surrogate"] - base["metrics"]["surrogate"]
        ),
        "d_teacher_kl": (
            arm["metrics"]["teacher_kl"] - base["metrics"]["teacher_kl"]
        ),
        "d_entropy": arm["metrics"]["entropy"] - base["metrics"]["entropy"],
        "value_loss": arm["value_metrics"].get("loss"),
        "d_value_loss": (
            arm["value_metrics"].get("loss", 0.0)
            - base["value_metrics"].get("loss", 0.0)
        ),
        "policy_grad_norm": arm["policy_grad_norm"],
        "d_policy_grad_norm": (
            arm["policy_grad_norm"] - base["policy_grad_norm"]
        ),
        "value_grad_norm": arm["value_grad_norm"],
        "d_value_grad_norm": (
            arm["value_grad_norm"] - base["value_grad_norm"]
        ),
        "anomalous_samples": arm["metrics"]["anomalous_samples"],
        "nonfinite": arm["nonfinite"],
        "dtypes": arm["dtypes"],
    }


_TABLE_ROWS = [
    ("max|d logprob|", "max_abs_dlogprob", "9.3g"),
    ("max|d ratio|", "max_abs_dratio", "9.3g"),
    ("max|d advantage|", "max_abs_dadvantage", "9.3g"),
    ("ratio_mean", "ratio_mean", "9.6f"),
    ("|ratio_mean - 1|", "ratio_mean_dev_from_1", "9.3g"),
    ("aKL mean", "akl_mean", "9.3g"),
    ("aKL max", "akl_max", "9.3g"),
    ("d aKL mean", "d_akl_mean", "9.3g"),
    ("loss", "loss", "9.6f"),
    ("d loss", "d_loss", "9.3g"),
    ("d surrogate", "d_surrogate", "9.3g"),
    ("d teacher_kl", "d_teacher_kl", "9.3g"),
    ("d entropy", "d_entropy", "9.3g"),
    ("d value_loss", "d_value_loss", "9.3g"),
    ("policy grad norm", "policy_grad_norm", "9.4f"),
    ("d policy gnorm", "d_policy_grad_norm", "9.3g"),
    ("value grad norm", "value_grad_norm", "9.4f"),
    ("d value gnorm", "d_value_grad_norm", "9.3g"),
]


def format_table(arms: dict) -> str:
    """Plain-text comparison table: one column per arm, vs-fp32 deltas."""
    names = [a for a in ARMS if a in arms]
    lines = []
    header = f"{'metric':<18}" + "".join(f"{n:>14}" for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for label, key, fmt in _TABLE_ROWS:
        cells = []
        for n in names:
            entry = arms[n]
            if "skipped" in entry:
                cells.append(f"{'skipped':>14}")
                continue
            val = entry.get(key)
            cells.append(
                f"{'-':>14}" if val is None else f"{format(val, fmt):>14}"
            )
        lines.append(f"{label:<18}" + "".join(cells))
    inv = []
    for n in names:
        entry = arms[n]
        if "skipped" in entry:
            inv.append(f"{n}: skipped ({entry['skipped']})")
        else:
            ok = "OK" if entry["ratio_invariant_ok"] else "VIOLATED"
            inv.append(f"{n}: {ok} (dev {entry['ratio_mean_dev_from_1']:.3g})")
        bad = sum(entry.get("nonfinite", {}).values())
        if bad:
            inv[-1] += f" [{bad} NONFINITE]"
    lines.append("")
    lines.append("ratio_mean==1 fresh-learner invariant: " + "; ".join(inv))
    return "\n".join(lines)


def run_fidelity(
    learner,
    trajectories,
    device_type: str,
    arms=ARMS,
    ratio_tol: float = 1e-3,
) -> dict:
    """All arms + comparisons. fp32 always runs first as the baseline."""
    arm_list = list(arms)
    assert arm_list[0] == "fp32", "fp32 must run first (it is the baseline)"
    raw: dict = {}
    report_arms: dict = {}
    for arm in arm_list:
        t0 = time.monotonic()
        try:
            raw[arm] = run_arm(learner, trajectories, arm, device_type)
        except (RuntimeError, ValueError) as e:
            if arm == "fp32":
                raise  # the baseline failing is a real error, not a skip
            # e.g. fp16 autocast unsupported on this device/build
            print(f"[{arm}] skipped: {e}", flush=True)
            report_arms[arm] = {"skipped": str(e)}
            continue
        report_arms[arm] = compare_arms(raw["fp32"], raw[arm], ratio_tol)
        print(f"[{arm}] done in {time.monotonic() - t0:.1f}s", flush=True)
    return report_arms


# ------------------------------------------------------------------ memcurve


def synth_trajectory(policy, rows: int, unroll: int, device, seed: int = 0):
    """Dummy Trajectory with production shapes: encoded state struct from
    the embedding's dummy(), zero (neutral) actions, and sample-time logits
    taken from the policy's OWN no-grad unroll over those frames — so the
    learner's recomputation matches them like real rollout data does
    (ratio ~ 1, finite losses, real optimizer steps: the memory curve
    exercises the full production allocation profile)."""
    import numpy as np

    from slippi_ai.types import Frames, StateAction

    from smashbot.rl.ppo import ActionData, Trajectory

    torch.manual_seed(seed)

    def to_t(x):
        x = np.asarray(x)
        if x.dtype.kind in "iu":
            x = x.astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    sae = policy.network.embed_state_action
    dummy = tree.map_structure(to_t, sae.dummy((rows, unroll + 1)))
    is_resetting = torch.zeros(
        rows, unroll + 1, dtype=torch.bool, device=device
    )
    rewards = 0.1 * torch.randn(rows, unroll, device=device)
    frames = Frames(
        state_action=StateAction(
            state=dummy.state, action=dummy.action, name=dummy.name
        ),
        is_resetting=is_resetting,
        reward=rewards,
    )
    with torch.no_grad():
        out = policy.unroll(frames, policy.initial_state(rows, device))
    # [B, T] -> [B, T+1]: repeat the last frame (position T is never read as
    # actor logits — _fixed_pass slices [:, :-1])
    logits = tree.map_structure(
        lambda t: torch.cat([t, t[:, -1:]], dim=1), out.logits
    )
    return Trajectory(
        states=dummy.state,
        name=dummy.name,
        actions=ActionData(controller_state=dummy.action, logits=logits),
        rewards=rewards,
        is_resetting=is_resetting,
        initial_state=policy.initial_state(rows, device),
    )


def run_memcurve(learner, rows_list, unroll: int, arm: str) -> list[dict]:
    """Full learner steps (fixed pass + policy loss + optimizer step +
    post-update pass, i.e. the real per-step allocation profile including
    the Adam moments and the revert snapshot) at increasing row counts;
    peak GPU memory per count, stopping at the first OOM."""
    results = []
    for rows in rows_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        try:
            traj = synth_trajectory(learner.policy, rows, unroll, "cuda")
            state = learner.initial_state(rows, "cuda")
            with autocast_ctx("cuda", arm):
                learner.step([traj], state)
            peak = torch.cuda.max_memory_allocated()
            entry = {
                "rows": rows,
                "peak_bytes": int(peak),
                "peak_gb": round(peak / 2**30, 3),
                "seconds": round(time.monotonic() - t0, 1),
                "oom": False,
            }
            print(f"rows {rows:4d}: peak {entry['peak_gb']:.2f} GiB "
                  f"({entry['seconds']}s)", flush=True)
        except torch.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            entry = {"rows": rows, "oom": True, "error": str(e)[:200]}
            print(f"rows {rows:4d}: OOM — ceiling reached", flush=True)
            results.append(entry)
            break
        results.append(entry)
    return results


# ----------------------------------------------------------------- CLI glue


def run_lambda_calib(learner, trajectories, target_shares=(0.10, 0.15, 0.20)):
    """Measure the PPO policy gradient vs the imitation gradient at lambda=1
    on the SAME rows (rows reused as pseudo-demonstrations: L_opp math is
    identical and NLL magnitude is what sets the scale, regardless of whose
    actions they are). recommended_lambda(share) = share * ||g_ppo|| /
    ||g_imit||. No optimizer steps; weights untouched."""
    freeze_value_updates(learner)
    ppo_trajs = [
        t for t in trajectories if getattr(t, "kind", "ppo") != "imitation"
    ]
    assert ppo_trajs, "batch contains no PPO trajectories"
    batch_size = ppo_trajs[0].rewards.shape[0]
    device = next(learner.policy.parameters()).device
    state = learner.initial_state(batch_size, device)

    learner.policy_optimizer.zero_grad(set_to_none=True)
    ppo_losses = []
    for traj in ppo_trajs:
        fixed, state, _ = learner._fixed_pass(traj, state)
        loss, _ = learner._policy_loss(fixed)
        loss.backward()
        ppo_losses.append(float(loss.detach()))
    g_ppo, nf = _grad_stats(learner.policy.parameters())
    assert nf == 0, "nonfinite PPO grads in calibration"

    learner.policy_optimizer.zero_grad(set_to_none=True)
    imit_losses = []
    for traj in ppo_trajs:
        imf = learner._imitation_fixed(traj)
        if imf is None:
            continue
        loss = learner._imitation_policy_loss(imf)
        loss.backward()
        imit_losses.append(float(loss.detach()))
    assert imit_losses, "no imitation-eligible trajectories"
    g_imit, nf = _grad_stats(learner.policy.parameters())
    assert nf == 0, "nonfinite imitation grads in calibration"
    learner.policy_optimizer.zero_grad(set_to_none=True)

    out = {
        "ppo_loss_mean": sum(ppo_losses) / len(ppo_losses),
        "imitation_loss_mean_at_lambda1": sum(imit_losses) / len(imit_losses),
        "grad_norm_ppo": g_ppo,
        "grad_norm_imitation_at_lambda1": g_imit,
        "grad_share_at_lambda1": g_imit / max(g_ppo, 1e-12),
        "recommended_lambda": {
            f"{int(s*100)}pct": s * g_ppo / max(g_imit, 1e-12)
            for s in target_shares
        },
    }
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    fid = sub.add_parser(
        "fidelity",
        help="fp32/bf16/fp16 numeric diff on a saved real batch (Tier 1)",
    )
    fid.add_argument("--batch", required=True,
                     help="saved batch from scripts/capture_batch.py")
    src = fid.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", default="",
                     help="full RL checkpoint (config+state)")
    src.add_argument("--snapshot", default="",
                     help="bare policy state_dict; needs --config-from")
    fid.add_argument("--config-from", default=DEFAULT_CONFIG_FROM,
                     help="full checkpoint for --snapshot mode")
    fid.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                     help="cpu is safe beside the live run; cuda only in a "
                          "training gap")
    fid.add_argument("--ratio-tol", type=float, default=1e-3,
                     help="tolerance for the ratio_mean==1 invariant flag")
    fid.add_argument("--out", default="",
                     help="JSON report path (default: <batch>.fidelity.json)")

    cal = sub.add_parser(
        "lambda-calib",
        help="imitation-vs-PPO gradient share on a saved batch -> "
             "recommended imitation_lambda (cpu-safe beside the live run)",
    )
    cal.add_argument("--batch", required=True)
    csrc = cal.add_mutually_exclusive_group(required=True)
    csrc.add_argument("--ckpt", default="")
    csrc.add_argument("--snapshot", default="")
    cal.add_argument("--config-from", default=DEFAULT_CONFIG_FROM)
    cal.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    cal.add_argument("--out", default="")

    mem = sub.add_parser(
        "memcurve",
        help="peak-memory vs learner rows under a precision (Tier 2, "
             "IDLE GPU ONLY)",
    )
    mem.add_argument("--config-from", default=DEFAULT_CONFIG_FROM,
                     help="full checkpoint that builds the learner")
    mem.add_argument("--precision", default="bf16", choices=ARMS)
    mem.add_argument("--rows", default="120,136,152,168,184,200,224,256",
                     help="comma-separated learner row counts to try")
    mem.add_argument("--unroll", type=int, default=240,
                     help="frames per chunk (T); production rollout length")
    mem.add_argument("--i-have-the-gpu", action="store_true",
                     help="required: asserts the GPU is IDLE (no live "
                          "training run)")
    mem.add_argument("--out", default="",
                     help="JSON report path (default: "
                          "memcurve-<precision>.json in cwd)")
    return ap.parse_args(argv)


def main_fidelity(args) -> None:
    saved = torch.load(args.batch, map_location="cpu", weights_only=False)
    trajectories = to_device(saved["trajectories"], args.device)
    meta = saved.get("meta", {})
    print(f"fidelity: batch {args.batch} "
          f"(step {meta.get('rl_step', '?')}, "
          f"{trajectories[0].rewards.shape[0]} rows), device {args.device}",
          flush=True)
    if args.ckpt and meta.get("student_ckpt") and (
        os.path.abspath(args.ckpt)
        != os.path.abspath(meta["student_ckpt"])
    ):
        print(f"warning: batch was captured with {meta['student_ckpt']} — "
              f"the ratio_mean==1 invariant only holds when the learner "
              f"loads the SAME weights", flush=True)
    learner, step = build_learner(
        ckpt=args.ckpt, snapshot=args.snapshot,
        config_from=args.config_from, device=args.device,
    )
    arms = run_fidelity(
        learner, trajectories, args.device, ratio_tol=args.ratio_tol
    )
    report = {
        "kind": "fidelity",
        "batch": os.path.abspath(args.batch),
        "batch_meta": {
            k: v for k, v in meta.items() if not isinstance(v, (list, dict))
        },
        "learner_step": step,
        "device": args.device,
        "rl_config": dataclasses.asdict(learner.config),
        "arms": arms,
        "notes": [
            "no optimizer steps ran: value_optimizer.step no-op'd, policy "
            "grads measured then zeroed — weights identical across arms",
            "fp16 arm ran WITHOUT GradScaler (no optimizer step here); a "
            "real fp16 training arm needs loss scaling",
            "network unrolls under autocast; log-probs/KLs/advantages fp32 "
            "via autocast's fp32 ops + value.py's fp32 island (see dtypes "
            "receipt per arm)",
        ],
        "torch_version": torch.__version__,
        "timestamp": __import__("datetime").datetime.now().isoformat(
            timespec="seconds"
        ),
    }
    print()
    print(format_table(arms))
    out = args.out or (args.batch + ".fidelity.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}", flush=True)


def main_memcurve(args) -> None:
    print("=" * 72)
    print("MEMCURVE ALLOCATES THE WHOLE GPU AND *WILL* OOM A LIVE TRAINING")
    print("RUN. Run it only in a training gap (tmux 'smash' stopped).")
    print("=" * 72)
    if not args.i_have_the_gpu:
        sys.exit("refusing to run without --i-have-the-gpu "
                 "(the live run owns the GPU by default)")
    if not torch.cuda.is_available():
        sys.exit("memcurve requires CUDA")
    rows_list = [int(r) for r in args.rows.split(",") if r.strip()]
    learner, step = build_learner(
        config_from=args.config_from, device="cuda", freeze=False
    )
    if args.precision == "fp16":
        print("note: fp16 memcurve steps run WITHOUT GradScaler (memory "
              "measurement only; a real fp16 arm needs loss scaling)",
              flush=True)
    print(f"memcurve: {args.config_from} (step {step}), "
          f"precision {args.precision}, T={args.unroll}, rows {rows_list}",
          flush=True)
    results = run_memcurve(learner, rows_list, args.unroll, args.precision)
    report = {
        "kind": "memcurve",
        "config_from": os.path.abspath(args.config_from),
        "learner_step": step,
        "precision": args.precision,
        "unroll": args.unroll,
        "gpu": torch.cuda.get_device_name(0),
        "total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 2
        ),
        "results": results,
        "torch_version": torch.__version__,
        "timestamp": __import__("datetime").datetime.now().isoformat(
            timespec="seconds"
        ),
    }
    ok = [r for r in results if not r["oom"]]
    if ok:
        ceiling = ok[-1]
        print(f"\nceiling: {ceiling['rows']} rows fit "
              f"({ceiling['peak_gb']} GiB peak) under {args.precision}",
              flush=True)
    out = args.out or f"memcurve-{args.precision}.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {out}", flush=True)


def main_lambda_calib(args) -> None:
    saved = torch.load(args.batch, map_location="cpu", weights_only=False)
    trajectories = to_device(saved["trajectories"], args.device)
    meta = saved.get("meta", {})
    print(f"lambda-calib: batch {args.batch} "
          f"(step {meta.get('rl_step', '?')}), device {args.device}",
          flush=True)
    learner, _ = build_learner(
        ckpt=args.ckpt, snapshot=args.snapshot,
        config_from=args.config_from, device=args.device,
    )
    out = run_lambda_calib(learner, trajectories)
    print(json.dumps(out, indent=2))
    dest = args.out or (args.batch + ".lambda_calib.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dest}", flush=True)


def main() -> None:
    args = parse_args()
    if args.cmd == "fidelity":
        main_fidelity(args)
    elif args.cmd == "lambda-calib":
        main_lambda_calib(args)
    else:
        main_memcurve(args)


if __name__ == "__main__":
    main()
