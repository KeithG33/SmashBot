"""v11 learning-health check: pull recent wandb history, apply
reasonableness bounds, print ALERT lines (or an OK digest with --digest).

Expectations encoded (Keith, 2026-09-15):
- phillips are much harder than v10's medium-only diet: tier winrates
  near/below 0.5 are EXPECTED; a collapse (<0.15) or a large fast drop is not.
- imitation lambda is FLAT 0.01 (v10 ended at 0.002, so ~5x stronger pull):
  tKL / imitation loss shifting regime would be the first sign of a
  learning-behavior change.
- self-play winrate ~0.5 is also the fp16 seat-B correctness diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import os

RUN = "keithg33/shinebot/rl-sim-v12b"
STATE = "/home/kage/drive2/ShineBot/runs/rl-sim-v12/health_state.json"

KEYS = [
    "rl/teacher_kl", "rl/actor_kl_mean", "rl/reverted",
    "rl/imitation/loss", "rl/imitation/lambda",
    "rl/self/win_rate_ema", "rl/frames_per_sec",
    "rl/phillip/medium/win_rate_ema", "rl/phillip/plat/win_rate_ema",
    "rl/phillip/diamond/win_rate_ema", "rl/phillip/master/win_rate_ema",
    "rl/phillip/gm/win_rate_ema",
    "rl/value_loss",
]


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", action="store_true")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=60)
    run = api.run(RUN)
    rows = list(run.history(samples=60, keys=KEYS, pandas=False))
    if len(rows) < 5:
        print("ALERT: fewer than 5 recent history rows — run logging stalled?")
        return
    last = rows[-1]
    step = last.get("_step", "?")

    state = {}
    if os.path.exists(STATE):
        with open(STATE) as f:
            state = json.load(f)
    hi = state.setdefault("rolling_max", {})
    alerts = []

    def recent(key, n=20):
        return mean([r.get(key) for r in rows[-n:]])

    # --- core losses / KLs ---
    tkl = recent("rl/teacher_kl")
    if tkl is None or not (0.002 <= tkl <= 0.05):
        alerts.append(f"tKL out of regime: {tkl} (healthy ~0.01, v10 band 0.002-0.05)")
    akl = recent("rl/actor_kl_mean")
    if akl is None or akl > 1e-4:
        alerts.append(f"actor KL high: {akl} (revert threshold territory)")
    rev = mean([r.get("rl/reverted") for r in rows[-20:]])
    if rev and rev > 0.05:
        alerts.append(f"reverts firing: {rev:.0%} of recent steps")
    im = recent("rl/imitation/loss")
    if im is None or math.isnan(im) or im > 10:
        alerts.append(f"imitation loss abnormal: {im}")
    vl = recent("rl/value_loss")
    if vl is not None and (math.isnan(vl) or vl > 5):
        alerts.append(f"value loss abnormal: {vl}")

    # --- winrates ---
    sw = recent("rl/self/win_rate_ema")
    if sw is not None and not (0.40 <= sw <= 0.60):
        alerts.append(f"SELF winrate {sw:.3f} outside [0.40,0.60] — symmetry/fp16 seat-B diagnostic")
    for tier in ("medium", "plat", "diamond", "master", "gm"):
        k = f"rl/phillip/{tier}/win_rate_ema"
        v = recent(k, n=10)
        if v is None:
            continue
        if v < 0.15:
            alerts.append(f"phillip:{tier} collapse: winrate {v:.3f}")
        peak = hi.get(k, v)
        if v < peak - 0.15:
            alerts.append(f"phillip:{tier} dropped {peak:.2f}->{v:.2f} (>0.15 off rolling max)")
        hi[k] = max(peak, v)

    fps = recent("rl/frames_per_sec", n=5)
    if fps is not None and fps < 4000:
        alerts.append(f"fps degraded: {fps:.0f} (steady ~6000)")

    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(state, f)

    for a in alerts:
        print(f"ALERT step {step}: {a}")
    if args.digest or not alerts:
        phil = " ".join(
            f"{t[:2]}{recent(f'rl/phillip/{t}/win_rate_ema', 5) or float('nan'):.2f}"
            for t in ("medium", "plat", "diamond", "master", "gm"))
        line = (f"step {step} | tKL {tkl:.4f} aKL {akl:.1e} imit {im:.3f} "
                f"self {sw if sw is not None else float('nan'):.2f} | {phil} | {fps:.0f} fps")
        print(("DIGEST " if args.digest else "OK ") + line)


if __name__ == "__main__":
    main()
