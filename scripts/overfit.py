"""M4 Check: memorize a handful of real games to ~0 loss.

Deliberately minimal training instrument (not the real train_bc.py): fixed
small replay set, no eval split, loss printed every log_interval steps with
per-component breakdown.

Usage (from repo root):
  .venv/bin/python scripts/overfit.py --num_games 4 --steps 2000
  .venv/bin/python scripts/overfit.py --num_games 4 --delay 0   # delay ablation
"""

import argparse
import time

import torch
import tree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kage/drive2/ShineBot/data/debug-fox/Root")
    ap.add_argument("--num_games", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--unroll_length", type=int, default=80)
    ap.add_argument("--delay", type=int, default=18)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="/home/kage/drive2/ShineBot/runs/overfit.pt")
    args = ap.parse_args()

    from slippi_ai import data as data_lib

    from shinebot import configs, embed as embed_lib
    from shinebot.data import loader
    from shinebot.policy import build_policy

    dataset_cfg = data_lib.DatasetConfig(
        data_dir=f"{args.root}/Parsed",
        meta_path=f"{args.root}/meta.json",
        allowed_characters="fox",
        allowed_opponents="all",
        swap=False,  # one perspective per game: purest memorization test
    )
    replays = data_lib.replays_from_meta(dataset_cfg)[: args.num_games]
    print(f"Memorizing {len(replays)} games:")
    for r in replays:
        print(f"  {r.meta.slp_md5[:8]}  chars=({r.meta.p0.character},{r.meta.p1.character})")

    source = data_lib.DataSource(
        replays=replays,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        extra_frames=args.delay + 1,
        num_workers=0,
    )

    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(),
        head_config=configs.ControllerHeadConfig(),
        policy_config=configs.PolicyConfig(delay=args.delay),
        num_names=1,
    ).to(args.device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params/1e6:.1f}M params, delay={args.delay}, device={args.device}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    hidden = policy.initial_state(args.batch_size, device=args.device)

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch_with_meta, epoch = next(source)
        frames = loader.batch_to_frames(batch_with_meta.batch, policy.network)
        frames = tree.map_structure(
            lambda t: t.to(args.device, non_blocking=True), frames
        )

        loss, hidden, metrics = policy.imitation_loss(frames, hidden)
        hidden = tree.map_structure(lambda t: t.detach(), hidden)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0 or step == 1:
            c = metrics["controller"]
            buttons = sum(c["buttons"]) / len(c["buttons"])
            sticks = (
                c["main_stick"].x + c["main_stick"].y + c["c_stick"].x + c["c_stick"].y
            ) / 4
            dt = time.perf_counter() - t0
            print(
                f"step {step:5d}  epoch {epoch:6.1f}  "
                f"policy_loss {metrics['policy_loss']:7.4f}  "
                f"buttons {buttons:6.4f}  sticks {sticks:6.4f}  "
                f"shoulder {c['shoulder']:6.4f}  "
                f"value_uev {metrics['value']['uev']:5.2f}  "
                f"({dt / step:.2f}s/step)"
            )

    source.shutdown()
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "delay": args.delay,
            "num_names": 1,
            "games": [r.meta.slp_md5 for r in replays],
        },
        args.save,
    )
    print(f"Saved checkpoint to {args.save}")


if __name__ == "__main__":
    main()
