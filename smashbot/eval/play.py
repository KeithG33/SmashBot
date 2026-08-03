"""Play a trained ShineBot checkpoint in Dolphin.

Bot (Fox) on port 1; opponent on port 2 is an in-game CPU by default, or a
human with --opponent human (plug in a controller / configure inputs in the
Dolphin window).

Usage:
  .venv/bin/python -m smashbot.eval.play                      # vs CPU 9, visible
  .venv/bin/python -m smashbot.eval.play --opponent human
  .venv/bin/python -m smashbot.eval.play --headless --max-frames 3600
"""

import argparse
import time

import melee
import numpy as np
import torch

from slippi_ai import controller_lib
from slippi_ai import dolphin as dolphin_lib

from smashbot import configs, embed as embed_lib, saving
from smashbot.eval.agent import DelayedAgent
from smashbot.paths import EXIAI_APPIMAGE, MELEE_ISO, NETPLAY_APPIMAGE
from smashbot.policy import build_policy


def load_policy(ckpt_path: str, device: str):
    ckpt = saving.load_checkpoint(ckpt_path)
    cfg = ckpt["config"]
    policy = build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(
            axis_spacing=cfg["head"]["axis_spacing"],
            shoulder_spacing=cfg["head"]["shoulder_spacing"],
        ),
        network_config=configs.NetworkConfig(**cfg["network"]),
        head_config=configs.ControllerHeadConfig(**cfg["head"]),
        policy_config=configs.PolicyConfig(**cfg["policy"]),
        num_names=cfg["data"]["max_names"],
    ).to(device)
    policy.load_state_dict(ckpt["state"]["policy"])
    policy.eval()
    name_map = ckpt["state"].get("name_map", {})
    return policy, name_map, ckpt["state"].get("step")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", default="/home/kage/drive2/ShineBot/runs/debug-fox-v0/best.pt"
    )
    ap.add_argument("--opponent", choices=["cpu", "human"], default="cpu")
    ap.add_argument("--cpu_level", type=int, default=9)
    ap.add_argument("--opponent_char", default="MARTH")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fullscreen", action="store_true", help="default is windowed")
    ap.add_argument("--gfx_backend", default="OGL", help="OGL | Vulkan | ''")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = play forever")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(policy.sample, mode='reduce-overhead'); "
                         "first ~100 frames are slow while compiling")
    ap.add_argument(
        "--name", default="Master Player",
        help="identity to condition on (looked up in the checkpoint's name_map)",
    )
    # CPU beats GPU for batch-1 inference (kernel-launch overhead dominates):
    # 13.8ms vs 17.4ms per frame on the 3090 box.
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    policy, name_map, step = load_policy(args.ckpt, args.device)
    if args.compile:
        import tree

        policy.sample = torch.compile(policy.sample, mode="reduce-overhead")
        print("torch.compile enabled; warming up...")

        import torch._dynamo

        torch._dynamo.config.recompile_limit = 128

        def _to_t(x):
            x = np.asarray(x)
            if x.dtype.kind in "iu":
                x = x.astype(np.int64)
            return torch.from_numpy(np.ascontiguousarray(x)).to(args.device)

        from slippi_ai.types import StateAction as _SA

        dummy = tree.map_structure(_to_t, policy.network.embed_state_action.dummy((1,)))
        dummy_sa = _SA(state=dummy.state, action=dummy.action, name=dummy.name)
        h = policy.initial_state(1, args.device)
        t0 = time.perf_counter()
        for _ in range(50):
            _, h = policy.sample(dummy_sa, h)
        print(f"warmup done in {time.perf_counter() - t0:.0f}s")
    if args.name in name_map:
        name_code = name_map[args.name]
    else:
        name_code = 0
        if name_map:
            print(f"'{args.name}' not in name_map {name_map}; using code 0")
    print(f"loaded {args.ckpt} (train step {step}), delay={policy.delay}, "
          f"conditioning on {args.name!r} -> code {name_code} (map: {name_map})")

    opponent_char = melee.Character[args.opponent_char.upper()]
    if args.opponent == "cpu":
        opponent = dolphin_lib.CPU(character=opponent_char, level=args.cpu_level)
    else:
        opponent = dolphin_lib.Human()

    players = {1: dolphin_lib.AI(character=melee.Character.FOX), 2: opponent}
    # ExiAI builds are inference-only (Null video, no rendering) — perfect for
    # headless, useless for watching. Visible play uses the standard netplay build.
    console_kwargs = {}
    if args.headless:
        dolphin_path = EXIAI_APPIMAGE
    else:
        dolphin_path = NETPLAY_APPIMAGE
        console_kwargs = dict(fullscreen=args.fullscreen)
        if args.gfx_backend:
            console_kwargs["gfx_backend"] = args.gfx_backend

    dolphin = dolphin_lib.Dolphin(
        path=str(dolphin_path),
        iso=str(MELEE_ISO),
        players=players,
        headless=args.headless,
        online_delay=0,
        emulation_speed=0 if args.headless else 1,
        **console_kwargs,
    )

    agent = DelayedAgent(
        policy,
        own_port=1,
        opponent_port=2,
        name_code=name_code,
        console_delay=0,
        temperature=args.temperature,
        device=args.device,
    )
    controller = dolphin.controllers[1]

    frames = 0
    last_game_frame = None
    step_times = []
    try:
        for gamestate in dolphin.iter_gamestates(skip_menu_frames=True):
            if last_game_frame is not None and gamestate.frame < last_game_frame:
                print("new game detected; resetting agent state")
                agent.reset()
            last_game_frame = gamestate.frame

            t0 = time.perf_counter()
            controller_state = agent.step(gamestate)
            controller_lib.send_controller(controller, controller_state)
            step_times.append(time.perf_counter() - t0)

            frames += 1
            if frames % 3600 == 0:
                mean_ms = 1e3 * float(np.mean(step_times[-3600:]))
                p1 = gamestate.players[1]
                p2 = gamestate.players[2]
                print(
                    f"frame {frames}: step {mean_ms:.1f}ms | "
                    f"bot {p1.stock} stocks {p1.percent:.0f}% | "
                    f"opp {p2.stock} stocks {p2.percent:.0f}%"
                )
                if mean_ms > 12 and not args.headless:
                    print("WARNING: too slow for real-time play")
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        dolphin.stop()
        if step_times:
            print(f"mean step time: {1e3 * float(np.mean(step_times)):.2f}ms "
                  f"over {frames} frames")


if __name__ == "__main__":
    main()
