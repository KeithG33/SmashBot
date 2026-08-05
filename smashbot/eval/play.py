"""Play a trained ShineBot checkpoint in Dolphin.

Bot (Fox) on port 1; opponent on port 2 is an in-game CPU by default, or a
human with --opponent human (plug in a controller / configure inputs in the
Dolphin window).

Thin wrapper over eval/game.py — live play and eval batteries share one
gamestate -> agent -> controller path, so they can never drift apart.

Usage:
  .venv/bin/python -m smashbot.eval.play                      # vs CPU 9, visible
  .venv/bin/python -m smashbot.eval.play --opponent human
  .venv/bin/python -m smashbot.eval.play --headless --max-frames 3600
"""

import argparse

import melee
import numpy as np

from slippi_ai import dolphin as dolphin_lib

from smashbot.eval import game as game_lib
from smashbot.eval.agent import DelayedAgent

# re-exported for backwards compatibility (older scripts import from play)
load_policy = game_lib.load_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt", default="/home/kage/drive2/ShineBot/runs/debug-fox-v0/best.pt"
    )
    ap.add_argument("--opponent", choices=["cpu", "human"], default="cpu")
    ap.add_argument("--cpu_level", type=int, default=9)
    ap.add_argument("--opponent_char", default="MARTH")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--fullscreen", action="store_true", help="default is windowed")
    ap.add_argument("--gfx_backend", default="OGL", help="OGL | Vulkan | ''")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = play forever")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(policy.sample). With the packed embed "
                         "path this is a ~2x win on CPU (13ms -> 7ms); "
                         "first ~100 frames are slow while compiling")
    ap.add_argument(
        "--name", default="Master Player",
        help="identity to condition on (looked up in the checkpoint's name_map)",
    )
    # CPU beats GPU for batch-1 inference (kernel-launch overhead dominates).
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    policy, name_map, step = game_lib.load_policy(args.ckpt, args.device)
    if args.compile:
        game_lib.maybe_compile(policy, args.device)
    name_code = game_lib.resolve_name_code(name_map, args.name)
    print(f"loaded {args.ckpt} (train step {step}), delay={policy.delay}, "
          f"conditioning on {args.name!r} -> code {name_code} (map: {name_map})")

    if args.opponent == "cpu":
        spec = f"cpu:{args.cpu_level}:{args.opponent_char}"
    else:
        spec = "human"
    opponent = game_lib.Opponent.parse(spec)
    players = {
        1: dolphin_lib.AI(character=melee.Character.FOX),
        2: opponent.make_player(),
    }
    dolphin = game_lib.make_dolphin(
        players,
        headless=args.headless,
        stage=args.stage,
        fullscreen=args.fullscreen,
        gfx_backend=args.gfx_backend,
    )
    agent = DelayedAgent(
        policy, own_port=1, opponent_port=2, name_code=name_code,
        console_delay=0, temperature=args.temperature, device=args.device,
    )

    frames = 0
    step_times: list[float] = []

    def on_frame(gamestate, _frames_this_game, step_seconds) -> bool:
        nonlocal frames
        step_times.append(step_seconds)
        frames += 1
        if frames % 3600 == 0:
            mean_ms = 1e3 * float(np.mean(step_times[-3600:]))
            p1, p2 = gamestate.players[1], gamestate.players[2]
            print(f"frame {frames}: step {mean_ms:.1f}ms | "
                  f"bot {p1.stock} stocks {p1.percent:.0f}% | "
                  f"opp {p2.stock} stocks {p2.percent:.0f}%")
            if mean_ms > 12 and not args.headless:
                print("WARNING: too slow for real-time play")
        return bool(args.max_frames and frames >= args.max_frames)

    try:
        for record in game_lib.run_games(dolphin, {1: agent}, on_frame=on_frame):
            print(f"game over: {record.winner or 'draw'} "
                  f"{record.bot_stocks}-{record.opp_stocks} "
                  f"(dealt {record.bot_damage_dealt:.0f}%, "
                  f"took {record.bot_damage_taken:.0f}%)")
    finally:
        dolphin.stop()
        if step_times:
            print(f"mean step time: {1e3 * float(np.mean(step_times)):.2f}ms "
                  f"over {frames} frames")


if __name__ == "__main__":
    main()
