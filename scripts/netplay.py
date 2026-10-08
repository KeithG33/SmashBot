"""Play a policy on Slippi Online Direct.

The bot signs in as its own Slippi account (--user-json, never yours) and
dials --connect-code; the other player dials the bot's code. The bot's
Dolphin takes --online-delay frames of Slippi's input delay and the agent
runs its policy delay minus that, so the bot reacts exactly as trained while
its inputs reach the other side early (fewer rollbacks there). Slippi seats
the players, so each game finds the bot's port by its connect code.

Usage:
  .venv/bin/python scripts/netplay.py --ckpt <ckpt> --user-json <bot user.json> --connect-code ABCD#123
"""

import argparse
import json
import os
import sys
import unicodedata

import melee
import torch
from melee.slippstream import EnetDisconnected

from slippi_ai import controller_lib
from slippi_ai import dolphin as dolphin_lib

from smashbot import saving
from smashbot.eval import game as game_lib
from smashbot.eval.agent import AsyncDelayedAgent


def _code(text: str) -> str:
    """Melee writes connect codes full-width; compare them as typed."""
    return unicodedata.normalize("NFKC", text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Play a policy on Slippi Online Direct.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--user-json", required=True,
                    help="the bot's own Slippi account (a second account, not yours)")
    ap.add_argument("--connect-code", required=True, help="the code the bot dials")
    ap.add_argument("--char", default="FOX")
    ap.add_argument("--stage", default="random",
                    help="the bot's pick when it chooses: a stage, or 'random' = a legal stage each game")
    ap.add_argument("--online-delay", type=int, default=15,
                    help="Slippi input delay on the bot's side, taken out of its policy delay")
    ap.add_argument("--name", default="Master Player", help="identity to condition on")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--replay-dir", default="/home/kage/drive2/ShineBot/replays/netplay")
    args = ap.parse_args(argv)

    with open(args.user_json) as f:
        bot_code = _code(json.load(f)["connectCode"])
    if bot_code == _code(args.connect_code):
        sys.exit("the bot can't dial its own code: --user-json must be a second account")
    device = args.device if torch.cuda.is_available() else "cpu"

    policy, name_map, step = game_lib.load_policy(args.ckpt, device)
    name_code = saving.resolve_name_code(name_map, args.name)
    agent = AsyncDelayedAgent(policy, own_port=1, opponent_port=2, name_code=name_code,
                              console_delay=args.online_delay, device=device)
    if device == "cuda":
        agent.capture()
    print(f"{args.ckpt} (step {step}) as {args.char.upper()}, {bot_code} dialing {args.connect_code} | "
          f"delay {policy.delay} = {args.online_delay} online + {agent.delay} agent | {device}",
          flush=True)

    os.makedirs(args.replay_dir, exist_ok=True)
    dolphin = game_lib.make_dolphin(
        {1: dolphin_lib.AI(character=melee.Character[args.char.upper()])},
        headless=False,
        stage=game_lib.resolve_stage(args.stage),
        online_delay=args.online_delay,
        mute=True,
        save_replays=True,
        replay_dir=args.replay_dir,
        connect_code=args.connect_code,
        user_json_path=args.user_json,
    )
    # the pinned slippi_ai checks each controller's port for its character,
    # which online is not the in-game port (upstream skips it with a connect code)
    dolphin.next_gamestate = dolphin.console.step
    controller = dolphin.controllers[1]   # the local controller; Slippi picks the in-game port
    last_frame = score = None

    def report():
        if score:
            print("game over: bot {} stocks {:.0f}% | {} {} stocks {:.0f}%".format(*score), flush=True)

    try:
        for gamestate in dolphin.iter_gamestates():
            if last_frame is None or gamestate.frame < last_frame:
                report()
                codes = {port: _code(p.connectCode) for port, p in gamestate.players.items()}
                own = [port for port, code in codes.items() if code == bot_code]
                if len(own) != 1:
                    raise RuntimeError(f"bot {bot_code} not found among {codes}")
                (opponent,) = [port for port in codes if port != own[0]]
                agent.set_ports(own[0], opponent)
                print(f"game: bot on port {own[0]}, {codes[opponent]} on port {opponent}, "
                      f"{gamestate.stage.name}", flush=True)
                dolphin.stage = melee.Stage[game_lib.resolve_stage(args.stage)]   # the next game's pick
            last_frame = gamestate.frame
            bot, opp = gamestate.players[own[0]], gamestate.players[opponent]
            score = (bot.stock, bot.percent, codes[opponent], opp.stock, opp.percent)
            controller_lib.send_controller(controller, agent.step(gamestate))
    except EnetDisconnected:
        print("the bot's Dolphin closed", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        report()
        dolphin.stop()


if __name__ == "__main__":
    main()
