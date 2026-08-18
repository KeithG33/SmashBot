"""Watch two policies fight each other live at ~60fps.

Bot-vs-bot exhibition mode. The old way to watch bot-vs-bot was the rollout
worker's watch path, which runs BOTH inferences serially inside the
frame-sync loop (20-30fps). Here both brains are AsyncDelayedAgents
(smashbot/eval/agent.py): each policy gets a persistent in-order compute
thread, step() just submits the frame and pops the delay queue, so Dolphin
never waits on inference. Occasional slow samples are absorbed by each
policy's delay-queue slack (18/21 frames = ~300ms of cushion), and the
emitted action sequence is identical to the sync agent's.

Defaults: McLaude (RL snapshot, Fox) vs Phillip (medium-v2, random main-12
character) on Final Destination, 1 game, replays saved, audio muted,
torch.compile on (both policies warmed BEFORE Dolphin boots, so there is no
compile stall at game start).

Usage:
  .venv/bin/python scripts/watch_live.py                     # default matchup
  .venv/bin/python scripts/watch_live.py --games 0           # play forever
  .venv/bin/python scripts/watch_live.py --p2-char MARTH --games 3
  .venv/bin/python scripts/watch_live.py --dry-run           # no Dolphin
"""

import argparse
import dataclasses
import os
import random
import sys
import time

import melee
import torch
from melee.slippstream import EnetDisconnected

from smashbot.eval import game as game_lib
from smashbot.eval.agent import AsyncDelayedAgent
from smashbot.rl.pool import MAIN_12

from slippi_ai import dolphin as dolphin_lib

DEFAULT_P1_SNAPSHOT = (
    "/home/kage/drive2/ShineBot/models/rl-best-step0010000-phillip56.pt"
)
DEFAULT_CONFIG_FROM = "/home/kage/drive2/ShineBot/runs/rl-pool-v3/latest.pt"
DEFAULT_P2 = "/home/kage/drive2/ShineBot/models/medium-v2-torch.pt"
DEFAULT_REPLAY_DIR = "/home/kage/drive2/ShineBot/replays/exhibition"

_EPILOG = """\
performance notes:
  * `system76-power profile performance` before watching helps hold 60fps
    (run it yourself; this script never touches power profiles).
  * both brains run on background compute threads (AsyncDelayedAgent), so
    the frame-sync loop only pays two queue pops per frame -- expect ~60fps
    where the serial rollout-worker watch path got 20-30.
  * --pin-cores separates inference cores from Dolphin's like play.py does;
    leave it off while the RL training run owns most of the machine.
"""


@dataclasses.dataclass
class SideSpec:
    """One player's policy source: a full checkpoint OR a bare snapshot
    state_dict that borrows its config/name_map from `config_from`
    (the battery.py --snapshot/--config-from pattern)."""

    ckpt: str = ""
    snapshot: str = ""
    config_from: str = ""

    @property
    def label(self) -> str:
        return self.snapshot or self.ckpt


@dataclasses.dataclass
class SideInfo:
    """Resolved facts about one seat, for matchup printing and tests."""

    port: int
    label: str
    delay: int
    name_code: int
    step: object = None  # train step from the checkpoint, if it carries one


def resolve_spec(
    ckpt: str,
    snapshot: str,
    config_from: str,
    default_ckpt: str = "",
    default_snapshot: str = "",
) -> SideSpec:
    """Turn one seat's CLI flags into a SideSpec. An explicit --pN or
    --pN-snapshot wins; with neither, the seat's default applies."""
    if ckpt and snapshot:
        raise ValueError(
            "give a full checkpoint OR a snapshot for a player, not both"
        )
    if not ckpt and not snapshot:
        ckpt, snapshot = default_ckpt, default_snapshot
    if snapshot and not config_from:
        raise ValueError("snapshot mode needs --config-from (a full checkpoint)")
    return SideSpec(ckpt=ckpt, snapshot=snapshot, config_from=config_from)


def resolve_specs(args) -> dict[int, SideSpec]:
    return {
        1: resolve_spec(
            args.p1, args.p1_snapshot, args.config_from,
            default_snapshot=DEFAULT_P1_SNAPSHOT,
        ),
        2: resolve_spec(
            args.p2, args.p2_snapshot, args.config_from,
            default_ckpt=DEFAULT_P2,
        ),
    }


def resolve_char(spec: str, rng=None) -> str:
    """'random' (or empty) draws from the main 12; otherwise validate the
    name against libmelee's roster."""
    if not spec or spec.lower() == "random":
        return (rng or random).choice(MAIN_12)
    name = spec.upper()
    if name not in melee.Character.__members__:
        raise ValueError(f"unknown character {spec!r}")
    return name


def load_side(spec: SideSpec, device: str = "cpu"):
    """Returns (policy, name_map, step). Mirrors battery.py's _load_student:
    bare snapshots borrow config/name_map from config_from."""
    if spec.snapshot:
        policy, name_map, _ = game_lib.load_policy(spec.config_from, device)
        state = torch.load(spec.snapshot, map_location=device, weights_only=True)
        policy.load_state_dict(state)
        policy.eval()
        return policy, name_map, None
    return game_lib.load_policy(spec.ckpt, device)


def build_agents(
    specs: dict[int, SideSpec],
    device: str = "cpu",
    compile_policies: bool = False,
    name: str = "Master Player",
    temperature: float | None = None,
) -> tuple[dict[int, AsyncDelayedAgent], dict[int, SideInfo]]:
    """Load both policies and wrap each in an AsyncDelayedAgent. Each agent
    keeps its own delay (from its checkpoint's policy config -- Phillip 21,
    ours 18) and its own name_code (resolved in its own name_map). Compiling
    here also WARMS each policy (game_lib.maybe_compile runs 50 dummy
    forwards), so it must happen before Dolphin boots."""
    ports = sorted(specs)
    agents: dict[int, AsyncDelayedAgent] = {}
    infos: dict[int, SideInfo] = {}
    for port in ports:
        policy, name_map, step = load_side(specs[port], device)
        if compile_policies:
            game_lib.maybe_compile(policy, device)
        code = game_lib.resolve_name_code(name_map, name)
        (opponent,) = [p for p in ports if p != port]
        agents[port] = AsyncDelayedAgent(
            policy,
            own_port=port,
            opponent_port=opponent,
            name_code=code,
            temperature=temperature,
            device=device,
        )
        infos[port] = SideInfo(
            port=port,
            label=specs[port].label,
            delay=agents[port].delay,
            name_code=code,
            step=step,
        )
    return agents, infos


def pin_cores(dolphin, n: int) -> None:
    """Same recipe as play.py: inference on the first n cores, Dolphin on
    the rest (minus SMT siblings). Shared cores caused collision spikes."""
    bot_cores = set(range(n))
    os.sched_setaffinity(0, bot_cores)
    try:
        siblings = set()
        for c in bot_cores:
            path = (f"/sys/devices/system/cpu/cpu{c}/topology/"
                    "thread_siblings_list")
            with open(path) as f:
                for part in f.read().strip().replace("-", ",").split(","):
                    siblings.add(int(part))
        dolphin_cores = set(range(os.cpu_count())) - bot_cores - siblings
        os.sched_setaffinity(dolphin.console._process.pid, dolphin_cores)
        print(f"cores: bots {sorted(bot_cores)}, dolphin gets "
              f"{len(dolphin_cores)} cpus (SMT siblings "
              f"{sorted(siblings - bot_cores)} excluded)")
    except (AttributeError, OSError, ValueError) as e:
        print(f"could not pin dolphin ({e}); shared cores")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Watch two policies play each other live at ~60fps "
                    "(default: McLaude RL snapshot vs Phillip medium-v2).",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--p1", default="",
                    help="full checkpoint for player 1 (default: snapshot "
                         f"mode with {DEFAULT_P1_SNAPSHOT})")
    ap.add_argument("--p2", default=DEFAULT_P2,
                    help="full checkpoint for player 2 (Phillip medium-v2)")
    ap.add_argument("--p1-snapshot", default="",
                    help="bare policy state_dict for player 1; config comes "
                         "from --config-from")
    ap.add_argument("--p2-snapshot", default="",
                    help="bare policy state_dict for player 2; config comes "
                         "from --config-from")
    ap.add_argument("--config-from", default=DEFAULT_CONFIG_FROM,
                    help="full checkpoint supplying config/name_map for "
                         "--pN-snapshot sides")
    ap.add_argument("--p1-char", default="FOX",
                    help="player 1 character ('random' = random main-12)")
    ap.add_argument("--p2-char", default="random",
                    help="player 2 character (default: random main-12)")
    ap.add_argument("--stage", default="FINAL_DESTINATION")
    ap.add_argument("--games", type=int, default=1,
                    help="games to play; 0 = play forever (close the window "
                         "to stop)")
    ap.add_argument("--save-replays", action=argparse.BooleanOptionalAction,
                    default=True, help="write .slp replays")
    ap.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    ap.add_argument("--mute", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="disable Dolphin audio (Pulse underruns disturb "
                         "frame pacing; default on for smooth 60fps)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="torch.compile both policies (cpu inductor) and "
                         "warm them BEFORE Dolphin boots (~2x faster "
                         "inference; avoids the compile stall at game start)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature for both policies")
    ap.add_argument("--name", default="Master Player",
                    help="identity to condition on (looked up in each "
                         "policy's own name_map)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--gfx-backend", default="OGL", help="OGL | Vulkan | ''")
    ap.add_argument("--threads", type=int, default=8,
                    help="torch intra-op threads (batch-1 CPU inference is "
                         "fastest around 8)")
    ap.add_argument("--pin-cores", type=int, default=0,
                    help="pin inference to the first N cores and Dolphin to "
                         "the rest, as in play.py; 0 = off (default -- the "
                         "RL training run may own the machine)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load, compile, and warm everything, print the "
                         "resolved matchup, then exit WITHOUT booting Dolphin")
    return ap.parse_args(argv)


def _winner_str(record) -> str:
    return {"bot": "P1", "opp": "P2"}.get(record.winner, "draw")


def _describe(record, index: int) -> str:
    mins, secs = divmod(record.frames // 60, 60)
    diff = abs(record.bot_stocks - record.opp_stocks)
    tail = " [timeout]" if record.timeout else ""
    if record.winner is None:
        verdict = "draw"
    else:
        verdict = f"{_winner_str(record)} wins by {diff} stock{'s' * (diff != 1)}"
    return (f"game {index}: P1 {record.bot_stocks} - "
            f"{record.opp_stocks} P2 -> {verdict} "
            f"({mins}:{secs:02d}){tail}")


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(args.threads)

    try:
        specs = resolve_specs(args)
        chars = {1: resolve_char(args.p1_char), 2: resolve_char(args.p2_char)}
    except ValueError as e:
        sys.exit(f"error: {e}")

    agents, infos = build_agents(
        specs,
        device="cpu",
        compile_policies=args.compile,
        name=args.name,
        temperature=args.temperature,
    )

    for port in sorted(infos):
        info = infos[port]
        step = f" (train step {info.step})" if info.step is not None else ""
        print(f"P{port}: {info.label}{step}")
        print(f"    char {chars[port]} | delay {info.delay} | "
              f"{args.name!r} -> code {info.name_code}")
    print(f"stage {args.stage} | games {args.games or 'forever'} | "
          f"compile {'on (cpu inductor, warmed)' if args.compile else 'off'} | "
          f"mute {'on' if args.mute else 'off'} | "
          f"replays {'-> ' + args.replay_dir if args.save_replays else 'off'}")

    if args.dry_run:
        print("dry run: stopping before Dolphin boot")
        return

    if args.save_replays:
        # single rendered dolphin: no Game_<timestamp>.slp collision, so no
        # per-env subdirs needed (that logic lives in the rollout path)
        os.makedirs(args.replay_dir, exist_ok=True)

    players = {
        port: dolphin_lib.AI(character=melee.Character[chars[port]])
        for port in (1, 2)
    }
    dolphin = game_lib.make_dolphin(
        players,
        headless=False,
        stage=args.stage,
        fullscreen=args.fullscreen,
        gfx_backend=args.gfx_backend,
        mute=args.mute,
        save_replays=args.save_replays,
        replay_dir=args.replay_dir if args.save_replays else "",
    )
    if args.pin_cores:
        pin_cores(dolphin, args.pin_cores)

    records = []
    frames = 0
    last_mark = time.perf_counter()

    def on_frame(gamestate, _frames_this_game, _step_seconds):
        nonlocal frames, last_mark
        frames += 1
        if frames % 1800 == 0:  # every 30s of game time
            now = time.perf_counter()
            fps = 1800.0 / (now - last_mark)
            last_mark = now
            p1, p2 = gamestate.players.get(1), gamestate.players.get(2)
            score = ""
            if p1 is not None and p2 is not None:
                score = (f" | P1 {p1.stock} stocks {p1.percent:.0f}% | "
                         f"P2 {p2.stock} stocks {p2.percent:.0f}%")
            waits = " ".join(
                f"p{p} wait {a.stage_ms['queue_wait']:.2f}ms"
                for p, a in agents.items() if hasattr(a, "stage_ms")
            )
            print(f"frame {frames}: {fps:.1f}fps{score} | {waits}")
        return None

    try:
        for record in game_lib.run_games(
            dolphin, agents, num_games=args.games, on_frame=on_frame
        ):
            records.append(record)
            print(_describe(record, len(records)))
            if record.timeout:
                print("frame-count timeout; stopping (restart to continue)")
    except EnetDisconnected:
        print(f"window closed, {len(records)} game"
              f"{'s' * (len(records) != 1)} recorded")
    except KeyboardInterrupt:
        print(f"interrupted, {len(records)} game"
              f"{'s' * (len(records) != 1)} recorded")
    finally:
        try:
            dolphin.stop()
        except Exception:
            pass  # already-dead dolphin (window closed) must not traceback

    if records:
        p1_wins = sum(1 for r in records if r.winner == "bot")
        p2_wins = sum(1 for r in records if r.winner == "opp")
        draws = len(records) - p1_wins - p2_wins
        mean_diff = sum(r.bot_stocks - r.opp_stocks for r in records) / len(records)
        print(f"summary: P1 {p1_wins} - {p2_wins} P2"
              + (f" ({draws} draws)" if draws else "")
              + f" over {len(records)} game{'s' * (len(records) != 1)}, "
              f"avg stock diff {mean_diff:+.1f}")


if __name__ == "__main__":
    main()
