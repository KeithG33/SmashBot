"""Core game engine shared by live play (play.py) and eval batteries
(evaluate.py): one implementation of policy loading, Dolphin setup, and the
gamestate -> agent -> controller loop, so eval measures exactly the bot that
plays.

A "game" here is one stock match; the Dolphin wrapper auto-restarts into the
next game, and `run_games` yields a GameRecord at each game boundary.
"""

from __future__ import annotations

import dataclasses
import time
import typing as tp

import melee
import numpy as np
import torch

from slippi_ai import controller_lib
from slippi_ai import dolphin as dolphin_lib

from smashbot import configs, embed as embed_lib, saving
from smashbot.eval.agent import DelayedAgent
from smashbot.eval.report import GameRecord
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


def maybe_compile(policy, device: str, verbose: bool = True) -> None:
    """torch.compile policy.sample in place and warm it up (~30-60s)."""
    import torch._dynamo
    import tree

    from slippi_ai.types import StateAction

    mode = "reduce-overhead" if device == "cuda" else "default"
    policy.sample = torch.compile(policy.sample, mode=mode)
    torch._dynamo.config.recompile_limit = 128
    if verbose:
        print("torch.compile enabled; warming up...")

    def to_t(x):
        x = np.asarray(x)
        if x.dtype.kind in "iu":
            x = x.astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(x)).to(device)

    dummy = tree.map_structure(to_t, policy.network.embed_state_action.dummy((1,)))
    dummy_sa = StateAction(state=dummy.state, action=dummy.action, name=dummy.name)
    h = policy.initial_state(1, device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(50):
            _, h = policy.sample(dummy_sa, h)
    if verbose:
        print(f"warmup done in {time.perf_counter() - t0:.0f}s")


def resolve_name_code(name_map: dict, name: str, verbose: bool = True) -> int:
    if name in name_map:
        return name_map[name]
    if name_map and verbose:
        print(f"'{name}' not in name_map {name_map}; using code 0")
    return 0


@dataclasses.dataclass
class Opponent:
    """Parsed opponent spec: cpu:<level>[:<CHAR>] | ckpt:<path>[:<CHAR>] | human."""

    kind: str  # "cpu" | "ckpt" | "human"
    level: int = 9
    character: str = "MARTH"
    ckpt_path: str = ""

    @classmethod
    def parse(cls, spec: str) -> "Opponent":
        parts = spec.split(":")
        kind = parts[0]
        if kind == "cpu":
            level = int(parts[1]) if len(parts) > 1 else 9
            char = parts[2] if len(parts) > 2 else "MARTH"
            if not 1 <= level <= 9:
                raise ValueError(f"cpu level must be 1-9, got {level}")
            return cls(kind="cpu", level=level, character=char)
        if kind == "ckpt":
            if len(parts) < 2 or not parts[1]:
                raise ValueError("ckpt spec needs a path: ckpt:/path/to/best.pt")
            # windows-free luxury: path may contain no colons on linux; keep
            # optional trailing :CHAR only if it parses as a character name
            char = "FOX"
            path = ":".join(parts[1:])
            if len(parts) > 2 and parts[-1].isalpha():
                char = parts[-1]
                path = ":".join(parts[1:-1])
            return cls(kind="ckpt", ckpt_path=path, character=char)
        if kind == "human":
            return cls(kind="human")
        raise ValueError(f"unknown opponent spec: {spec!r}")

    def make_player(self):
        if self.kind == "cpu":
            return dolphin_lib.CPU(
                character=melee.Character[self.character.upper()], level=self.level
            )
        if self.kind == "ckpt":
            return dolphin_lib.AI(character=melee.Character[self.character.upper()])
        return dolphin_lib.Human()


def make_dolphin(
    players: dict,
    headless: bool,
    stage: str = "FINAL_DESTINATION",
    fullscreen: bool = False,
    gfx_backend: str = "OGL",
    online_delay: int = 0,
    mute: bool = False,
) -> dolphin_lib.Dolphin:
    """One Dolphin. Headless uses the ExiAI build (Null video, fast-forward);
    visible play uses the standard netplay build."""
    console_kwargs: dict = {"stage": melee.Stage[stage.upper()]}
    # Slippi builds open slippi.gg/online/enable in a browser when their
    # (fresh temp) user dir has no linked account. All our games are local
    # direct-mode (headless fleets AND rendered watch sessions), so make
    # URL-opening a no-op for every Dolphin we spawn. $BROWSER alone is NOT
    # enough: on KDE, xdg-open delegates to kde-open and ignores it — so we
    # also shadow the openers via PATH (shim dir of exit-0 scripts).
    import os

    os.environ["BROWSER"] = "true"
    _noopen = "/home/kage/drive2/ShineBot/noopen"
    if not os.environ.get("PATH", "").startswith(_noopen):
        os.environ["PATH"] = _noopen + os.pathsep + os.environ.get("PATH", "")
    if headless:
        path = EXIAI_APPIMAGE
    else:
        path = NETPLAY_APPIMAGE
        console_kwargs["fullscreen"] = fullscreen
        if gfx_backend:
            console_kwargs["gfx_backend"] = gfx_backend
        if mute:
            # Pulse underruns ("Dropping OutputStream") disturb Dolphin's
            # frame pacing; muting removes the audio path entirely.
            console_kwargs["disable_audio"] = True
    return dolphin_lib.Dolphin(
        path=str(path),
        iso=str(MELEE_ISO),
        players=players,
        headless=headless,
        online_delay=online_delay,
        emulation_speed=0 if headless else 1,
        copy_home_directory=copy_home_config,
        **console_kwargs,
    )


def run_games(
    dolphin: dolphin_lib.Dolphin,
    agents: dict[int, DelayedAgent],
    num_games: int = 0,
    max_frames_per_game: int = 8 * 60 * 60,
    on_frame: tp.Callable[[melee.GameState, int, float], bool | None] | None = None,
) -> tp.Iterator[GameRecord]:
    """Yields one GameRecord per completed game; stops after `num_games`
    (0 = run forever, for live play). On timeout the record is yielded with
    timeout=True and iteration stops — the caller should restart Dolphin.

    on_frame(gamestate, frames_this_game, agent_step_seconds) is called every
    frame; returning True stops iteration immediately (mid-game, no record).

    Stats are tracked from the bot's perspective: the lowest agent port is
    "us", the other agent's opponent port is the opponent.
    """
    bot_port = min(agents)
    opp_port = agents[bot_port]._ports[1]

    completed = 0
    last_frame: int | None = None
    frames_this_game = 0
    prev_pct = {p: 0.0 for p in (bot_port, opp_port)}
    dealt = taken = 0.0
    last_stocks = {bot_port: 4, opp_port: 4}

    def finalize(timeout: bool = False) -> GameRecord:
        b, o = last_stocks[bot_port], last_stocks[opp_port]
        winner = None
        if b != o:
            winner = "bot" if b > o else "opp"
        return GameRecord(
            winner=winner,
            bot_stocks=b,
            opp_stocks=o,
            bot_damage_dealt=dealt,
            bot_damage_taken=taken,
            frames=frames_this_game,
            timeout=timeout,
        )

    for gamestate in dolphin.iter_gamestates(skip_menu_frames=True):
        if last_frame is not None and gamestate.frame < last_frame:
            # game boundary: previous game is over
            yield finalize()
            completed += 1
            if num_games and completed >= num_games:
                return
            for agent in agents.values():
                agent.reset()
            frames_this_game = 0
            prev_pct = {p: 0.0 for p in prev_pct}
            dealt = taken = 0.0
        last_frame = gamestate.frame

        t0 = time.perf_counter()
        for port, agent in agents.items():
            controller_state = agent.step(gamestate)
            controller_lib.send_controller(dolphin.controllers[port], controller_state)
        step_seconds = time.perf_counter() - t0

        frames_this_game += 1
        for port in (bot_port, opp_port):
            player = gamestate.players.get(port)
            if player is None:
                continue
            delta = float(player.percent) - prev_pct[port]
            if delta > 0:
                if port == bot_port:
                    taken += delta
                else:
                    dealt += delta
            prev_pct[port] = float(player.percent)
            last_stocks[port] = int(player.stock)

        if on_frame is not None:
            if on_frame(gamestate, frames_this_game, step_seconds):
                return

        if frames_this_game >= max_frames_per_game:
            yield finalize(timeout=True)
            return
