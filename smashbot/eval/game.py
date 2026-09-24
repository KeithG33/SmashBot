"""Core game engine shared by live play (play.py) and eval batteries
one implementation of policy loading, Dolphin setup, and the
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

from slippi_ai import controller_lib
from slippi_ai import dolphin as dolphin_lib

from smashbot import saving
from smashbot.eval.agent import DelayedAgent
from smashbot.eval.dolphin_setup import make_dolphin  # noqa: F401  (re-export)
from smashbot.networks import check_loadable
from smashbot.policy import build_policy_from_config


@dataclasses.dataclass
class GameRecord:
    """One game, from the bot's perspective (bot on port 1)."""

    winner: str | None  # "bot" | "opp" | None (draw)
    bot_stocks: int
    opp_stocks: int
    bot_damage_dealt: float  # sum of opponent percent gains
    bot_damage_taken: float
    frames: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_policy(ckpt_path: str, device: str):
    ckpt = saving.load_checkpoint(ckpt_path)
    check_loadable(ckpt["config"]["network"], ckpt["state"]["policy"])
    policy = build_policy_from_config(ckpt["config"]).to(device)
    policy.load_state_dict(ckpt["state"]["policy"])
    policy.eval()
    name_map = ckpt["state"].get("name_map", {})
    return policy, name_map, ckpt["state"].get("step")


def compile_policy(policy) -> None:
    """torch.compile policy.sample in place; the agent's warm_up() compiles it
    through the live call before play (~30-60s)."""
    import torch._dynamo

    # "default" (inductor fusion, no cudagraphs) on both devices. cudagraphs
    # (mode="reduce-overhead") reuse output-tensor storage across runs, which
    # collides with the async agent holding the previous frame's action while
    # the next forward runs ("accessing tensor output of CUDAGraphs that has
    # been overwritten"). Fusion alone is enough for batch-1 live inference.
    policy.sample = torch.compile(policy.sample, mode="default")
    torch._dynamo.config.recompile_limit = 128


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


def run_games(
    dolphin: dolphin_lib.Dolphin,
    agents: dict[int, DelayedAgent],
    num_games: int = 0,
    on_frame: tp.Callable[[melee.GameState, int, float], bool | None] | None = None,
) -> tp.Iterator[GameRecord]:
    """Yields one GameRecord per completed game; stops after `num_games`
    (0 = run forever, for live play).

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

    def finalize() -> GameRecord:
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

