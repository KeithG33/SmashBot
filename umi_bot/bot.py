"""SmashBot as a UMI bot (Computer Melee Championship): one checkpoint, one bot.

umi.Bot keeps each game's state between frames: act gets the state of this
batch's games and returns their new state. The arena applies the input delay,
so unlike eval/agent.py's DelayedAgent there is no action queue: each
observation's sample is the reply.

An entry subclasses SmashBot with what it plays, in the class body, where the
arena reads it; computer-melee-championship's examples/smashbot is an uploaded
zip's version of this file. From the command line (an image), main() builds
that subclass from its arguments:

  python -m umi_bot.bot --checkpoint models/qk-1057011.pt --delay 21 --scope 12
"""
import argparse
import typing as tp

import numpy as np
import torch
import tree

import umi
from umi.convert import to_slippi_ai

from slippi_ai.types import StateAction
from smashbot import encode
from smashbot.eval.agent import _neutral_controller
from smashbot.eval.sim_arena import load_player
from smashbot.policy import Policy


class State(tp.NamedTuple):
    hidden: tp.Any        # the network's recurrent state: a nest of tensors
    prev_action: tp.Any   # the controller it sent last frame, encoded (the policy's input)


class SmashBot(umi.Bot):
    checkpoint: str                       # trained at the bot's one delay
    temperature: float | None = None
    player_name = "Master Player"         # the player name the policy is conditioned on

    def setup(self, session: umi.Session) -> None:
        """Load the policy once, before the first game.

        Args:
            session: This session's settings. The arena delays every input itself, so the
                policy just answers each frame; session.delay must be the one it was
                trained at.
        """

        from smashbot import saving
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        policy, _, _ = load_player(self.checkpoint, self.device, "")
        assert policy.delay == session.delay, f"{self.checkpoint} was trained at delay {policy.delay}"
        self.policy: Policy = policy
        name_map = torch.load(self.checkpoint, map_location="cpu", weights_only=False)["state"].get("name_map", {})
        self.name_code = torch.tensor(saving.resolve_name_code(name_map, self.player_name, verbose=False),
                                      device=self.device)

    def initial_state(self, n: int) -> State:
        """Return the memory of n games that are just starting.

        umi.Bot calls this once to allocate a row for every game, then again as games
        start; act() then receives each game's memory and returns the updated one.

        Args:
            n: How many games are starting.

        Returns:
            The network's recurrent state and the controller the bot sent last frame
            (the policy's input), neutral at the start: tensors with n rows.
        """

        neutral = tree.map_structure(lambda x: np.asarray(x)[None].repeat(n, 0), _neutral_controller())
        encoded = self.policy.controller_head.controller_embedding.from_state(neutral)
        return State(self.policy.initial_state(n, self.device), tree.map_structure(self._tensor, encoded))

    def act(self, batch: umi.Batch, state: State) -> tuple[umi.Actions, State]:
        """Play one frame of N games.

        Args:
            batch: This frame's games. batch.frames is Frame[N], seen from this bot's
                seat: players[:, 0] is this bot.
            state: The memory of the batch's games, row k for batch game k. A new game's
                row comes from initial_state; otherwise it is what the last act()
                returned for that game.

        Returns:
            The actions and the new state, which umi.Bot keeps for each game's next
            frame. The actions hold one controller row of 13 per game: main_x, main_y,
            c_x, c_y and shoulder in [0, 1], then A, B, X, Y, Z, L, R and D_UP, each
            pressed when > 0.5.
        """

        game = tree.map_structure(self._tensor, self.policy.network.encode_game(to_slippi_ai(batch.frames)))
        inputs = StateAction(state=game, action=state.prev_action, name=self.name_code.expand(len(batch.games)))
        sampled, hidden = self.policy.sample(inputs, state.hidden, temperature=self.temperature)
        controller = sampled.controller_state
        decoded = self.policy.controller_head.controller_embedding.decode(
            tree.map_structure(lambda t: t.cpu().numpy(), controller))
        return umi.Actions(ctrl=encode.controller_rows(decoded)), State(hidden, controller)

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x)).to(self.device)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="the policy, e.g. models/qk-1057011.pt")
    parser.add_argument("--delay", type=int, required=True, help="the delay it was trained at: 18 or 21")
    parser.add_argument("--scope", default="12")
    parser.add_argument("--name", default="smashbot")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--player-name", default="Master Player", help="the player name to condition on")
    args, rest = parser.parse_known_args(argv)
    entry = type("SmashBotEntry", (SmashBot,), {
        "name": args.name, "scope": args.scope, "delay": args.delay, "checkpoint": args.checkpoint,
        "temperature": args.temperature, "player_name": args.player_name})
    umi.serve(entry(), argv=rest)


if __name__ == "__main__":
    main()
