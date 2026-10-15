"""SmashBot as a UMI bot (Computer Melee Championship).

TorchBot keeps each game's state between frames: act gets the state of this
batch's games and returns their new state. The arena applies the input delay,
so unlike eval/agent.py's DelayedAgent there is no action queue: each
observation's sample is the reply.

  python -m umi_bot.bot --checkpoint 21=models/qk-1057011.pt --scope 12
"""
import argparse
import typing as tp

import numpy as np
import torch
import tree

import umi
from umi.convert import to_slippi_ai
from umi.torch import TorchBot

from slippi_ai.types import StateAction
from smashbot import encode
from smashbot.eval.agent import _neutral_controller
from smashbot.eval.sim_arena import load_player
from smashbot.policy import Policy


class State(tp.NamedTuple):
    hidden: tp.Any        # the network's recurrent state: a nest of tensors
    prev_action: tp.Any   # the controller it sent last frame, encoded (the policy's input)


class SmashBot(TorchBot):
    name, author, framework, max_games = "smashbot", "KeithG33", "torch", 256

    def __init__(self, checkpoints: dict[int, str], scope: str = "12", name: str = "smashbot",
                 temperature: float | None = None, player_name: str = "Master Player", device: str | None = None):
        super().__init__(device)
        self.checkpoints, self.scope, self.name, self.temperature = checkpoints, scope, name, temperature
        self.player_name = player_name
        self.delays = tuple(sorted(checkpoints))

    def setup(self, session: umi.Session) -> None:
        """Once, before the first game. session.delay is the division being played, so load
        the checkpoint trained for it; the arena delays every input itself."""
        from smashbot import saving
        path = self.checkpoints[session.delay]
        policy, _, _ = load_player(path, self.device, "")
        assert policy.delay == session.delay, f"{path} was trained at delay {policy.delay}"
        self.policy: Policy = policy
        name_map = torch.load(path, map_location="cpu", weights_only=False)["state"].get("name_map", {})
        self.name_code = torch.tensor(saving.resolve_name_code(name_map, self.player_name, verbose=False),
                                      device=self.device)

    def initial_state(self, n: int) -> State:
        """The memory of n games that are just starting, as a nest of tensors with n rows.

        TorchBot calls it once for every slot to allocate storage, then again as games
        start; act() then receives each game's memory and returns the updated one. Here
        the memory is the network's recurrent state and the controller the bot sent last
        frame (the policy's input), neutral at the start."""
        neutral = tree.map_structure(lambda x: np.asarray(x)[None].repeat(n, 0), _neutral_controller())
        encoded = self.policy.controller_head.controller_embedding.from_state(neutral)
        return State(self.policy.initial_state(n, self.device), tree.map_structure(self._tensor, encoded))

    def act(self, batch: umi.Batch, state: State) -> tuple[umi.Actions, State]:
        """One frame of N games.

        In:  batch.frames, Frame[N] seen from this bot's seat (players[:, 0] is this bot);
             state, row k the memory of batch game k: initial_state's for a new game,
             otherwise what the last act() returned for it.
        Out: controller rows [N, 13] (main_x, main_y, c_x, c_y, shoulder in [0, 1];
             A, B, X, Y, Z, L, R, D_UP pressed when > 0.5), and the new state, which
             TorchBot keeps for each game's next frame."""
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
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="DELAY=PATH; one per division the bot enters, e.g. 21=models/qk.pt")
    parser.add_argument("--scope", default="12")
    parser.add_argument("--name", default="smashbot")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--player-name", default="Master Player", help="the player name to condition on")
    args, rest = parser.parse_known_args(argv)
    checkpoints = {int(d): p for d, p in (c.split("=", 1) for c in args.checkpoint)}
    umi.serve(SmashBot(checkpoints, args.scope, args.name, args.temperature, args.player_name), argv=rest)


if __name__ == "__main__":
    main()
