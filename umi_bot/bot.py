"""SmashBot as a UMI bot (Computer Melee Championship).

One process serves every game of a session: per-game recurrent state and
previous action live in [num_slots, ...] tensors, gathered by slot for each
batch. The arena applies the input delay, so unlike eval/agent.py's
DelayedAgent there is no action queue: each observation's sample is returned
as the reply.

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

Nest = tp.Any   # a nest (dicts, tuples, named tuples) of tensors


def batch_axes(policy: Policy, device: str) -> list[int]:
    """Each recurrent-state leaf's batch axis: the one that changes with the
    batch size (batch-first leaves, and [layers, B, H] torch RNN states)."""
    a, b = (tree.flatten(policy.initial_state(n, device)) for n in (3, 5))
    axes = []
    for x, y in zip(a, b):
        diff = [d for d, (p, q) in enumerate(zip(x.shape, y.shape)) if p != q]
        assert len(diff) == 1, f"no unique batch axis for a state leaf of shape {tuple(x.shape)}"
        axes.append(diff[0])
    return axes


def _torch(x: np.ndarray, device: str) -> torch.Tensor:
    x = np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x)
    return torch.from_numpy(x).to(device)


class SmashBot(TorchBot):
    name, author, framework, max_games = "smashbot", "KeithG33", "torch", 256

    def __init__(self, checkpoints: dict[int, str], scope: str = "12", name: str = "smashbot",
                 temperature: float | None = None, player_name: str = "", device: str | None = None):
        super().__init__(device)
        self.checkpoints, self.scope, self.name, self.temperature = checkpoints, scope, name, temperature
        self.player_name = player_name
        self.delays = tuple(sorted(checkpoints))

    def setup(self, session: umi.Session) -> None:
        """Once per session, before `ready`: the checkpoint for the session's
        delay, and one row of state per game slot."""
        path = self.checkpoints[session.delay]
        policy, name_code, _ = load_player(path, self.device, "")
        self.policy: Policy = policy
        if self.player_name:   # condition on a player name the checkpoint was trained with
            from smashbot import saving
            name_map = torch.load(path, map_location="cpu", weights_only=False)["state"].get("name_map", {})
            name_code = saving.resolve_name_code(name_map, self.player_name, verbose=False)
        assert self.policy.delay == session.delay, f"{path} was trained at delay {self.policy.delay}"
        self.embed_controller = self.policy.controller_head.controller_embedding
        self.axes: list[int] = batch_axes(self.policy, self.device)
        self.hidden: Nest = self.policy.initial_state(session.num_slots, self.device)
        neutral = tree.map_structure(lambda x: np.asarray(x)[None].repeat(session.num_slots, 0), _neutral_controller())
        self.neutral_action: Nest = tree.map_structure(lambda x: _torch(x, self.device),
                                                       self.embed_controller.from_state(neutral))
        self.prev_action: Nest = tree.map_structure(torch.clone, self.neutral_action)
        self.names: torch.Tensor = torch.full((session.num_slots,), name_code, dtype=torch.int64, device=self.device)

    def _take(self, state: Nest, rows: torch.Tensor) -> Nest:
        return tree.unflatten_as(state, [x.index_select(ax, rows) for x, ax in zip(tree.flatten(state), self.axes)])

    def _put(self, state: Nest, rows: torch.Tensor, values: Nest) -> None:
        for x, ax, v in zip(tree.flatten(state), self.axes, tree.flatten(values)):
            x.index_copy_(ax, rows, v.to(x.dtype))

    def act(self, batch: umi.Batch) -> umi.Actions:
        """One frame of N games: their observations in, [N, 13] controller rows out."""
        rows = torch.as_tensor(batch.slots, device=self.device)
        first = torch.as_tensor(batch.first, device=self.device)
        game = self.policy.network.encode_game(to_slippi_ai(batch.frames))
        state = tree.map_structure(lambda x: _torch(x, self.device), game)
        prev = tree.map_structure(lambda p, n: torch.where(first.view(-1, *[1] * (p.dim() - 1)), n[rows], p[rows]),
                                  self.prev_action, self.neutral_action)
        sampled, hidden = self.policy.sample(
            StateAction(state=state, action=prev, name=self.names[rows]), self._take(self.hidden, rows),
            is_resetting=first, temperature=self.temperature)
        controller = sampled.controller_state
        self._put(self.hidden, rows, hidden)
        for p, c in zip(tree.flatten(self.prev_action), tree.flatten(controller)):
            p[rows] = c.to(p.dtype)
        decoded = self.embed_controller.decode(tree.map_structure(lambda t: t.cpu().numpy(), controller))
        return umi.Actions(ctrl=encode.controller_rows(decoded))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="DELAY=PATH; one per division the bot enters, e.g. 21=models/qk.pt")
    parser.add_argument("--scope", default="12")
    parser.add_argument("--name", default="smashbot")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--player-name", default="", help="name to condition on (default: Master Player)")
    args, rest = parser.parse_known_args(argv)
    checkpoints = {int(d): p for d, p in (c.split("=", 1) for c in args.checkpoint)}
    umi.serve(SmashBot(checkpoints, args.scope, args.name, args.temperature, args.player_name), argv=rest)


if __name__ == "__main__":
    main()
