"""DelayedAgent: turns delay-trained predictions into properly-timed live inputs.

Each frame: encode the live gamestate (via slippi-ai's libmelee Parser — the
same encoding training data went through), sample an action, push it onto a
queue, and pop the action sampled `delay` frames ago to actually execute.
The queue is pre-filled with neutral inputs, so the first `delay` frames of a
game are no-ops — exactly how the model saw the world during training.
"""

import collections

import numpy as np
import torch
import tree

from slippi_ai.types import Buttons, Controller, StateAction, Stick
from slippi_db.parse_libmelee import Parser

from smashbot.policy import Policy


def _neutral_controller() -> Controller:
    return Controller(
        main_stick=Stick(x=np.float32(0.5), y=np.float32(0.5)),
        c_stick=Stick(x=np.float32(0.5), y=np.float32(0.5)),
        shoulder=np.float32(0.0),
        buttons=Buttons(*(np.bool_(False) for _ in Buttons._fields)),
    )


class DelayedAgent:
    def __init__(
        self,
        policy: Policy,
        own_port: int,
        opponent_port: int,
        name_code: int = 0,
        console_delay: int = 0,
        temperature: float | None = None,
        device: str = "cuda",
    ):
        self.policy = policy
        self.device = device
        self.temperature = temperature
        self.delay = policy.delay - console_delay
        assert self.delay >= 0, "console delay exceeds policy delay"
        self._ports = (own_port, opponent_port)
        self._name = torch.tensor([name_code], dtype=torch.int32, device=device)
        self._embed_controller = policy.controller_head.controller_embedding
        self.reset()

    def reset(self) -> None:
        self.parser = Parser(ports=list(self._ports))
        self.hidden = self.policy.initial_state(1, self.device)
        neutral = tree.map_structure(
            lambda x: np.asarray(x)[None], _neutral_controller()
        )
        encoded_neutral = self._embed_controller.from_state(neutral)
        self._prev_action = tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(
                    x.astype(np.int64) if x.dtype.kind in "iu" else x
                )
            ).to(self.device),
            encoded_neutral,
        )
        self._queue = collections.deque(
            [_neutral_controller()] * self.delay
        )

    @torch.no_grad()
    def step(self, gamestate) -> Controller:
        """Consumes a live gamestate, returns the controller to execute NOW."""
        game = self.parser.get_game(gamestate)
        game = tree.map_structure(lambda x: np.asarray(x)[None], game)
        state = self.policy.network.encode_game(game)
        state = tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(
                    x.astype(np.int64) if x.dtype.kind in "iu" else x
                )
            ).to(self.device),
            state,
        )

        sampled, self.hidden = self.policy.sample(
            StateAction(state=state, action=self._prev_action, name=self._name),
            self.hidden,
            temperature=self.temperature,
        )
        # clone: retained across steps, and cudagraph replay reuses output buffers.
        # int64 keeps dtypes uniform for dynamo guards (bools stay bool).
        self._prev_action = tree.map_structure(
            lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(),
            sampled.controller_state,
        )

        encoded_np = tree.map_structure(
            lambda t: t[0].cpu().numpy(), sampled.controller_state
        )
        self._queue.append(self._embed_controller.decode(encoded_np))
        return self._queue.popleft()
