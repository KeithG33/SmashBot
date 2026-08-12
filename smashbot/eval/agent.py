"""DelayedAgent: turns delay-trained predictions into properly-timed live inputs.

Each frame: encode the live gamestate (via slippi-ai's libmelee Parser — the
same encoding training data went through), sample an action, push it onto a
queue, and pop the action sampled `delay` frames ago to actually execute.
The queue is pre-filled with neutral inputs, so the first `delay` frames of a
game are no-ops — exactly how the model saw the world during training.
"""

import collections
import time

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
        """Consumes a live gamestate, returns the controller to execute NOW.

        Per-stage wall times land in self.stage_ms (running means) for the
        --profile flag in play.py."""
        t0 = time.perf_counter()
        game = self.parser.get_game(gamestate)
        game = tree.map_structure(lambda x: np.asarray(x)[None], game)
        t1 = time.perf_counter()
        state = self.policy.network.encode_game(game)
        t2 = time.perf_counter()
        state = tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(
                    x.astype(np.int64) if x.dtype.kind in "iu" else x
                )
            ).to(self.device),
            state,
        )
        t3 = time.perf_counter()

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

        t4 = time.perf_counter()
        encoded_np = tree.map_structure(
            lambda t: t[0].cpu().numpy(), sampled.controller_state
        )
        self._queue.append(self._embed_controller.decode(encoded_np))
        t5 = time.perf_counter()
        n = self._stage_count = getattr(self, "_stage_count", 0) + 1
        stages = dict(parse=t1 - t0, encode=t2 - t1, to_torch=t3 - t2,
                      sample=t4 - t3, decode=t5 - t4)
        acc = getattr(self, "_stage_acc", {k: 0.0 for k in stages})
        for k, v in stages.items():
            acc[k] += v
        self._stage_acc = acc
        self.stage_ms = {k: 1e3 * v / n for k, v in acc.items()}
        return self._queue.popleft()


class AsyncDelayedAgent(DelayedAgent):
    """DelayedAgent with inference moved off the frame-critical path.

    The Slippi build frame-syncs with the bot: it will not emulate frame N+1
    until frame N's inputs arrive, so with the sync agent the frame period is
    emulation + inference IN SERIES (measured: ~8ms + ~14ms = 45fps live).

    Here step() returns a controller immediately (queue pop) and computes the
    new sample on a background thread during the NEXT frame's emulation. The
    output sequence is IDENTICAL to the sync agent's (join-before-pop: the
    sample from frame N is enqueued by frame N+1, and it isn't executed until
    frame N+delay, delay >= 1) — the bot's play is unchanged, Dolphin just
    never waits. Requires effective delay >= 1 (we run 18).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.delay >= 1, "async agent needs >= 1 frame of delay"
        self._worker = None
        self._wait_acc = 0.0
        self._wait_n = 0

    def reset(self) -> None:
        # finish any in-flight compute before wiping state it writes to
        w = getattr(self, "_worker", None)
        if w is not None:
            w.join()
            self._worker = None
        super().reset()

    def _compute(self, game) -> None:
        """encode -> sample -> enqueue; runs on the background thread.
        Sequenced by join-before-launch, so self.hidden/_prev_action/_queue
        are never touched concurrently."""
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
        with torch.no_grad():
            sampled, self.hidden = self.policy.sample(
                StateAction(
                    state=state, action=self._prev_action, name=self._name
                ),
                self.hidden,
                temperature=self.temperature,
            )
        self._prev_action = tree.map_structure(
            lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(),
            sampled.controller_state,
        )
        encoded_np = tree.map_structure(
            lambda t: t[0].cpu().numpy(), sampled.controller_state
        )
        self._queue.append(self._embed_controller.decode(encoded_np))

    def step(self, gamestate) -> Controller:
        import threading

        t0 = time.perf_counter()
        if self._worker is not None:
            self._worker.join()  # a_{N-1}: normally already done
        self._wait_acc += time.perf_counter() - t0
        self._wait_n += 1
        self.stage_ms = {"join_wait": 1e3 * self._wait_acc / self._wait_n}
        # parse synchronously: the Parser is stateful and frame-ordered
        game = self.parser.get_game(gamestate)
        out = self._queue.popleft()
        self._worker = threading.Thread(target=self._compute, args=(game,))
        self._worker.start()
        return out
