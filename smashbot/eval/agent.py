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
    """DelayedAgent with inference fully off the frame-critical path.

    The Slippi build frame-syncs with the bot, so with the sync agent every
    frame costs emulation + inference IN SERIES (~45fps live). Here step()
    hands the frame to a persistent compute thread and immediately pops the
    action queue — it never waits on inference. The compute thread processes
    frames strictly in order at its own pace; occasional slow samples (page
    faults, allocator spikes) simply lag a frame or two behind and catch up,
    absorbed by the delay queue's slack (delay=18 frames ~ 300ms of cushion).
    The emitted action sequence is IDENTICAL to the sync agent's, frame for
    frame (equivalence + spike tests assert it); only the arrival timing of
    queue entries differs, which the pre-filled delay queue makes invisible.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.delay >= 1, "async agent needs >= 1 frame of delay"

    def reset(self) -> None:
        import queue as queue_lib
        import threading

        # stop any prior worker before wiping the state it writes
        if getattr(self, "_in_q", None) is not None:
            self._in_q.put(None)
            self._thread.join()
        super().reset()
        # thread-safe handoff queues; _queue (deque from super) holds the
        # delay-queue prefill and receives computed actions in order
        self._in_q = queue_lib.Queue()
        self._out_ready = threading.Semaphore(len(self._queue))
        self._wait_acc = 0.0
        self._wait_n = 0
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        while True:
            game = self._in_q.get()
            if game is None:
                return
            self._compute(game)
            self._out_ready.release()
            self._in_q.task_done()

    def _compute(self, game) -> None:
        """encode -> sample -> enqueue; runs only on the worker thread, which
        processes frames strictly in order (single consumer)."""
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

    def drain(self) -> None:
        """Block until every submitted frame has been computed (tests)."""
        self._in_q.join()

    def step(self, gamestate) -> Controller:
        # parse on the caller: the Parser is stateful and frame-ordered
        game = self.parser.get_game(gamestate)
        self._in_q.put(game)
        # blocks ONLY if compute has lagged a full `delay` frames (~300ms)
        t0 = time.perf_counter()
        self._out_ready.acquire()
        self._wait_acc += time.perf_counter() - t0
        self._wait_n += 1
        self.stage_ms = {"queue_wait": 1e3 * self._wait_acc / self._wait_n}
        return self._queue.popleft()
