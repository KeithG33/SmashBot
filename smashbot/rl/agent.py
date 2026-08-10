"""Batched policy agent for RL rollouts: one forward pass drives N envs.

Interface is ENCODED states (the env thread owns libmelee parsing), which
keeps this module Dolphin-free and unit-testable. Per-env delay queues and
recurrent-state resets are handled here; every step also emits the streams
the PPO Trajectory needs (prev-action inputs, sample-time logits).

batch_steps > 1 (amortizing multiple frames per forward via delay slack, as
slippi-ai does in RL) is a planned optimization; the env-batching here is
the dominant win (N thin forwards -> one wide one).
"""

from __future__ import annotations

import collections
import typing as tp

import numpy as np
import torch
import tree

from slippi_ai.types import Controller, StateAction

from smashbot.eval.agent import _neutral_controller
from smashbot.networks import _mask_state
from smashbot.policy import Policy


class FrameRecord(tp.NamedTuple):
    """Per-frame streams for trajectory assembly (all batched [N, ...])."""

    state: tp.Any  # encoded Game struct
    prev_action: tp.Any  # encoded controller struct — the policy's input
    logits: tp.Any  # controller struct — logits that sampled this frame
    name: torch.Tensor  # [N]


class BatchedPolicyAgent:
    def __init__(
        self,
        policy: Policy,
        num_envs: int,
        name_code: int = 0,
        temperature: float | None = None,
        device: str = "cpu",
        batch_steps: int = 1,
    ):
        self.policy = policy
        self.num_envs = num_envs
        self.device = device
        self.temperature = temperature
        self.delay = policy.delay
        self._embed_controller = policy.controller_head.controller_embedding
        self._name = torch.full((num_envs,), name_code, dtype=torch.int64, device=device)

        neutral = tree.map_structure(
            lambda x: np.asarray(x)[None].repeat(num_envs, axis=0),
            _neutral_controller(),
        )
        self._neutral_encoded = tree.map_structure(
            lambda x: torch.from_numpy(
                np.ascontiguousarray(x.astype(np.int64) if x.dtype.kind in "iu" else x)
            ).to(device),
            self._embed_controller.from_state(neutral),
        )

        self.hidden = policy.initial_state(num_envs, device)
        self._prev_action = tree.map_structure(lambda t: t.clone(), self._neutral_encoded)
        self._queues: list[collections.deque[Controller]] = [
            collections.deque([_neutral_controller()] * self.delay)
            for _ in range(num_envs)
        ]
        assert self.delay >= batch_steps, (
            "delay must cover batch_steps (queue runs batch_steps-1 short "
            "between flushes)"
        )
        self.batch_steps = batch_steps
        self._buf_states: list = []
        self._buf_resets: list[torch.Tensor] = []

    def reset_env(self, i: int) -> None:
        """Fresh game in env i: zero its recurrent state, queue, and prev action."""
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self._name.device)
        mask[i] = True
        self.hidden = _mask_state(
            mask, self.policy.initial_state(self.num_envs, self.device), self.hidden
        )
        self._queues[i] = collections.deque([_neutral_controller()] * self.delay)
        tree.map_structure(
            lambda dst, src: dst[i].copy_(src[i]),
            self._prev_action, self._neutral_encoded,
        )

    @torch.no_grad()
    def step(
        self, states: tp.Any, resets: torch.Tensor | None = None
    ) -> tuple[list[Controller], list[FrameRecord], tp.Any]:
        """states: encoded Game struct batched [N, ...]; resets: [N] bool.

        Buffers the frame; every `batch_steps` frames one sample_n call
        processes the buffer (amortizing launch overhead). Returns the
        controllers to execute NOW (popped from the delay queue — instant,
        never waits on inference), the flushed FrameRecords ([] between
        flushes), and the recurrent snapshot from just before the flush
        (None between flushes) for chunk-boundary bookkeeping.
        """
        if resets is None:
            resets = torch.zeros(self.num_envs, dtype=torch.bool, device=self._name.device)
        for i in torch.nonzero(resets).flatten().tolist():
            self._queues[i] = collections.deque(
                [_neutral_controller()] * self.delay
            )

        self._buf_states.append(states)
        self._buf_resets.append(resets)

        records: list[FrameRecord] = []
        hidden_before = None
        if len(self._buf_states) == self.batch_steps:
            hidden_before = self.hidden_snapshot()
            stack = lambda seq: tree.map_structure(
                lambda *xs: torch.stack(xs, dim=1), *seq
            )
            outs, hidden, used_prevs = self.policy.sample_n(
                states=stack(self._buf_states),
                names=self._name[:, None].expand(-1, self.batch_steps),
                prev_action=self._prev_action,
                neutral_action=self._neutral_encoded,
                initial_state=self.hidden,
                is_resetting=torch.stack(self._buf_resets, dim=1),
                temperature=self.temperature,
            )
            # clones: retained across flushes / fed back next flush, and
            # compiled (cudagraph) replay reuses output buffers
            self.hidden = tree.map_structure(
                lambda t: t.clone() if isinstance(t, torch.Tensor) else t, hidden
            )
            self._prev_action = tree.map_structure(
                lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(),
                outs[-1].controller_state,
            )
            for t, out in enumerate(outs):
                records.append(
                    FrameRecord(
                        state=self._buf_states[t],
                        prev_action=tree.map_structure(
                            lambda x: x.clone() if x.dtype == torch.bool
                            else x.long().clone(),
                            used_prevs[t],
                        ),
                        logits=tree.map_structure(lambda x: x.clone(), out.logits),
                        name=self._name.clone(),
                    )
                )
                encoded_np = tree.map_structure(
                    lambda x: x.cpu().numpy(), out.controller_state
                )
                decoded = self._embed_controller.decode(encoded_np)
                for i in range(self.num_envs):
                    self._queues[i].append(
                        tree.map_structure(lambda x: x[i], decoded)
                    )
            self._buf_states, self._buf_resets = [], []

        to_execute = [self._queues[i].popleft() for i in range(self.num_envs)]
        return to_execute, records, hidden_before

    def hidden_snapshot(self) -> tp.Any:
        """Detached copy of the recurrent state (for Trajectory.initial_state)."""
        return tree.map_structure(
            lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
            self.hidden,
        )
