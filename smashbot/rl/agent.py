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


def _make_builder(struct):
    """Compile a structure (nested NamedTuples/dicts/lists of leaves) into
    a function leaves_iter -> struct, walking the structure ONCE so per-row
    rebuilds are plain constructor calls (dm-tree's unflatten_as re-walks
    with isinstance checks every time)."""
    if isinstance(struct, tuple) and hasattr(struct, "_fields"):
        kids = [_make_builder(v) for v in struct]
        ctor = type(struct)
        return lambda it: ctor(*[k(it) for k in kids])
    if isinstance(struct, dict):
        keys = list(struct.keys())
        kids = [_make_builder(struct[k]) for k in keys]
        return lambda it: {k: b(it) for k, b in zip(keys, kids)}
    if isinstance(struct, (list, tuple)):
        kids = [_make_builder(v) for v in struct]
        ctor = type(struct)
        return lambda it: ctor(k(it) for k in kids)
    return next


_BUILDERS: dict = {}


def _split_rows(struct, n: int) -> list:
    """Per-row structs of a batched struct: one flatten plus n cheap
    rebuilds through a cached compiled constructor."""
    leaves = tree.flatten(struct)
    key = id(type(struct)), len(leaves)
    builder = _BUILDERS.get(key)
    if builder is None:
        builder = _BUILDERS[key] = _make_builder(struct)
    return [builder(iter([leaf[i] for leaf in leaves])) for i in range(n)]


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
        # flat_controllers=True (the rollout worker): queues hold 13-float
        # rows and step() returns rows (env rebuilds the struct) — no
        # per-env struct construction on the worker
        self.flat_controllers = False
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
        self._queues[i] = collections.deque([self._neutral()] * self.delay)
        tree.map_structure(
            lambda dst, src: dst[i].copy_(src[i]),
            self._prev_action, self._neutral_encoded,
        )

    @torch.no_grad()
    def set_flat_controllers(self, flat: bool = True) -> None:
        """Switch the controller output format (rows vs structs); the
        pre-filled delay queues are rebuilt in the new format."""
        self.flat_controllers = flat
        self._queues = [
            collections.deque([self._neutral()] * self.delay)
            for _ in range(self.num_envs)
        ]

    def _neutral(self):
        if self.flat_controllers:
            from smashbot import encode

            return encode.controller_rows(
                tree.map_structure(lambda x: np.asarray(x)[None], _neutral_controller())
            )[0]
        return _neutral_controller()

    def _enqueue(self, decoded) -> None:
        if self.flat_controllers:
            from smashbot import encode

            rows = encode.controller_rows(decoded)
            for i in range(self.num_envs):
                self._queues[i].append(rows[i])
        else:
            for i, c in enumerate(_split_rows(decoded, self.num_envs)):
                self._queues[i].append(c)

    @torch.no_grad()  # rollout stepping is inference: without this the
    # compiled sample runs its TRAINING graph and every frame's activations
    # are saved for a backward that never comes (live-caught: 21GB OOM at
    # 200 rows, and cudagraph trees' "pending, uninvoked backwards" stall)
    def step(
        self, states: tp.Any, resets: torch.Tensor | None = None,
        reset_indices: tp.Sequence[int] | None = None,
        want_snapshot: bool = True,
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
        if reset_indices is None:  # caller without a CPU copy: one sync
            reset_indices = torch.nonzero(resets).flatten().tolist()
        for i in reset_indices:
            self._queues[i] = collections.deque([self._neutral()] * self.delay)

        self._buf_states.append(states)
        self._buf_resets.append(resets)

        records: list[FrameRecord] = []
        hidden_before = None
        if self.batch_steps == 1:
            # fast path: skip the sample_n wrapper (measured ~20% faster
            # under reduce-overhead compile at S=1)
            hidden_before = self.hidden_snapshot() if want_snapshot else None
            reset_t = resets
            prev = tree.map_structure(
                lambda pv, n: torch.where(
                    reset_t.view(-1, *([1] * (pv.dim() - 1))), n, pv
                ),
                self._prev_action, self._neutral_encoded,
            )
            out, hidden = self.policy.sample(
                StateAction(state=states, action=prev, name=self._name),
                self.hidden, is_resetting=reset_t, temperature=self.temperature,
            )
            self.hidden = tree.map_structure(
                lambda t: t.clone() if isinstance(t, torch.Tensor) else t, hidden
            )
            self._prev_action = tree.map_structure(
                lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(),
                out.controller_state,
            )
            records.append(FrameRecord(
                state=states,
                prev_action=tree.map_structure(
                    lambda x: x.clone() if x.dtype == torch.bool else x.long().clone(),
                    prev,
                ),
                logits=tree.map_structure(lambda x: x.clone(), out.logits),
                name=self._name.clone(),
            ))
            encoded_np = tree.map_structure(
                lambda x: x.cpu().numpy(), out.controller_state
            )
            decoded = self._embed_controller.decode(encoded_np)
            self._enqueue(decoded)
            self._buf_states, self._buf_resets = [], []
            to_execute = [self._queues[i].popleft() for i in range(self.num_envs)]
            return to_execute, records, hidden_before

        if len(self._buf_states) == self.batch_steps:
            hidden_before = self.hidden_snapshot() if want_snapshot else None
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
                self._enqueue(decoded)
            self._buf_states, self._buf_resets = [], []

        to_execute = [self._queues[i].popleft() for i in range(self.num_envs)]
        return to_execute, records, hidden_before

    def hidden_snapshot(self) -> tp.Any:
        """Detached copy of the recurrent state (for Trajectory.initial_state)."""
        return tree.map_structure(
            lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
            self.hidden,
        )


class LeagueAgent:
    """The league GRID: S weight slices x N cells, stepped as ONE forward.

    A slice holds one league member's weights (a stacked copy, loaded in
    place by load_slice); a cell is a seat with its own recurrent state,
    prev action and delay queue. Who sits where is the worker's business
    (rollouts._Grid routes envs to cells per match); this class only knows
    the [S, N] batch. CUDA: the vmap forward over the stacked parameters is
    captured once into a manual CUDA graph and replayed per frame. CPU: the
    same vmap forward runs eagerly — one code path, no per-slice loop.
    """

    def __init__(
        self, template: Policy, slices: int, cells: int, name_code: int,
        device, temperature=None, capture: bool | None = None,
    ):
        import copy

        self.S, self.N = slices, cells
        self.device = torch.device(device)
        self.temperature = temperature
        self.delay = template.delay
        self._embed_controller = template.controller_head.controller_embedding
        # functional_call's skeleton: a THROWAWAY copy (never read back).
        # The policy has tied parameters (one item MLP shared across item
        # slots and between the game/state-action embeddings) and
        # functional_call under vmap leaves a tied template holding an
        # escaped BatchedTensor — live-caught at the first park.
        self._template = copy.deepcopy(template).to("cpu")
        self._template.__dict__.pop("sample", None)  # any compiled wrapper
        self._template.requires_grad_(False).eval()
        # stacked weights [S, ...]: slice s serves cells (s, 0..N-1)
        with torch.no_grad():
            params = dict(self._template.named_parameters())
            buffers = dict(self._template.named_buffers())
            stack = lambda t: t.detach().to(self.device).unsqueeze(0).repeat(
                self.S, *([1] * t.dim())
            ).clone()
            self._stacked_params = {k: stack(v) for k, v in params.items()}
            self._stacked_buffers = {k: stack(v) for k, v in buffers.items()}
        self._name = torch.full(
            (self.S, self.N), name_code, dtype=torch.int64, device=self.device
        )
        neutral = tree.map_structure(
            lambda x: np.asarray(x)[None].repeat(self.N, axis=0),
            _neutral_controller(),
        )
        self._neutral = tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x
            )).to(self.device)[None].expand(self.S, *x.shape).clone(),
            self._embed_controller.from_state(neutral),
        )
        self._prev = tree.map_structure(lambda t: t.clone(), self._neutral)
        from smashbot import encode

        self._neutral_row = encode.controller_rows(
            tree.map_structure(lambda x: np.asarray(x)[None], _neutral_controller())
        )[0]
        self._queues = [
            collections.deque([self._neutral_row] * self.delay)
            for _ in range(self.S * self.N)
        ]
        if capture is None:
            capture = self.device.type == "cuda"
        assert not capture or self.device.type == "cuda", (
            "LeagueAgent capture=True needs a CUDA device"
        )
        self._use_capture = capture
        self._graph = None
        self._vm = self._make_vmap()
        # eager-path recurrent state [S, N, ...] (the captured path keeps
        # it in static buffers)
        self._hidden = self._initial_hidden()

    # ---------------------------------------------------------- weights

    @torch.no_grad()
    def load_slice(self, s: int, state_dict: dict) -> None:
        """Copy a member's weights (any device) into slice s, in place —
        captured replays read the stack by pointer, so they see it."""
        for name, t in self._stacked_params.items():
            t[s].copy_(state_dict[name])
        for name, t in self._stacked_buffers.items():
            if name in state_dict:
                t[s].copy_(state_dict[name])

    # ---------------------------------------------------------- stepping

    def reset_cell(self, s: int, n: int) -> None:
        """Fresh game in cell (s, n): neutral prev action + delay queue (the
        recurrent state is zeroed by the forward's reset mask)."""
        self._queues[s * self.N + n] = collections.deque(
            [self._neutral_row] * self.delay
        )
        tree.map_structure(
            lambda dst, src: dst[s, n].copy_(src[s, n]), self._prev, self._neutral
        )

    @torch.no_grad()
    def step(self, views, resets: torch.Tensor):
        """views: encoded Game struct batched [S, N, ...] on device; resets:
        [S, N] bool on device (True on a cell's first frame of a game —
        zeroes its recurrent state and substitutes the neutral prev action).
        Returns controller rows to execute NOW ([S*N, 13] numpy, popped from
        the delay queues) and ONE FrameRecord over all S*N cells."""
        prev = tree.map_structure(
            lambda pv, n: torch.where(
                resets.view(self.S, self.N, *([1] * (pv.dim() - 2))), n, pv
            ),
            self._prev, self._neutral,
        )
        if self._use_capture:
            ctrl, logits = self._captured_forward(views, prev, resets)
        else:
            ctrl, logits, self._hidden = self._vm(
                self._stacked_params, self._stacked_buffers, views, prev,
                self._hidden, resets,
            )
        self._prev = tree.map_structure(
            lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(), ctrl
        )
        flat = lambda t: t.reshape(self.S * self.N, *t.shape[2:])
        record = FrameRecord(
            state=tree.map_structure(flat, views),
            prev_action=tree.map_structure(
                lambda x: flat(x.clone() if x.dtype == torch.bool else x.long().clone()),
                prev,
            ),
            logits=tree.map_structure(flat, logits),
            name=flat(self._name).clone(),
        )
        from smashbot import encode

        encoded_np = tree.map_structure(lambda x: flat(x).cpu().numpy(), ctrl)
        rows = encode.controller_rows(self._embed_controller.decode(encoded_np))
        for q, row in zip(self._queues, rows):
            q.append(row)
        return np.stack([q.popleft() for q in self._queues]), record

    # ---------------------------------------------------------- forward

    def _initial_hidden(self):
        h0 = [self._template.initial_state(self.N, self.device) for _ in range(self.S)]
        return tree.map_structure(
            lambda *xs: torch.stack(xs) if isinstance(xs[0], torch.Tensor) else xs[0],
            *h0,
        )

    def _make_vmap(self):
        from torch.func import functional_call, vmap

        base, name, temperature = self._template, self._name, self.temperature

        def fmodel(p, b, st, ac, hid, rst):
            out, hid2 = functional_call(base, (p, b), (
                StateAction(state=st, action=ac, name=name[0]), hid, rst, temperature,
            ))
            return out.controller_state, out.logits, hid2

        return vmap(fmodel, in_dims=(0, 0, 0, 0, 0, 0), randomness="different")

    def _captured_forward(self, views, prev, resets):
        """Copy this frame's inputs into the static buffers, replay, return
        clones of the static outputs. Captured once at first use (shapes
        never change); in-place slice loads are visible to replays."""
        if self._graph is None:
            self._capture(views, prev, resets)
        tree.map_structure(lambda d, s: d.copy_(s), self._in_views, views)
        tree.map_structure(lambda d, s: d.copy_(s), self._in_prev, prev)
        self._in_resets.copy_(resets)
        # recurrent state: static in <- last replay's static out
        tree.map_structure(
            lambda d, s: d.copy_(s) if isinstance(d, torch.Tensor) else None,
            self._in_hidden, self._out_hidden,
        )
        self._graph.replay()
        ctrl = tree.map_structure(lambda t: t.clone(), self._out_ctrl)
        logits = tree.map_structure(lambda t: t.clone(), self._out_logits)
        return ctrl, logits

    def _capture(self, views, prev, resets):
        self._in_views = tree.map_structure(lambda t: t.clone(), views)
        self._in_prev = tree.map_structure(lambda t: t.clone(), prev)
        self._in_resets = resets.clone()
        self._in_hidden = self._initial_hidden()
        args = (self._stacked_params, self._stacked_buffers, self._in_views,
                self._in_prev, self._in_hidden, self._in_resets)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._vm(*args)
        torch.cuda.current_stream().wait_stream(s)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._out_ctrl, self._out_logits, self._out_hidden = self._vm(*args)
        # the just-captured pass ran with warm-up inputs; hidden restarts
        # from the initial state on the first real replay
        tree.map_structure(
            lambda d, s_: d.copy_(s_) if isinstance(d, torch.Tensor) else None,
            self._out_hidden, self._in_hidden,
        )
