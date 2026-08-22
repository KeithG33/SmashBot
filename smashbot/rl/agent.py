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
    """All same-config league slots stepped in ONE Python pass per frame:
    S slot policies x N rows. Per-row state (recurrent state, prev action,
    delay queues) lives here for every slot; each frame runs one forward per
    slot (a stacked forward can replace that loop), then a single batched
    controller transfer/decode/queue update for all S*N rows. Seats point at
    slot refs (slot_ref) for weight loads and harvest metadata."""

    def __init__(self, policies, num_envs: int, name_code: int, device, temperature=None):
        assert policies, "LeagueAgent needs at least one slot policy"
        self.policies = list(policies)
        self.S, self.N = len(self.policies), num_envs
        self.device = device
        self.temperature = temperature
        p0 = self.policies[0]
        self.delay = p0.delay
        self._embed_controller = p0.controller_head.controller_embedding
        self._name = torch.full((self.S, num_envs), name_code, dtype=torch.int64, device=device)
        # same construction as BatchedPolicyAgent: the neutral controller
        # tiled to the row batch, then to every slot -> [S, N, ...]
        neutral = tree.map_structure(
            lambda x: np.asarray(x)[None].repeat(num_envs, axis=0), _neutral_controller()
        )
        self._neutral = tree.map_structure(
            lambda x: torch.from_numpy(np.ascontiguousarray(
                x.astype(np.int64) if x.dtype.kind in "iu" else x
            )).to(device)[None].expand(self.S, *x.shape).clone(),
            self._embed_controller.from_state(neutral),
        )
        self._prev = tree.map_structure(lambda t: t.clone(), self._neutral)  # [S, N, ...]
        self.hidden = [p.initial_state(num_envs, device) for p in self.policies]
        from smashbot import encode

        self._neutral_row = encode.controller_rows(
            tree.map_structure(lambda x: np.asarray(x)[None], _neutral_controller())
        )[0]
        self._queues = [
            collections.deque([self._neutral_row] * self.delay)
            for _ in range(self.S * num_envs)
        ]
        # CUDA production path: ONE eager-vmap forward over stacked per-slot
        # parameters, captured into a manual CUDA graph (inductor's cudagraph
        # trees choke on 12 compiled callables in a tight loop). The slot
        # MODULES stay the source of truth for weights; slot_weights_changed
        # refreshes the stack slice in place, which captured replays see
        # (graphs hold pointers). CPU / tests use the per-slot loop below.
        self._use_capture = (
            torch.device(device).type == "cuda" and len(self.policies) > 1
        )
        self._graph = None
        if self._use_capture:
            from torch.func import stack_module_state

            self._stacked_params, self._stacked_buffers = stack_module_state(
                [p for p in self.policies]
            )

    def slot_weights_changed(self, k: int) -> None:
        """Refresh stack slice k from the slot's module (weights are loaded
        into modules by apply_assignments/parking; the captured graph reads
        the stack in place)."""
        if not self._use_capture:
            return
        sd = self.policies[k].state_dict()
        with torch.no_grad():
            for name, t in self._stacked_params.items():
                t[k].copy_(sd[name])
            for name, t in self._stacked_buffers.items():
                if name in sd:
                    t[k].copy_(sd[name])

    def slot_ref(self, k: int) -> "_SlotRef":
        return _SlotRef(self, k)

    @torch.no_grad()
    def step(self, views, resets, reset_indices):
        """views: per-slot encoded structs [N, ...]; resets: [S, N] bool on
        device; reset_indices: iterable of (slot, row). Returns per-slot
        controller rows (list of [N, 13] numpy) and per-slot FrameRecords."""
        for s, i in reset_indices:
            self._queues[s * self.N + i] = collections.deque(
                [self._neutral_row] * self.delay
            )
        prev = tree.map_structure(
            lambda pv, n: torch.where(
                resets.view(self.S, self.N, *([1] * (pv.dim() - 2))), n, pv
            ),
            self._prev, self._neutral,
        )
        if self._use_capture:
            ctrl, logits = self._captured_forward(views, prev, resets)
        else:
            ctrl, logits = self._loop_forward(views, prev, resets)
        self._prev = tree.map_structure(
            lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(), ctrl
        )
        prev_rec = self._prev_record(prev)
        records = [
            FrameRecord(
                state=views[k],
                prev_action=tree.map_structure(lambda t: t[k], prev_rec),
                logits=tree.map_structure(lambda t: t[k], logits),
                name=self._name[k],
            )
            for k in range(self.S)
        ]
        # ONE host transfer + decode for all S*N rows
        encoded_np = tree.map_structure(
            lambda x: x.reshape(self.S * self.N, *x.shape[2:]).cpu().numpy(), ctrl
        )
        from smashbot import encode

        rows = encode.controller_rows(self._embed_controller.decode(encoded_np))
        for q, row in zip(self._queues, rows):
            q.append(row)
        execute = np.stack([q.popleft() for q in self._queues]).reshape(self.S, self.N, -1)
        return [execute[k] for k in range(self.S)], records

    def _captured_forward(self, views, prev, resets):
        """One vmap'd forward over the stacked slot parameters, executed as a
        manual CUDA-graph replay: copy this frame's inputs into the static
        buffers, replay, return clones of the static outputs. Captured once
        at first use (shapes never change); in-place stack-slice weight
        updates are visible to replays."""
        stk_views = tree.map_structure(lambda *xs: torch.stack(xs), *views)
        if self._graph is None:
            self._capture(stk_views, prev, resets)
        tree.map_structure(lambda d, s: d.copy_(s), self._in_views, stk_views)
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

    def _capture(self, stk_views, prev, resets):
        from torch.func import functional_call, vmap

        base = self.policies[0]
        name = self._name

        def fmodel(p, b, st, ac, hid, rst):
            out, hid2 = functional_call(base, (p, b), (
                StateAction(state=st, action=ac, name=name[0]), hid, rst, self.temperature,
            ))
            return out.controller_state, out.logits, hid2

        self._vm = vmap(fmodel, in_dims=(0, 0, 0, 0, 0, 0), randomness="different")
        self._in_views = tree.map_structure(lambda t: t.clone(), stk_views)
        self._in_prev = tree.map_structure(lambda t: t.clone(), prev)
        self._in_resets = resets.clone()
        h0 = [p.initial_state(self.N, self.device) for p in self.policies]
        self._in_hidden = tree.map_structure(
            lambda *xs: torch.stack(xs) if isinstance(xs[0], torch.Tensor) else xs[0], *h0
        )
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
        # the just-captured pass ran with the warm-up inputs; hidden restarts
        # from the stacked initial state on the first real replay
        tree.map_structure(
            lambda d, s_: d.copy_(s_) if isinstance(d, torch.Tensor) else None,
            self._out_hidden, self._in_hidden,
        )

    def _loop_forward(self, views, prev, resets):
        outs, hiddens = [], []
        for k, policy in enumerate(self.policies):
            # contiguous per-slot inputs: a strided slice of the [S, N, ...]
            # stack would miss the compiled sample's guards and re-record a
            # CUDA graph per slot (live-caught as an OOM)
            out, hid = policy.sample(
                StateAction(
                    state=views[k],
                    action=tree.map_structure(lambda t: t[k].contiguous(), prev),
                    name=self._name[k].contiguous(),
                ),
                self.hidden[k], is_resetting=resets[k].contiguous(),
                temperature=self.temperature,
            )
            outs.append(out)
            hiddens.append(hid)
        self.hidden = [
            tree.map_structure(lambda t: t.clone() if isinstance(t, torch.Tensor) else t, h)
            for h in hiddens
        ]
        ctrl = tree.map_structure(lambda *xs: torch.stack(xs), *[o.controller_state for o in outs])
        logits = tree.map_structure(lambda *xs: torch.stack(xs), *[o.logits for o in outs])
        return ctrl, logits

    @staticmethod
    def _prev_record(prev):
        return tree.map_structure(
            lambda x: x.clone() if x.dtype == torch.bool else x.long().clone(), prev
        )


class _SlotRef:
    """A seat's handle on one slot of a LeagueAgent: exposes what the worker
    needs for weight loads (policy), harvest grouping (delay, embedding) and
    row count; stepping happens in LeagueAgent.step."""

    def __init__(self, league: LeagueAgent, k: int):
        self.league, self.k = league, k
        self.policy = league.policies[k]
        self.num_envs = league.N
        self.delay = league.delay
        self._embed_controller = league._embed_controller
        self.flat_controllers = True

    def set_flat_controllers(self, flat: bool = True) -> None:
        assert flat, "league slots always speak flat controller rows"
