"""Batched policy agent for RL rollouts: one forward pass drives N envs.

Interface is ENCODED states (the env thread owns libmelee parsing), which
keeps this module Dolphin-free and unit-testable. Per-env delay queues and
recurrent-state resets are handled here; every step also emits the streams
the PPO Trajectory needs (prev-action inputs, sample-time logits).

"""

from __future__ import annotations

import collections
import typing as tp

import numpy as np
import torch
import tree

from slippi_ai.types import Controller, StateAction

from smashbot.eval.agent import _neutral_controller
from smashbot.networks import current_names, use_manual_recurrent_step
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


def _controller_to_host(ctrl):
    """One device-to-host copy for the whole sampled controller (13 uint8/bool
    leaves -> one [.., 13] tensor); each leaf comes back in its own dtype."""
    leaves = tree.flatten(ctrl)
    pack = (torch.uint8 if all(t.dtype in (torch.bool, torch.uint8) for t in leaves)
            else torch.int64)
    packed = torch.stack([t.to(pack) for t in leaves], dim=-1).cpu().numpy()
    return tree.unflatten_as(ctrl, [
        packed[..., k].astype(torch.empty(0, dtype=t.dtype).numpy().dtype)
        for k, t in enumerate(leaves)
    ])


class BatchedPolicyAgent:
    def __init__(
        self,
        policy: Policy,
        num_envs: int,
        name_code: int = 0,
        temperature: float | None = None,
        device: str = "cpu",
        precision: str = "fp32",
        state_dtype: torch.dtype | None = None,
        capture: bool = False,
        ring: bool | None = None,
    ):
        self.policy = policy
        self.num_envs = num_envs
        self.device = device
        self.temperature = temperature
        # fp16 carried state — OPPONENT seats only (nothing downstream of
        # them enters a loss; the student's logits feed the PPO ratio). The
        # fp16-autocast forward computes state in fp16 anyway; seeding the
        # initial zeros fp16 keeps the KV cat in fp16 (fp32 zeros would
        # promote it back). Bit-identical vs fp32 storage:
        # scripts/check_fp16_state.py.
        assert state_dtype is None or precision in ("fp16", "bf16"), (
            "state_dtype override requires a half-precision autocast forward"
        )
        self.state_dtype = state_dtype
        # "fp16": the network runs under fp16 autocast (sampling math stays
        # fp32 — embed.py casts logits up); logits are stored fp16. Gated by
        # the precision probe (docs/precision): the learner's ratio
        # invariant must hold on batches captured this way.
        assert precision in ("fp32", "fp16", "bf16"), precision
        self.precision = precision
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

        self.hidden = self._cast_state(policy.initial_state(num_envs, device))
        self._prev_action = tree.map_structure(lambda t: t.clone(), self._neutral_encoded)
        # flat_controllers=True (the rollout worker): queues hold 13-float
        # rows and step() returns rows (env rebuilds the struct) — no
        # per-env struct construction on the worker
        self.flat_controllers = False
        self._queues: list[collections.deque[Controller]] = [
            collections.deque([_neutral_controller()] * self.delay)
            for _ in range(num_envs)
        ]
        # manual CUDA-graph capture with STATIC buffers.
        # torch.compile's cudagraph trees hand back outputs that the next
        # replay overwrites, forcing a full clone of the carried state every
        # frame (3.4 ms at 400 rows for the windowed cores). Owning the
        # buffers lets the graph carry the state in place instead.
        self._use_capture = capture and torch.device(device).type == "cuda"
        # serving ring for the SGU v-cache (see SGUCore.initial_ring_state):
        # the captured graph writes one slot per frame instead of carrying a
        # shifted copy of every cache. Capture-only — under cudagraph trees
        # inductor functionalizes the in-place write back into a copy.
        self._core = getattr(getattr(policy, "network", None), "core", None)
        self._ring = (self._use_capture and hasattr(self._core, "initial_ring_state")
                      and ring is not False)
        if self._ring:   # the ring state exists from the start: snapshots precede the first replay
            self.hidden = self._cast_state(self._core.initial_ring_state(num_envs, device))
        self._graph = None
        # flat inputs (capture): the static inputs are the worker's three typed
        # flats and the struct the forward reads is a view of them, built once —
        # three copies per frame instead of one per leaf. Int storage is int64
        # because unflatten's .long() would otherwise COPY, freezing the views.
        self._view_fn = None
        self._in_flats = None

    def _cast_state(self, state):
        if self.state_dtype is None:
            return state
        return self.policy.network.cache_state(state, self.state_dtype)

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

    def execute(self, reset_indices: tp.Sequence[int] = ()) -> list:
        """Controllers to execute NOW: one pop per env from the delay
        queues (envs whose game just reset get a fresh neutral queue first).
        Instant — never waits on inference — so the worker sends these
        BEFORE running this frame's forward and the envs step while the
        GPU works. The controller popped is the same whether infer() has
        appended this frame's output yet or not (FIFO of length delay)."""
        for i in reset_indices:
            self._queues[i] = collections.deque([self._neutral()] * self.delay)
        return [self._queues[i].popleft() for i in range(self.num_envs)]

    def step(
        self, states: tp.Any, resets: torch.Tensor | None = None,
        reset_indices: tp.Sequence[int] | None = None,
        want_snapshot: bool = True,
    ) -> tuple[list[Controller], list[FrameRecord], tp.Any]:
        """execute() then infer(): returns (controllers to execute now,
        flushed FrameRecords, recurrent snapshot). Convenience for callers
        that do not pipeline the send (tests, eval)."""
        if resets is None:
            resets = torch.zeros(self.num_envs, dtype=torch.bool, device=self._name.device)
        if reset_indices is None:  # caller without a CPU copy: one sync
            reset_indices = torch.nonzero(resets).flatten().tolist()
        to_execute = self.execute(reset_indices)
        records, hidden_before = self.infer(states, resets, want_snapshot)
        return to_execute, records, hidden_before

    @torch.no_grad()  # rollout stepping is inference: without this the
    # compiled sample saves every frame's activations for a backward that
    # never comes
    def infer(
        self, states: tp.Any, resets: torch.Tensor, want_snapshot: bool = True,
        flats: tuple | None = None,
    ) -> tuple[list[FrameRecord], tp.Any]:
        """states: encoded Game struct batched [N, ...]; resets: [N] bool.

        One forward; the sampled controllers are appended to the delay
        queues. Returns [FrameRecord] and the recurrent snapshot from just
        before the forward (None unless wanted) for chunk-boundary bookkeeping."""
        hidden_before = self.hidden_snapshot() if want_snapshot else None
        prev = tree.map_structure(
            lambda pv, n: torch.where(
                resets.view(-1, *([1] * (pv.dim() - 1))), n, pv
            ),
            self._prev_action, self._neutral_encoded,
        )
        if self._use_capture:
            ctrl, logits = self._graph_step(states, prev, resets, flats)
        else:
            with self._autocast():
                out, hidden = self.policy.sample(
                    StateAction(state=states, action=prev, name=self._name),
                    self.hidden, is_resetting=resets, temperature=self.temperature,
                )
            ctrl, logits = out.controller_state, out.logits
            # carried state MUST be cloned on this path: with cudagraph
            # trees the forward's output lives in the graph's pool and the
            # next replay overwrites it (torch raises "accessing tensor
            # output of CUDAGraphs that has been overwritten"). Measured
            # 3.4 ms/frame at 400 rows for the windowed cores, 0 for the
            # LSTM. The capture path above avoids it entirely.
            self.hidden = tree.map_structure(
                lambda t: t.clone() if isinstance(t, torch.Tensor) else t, hidden
            )
        self._prev_action = tree.map_structure(
            lambda t: t.clone() if t.dtype == torch.bool else t.long().clone(),
            ctrl,
        )
        record = FrameRecord(
            state=states,
            prev_action=tree.map_structure(
                lambda x: x.clone() if x.dtype == torch.bool else x.long().clone(),
                prev,
            ),
            logits=tree.map_structure(lambda x: x.clone(), logits),
            name=self._name.clone(),
        )
        encoded_np = _controller_to_host(ctrl)
        decoded = self._embed_controller.decode(encoded_np)
        self._enqueue(decoded)
        return [record], hidden_before

    def _autocast(self):
        dev = torch.device(self.device).type
        return torch.autocast(dev, dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16,
                              enabled=self.precision in ("fp16", "bf16") and dev == "cuda")

    def set_flat_inputs(self, view_fn) -> None:
        """view_fn(flats) -> state struct (FlatFrames.view); pass flats= to infer."""
        self._view_fn = view_fn

    def _capture_step(self, states, prev, resets, flats=None) -> None:
        """Record one policy.sample into a manual CUDA graph over static
        input buffers. The graph's LAST op copies the new recurrent state
        back into the buffer it read from, so a replay both consumes and
        advances the state with no python-side clone and no placeholder
        copy. policy.sample must be uncompiled or compiled WITHOUT cudagraph
        trees (a graph inside a graph is not capturable)."""
        if flats is not None:
            assert self._view_fn is not None, "flats= needs set_flat_inputs(view_fn)"
            self._in_flats = tuple(
                t.clone().long() if not (t.is_floating_point() or t.dtype == torch.bool) else t.clone()
                for t in flats)
            self._in_states = self._view_fn(self._in_flats)
            bases = {t.untyped_storage().data_ptr() for t in self._in_flats}
            assert all(l.untyped_storage().data_ptr() in bases for l in tree.flatten(self._in_states)), \
                "state views do not alias the static flats"
        else:
            self._in_states = tree.map_structure(lambda t: t.clone(), states)
        self._in_prev = tree.map_structure(lambda t: t.clone(), prev)
        self._in_resets = resets.clone()

        def _forward():
            with self._autocast():
                return self.policy.sample(
                    StateAction(state=self._in_states, action=self._in_prev,
                                name=self._name),
                    self.hidden, is_resetting=self._in_resets,
                    temperature=self.temperature,
                )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm up allocations/autotuning before capture
                _forward()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, new_hidden = _forward()
            self._out_ctrl = out.controller_state
            self._out_logits = out.logits
            if self._ring:
                self._ring_carry(new_hidden)
            else:
                tree.map_structure(
                    lambda dst, src: dst.copy_(src) if isinstance(dst, torch.Tensor) else None,
                    self.hidden, new_hidden,
                )
        self._graph = graph
        # warmup + capture ran on whatever frame arrived first: restart the
        # carried state so the first replay begins from zeros
        init = self._cast_state(
            self._core.initial_ring_state(self.num_envs, self.device) if self._ring
            else self.policy.initial_state(self.num_envs, self.device))
        tree.map_structure(
            lambda dst, src: dst.copy_(src) if isinstance(dst, torch.Tensor) else None,
            self.hidden, init,
        )

    def _ring_carry(self, new_hidden):
        ptr = self.hidden["ptr"]
        for layer, new in zip(self.hidden["layers"], new_hidden["layers"]):
            if not isinstance(layer, tuple):   # recurrent layer: one state tensor
                layer.copy_(new)
                continue
            (v_ring, kv), (v_new, kv_new) = layer, new
            assert v_new.dim() == 2 and v_ring.dim() == 3, "ring carry takes a [B, d] slot, never a cache"
            v_ring.index_copy_(1, ptr.view(1), v_new.unsqueeze(1).to(v_ring.dtype))
            kv.copy_(kv_new)
        self.hidden["cache_len"].copy_(new_hidden["cache_len"])
        ptr.add_(1)
        ptr.remainder_(self._core.window - 1)

    def _graph_step(self, states, prev, resets, flats=None):
        """Replay the captured graph on this frame's inputs. Returns the
        STATIC output buffers — every consumer below copies out of them
        (prev_action clone, logits clone, .cpu() for the queues) before the
        next replay overwrites them."""
        if self._graph is None:
            self._capture_step(states, prev, resets, flats)
        if self._in_flats is not None:
            for dst, src in zip(self._in_flats, flats):
                dst.copy_(src)
        else:
            tree.map_structure(lambda dst, src: dst.copy_(src), self._in_states, states)
        tree.map_structure(lambda dst, src: dst.copy_(src), self._in_prev, prev)
        self._in_resets.copy_(resets)
        self._graph.replay()
        return self._out_ctrl, self._out_logits

    def hidden_snapshot(self) -> tp.Any:
        """Detached copy of the recurrent state (for Trajectory.initial_state)."""
        hidden = self._core.canonical_state(self.hidden) if self._ring else self.hidden
        return tree.map_structure(
            lambda t: t.detach().clone() if isinstance(t, torch.Tensor) else t,
            hidden,
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
        weights_dtype: torch.dtype = torch.float32,
        state_dtype: torch.dtype | None = None,
    ):
        import copy

        self.S, self.N = slices, cells
        self.device = torch.device(device)
        self.temperature = temperature
        # fp16 stacked weights halve the per-slice VRAM (107 -> 54 MB); the
        # forward then runs under fp16 autocast (norms stay fp32)
        self.weights_dtype = weights_dtype
        # optional recurrent-state storage dtype (see _initial_hidden); only
        # meaningful with the fp16-autocast forward
        self.state_dtype = state_dtype
        assert state_dtype is None or weights_dtype == torch.float16, (
            "state_dtype override is for the fp16 forward"
        )
        assert weights_dtype == torch.float32 or self.device.type == "cuda", (
            "fp16 league weights need CUDA (fp16 autocast)"
        )
        self.delay = template.delay
        self._embed_controller = template.controller_head.controller_embedding
        # functional_call's skeleton: a THROWAWAY copy (never read back).
        # The policy has tied parameters, and functional_call under vmap
        # leaves a tied template holding an escaped BatchedTensor — so the
        # template must be a throwaway copy.
        self._template = copy.deepcopy(template).to("cpu")
        self._template.__dict__.pop("sample", None)  # any compiled wrapper
        self._template.requires_grad_(False).eval()
        use_manual_recurrent_step(self._template)   # cuDNN steps have no vmap rule
        # stacked weights [S, ...]: slice s serves cells (s, 0..N-1)
        with torch.no_grad():
            params = dict(self._template.named_parameters())
            buffers = dict(self._template.named_buffers())
            def stack(t, dtype=None):
                t = t.detach().to(self.device)
                if dtype is not None and t.is_floating_point():
                    t = t.to(dtype)  # parameters only; buffers are constants
                return t.unsqueeze(0).repeat(self.S, *([1] * t.dim())).clone()
            self._stacked_params = {
                k: stack(v, self.weights_dtype) for k, v in params.items()
            }
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
        self._timer = None  # optional profiler callback (name) -> None
        # eager-path recurrent state [S, N, ...]; the captured path keeps
        # state in its static in/out buffers — lazy, so capture-mode never
        # allocates this third full copy
        self._hidden = None if self._use_capture else self._initial_hidden()

    # ---------------------------------------------------------- weights

    @torch.no_grad()
    def move_cell(self, src: tuple[int, int], dst: tuple[int, int]) -> None:
        """Move a seat's state (recurrent state, prev action, delay queue)
        to another cell — between frames, and only between slices holding
        the SAME weights (compaction: a member donates a slice by packing
        its few occupants into its other slices). Bit-exact for the env."""
        (s0, n0), (s1, n1) = src, dst
        tree.map_structure(lambda t: t[s1, n1].copy_(t[s0, n0]), self._prev)
        if not (self._use_capture and self._graph is not None) and self._hidden is None:
            self._hidden = self._initial_hidden()   # lazy (pre-capture move)
        hidden = self._out_hidden if self._use_capture and self._graph is not None else self._hidden

        def cell(t, s, n):   # state leaves are [S, N, ...], torch RNN state [S, layers, N, H]
            x = t[s]
            return x[n] if x.shape[0] == self.N else x[:, n]

        tree.map_structure(
            lambda t: cell(t, s1, n1).copy_(cell(t, s0, n0)) if isinstance(t, torch.Tensor) else None,
            hidden,
        )
        self._queues[s1 * self.N + n1] = self._queues[s0 * self.N + n0]
        self._queues[s0 * self.N + n0] = collections.deque(
            [self._neutral_row] * self.delay
        )

    @torch.no_grad()
    def load_slice(self, s: int, state_dict: dict) -> None:
        """Copy a member's weights (any device) into slice s, in place —
        captured replays read the stack by pointer, so they see it."""
        state_dict = current_names(state_dict)
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

    def execute(self) -> np.ndarray:
        """Controller rows to execute NOW, [S*N, 13]: one pop per cell
        (reset_cell first for cells starting a game). Sent before this
        frame's forward — see BatchedPolicyAgent.execute."""
        return np.stack([q.popleft() for q in self._queues])

    def step(self, views, resets: torch.Tensor):
        """execute() then infer(): (rows to execute now, FrameRecord)."""
        rows = self.execute()
        return rows, self.infer(views, resets)

    @torch.no_grad()
    def infer(self, views, resets: torch.Tensor, flats=None) -> FrameRecord:
        """views: encoded Game struct batched [S, N, ...] on device; resets:
        [S, N] bool on device (True on a cell's first frame of a game —
        zeroes its recurrent state and substitutes the neutral prev action).
        One forward; the sampled controllers are appended to the delay
        queues. Returns ONE FrameRecord over all S*N cells."""
        prev = tree.map_structure(
            lambda pv, n: torch.where(
                resets.view(self.S, self.N, *([1] * (pv.dim() - 2))), n, pv
            ),
            self._prev, self._neutral,
        )
        if self._use_capture:
            ctrl, logits = self._captured_forward(views, prev, resets, flats)
        else:
            ctrl, logits, self._hidden = self._vm(
                self._stacked_params, self._stacked_buffers, views, prev,
                self._hidden, resets,
            )
        if self._timer is not None:
            self._timer("forward")
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

        encoded_np = _controller_to_host(tree.map_structure(flat, ctrl))
        if self._timer is not None:
            self._timer("record+to_cpu")
        rows = encode.controller_rows(self._embed_controller.decode(encoded_np))
        for q, row in zip(self._queues, rows):
            q.append(row)
        if self._timer is not None:
            self._timer("decode+queues")
        return record

    # ---------------------------------------------------------- forward

    def _initial_hidden(self):
        h0 = [self._template.initial_state(self.N, self.device) for _ in range(self.S)]
        stacked = tree.map_structure(
            lambda *xs: torch.stack(xs) if isinstance(xs[0], torch.Tensor) else xs[0],
            *h0,
        )
        if self.state_dtype is not None:
            # fp16 window caches: the in/out static buffers are the grid's
            # biggest resident term (recurrent memory stays fp32, see cache_state)
            stacked = self._template.network.cache_state(stacked, self.state_dtype)
        return stacked

    def _make_vmap(self):
        from torch.func import functional_call, vmap

        base, name, temperature = self._template, self._name, self.temperature

        half = self.weights_dtype == torch.float16 and self.device.type == "cuda"

        def fmodel(p, b, st, ac, hid, rst):
            with torch.autocast("cuda", dtype=torch.float16, enabled=half):
                out, hid2 = functional_call(base, (p, b), (
                    StateAction(state=st, action=ac, name=name[0]), hid, rst, temperature,
                ))
            return out.controller_state, out.logits, hid2

        if self.S > 1:
            return vmap(fmodel, in_dims=(0, 0, 0, 0, 0, 0), randomness="different")

        # one slice: no vmap needed (and none possible for cores without a
        # batching rule, e.g. the ported medium-v2's cuDNN LSTM). Same
        # [1, N, ...] signature: squeeze the slice dim in, add it back out.
        sq = lambda t: t[0] if isinstance(t, torch.Tensor) else t
        un = lambda t: t[None] if isinstance(t, torch.Tensor) else t

        def single(p, b, st, ac, hid, rst):
            ctrl, logits, hid2 = fmodel(
                {k: v[0] for k, v in p.items()}, {k: v[0] for k, v in b.items()},
                tree.map_structure(sq, st), tree.map_structure(sq, ac),
                tree.map_structure(sq, hid), rst[0],
            )
            return (tree.map_structure(un, ctrl), tree.map_structure(un, logits),
                    tree.map_structure(un, hid2))

        return single

    def _captured_forward(self, views, prev, resets, flats=None):
        """Copy this frame's inputs into the static buffers, replay, return
        clones of the static outputs. Captured once at first use (shapes
        never change); in-place slice loads are visible to replays."""
        if self._graph is None:
            self._capture(views, prev, resets, flats)
        if getattr(self, "_in_flats", None) is not None:
            for d, src in zip(self._in_flats, flats):
                d.copy_(src)
        else:
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

    def set_flat_inputs(self, view_fn) -> None:
        """view_fn(flats) -> views struct; pass the grid's three flats to infer."""
        self._view_fn = view_fn
        self._in_flats = None

    def _capture(self, views, prev, resets, flats=None):
        if flats is not None:
            self._in_flats = tuple(
                t.clone().long() if not (t.is_floating_point() or t.dtype == torch.bool) else t.clone()
                for t in flats)
            self._in_views = self._view_fn(self._in_flats)
            bases = {t.untyped_storage().data_ptr() for t in self._in_flats}
            assert all(l.untyped_storage().data_ptr() in bases for l in tree.flatten(self._in_views)), \
                "grid views do not alias the static flats"
        else:
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
