"""Pieces shared by the BC and RL learners: restore resolution, gradient
clipping and compile."""
from __future__ import annotations

import heapq
import math
import os

import torch


def resolve_restore(run_dir: str, restore: str) -> str:
    """The checkpoint a run resumes from, or "" for a fresh start. A fresh
    start into a run directory that already holds checkpoints is refused, so
    a relaunch that forgets --runtime.restore cannot overwrite them; "auto"
    and an explicit path must name a checkpoint that exists."""
    if not restore:
        existing = [f for f in ("latest.pt", "best.pt") if os.path.exists(os.path.join(run_dir, f))]
        if existing:
            raise FileExistsError(
                f"{run_dir} already holds {' and '.join(existing)}: resume it with "
                "--runtime.restore auto, or start a new experiment under a new --runtime.tag")
        return ""
    path = os.path.join(run_dir, "latest.pt") if restore == "auto" else restore
    if not os.path.exists(path):
        raise FileNotFoundError(f"--runtime.restore {restore}: no checkpoint at {path}")
    return path


class RunningPercentile:
    """np.percentile (method "linear") of every value added so far, equal to
    the bit, at O(log n) per value instead of numpy's O(n). numpy
    interpolates between the sorted values at ranks floor(v) and floor(v)+1,
    v = (n - 1) * q / 100; a max-heap keeps the values up to rank floor(v)
    and a min-heap the rest, so both endpoints are heap tops."""

    def __init__(self, percentile: float, values=()):
        assert 0 <= percentile < 100
        self.fraction = percentile / 100
        ordered = sorted(values)
        self.n = len(ordered)
        split = self._split()
        self._low = [-v for v in reversed(ordered[:split])]   # max-heap as negations
        self._high = ordered[split:]

    def _split(self) -> int:
        return math.floor((self.n - 1) * self.fraction) + 1 if self.n else 0

    def add(self, value: float) -> None:
        if self._low and value <= -self._low[0]:
            heapq.heappush(self._low, -value)
        else:
            heapq.heappush(self._high, value)
        self.n += 1
        while len(self._low) > self._split():
            heapq.heappush(self._high, -heapq.heappop(self._low))
        while len(self._low) < self._split():
            heapq.heappush(self._low, -heapq.heappop(self._high))

    @property
    def value(self) -> float:
        v = (self.n - 1) * self.fraction
        t = v - math.floor(v)
        a = -self._low[0]
        b = self._high[0] if self._high else a
        # numpy's _lerp works from the nearer endpoint, and so rounds differently
        return b - (b - a) * (1 - t) if t >= 0.5 else a + (b - a) * t


class GradClipper:
    """Clips a network's gradient to `max_norm`, or with AutoClip
    (Seetharaman et al. 2020) to a percentile of the run's own gradient-norm
    history. The history is checkpointed so a resume clips exactly as the
    uninterrupted run would.

    `measure` reads the norm without touching the gradients, so the caller
    decides what a non-finite one means (BC stops, the RL learner skips the
    step); `clip` then records it and clips with it."""

    def __init__(self, params, max_norm: float, percentile: float = 0.0, history=None):
        assert not (max_norm > 0 and percentile > 0), "one clipping rule at a time"
        self.params = list(params)
        self.max_norm, self.percentile = max_norm, percentile
        self.restore_history(history or [])

    def restore_history(self, history) -> None:
        self._history = list(history)
        self._running = RunningPercentile(self.percentile, self._history)

    @property
    def history(self) -> list:
        return self._history

    def norm(self) -> torch.Tensor:
        return torch.nn.utils.get_total_norm([p.grad for p in self.params if p.grad is not None])

    def measure(self) -> float:
        return self.norm().item()

    @property
    def threshold(self) -> float:
        if self.percentile > 0:
            return self._running.value if self._history else math.inf
        return self.max_norm if self.max_norm > 0 else math.inf

    def clip(self, norm: float) -> dict:
        if self.percentile > 0:
            assert math.isfinite(norm), "a non-finite norm must never reach the history"
            self._history.append(norm)
            self._running.add(norm)
        clip = self.threshold
        if math.isfinite(clip):
            torch.nn.utils.clip_grads_with_norm_(self.params, clip, torch.tensor(norm))
        return {"grad_norm": norm, "clip_norm": clip}


def compile_cores(policy, value_fn) -> None:
    """Compile the pieces with a fixed structure, dynamic over the chunk
    length: an SGU core's per-chunk forward (cuDNN recurrent layers stay
    eager inside it), a tx_like core's stateless layers (the encoder and the
    FFW blocks between its LSTMs), and the controller head. The reset
    chunking, tree maps and metric .item()s around them stay in Python. For
    the learner's copies only: a serving copy has its own compile inside its
    CUDA-graph capture."""
    from smashbot.networks import FFWWrapper

    torch._dynamo.config.recompile_limit = 64  # two cores x chunk shapes x cache dtypes
    for net in (policy.network, value_fn.network):
        core = net.core
        if hasattr(core, "_forward"):
            core._forward = torch.compile(core._forward, dynamic=True)
        else:
            for layer in core.modules():
                if isinstance(layer, FFWWrapper):
                    layer._module.forward = torch.compile(layer._module.forward, dynamic=True)
    policy.controller_head.distance = torch.compile(policy.controller_head.distance, dynamic=True)
