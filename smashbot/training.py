"""Pieces shared by the BC and RL learners: gradient clipping and compile."""
from __future__ import annotations

import math

import numpy as np
import torch


class GradClipper:
    """Clips a network's gradient to `max_norm`, or with AutoClip
    (Seetharaman et al. 2020) to a percentile of the run's own gradient-norm
    history. The history is checkpointed so a resume clips exactly as the
    uninterrupted run would.

    `measure` then `clip` lets a caller inspect the norm first (the RL
    learner skips non-finite steps, which must not enter the history);
    `__call__` does both."""

    def __init__(self, params, max_norm: float, percentile: float = 0.0, history=None):
        assert not (max_norm > 0 and percentile > 0), "one clipping rule at a time"
        self.params = list(params)
        self.max_norm, self.percentile = max_norm, percentile
        self.history = list(history or [])

    def measure(self) -> float:
        return torch.nn.utils.clip_grad_norm_(self.params, math.inf).item()

    @property
    def threshold(self) -> float:
        if self.percentile > 0:
            return float(np.percentile(self.history, self.percentile)) if self.history else math.inf
        return self.max_norm if self.max_norm > 0 else math.inf

    def clip(self, norm: float) -> dict:
        if self.percentile > 0:
            self.history.append(norm)
        clip = self.threshold
        if math.isfinite(clip):
            torch.nn.utils.clip_grad_norm_(self.params, clip)
        return {"grad_norm": norm, "clip_norm": clip}

    def __call__(self) -> dict:
        norm = self.measure()
        if math.isfinite(norm):
            return self.clip(norm)
        return {"grad_norm": norm, "clip_norm": self.threshold}


def compile_cores(policy, value_fn) -> None:
    """Compile the pieces with a fixed structure: each core's per-chunk
    forward (dynamic over the chunk length; cuDNN recurrent layers stay eager
    inside it) and the controller head. The reset chunking, tree maps and
    metric .item()s around them stay in Python. For the learner's copies
    only: a serving copy has its own compile inside its CUDA-graph capture."""
    torch._dynamo.config.cache_size_limit = 64  # two cores x chunk shapes x cache dtypes
    for net in (policy.network, value_fn.network):
        net.core._forward = torch.compile(net.core._forward, dynamic=True)
    policy.controller_head.distance = torch.compile(policy.controller_head.distance, dynamic=True)
