"""Torch-free numpy encoder rebuilt from an Embedding spec.

Env processes encode frames with this instead of importing smashbot.embed
(whose classes are nn.Modules, pulling torch's ~0.26 GB private RSS into
every env). The spec is produced by Embedding.spec(); the four leaf rules
here mirror the from_state methods exactly and are pinned by a golden test
(smashbot/tests/test_encode.py).
"""

from __future__ import annotations

import importlib
import typing as tp

import numpy as np


class _Astype:
    def __init__(self, dtype: str):
        self.dtype = np.dtype(dtype)

    def from_state(self, state):
        return state.astype(self.dtype)


class _OneHot:
    def __init__(self, policy: str, input_size: int, dtype: str, name: str):
        self.policy = policy
        self.input_size = input_size
        self.dtype = np.dtype(dtype)
        self.name = name

    def from_state(self, state):
        if self.policy == "CLAMP":
            state = np.clip(state, 0, self.input_size - 1)
        elif self.policy == "ERROR":
            if np.any(state < 0):
                raise ValueError(f"Got negative input in {self.name}")
            if np.any(state >= self.input_size):
                x = np.max(state)
                raise ValueError(
                    f"Invalid input {x} >= {self.input_size} in {self.name}"
                )
        elif self.policy == "EXTRA":
            invalid = (state < 0) | (state >= self.input_size)
            if np.any(invalid):
                state = state.copy()
                state[invalid] = self.input_size
        return state.astype(self.dtype)


class _Discrete:
    def __init__(self, n: int, dtype: str):
        self.n = n
        self.dtype = np.dtype(dtype)

    def from_state(self, state):
        assert state.dtype == np.float32
        return (state * self.n + 0.5).astype(self.dtype)


class _Struct:
    def __init__(self, fields, ctor, fixed_kwargs):
        self.fields = fields  # [(name, encoder)]
        self.ctor = ctor
        self.fixed_kwargs = fixed_kwargs

    def from_state(self, state):
        out = {k: e.from_state(getattr(state, k)) for k, e in self.fields}
        return self.ctor(**out, **self.fixed_kwargs)


def build(spec: tuple):
    """Encoder with .from_state(raw) from an Embedding.spec()."""
    kind = spec[0]
    if kind == "astype":
        return _Astype(spec[1])
    if kind == "onehot":
        return _OneHot(*spec[1:])
    if kind == "discrete":
        return _Discrete(*spec[1:])
    if kind == "struct":
        _, fields, (mod, qual), fixed = spec
        ctor: tp.Any = importlib.import_module(mod)
        for part in qual.split("."):
            ctor = getattr(ctor, part)
        return _Struct([(k, build(sub)) for k, sub in fields], ctor, fixed)
    raise ValueError(f"unknown encoder spec {kind!r}")
