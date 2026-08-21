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


# ---------------------------------------------------------------------------
# Typed flat layout: env processes ship an encoded frame as three 1-D arrays
# (bool / int32 / float32, leaves in tree order) instead of a ~150-leaf
# nested struct — one pickle each side, three host->GPU copies per frame
# for the whole fleet instead of ~150. The worker rebuilds the struct from a
# layout computed once (layout_of) on a dummy of the same embedding.
# ---------------------------------------------------------------------------

_KIND = {"b": "bool", "i": "int", "u": "int", "f": "float"}


def flatten_typed(struct) -> tuple:
    """(bools, ints, floats) for one encoded frame (leaves ravelled in tree
    order). Int leaves are widened to int32 (every one-hot fits)."""
    import tree

    parts: dict = {"bool": [], "int": [], "float": []}
    for leaf in tree.flatten(struct):
        a = np.asarray(leaf)
        parts[_KIND[a.dtype.kind]].append(a.ravel())
    return (
        np.concatenate(parts["bool"]).astype(np.bool_) if parts["bool"] else np.zeros(0, np.bool_),
        np.concatenate(parts["int"]).astype(np.int32) if parts["int"] else np.zeros(0, np.int32),
        np.concatenate(parts["float"]).astype(np.float32) if parts["float"] else np.zeros(0, np.float32),
    )


def layout_of(struct) -> list:
    """Per leaf in tree order: (kind, offset, size, shape) — computed from a
    dummy struct of the embedding (shapes are fixed for a run)."""
    import tree

    off = {"bool": 0, "int": 0, "float": 0}
    out = []
    for leaf in tree.flatten(struct):
        a = np.asarray(leaf)
        k = _KIND[a.dtype.kind]
        out.append((k, off[k], a.size, tuple(a.shape)))
        off[k] += a.size
    return out


def unflatten_typed_torch(struct_template, layout, bools, ints, floats):
    """Rebuild a batched struct [N, ...] from the three batched flat tensors
    ([N, L_kind], already on the target device). Int leaves come back as
    int64, bools as bool, floats as float32 — the learner's conventions."""
    import tree

    src = {"bool": bools, "int": ints, "float": floats}
    n = bools.shape[0] if bools.numel() else (ints.shape[0] if ints.numel() else floats.shape[0])
    leaves = []
    for kind, off, size, shape in layout:
        t = src[kind][:, off:off + size].reshape((n,) + shape)
        if kind == "int":
            t = t.long()
        leaves.append(t)
    return tree.unflatten_as(struct_template, leaves)


# ---------------------------------------------------------------------------
# Flat controller: the worker ships each env's controller as a 13-float row
# (tree order of slippi_ai.types.Controller: main_stick.x/y, c_stick.x/y,
# shoulder, then the Buttons fields) instead of a nested NamedTuple of
# numpy scalars; the env rebuilds the struct with one constructor call.
# ---------------------------------------------------------------------------

def controller_from_flat(v):
    """Controller struct from a flat float row (tree order)."""
    from slippi_ai.types import Buttons, Controller, Stick

    nb = len(Buttons._fields)
    return Controller(
        main_stick=Stick(x=np.float32(v[0]), y=np.float32(v[1])),
        c_stick=Stick(x=np.float32(v[2]), y=np.float32(v[3])),
        shoulder=np.float32(v[4]),
        buttons=Buttons(*(bool(v[5 + k] > 0.5) for k in range(nb))),
    )


def controller_rows(decoded) -> np.ndarray:
    """[N, 13] float32 rows from a batched decoded controller struct."""
    import tree

    return np.stack([np.asarray(x, dtype=np.float32) for x in tree.flatten(decoded)], axis=-1)
