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


class RowEncoder:
    """build(spec) + flatten_typed in ONE pass over the raw game: the leaf
    list is compiled once (getter chain + scalar rule per leaf, in the exact
    order tree.flatten yields the encoded struct), so a frame costs ~122
    attribute lookups and three np.array calls instead of a dm-tree walk
    with per-leaf numpy ops (measured 4 ms -> <1 ms per env frame). The
    scalar rules reproduce the numpy casts bit for bit (uint8 wrap, float32
    discrete rounding); test_encode pins equality with the slow path."""

    def __init__(self, spec: tuple):
        import operator

        self._leaves: list = []  # (kind, getter, rule)
        self._compile(spec, ())
        for i, (kind, path, rule) in enumerate(self._leaves):
            self._leaves[i] = (kind, operator.attrgetter(".".join(path)), rule)
        self.counts = {
            k: sum(1 for kind, _, _ in self._leaves if kind == k)
            for k in ("bool", "int", "float")
        }

    def _compile(self, spec, path):
        kind = spec[0]
        if kind == "struct":
            _, fields, (mod, qual), fixed = spec
            ctor: tp.Any = importlib.import_module(mod)
            for part in qual.split("."):
                ctor = getattr(ctor, part)
            subs = dict(fields)
            # tree.flatten walks the encoded namedtuple in ctor field order
            for name in ctor._fields:
                if name in subs:
                    self._compile(subs[name], path + (name,))
                else:
                    v = fixed[name]
                    assert v == () or v is None or isinstance(v, (tuple, list)) and not v, (
                        f"RowEncoder: fixed struct field {name}={v!r} is a leaf"
                    )  # empty containers flatten to nothing
            return
        dtype = np.dtype(spec[1] if kind != "onehot" else spec[3])
        k = _KIND[dtype.kind]
        self._leaves.append((k, path, self._rule(kind, spec, dtype)))

    @staticmethod
    def _rule(kind, spec, dtype):
        wrap = _int_wrap(dtype)
        if kind == "astype":
            if dtype.kind == "b":
                return bool
            if dtype.kind == "f":
                return float
            return lambda v: wrap(int(v))
        if kind == "onehot":
            _, policy, n, _d, name = spec
            if policy == "CLAMP":
                return lambda v: wrap(min(max(int(v), 0), n - 1))
            if policy == "EXTRA":
                return lambda v: wrap(int(v) if 0 <= int(v) < n else n)
            if policy == "ERROR":
                def rule(v):
                    v = int(v)
                    if v < 0:
                        raise ValueError(f"Got negative input in {name}")
                    if v >= n:
                        raise ValueError(f"Invalid input {v} >= {n} in {name}")
                    return wrap(v)
                return rule
            raise ValueError(f"unknown one-hot policy {policy!r}")
        if kind == "discrete":
            _, n, _d = spec
            f32 = np.float32
            # (state * n + 0.5) in float32, then truncating cast
            return lambda v: wrap(int(f32(f32(v) * n) + f32(0.5)))
        raise ValueError(f"unknown encoder spec {kind!r}")

    def encode(self, raw) -> tuple:
        """(bools, ints, floats) for one raw game — flatten_typed's output."""
        parts: dict = {"bool": [], "int": [], "float": []}
        for kind, get, rule in self._leaves:
            parts[kind].append(rule(get(raw)))
        return (
            np.array(parts["bool"], dtype=np.bool_),
            np.array(parts["int"], dtype=np.int32),
            np.array(parts["float"], dtype=np.float32),
        )


def _int_wrap(dtype: np.dtype):
    """Python equivalent of numpy's integer cast to `dtype` (low bits,
    two's complement), as the slow path's .astype does before flatten_typed
    widens to int32."""
    if dtype.kind == "u":
        mask = (1 << (8 * dtype.itemsize)) - 1
        return lambda v: v & mask
    if dtype.kind == "i":
        bits = 8 * dtype.itemsize
        half, full = 1 << (bits - 1), 1 << bits
        return lambda v: ((v + half) % full) - half
    return lambda v: v


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
    """Rebuild a batched struct [*leading, ...] from the three batched flat
    tensors ([*leading, L_kind], already on the target device; any number of
    leading dims). Int leaves come back as int64, bools as bool, floats as
    float32 — the learner's conventions. One kernel (the int64 cast); every
    leaf is a view."""
    import tree

    src = {"bool": bools, "int": ints.long(), "float": floats}
    lead = tuple(bools.shape[:-1])
    leaves = []
    for kind, off, size, shape in layout:
        leaves.append(src[kind][..., off:off + size].reshape(lead + shape))
    return tree.unflatten_as(struct_template, leaves)


def swap_perm(struct_template, layout) -> dict:
    """Per-kind column permutations implementing the p0 <-> p1 perspective
    swap AT THE FLAT LEVEL: applying them to the typed flat tensors equals
    swapping the players in the struct (bit-exact — it is a permutation).
    Returns {"bool"/"int"/"float": np.ndarray | None} (None = identity)."""
    import tree

    paths = [p for p, _ in tree.flatten_with_path(struct_template)]
    assert len(paths) == len(layout)
    sizes = {"bool": 0, "int": 0, "float": 0}
    for kind, off, size, _ in layout:
        sizes[kind] = max(sizes[kind], off + size)
    perm = {k: np.arange(n) for k, n in sizes.items()}
    spans = {}  # (player, suffix) -> (kind, off, size)
    for path, (kind, off, size, _shape) in zip(paths, layout):
        if path and path[0] in ("p0", "p1"):
            spans[(path[0], path[1:])] = (kind, off, size)
    for (player, suffix), (kind, off, size) in spans.items():
        if player != "p0":
            continue
        okind, ooff, osize = spans[("p1", suffix)]
        assert okind == kind and osize == size, suffix
        perm[kind][off:off + size] = np.arange(ooff, ooff + size)
        perm[kind][ooff:ooff + size] = np.arange(off, off + size)
    return {
        k: (None if np.array_equal(p_, np.arange(len(p_))) else p_)
        for k, p_ in perm.items()
    }


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
