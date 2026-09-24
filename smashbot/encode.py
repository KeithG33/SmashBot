"""Flat layouts between the sim worker and the networks: an encoded frame
as three typed arrays (bool / int32 / float32) instead of a ~150-leaf
struct, and a controller as a 13-float row."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Typed flat layout: an encoded frame as three arrays (bool / int32 /
# float32, leaves in tree order) instead of a ~150-leaf nested struct, so a
# frame moves to the GPU in three copies instead of ~150. The struct is
# rebuilt from a layout computed once (layout_of) on a dummy of the same
# embedding.
# ---------------------------------------------------------------------------

_KIND = {"b": "bool", "i": "int", "u": "int", "f": "float"}


def flatten_typed_batched(struct, batch: int) -> tuple:
    """Leaves are [batch, ...]; returns (bools[batch, B], ints[batch, I],
    floats[batch, F]), leaves ravelled in tree order and ints widened to
    int32, with per-frame column offsets IDENTICAL to layout_of(per-frame
    struct) — so the flat tensors reconstruct through layout / swap_perm."""
    import tree

    parts: dict = {"bool": [], "int": [], "float": []}
    for leaf in tree.flatten(struct):
        a = np.asarray(leaf)
        assert a.shape[0] == batch, (a.shape, batch)
        parts[_KIND[a.dtype.kind]].append(a.reshape(batch, -1))
    cat = lambda k, dt: (
        np.concatenate(parts[k], axis=1).astype(dt, copy=False)
        if parts[k] else np.zeros((batch, 0), dt)
    )
    return cat("bool", np.bool_), cat("int", np.int32), cat("float", np.float32)


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
# numpy scalars, written straight into the sim (sim_env.write_controller_rows).
# ---------------------------------------------------------------------------

def controller_rows(decoded) -> np.ndarray:
    """[N, 13] float32 rows from a batched decoded controller struct."""
    import tree

    return np.stack([np.asarray(x, dtype=np.float32) for x in tree.flatten(decoded)], axis=-1)


def controller_from_rows(rows: np.ndarray):
    """Batched decoded controller struct from [..., 13] rows (controller_rows' inverse)."""
    from slippi_ai.types import Buttons, Controller, Stick

    axis = lambda k: rows[..., k].astype(np.float32)
    return Controller(
        main_stick=Stick(x=axis(0), y=axis(1)),
        c_stick=Stick(x=axis(2), y=axis(3)),
        shoulder=axis(4),
        buttons=Buttons(*(rows[..., 5 + k] > 0.5 for k in range(len(Buttons._fields)))),
    )
