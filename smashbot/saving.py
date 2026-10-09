"""Versioned checkpointing.

Checkpoint = torch.save of {"config", "state", "best_eval_loss", "version"},
mirroring slippi-ai's shape. `upgrade_checkpoint` maps old versions forward so
early checkpoints stay loadable as the config schema evolves.
"""

import dataclasses
import os
import typing as tp

import torch

VERSION = 2


def _upgraders() -> dict[int, tp.Callable[[dict], dict]]:
    return {1: _v1_to_v2}


# a value net's attention tensors, by name suffix: (dim holding the heads,
# blocks along it, what a new head's slice starts as)
_ATTENTION = {
    "attn_qkv.weight": (0, 3, "init"),   # [q; k; v] rows, each head-major
    "attn_out.weight": (1, 1, "zeros"),  # a new head adds nothing until trained
    "q_norm.weight": (0, 1, "ones"),
    "k_norm.weight": (0, 1, "ones"),
}


def _grow(t: torch.Tensor, dim: int, blocks: int, old: int, new: int, fill: str) -> torch.Tensor:
    """Each of t's `blocks` equal slices along `dim` grows from `old` to
    `new` entries; the added entries start as `fill`."""
    parts, generator = [], torch.Generator().manual_seed(0)
    for part in t.split(t.shape[dim] // blocks, dim=dim):
        shape = list(part.shape)
        shape[dim] = part.shape[dim] // old * (new - old)
        if fill == "init":   # nn.Linear's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            bound = part.shape[1] ** -0.5
            added = torch.empty(shape, dtype=t.dtype).uniform_(-bound, bound, generator=generator)
        else:
            added = (torch.zeros if fill == "zeros" else torch.ones)(shape, dtype=t.dtype)
        parts += [part, added]
    return torch.cat(parts, dim=dim)


def widen_value_heads(state: dict, old: int, new: int, head_dim: int) -> None:
    """Grows a value net's attention from `old` to `new` heads in a saved
    state, computing exactly what it did: the new heads' output starts at
    zero. Weights, Adam moments (located by shape, refusing ambiguity) and
    the carried attention cache all grow."""
    value = state["value"]
    layouts = {}   # shape -> how its tensors grow (None: not attention)
    for key, t in value.items():
        kind = next((s for s in _ATTENTION if key.endswith(s)), None)
        layouts.setdefault(tuple(t.shape), set()).add(_ATTENTION[kind][:2] if kind else None)
        if kind:
            value[key] = _grow(t, *_ATTENTION[kind][:2], old, new, _ATTENTION[kind][2])
    for slots in (state.get("value_opt") or {}).get("state", {}).values():
        found = layouts.get(tuple(slots["exp_avg"].shape), {None})
        if found == {None}:
            continue
        if len(found) != 1:
            raise ValueError(f"value optimizer slot of shape {tuple(slots['exp_avg'].shape)} "
                             f"is ambiguous between {found}")
        ((dim, blocks),) = found
        for m in ("exp_avg", "exp_avg_sq"):
            slots[m] = _grow(slots[m], dim, blocks, old, new, "zeros")
    hidden = state.get("value_hidden")
    if hidden is not None:   # an attention block's cache: [..., keys; values], each head-major
        core = hidden["core"] if "core" in hidden else hidden
        core["layers"] = [
            (layer[0], _grow(layer[1], -1, 2, old, new, "zeros"))
            if isinstance(layer, tuple) and layer[1].shape[-1] == 2 * old * head_dim else layer
            for layer in core["layers"]]


def _v1_to_v2(ckpt: dict) -> dict:
    """A "match" value net became the policy's network in full: before, the
    policy's attention heads were not carried over and value nets kept one."""
    from smashbot.value import value_network_config

    state, config = ckpt["state"], ckpt["config"]
    if "value" in state and "value" in config:
        want = value_network_config(config["network"], config["value"])
        qkv = next((t for k, t in state["value"].items() if k.endswith("attn_qkv.weight")), None)
        if want.name == "sgu" and qkv is not None:
            have = qkv.shape[0] // (3 * want.attn_head_dim)
            if have < want.attn_heads:
                widen_value_heads(state, have, want.attn_heads, want.attn_head_dim)
    return {**ckpt, "version": 2}


def upgrade_checkpoint(ckpt: dict) -> dict:
    version = ckpt.get("version", 0)
    while version < VERSION:
        upgrader = _upgraders().get(version)
        if upgrader is None:
            raise ValueError(f"No upgrader from checkpoint version {version}")
        ckpt = upgrader(ckpt)
        version = ckpt["version"]
    return ckpt


def save_checkpoint(
    path: str,
    config: tp.Any,  # dataclass
    state: dict,  # module/optimizer state_dicts + step
    best_eval_loss: float,
) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "config": dataclasses.asdict(config),
            "state": state,
            "best_eval_loss": best_eval_loss,
            "version": VERSION,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic: never leave a torn checkpoint


def load_checkpoint(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return upgrade_checkpoint(ckpt)


def resolve_name_code(name_map: dict, name: str, verbose: bool = True) -> int:
    if name in name_map:
        return name_map[name]
    if name_map and verbose:
        print(f"'{name}' not in name_map {name_map}; using code 0")
    return 0


def load_optimizer(optimizer, saved_optimizer: dict, saved_module_state: dict, prefix: str = "value_head.") -> None:
    """Restore an optimizer whose module may have lost parameters since the
    checkpoint: the built-in value head's two tensors registered last, so a
    checkpoint that still carries them has two trailing parameter ids to drop."""
    dropped = sum(k.startswith(prefix) for k in saved_module_state)
    if dropped:
        saved_optimizer = {
            "state": dict(saved_optimizer["state"]),
            "param_groups": [dict(g) for g in saved_optimizer["param_groups"]],
        }
        group = saved_optimizer["param_groups"][-1]
        stale, group["params"] = group["params"][-dropped:], group["params"][:-dropped]
        for i in stale:
            saved_optimizer["state"].pop(i, None)
    optimizer.load_state_dict(saved_optimizer)

