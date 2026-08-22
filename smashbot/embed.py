"""PyTorch port of slippi-ai's composable embeddings (vendor: slippi_ai/tf/embed.py).

Semantics are kept exactly: same scales, sizes, one-hot policies, and struct
field ordering (which drives autoregressive sampling order). `from_state`
operates on numpy (used in the data thread and at inference to encode raw
game structs); `__call__`/`distance`/`sample` operate on torch tensors.
"""

import abc
import dataclasses
import enum
import typing as tp
from typing import Any, Callable, Generic, Iterator, Mapping, Sequence, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from slippi_ai import types
from slippi_ai.controller_lib import LEGAL_BUTTONS
from slippi_ai.types import (
    Buttons,
    Controller,
    FoDPlatforms,
    Game,
    Item,
    Items,
    Player,
    Randall,
    StateAction,
    Stick,
)

from slippi_ai.types import NAME_DTYPE

In = TypeVar("In")
Out = TypeVar("Out")


class Embedding(Generic[In, Out], nn.Module, abc.ABC):
    """Embeds game type (In) into a torch-ready type (Out)."""

    size: int
    dtype: Any

    def from_state(self, state: In) -> Out:
        return state.astype(self.dtype)

    def spec(self) -> tuple:
        """Pure-data description of this embedding's from_state rule, for
        smashbot.encode (torch-free numpy encoder used by env processes)."""
        return ("astype", np.dtype(self.dtype).str)

    @abc.abstractmethod
    def forward(self, x: Out) -> torch.Tensor:
        """Embed the input as a flat float tensor."""

    def map(self, f, *args: Out) -> Out:
        return f(self, *args)

    def flatten(self, struct: Out) -> Iterator[Any]:
        yield struct

    def unflatten(self, seq: Iterator[Any]) -> Out:
        return next(seq)

    def decode(self, out: Out) -> In:
        return out

    def dummy(self, shape: Sequence[int] = ()) -> Out:
        return np.zeros(shape, self.dtype)

    def dummy_embedding(self, shape: Sequence[int]) -> Out:
        return np.zeros(list(shape) + [self.size], np.float32)

    def sample(self, embedded: torch.Tensor, temperature=None) -> Out:
        raise NotImplementedError

    def distance(self, embedded: torch.Tensor, target: Out) -> Out:
        """Negative log-prob of the target sample."""
        raise NotImplementedError

    def logits_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy of the distribution parameterized by `logits`. [RL]"""
        raise NotImplementedError

    def logits_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """KL(p || q) between distributions parameterized by logits. [RL]"""
        raise NotImplementedError


class BoolEmbedding(Embedding[bool, np.bool_]):
    size = 1
    dtype = np.bool_

    def __init__(self, name="bool", on=1.0, off=0.0):
        super().__init__()
        self.name = name
        self.on = on
        self.off = off

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.where(t, self.on, self.off).unsqueeze(-1).float()

    def distance(self, embedded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = embedded.squeeze(-1)
        labels = target.float()
        logits, labels = torch.broadcast_tensors(logits, labels)
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    def sample(self, embedded: torch.Tensor, temperature=None) -> torch.Tensor:
        logits = embedded.squeeze(-1)
        if temperature is not None:
            logits = logits / temperature
        return torch.bernoulli(torch.sigmoid(logits)).bool()

    def logits_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.squeeze(-1)
        p = torch.sigmoid(x)
        # H = -p log p - (1-p) log(1-p), via logsigmoid for stability
        return -(p * F.logsigmoid(x) + (1 - p) * F.logsigmoid(-x))

    def logits_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        x, y = p_logits.squeeze(-1), q_logits.squeeze(-1)
        p = torch.sigmoid(x)
        return p * (F.logsigmoid(x) - F.logsigmoid(y)) + (1 - p) * (
            F.logsigmoid(-x) - F.logsigmoid(-y)
        )


embed_bool = BoolEmbedding()


class FloatEmbedding(Embedding[float, np.float32]):
    dtype = np.float32
    size = 1

    def __init__(self, name, scale=None, bias=None, lower=-10.0, upper=10.0):
        super().__init__()
        self.name = name
        self.scale = scale
        self.bias = bias
        self.lower = lower
        self.upper = upper

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        if self.bias is not None:
            t = t + self.bias
        if self.scale is not None:
            t = t * self.scale
        if self.lower:
            t = torch.clamp(t, min=self.lower)
        if self.upper:
            t = torch.clamp(t, max=self.upper)
        return t

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.encode(t).unsqueeze(-1)

    def distance(self, embedded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = self.encode(target)
        predicted = embedded.squeeze(-1)
        return torch.square(predicted - target)


embed_float = FloatEmbedding("float")


class OneHotPolicy(enum.Enum):
    CLAMP = 0  # clamp to [0, size-1]
    ERROR = 1  # raise on invalid inputs
    EXTRA = 2  # extra dimension for invalid inputs
    EMPTY = 3  # invalid inputs embed as all-zeros


class OneHotEmbedding(Embedding[int, np.ndarray]):
    def __init__(
        self,
        name: str,
        size: int,
        dtype=np.int32,
        one_hot_policy: OneHotPolicy = OneHotPolicy.ERROR,
    ):
        super().__init__()
        self.name = name
        self.one_hot_policy = one_hot_policy
        self.size = size
        if one_hot_policy is OneHotPolicy.EXTRA:
            self.size += 1
        self.input_size = size
        self.dtype = dtype

    def spec(self) -> tuple:
        return ("onehot", self.one_hot_policy.name, self.input_size,
                np.dtype(self.dtype).str, self.name)

    def from_state(self, state: np.ndarray) -> np.ndarray:
        if self.one_hot_policy is OneHotPolicy.CLAMP:
            state = np.clip(state, 0, self.input_size - 1)
        elif self.one_hot_policy is OneHotPolicy.ERROR:
            if np.any(state < 0):
                raise ValueError(f"Got negative input in {self.name}")
            if np.any(state >= self.input_size):
                x = np.max(state)
                raise ValueError(f"Invalid input {x} >= {self.input_size} in {self.name}")
        elif self.one_hot_policy is OneHotPolicy.EXTRA:
            invalid = (state < 0) | (state >= self.input_size)
            if np.any(invalid):
                state = state.copy()
                state[invalid] = self.input_size
        return state.astype(self.dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long()
        if self.one_hot_policy is OneHotPolicy.EMPTY:
            valid = (t >= 0) & (t < self.size)
            one_hot = F.one_hot(t.clamp(0, self.size - 1), self.size).float()
            return one_hot * valid.unsqueeze(-1)
        return F.one_hot(t, self.size).float()

    def distance(self, embedded: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(embedded, dim=-1)
        return -logprobs.gather(-1, target.long().unsqueeze(-1)).squeeze(-1)

    def sample(self, embedded: torch.Tensor, temperature=None) -> torch.Tensor:
        logits = embedded
        if temperature is not None:
            logits = logits / temperature
        flat = logits.reshape(-1, logits.shape[-1])
        samples = torch.multinomial(F.softmax(flat, dim=-1), 1).squeeze(-1)
        samples = samples.reshape(logits.shape[:-1])
        # match the embedding's declared dtype so decode()'s round-trip holds
        torch_dtype = {"uint8": torch.uint8, "int32": torch.int32}[
            np.dtype(self.dtype).name
        ]
        return samples.to(torch_dtype)

    def logits_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(-1)

    def logits_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(p_logits, dim=-1)
        logq = F.log_softmax(q_logits, dim=-1)
        return (logp.exp() * (logp - logq)).sum(-1)


NT = TypeVar("NT")


class SplatKwargs(Generic[NT]):
    """Wraps a constructor taking kwargs (lambdas don't pickle)."""

    def __init__(self, f: Callable[..., NT], fixed_kwargs: Mapping[str, Any] = {}):
        self._func = f
        self._fixed_kwargs = fixed_kwargs

    def __call__(self, kwargs: Mapping[str, Any]) -> NT:
        return self._func(**kwargs, **self._fixed_kwargs)


class StructEmbedding(Embedding[NT, NT]):
    """Embeds dicts/NamedTuples. Sub-embedding order = traversal order,
    which determines autoregressive sampling order."""

    def __init__(
        self,
        name: str,
        embedding: Sequence[tuple[str, Embedding]],
        builder: Callable[[Mapping[str, Any]], NT],
        getter: Callable[[NT, str], Any],
    ):
        super().__init__()
        self.name = name
        self.embedding = list(embedding)
        self.builder = builder
        self.getter = getter
        # Register children for parameter tracking (shared modules dedupe fine).
        self._children = nn.ModuleList([e for _, e in self.embedding])
        self.size = sum(op.size for _, op in self.embedding)

    def map(self, f, *args: NT) -> NT:
        result = {
            k: e.map(f, *(self.getter(x, k) for x in args)) for k, e in self.embedding
        }
        return self.builder(result)

    def flatten(self, struct: NT):
        for k, e in self.embedding:
            yield from e.flatten(self.getter(struct, k))

    def unflatten(self, seq: Iterator[Any]) -> NT:
        return self.builder({k: e.unflatten(seq) for k, e in self.embedding})

    def from_state(self, state: NT) -> NT:
        struct = {k: e.from_state(self.getter(state, k)) for k, e in self.embedding}
        return self.builder(struct)

    def spec(self) -> tuple:
        assert isinstance(self.builder, SplatKwargs) and self.getter is getattr, (
            "encode.spec supports SplatKwargs/getattr structs only"
        )
        f = self.builder._func
        return ("struct", [(k, e.spec()) for k, e in self.embedding],
                (f.__module__, f.__qualname__), dict(self.builder._fixed_kwargs))

    def forward(self, struct: NT) -> torch.Tensor:
        embed = []
        for field, op in self.embedding:
            if op.size == 0:
                continue
            embed.append(op(self.getter(struct, field)))
        assert embed, "Embedding must not be empty"
        return torch.cat(embed, dim=-1)

    def dummy(self, shape=()):
        return self.map(lambda e: e.dummy(shape))

    def dummy_embedding(self, shape):
        return self.map(lambda e: e.dummy_embedding(shape))

    def decode(self, struct: NT) -> NT:
        return self.map(lambda e, x: e.decode(x), struct)


def struct_embedding_from_nt(name: str, nt: NT) -> StructEmbedding[NT]:
    return StructEmbedding(
        name=name,
        embedding=list(zip(nt._fields, nt)),
        builder=SplatKwargs(type(nt)),
        getter=getattr,
    )


def ordered_struct_embedding(
    name: str,
    embedding: Sequence[tuple[str, Embedding]],
    nt_type: type[NT],
) -> StructEmbedding[NT]:
    """Supports missing fields, which appear as ()."""
    existing = set(k for k, _ in embedding)
    missing_kwargs = {k: () for k in set(nt_type._fields) - existing}
    return StructEmbedding(
        name=name,
        embedding=embedding,
        builder=SplatKwargs(nt_type, missing_kwargs),
        getter=getattr,
    )


class MLPWrapper(Embedding[In, Out]):
    def __init__(self, output_sizes: Sequence[int], embed: Embedding[In, Out]):
        super().__init__()
        self.name = f"MLP_{embed.name}"
        self.size = output_sizes[-1]
        self._embed = embed

        layers: list[nn.Module] = []
        in_size = embed.size
        for out_size in output_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())  # activate_final=True upstream
            in_size = out_size
        self._mlp = nn.Sequential(*layers)

    def from_state(self, state: In) -> Out:
        return self._embed.from_state(state)

    def spec(self) -> tuple:
        return self._embed.spec()

    def forward(self, inputs: Out) -> torch.Tensor:
        return self._mlp(self._embed(inputs))

    def dummy(self, shape=()):
        return self._embed.dummy(shape)


# Note: some Kirby ability-copy action states go beyond this (CLAMP policy).
def make_embed_action():
    return OneHotEmbedding(
        "Action", size=0x18F, dtype=np.int32, one_hot_policy=OneHotPolicy.CLAMP
    )


def make_embed_char():
    return OneHotEmbedding("Character", size=0x21, dtype=np.uint8)


# puff and kirby have 6 jumps; one-hot of 7 (the +1 was fixed upstream Sep 2025)
def make_embed_jumps_left():
    return OneHotEmbedding("jumps_left", 7, dtype=np.uint8)


def _base_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
) -> list[tuple[str, Embedding]]:
    embed_xy = FloatEmbedding("xy", scale=xy_scale)

    embedding = [
        ("percent", FloatEmbedding("percent", scale=0.01)),
        ("facing", BoolEmbedding("facing", off=-1.0)),
        ("x", embed_xy),
        ("y", embed_xy),
        ("action", make_embed_action()),
        ("character", make_embed_char()),
        ("invulnerable", BoolEmbedding()),
        ("jumps_left", make_embed_jumps_left()),
        ("shield_strength", FloatEmbedding("shield_size", scale=shield_scale)),
        ("on_ground", BoolEmbedding()),
    ]

    if with_speeds:
        embed_speed = FloatEmbedding("speed", scale=speed_scale)
        embedding.extend(
            [
                ("speed_air_x_self", embed_speed),
                ("speed_ground_x_self", embed_speed),
                ("speed_y_self", embed_speed),
                ("speed_x_attack", embed_speed),
                ("speed_y_attack", embed_speed),
            ]
        )

    return embedding


def make_player_embedding(
    xy_scale: float = 0.05,
    shield_scale: float = 0.01,
    speed_scale: float = 0.5,
    with_speeds: bool = False,
    with_controller: bool = False,
    with_nana: bool = True,
) -> StructEmbedding[Player]:
    embedding = _base_player_embedding(
        xy_scale=xy_scale,
        shield_scale=shield_scale,
        speed_scale=speed_scale,
        with_speeds=with_speeds,
    )

    if with_nana:
        nana_embedding = embedding.copy()
        nana_embedding.append(("exists", BoolEmbedding()))
        embedding.append(
            ("nana", ordered_struct_embedding("nana", nana_embedding, types.Nana))
        )

    if with_controller:
        embedding.append(("controller", get_controller_embedding()))

    return ordered_struct_embedding("player", embedding, Player)


@dataclasses.dataclass
class PlayerConfig:
    xy_scale: float = 0.05
    shield_scale: float = 0.01
    speed_scale: float = 0.5
    with_speeds: bool = False
    # opponent's controller is not embedded; our own prev action is separate
    with_controller: bool = False
    with_nana: bool = True


def make_embed_stage():
    return OneHotEmbedding("Stage", size=64, dtype=np.uint8)


MAX_ITEM_TYPE = 0xEC
MAX_ITEM_STATE = 11  # empirically determined upstream


def make_item_embedding(xy_scale: float) -> StructEmbedding[Item]:
    embed_xy = FloatEmbedding("xy", scale=xy_scale)
    return struct_embedding_from_nt(
        "Item",
        Item(
            exists=BoolEmbedding(),
            type=OneHotEmbedding(
                "ItemType", size=MAX_ITEM_TYPE + 1, dtype=np.int32,
                one_hot_policy=OneHotPolicy.EXTRA,
            ),
            state=OneHotEmbedding(
                "ItemState", size=MAX_ITEM_STATE + 1, dtype=np.uint8,
                one_hot_policy=OneHotPolicy.EXTRA,
            ),
            x=embed_xy,
            y=embed_xy,
        ),
    )


class ItemsType(enum.Enum):
    SKIP = "skip"
    FLAT = "flat"
    MLP = "mlp"


@dataclasses.dataclass
class ItemsConfig:
    type: ItemsType = ItemsType.MLP
    mlp_sizes: tuple[int, ...] = (128, 32)


def make_items_embedding(items_config: ItemsConfig, xy_scale: float) -> Embedding:
    if items_config.type is ItemsType.SKIP:
        return ordered_struct_embedding("items", [], Items)

    embed_item_flat = make_item_embedding(xy_scale)

    if items_config.type is ItemsType.FLAT:
        embed_item = embed_item_flat
    elif items_config.type is ItemsType.MLP:
        embed_item = MLPWrapper(
            output_sizes=items_config.mlp_sizes, embed=embed_item_flat
        )
    else:
        raise ValueError(f"Unsupported items type: {items_config.type}")

    # All 15 slots share one embedding module (and its MLP weights).
    return ordered_struct_embedding(
        "items", [(field, embed_item) for field in Items._fields], Items
    )


def make_game_embedding(
    with_randall: bool = True,
    with_fod: bool = True,
    items_config: ItemsConfig = ItemsConfig(),
    player_config: dict = dataclasses.asdict(PlayerConfig()),
) -> StructEmbedding[Game]:
    embed_player = make_player_embedding(**player_config)

    if with_randall:
        embed_xy = FloatEmbedding("randall_xy", scale=player_config["xy_scale"])
        embed_randall = struct_embedding_from_nt("randall", Randall(x=embed_xy, y=embed_xy))
    else:
        embed_randall = ordered_struct_embedding("randall", [], Randall)

    if with_fod:
        embed_height = FloatEmbedding("fod_height", scale=player_config["xy_scale"])
        embed_fod = struct_embedding_from_nt(
            "fod", FoDPlatforms(left=embed_height, right=embed_height)
        )
    else:
        embed_fod = ordered_struct_embedding("fod", [], FoDPlatforms)

    embed_items = make_items_embedding(items_config, xy_scale=player_config["xy_scale"])

    return struct_embedding_from_nt(
        "game",
        Game(
            p0=embed_player,
            p1=embed_player,
            stage=make_embed_stage(),
            randall=embed_randall,
            fod_platforms=embed_fod,
            items=embed_items,
        ),
    )


class DiscreteEmbedding(OneHotEmbedding):
    """Buckets float inputs in [0, 1] into n+1 one-hot bins."""

    def __init__(self, n=16):
        super().__init__("DiscreteEmbedding", n + 1, dtype=np.uint8)
        self.n = n

    def from_state(self, state: np.ndarray) -> np.ndarray:
        assert state.dtype == np.float32
        return (state * self.n + 0.5).astype(self.dtype)

    def spec(self) -> tuple:
        return ("discrete", self.n, np.dtype(self.dtype).str)

    def decode(self, out: np.ndarray) -> np.ndarray:
        assert out.dtype == self.dtype
        return (out / self.n).astype(np.float32)


NATIVE_AXIS_SPACING = 160
NATIVE_SHOULDER_SPACING = 140


def make_embed_buttons() -> StructEmbedding[Buttons]:
    return ordered_struct_embedding(
        "buttons",
        [(b.value, BoolEmbedding(name=b.value)) for b in LEGAL_BUTTONS],
        Buttons,
    )


def get_controller_embedding(
    axis_spacing: int = 0,
    shoulder_spacing: int = 4,
) -> StructEmbedding[Controller]:
    """Controller embedding. Used for autoregressive sampling, so order matters."""
    if axis_spacing:
        if NATIVE_AXIS_SPACING % axis_spacing != 0:
            raise ValueError(f"Axis spacing must divide {NATIVE_AXIS_SPACING}")

        def make_axis():
            return DiscreteEmbedding(axis_spacing)
    else:
        def make_axis():
            return embed_float

    def make_stick():
        return struct_embedding_from_nt("stick", Stick(x=make_axis(), y=make_axis()))

    if NATIVE_SHOULDER_SPACING % shoulder_spacing != 0:
        raise ValueError(f"Shoulder spacing must divide {NATIVE_SHOULDER_SPACING}")

    return ordered_struct_embedding(
        "controller",
        [
            ("buttons", make_embed_buttons()),
            ("main_stick", make_stick()),
            ("c_stick", make_stick()),
            ("shoulder", DiscreteEmbedding(shoulder_spacing)),
        ],
        Controller,
    )


@dataclasses.dataclass
class ControllerConfig:
    axis_spacing: int = 16
    shoulder_spacing: int = 4

    def make_embedding(self) -> StructEmbedding[Controller]:
        return get_controller_embedding(
            axis_spacing=self.axis_spacing,
            shoulder_spacing=self.shoulder_spacing,
        )


@dataclasses.dataclass
class EmbedConfig:
    player: PlayerConfig = dataclasses.field(default_factory=PlayerConfig)
    controller: ControllerConfig = dataclasses.field(default_factory=ControllerConfig)
    with_randall: bool = True
    with_fod: bool = True
    items: ItemsConfig = dataclasses.field(default_factory=ItemsConfig)
    # Vectorized state-action forward (PackedStructForward); bitwise-identical
    # to the per-leaf path, kill switch only.
    packed: bool = True

    def make_game_embedding(self) -> StructEmbedding[Game]:
        return make_game_embedding(
            player_config=dataclasses.asdict(self.player),
            with_randall=self.with_randall,
            with_fod=self.with_fod,
            items_config=self.items,
        )


def get_state_action_embedding(
    embed_game: Embedding,
    embed_action: Embedding,
    num_names: int,
) -> StructEmbedding[StateAction]:
    return struct_embedding_from_nt(
        "state_action",
        StateAction(
            state=embed_game,
            action=embed_action,
            name=OneHotEmbedding(
                "name", num_names, dtype=NAME_DTYPE,
                one_hot_policy=OneHotPolicy.EMPTY,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Packed fast-path forward.
#
# StructEmbedding.forward dispatches one-per-leaf into a zoo of tiny modules
# (~150 ops for the full state-action embed), which dominates batch-1 play
# latency and fragments torch.compile. PackedStructForward computes the exact
# same output vector with a handful of vectorized ops. The embedding tree
# stays the schema authority; this is a compiled view of it.
# ---------------------------------------------------------------------------


def _compose(fetch, getter, key):
    def fetch_child(root, _fetch=fetch, _getter=getter, _key=key):
        return _getter(_fetch(root), _key)

    return fetch_child


class _PackedSpec(nn.Module):
    """Vectorized equivalent of a static set of scalar/one-hot leaves.

    Holds only non-persistent buffers (column indices and per-leaf constants),
    so it never appears in checkpoints.
    """

    def __init__(self, leaves: list, total_size: int):
        super().__init__()
        self.total_size = total_size
        bools = [(f, e, o) for f, e, o in leaves if isinstance(e, BoolEmbedding)]
        onehots = [(f, e, o) for f, e, o in leaves if isinstance(e, OneHotEmbedding)]
        floats = [(f, e, o) for f, e, o in leaves if isinstance(e, FloatEmbedding)]
        unknown = [e for _, e, _ in leaves if not isinstance(e, (BoolEmbedding, OneHotEmbedding, FloatEmbedding))]
        if unknown:
            raise TypeError(f"unsupported leaves in packed embedding: {unknown}")

        def buf(name, values, dtype):
            self.register_buffer(name, torch.tensor(values, dtype=dtype), persistent=False)

        self._bool_fetch = [f for f, _, _ in bools]
        buf("bool_cols", [o for _, _, o in bools], torch.long)
        buf("bool_on", [e.on for _, e, _ in bools], torch.float32)
        buf("bool_off", [e.off for _, e, _ in bools], torch.float32)

        self._float_fetch = [f for f, _, _ in floats]
        buf("float_cols", [o for _, _, o in floats], torch.long)
        # Mirror FloatEmbedding.encode exactly: (+bias?) then (*scale?), then
        # clamp guarded by truthiness (0.0 bounds disable clamping upstream).
        buf("float_bias", [e.bias if e.bias is not None else 0.0 for _, e, _ in floats], torch.float32)
        buf("float_scale", [e.scale if e.scale is not None else 1.0 for _, e, _ in floats], torch.float32)
        buf("float_lo", [e.lower if e.lower else float("-inf") for _, e, _ in floats], torch.float32)
        buf("float_hi", [e.upper if e.upper else float("inf") for _, e, _ in floats], torch.float32)

        self._onehot_fetch = [f for f, _, _ in onehots]
        buf("oh_offsets", [o for _, _, o in onehots], torch.long)
        buf("oh_sizes", [e.size for _, e, _ in onehots], torch.long)
        # EMPTY-policy leaves embed invalid ids as all-zeros; others are
        # guaranteed in-range by from_state (we clamp defensively either way).
        buf("oh_checked", [e.one_hot_policy is OneHotPolicy.EMPTY for _, e, _ in onehots], torch.bool)

        fetches = self._bool_fetch or self._float_fetch or self._onehot_fetch
        assert fetches, "packed spec must have at least one leaf"
        self._shape_fetch = fetches[0]

    def compute(self, struct) -> torch.Tensor:
        # out-of-place writes (index_copy / scatter, identical results) so the
        # whole forward is vmap-able over stacked per-slot parameters
        lead = self._shape_fetch(struct).shape
        out = torch.zeros(
            *lead, self.total_size, dtype=torch.float32, device=self.bool_cols.device
        )
        if self._bool_fetch:
            b = torch.stack([f(struct) for f in self._bool_fetch], dim=-1)
            out = out.index_copy(-1, self.bool_cols, torch.where(b, self.bool_on, self.bool_off))
        if self._float_fetch:
            x = torch.stack([f(struct).float() for f in self._float_fetch], dim=-1)
            x = torch.clamp((x + self.float_bias) * self.float_scale, self.float_lo, self.float_hi)
            out = out.index_copy(-1, self.float_cols, x)
        if self._onehot_fetch:
            idx = torch.stack([f(struct).long() for f in self._onehot_fetch], dim=-1)
            valid = (idx >= 0) & (idx < self.oh_sizes)
            src = (valid | ~self.oh_checked).float()
            dest = self.oh_offsets + torch.minimum(idx.clamp(min=0), self.oh_sizes - 1)
            out = out.scatter(-1, dest, src)
        return out


class PackedStructForward(nn.Module):
    """Drop-in replacement for `root(struct)` producing the identical tensor.

    Scalars go through two stacked vectorized paths, one-hots through a single
    scatter, and slots sharing an MLPWrapper module (the 15 item slots) run as
    one batched MLP call. Registers no parameters and only non-persistent
    buffers: checkpoints and RNG streams are unaffected.
    """

    def __init__(self, root: StructEmbedding):
        super().__init__()
        leaves: list = []

        def walk(emb, fetch):
            if isinstance(emb, StructEmbedding):
                for k, e in emb.embedding:
                    if e.size == 0:
                        continue
                    walk(e, _compose(fetch, emb.getter, k))
            else:
                leaves.append((fetch, emb))

        walk(root, lambda s: s)

        offset = 0
        main: list = []
        mlp_groups: dict[int, dict] = {}
        for fetch, emb in leaves:
            if isinstance(emb, MLPWrapper):
                grp = mlp_groups.setdefault(id(emb), {"emb": emb, "slots": []})
                grp["slots"].append((fetch, offset))
            else:
                main.append((fetch, emb, offset))
            offset += emb.size
        assert offset == root.size, (offset, root.size)

        self.spec = _PackedSpec(main, root.size)

        group_specs = []
        self._groups: list[dict] = []  # plain list: MLP modules stay registered
        for grp in mlp_groups.values():  # under the original embedding only
            emb, slots = grp["emb"], grp["slots"]
            inner: Embedding = emb._embed
            start = slots[0][1]
            for i, (_, off) in enumerate(slots):
                assert off == start + i * emb.size, "MLP slots must be contiguous"

            inner_leaves: list = []
            inner_off = 0
            for slot_fetch, _ in slots:
                base = inner_off

                def walk_inner(e, fetch):
                    nonlocal inner_off
                    if isinstance(e, StructEmbedding):
                        for k, sub in e.embedding:
                            if sub.size == 0:
                                continue
                            walk_inner(sub, _compose(fetch, e.getter, k))
                    elif isinstance(e, MLPWrapper):
                        raise TypeError("nested MLPWrapper is not supported")
                    else:
                        inner_leaves.append((fetch, e, inner_off))
                        inner_off += e.size

                walk_inner(inner, slot_fetch)
                assert inner_off - base == inner.size

            spec = _PackedSpec(inner_leaves, len(slots) * inner.size)
            group_specs.append(spec)
            self._groups.append(
                {
                    "spec_idx": len(group_specs) - 1,
                    "mlp": emb._mlp,
                    "start": start,
                    "k": len(slots),
                    "in_size": inner.size,
                    "out_size": emb.size,
                }
            )
        self.group_specs = nn.ModuleList(group_specs)

    def forward(self, struct) -> torch.Tensor:
        out = self.spec.compute(struct)
        for g in self._groups:
            x = self.group_specs[g["spec_idx"]].compute(struct)
            x = x.reshape(*x.shape[:-1], g["k"], g["in_size"])
            y = g["mlp"](x)
            width = g["k"] * g["out_size"]
            out[..., g["start"] : g["start"] + width] = y.reshape(*y.shape[:-2], width)
        return out
