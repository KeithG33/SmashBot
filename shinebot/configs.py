"""Typed configs for ShineBot. Mirrors slippi-ai's nested flag structure."""

import dataclasses

from slippi_ai.data import DatasetConfig


@dataclasses.dataclass
class DataConfig:
    """Wraps slippi-ai's DatasetConfig plus DataSource/bridge options."""

    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)

    batch_size: int = 512
    unroll_length: int = 80
    # DataSource decode workers (0 = decode in main process).
    num_workers: int = 8
    damage_ratio: float = 0.01
    # Chunks start at a random offset within [0, random_offset) of each game.
    random_offset: int = 0
    balance_characters: bool = False
    max_names: int = 16

    # Torch bridge
    prefetch: int = 4
    pin_memory: bool = True


@dataclasses.dataclass
class PolicyConfig:
    delay: int = 18


@dataclasses.dataclass
class NetworkConfig:
    hidden_size: int = 512
    num_layers: int = 3
    ffw_multiplier: int = 2
    recurrent_layer: str = "lstm"  # or "gru"


@dataclasses.dataclass
class ControllerHeadConfig:
    residual_size: int = 128
    component_depth: int = 2
    axis_spacing: int = 16  # 17 bins per stick axis
    shoulder_spacing: int = 4  # 5 shoulder bins


@dataclasses.dataclass
class ValueConfig:
    hidden_size: int = 512
    num_layers: int = 1
    reward_halflife: float = 4.0  # seconds; discount = 0.5 ** (1 / (halflife * 60))


@dataclasses.dataclass
class LearnerConfig:
    learning_rate: float = 1e-4
    value_cost: float = 0.5
