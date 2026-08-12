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
    name: str = "tx_like"  # tx_like | transformer | sgu
    hidden_size: int = 512
    num_layers: int = 3
    ffw_multiplier: int = 2
    recurrent_layer: str = "lstm"  # or "gru" (tx_like only)
    # LayerNorm epsilon in tx_like ResBlocks. slippi-ai's LayerNorm has no
    # epsilon, so checkpoints ported from TF use 0.0; ours keep torch's 1e-5.
    ln_eps: float = 1e-5
    # transformer only:
    num_heads: int = 8
    window: int = 256  # KV-cache length (frames of memory carried at play time)


@dataclasses.dataclass
class ControllerHeadConfig:
    residual_size: int = 128
    component_depth: int = 2
    axis_spacing: int = 16  # 17 bins per stick axis
    shoulder_spacing: int = 4  # 5 shoulder bins


@dataclasses.dataclass
class ValueConfig:
    # slippi-ai pattern: value = smaller instance of the POLICY's family.
    # "match" mirrors the policy core's name/window/heads at num_layers depth.
    name: str = "match"  # match | tx_like | transformer | sgu
    hidden_size: int = 512
    num_layers: int = 1
    # 0 = inherit the policy's window (back-compat with old checkpoints).
    # Long windows mildly hurt value estimation (uev 0.337 @W256 vs 0.325
    # @W64), so big trains pass an explicit smaller window here.
    window: int = 0
    reward_halflife: float = 4.0  # seconds; discount = 0.5 ** (1 / (halflife * 60))


@dataclasses.dataclass
class LearnerConfig:
    learning_rate: float = 1e-4
    value_cost: float = 0.5
    # Faithful slippi-ai defaults: fp32, no clipping. The bf16+clip experiment
    # (debug-fox-v0-bf16) tracked slightly worse on eval and value loss; revisit
    # with seeded A/Bs if the +35% throughput is ever needed.
    max_grad_norm: float = 0.0
    precision: str = "fp32"  # bf16 | fp32
    # Measured SLOWER than eager (125k vs 152k frames/s): dynamo graph-breaks
    # on tree-structured code + cuDNN LSTM boundary. Left as opt-in.
    compile: bool = False
