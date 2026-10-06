"""Typed configs for SmashBot. Mirrors slippi-ai's nested flag structure."""

import dataclasses

from slippi_ai.data import DatasetConfig


def from_dict(cls, saved: dict):
    """A config from a checkpoint's saved dict. A key the class no longer
    has is a setting whose behaviour became unconditional (gate_gelu, v_norm)
    and is dropped; a key the checkpoint predates takes its default."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in saved.items() if k in fields})


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
    # sgu only: the tiny attention's shape (aMLP's is one head of 64)
    attn_heads: int = 1
    attn_head_dim: int = 64
    # sgu only: one letter per layer, s = SGU block, g = GRU, l = LSTM (a
    # residual fp32 cell in place of the conv + attention mixing); empty = all SGU
    layout: str = ""
    # sgu only: RMS-normalize each tiny-attention head's q and k (QK-norm), so
    # its logits stay bounded however attn_qkv grows; the value net follows
    qk_norm: bool = False
    # tx_like only: the FFW blocks' GELU as the tanh approximation
    gelu_approximate: bool = False
    # the input embedding: "simple" (every leaf's default embedding) or
    # "enhanced" (embed.EnhancedEmbed: learned action and character vectors,
    # items summed), the big RL Phillips'
    embed: str = "simple"
    embed_hidden_size: int = 128
    # enhanced only: a constant slippi ranked rating input; None = no input
    rating: float | None = None
    # !!! enhanced only: reproduces slippi-ai's uint8 character-action index
    # BUG for the ported Phillips, which were trained on it. NEVER set it for a
    # model we train (see embed.EnhancedEmbed's NOTE).
    embed_joint_index_wraps: bool = False
    # the opponent's tech hidden as a neutral tech for its first frames
    # (slippi-ai's AnimationFilter; the big RL Phillips' observation); 0 = off
    tech_mask_window: int = 0


@dataclasses.dataclass
class ControllerHeadConfig:
    residual_size: int = 128
    component_depth: int = 2
    axis_spacing: int = 16  # 17 bins per stick axis
    shoulder_spacing: int = 4  # 5 shoulder bins
    # or "custom_v1" (smashbot.custom_v1), or "balanced_v6_157": each stick one
    # choice among embed.STICK_TABLES' buckets (157 main, 75 c-stick)
    controller_type: str = "default"


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
    # sgu only: one letter per layer as network.layout; empty = all SGU
    layout: str = ""
    reward_halflife: float = 4.0  # seconds; discount = 0.5 ** (1 / (halflife * 60))
    # "policy": the policy network's input embedding and tech mask; "simple":
    # one-hot leaves, unmasked (what every value net before 2026-10-02 had,
    # whatever the policy used: configs saved without this key mean it)
    inputs: str = "policy"


@dataclasses.dataclass
class LearnerConfig:
    learning_rate: float = 1e-4
    # Faithful slippi-ai defaults: fp32, no clipping.
    max_grad_norm: float = 0.0
    # AutoClip (Seetharaman et al. 2020): clip each network to this percentile
    # of its own gradient-norm history instead of a constant; 0 = off
    autoclip_percentile: float = 0.0
    precision: str = "fp32"  # bf16 | fp32
    compile: bool = False
    # BC: add these weights x each joint stick's energy score
    # (policy.EnergyScore) and action loss (policy.ActionLoss) to the loss;
    # they need a joint stick controller type; 0 = off
    energy_score_weight: float = 0.0
    action_loss_weight: float = 0.0
