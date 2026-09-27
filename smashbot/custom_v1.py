"""slippi-ai's custom_v1 action space (upstream slippi_ai/action_space/
custom_v1.py at 275c072, MIT), which the big RL Phillips sample in.

A controller becomes two labels. Buttons: B, X|Y, L|R, a combined Z/A/shoulder
bucket (7: A x {no, light, full} shoulder, or Z, which implies A and a light
shoulder) and the c-stick in polar buckets (the origin, 4 at a small radius, 8
at a large one), 2 * 2 * 2 * 7 * 13 = 728. Main stick: polar buckets with
[1, 4, 16, 64] angles over the origin and three log-spaced radii, 85."""
import dataclasses
import enum
import typing as tp

import numpy as np

from slippi_ai.types import Buttons, Controller, Stick


class AnalogShoulder(enum.IntEnum):
    NONE = 0
    LIGHT = 1
    FULL = 2


LIGHT_SHOULDER_THRESHOLD = np.float32(0.3)
FULL_SHOULDER_THRESHOLD = np.float32(0.9)
SHOULDER_TABLE = np.array([0, 0.35, 1], dtype=np.float32)   # 0.35: what pressing Z reads


def bucket_analog_shoulder(shoulder: np.ndarray) -> np.ndarray:
    buckets = np.zeros_like(shoulder, dtype=np.uint8)
    buckets[shoulder > LIGHT_SHOULDER_THRESHOLD] = AnalogShoulder.LIGHT
    buckets[shoulder > FULL_SHOULDER_THRESHOLD] = AnalogShoulder.FULL
    return buckets


class Cartesian:
    """Labels for a product of small axes, the last one fastest."""

    def __init__(self, axis_specs: tp.Sequence[tuple[int, type]]):
        self.axis_specs = axis_specs
        self.axis_sizes = [size for size, _ in axis_specs]
        self.num_labels = int(np.prod(self.axis_sizes))

    def flatten(self, components: tp.Sequence[np.ndarray]) -> np.ndarray:
        label = np.zeros(components[0].shape, dtype=np.uint16)
        for size, component in zip(self.axis_sizes, components):
            assert np.all(component < size)
            label *= np.uint16(size)
            label += component.astype(np.uint16)
        return label

    def unflatten(self, label: np.ndarray) -> list[np.ndarray]:
        components = []
        for size, dtype in reversed(self.axis_specs):
            label, component = np.divmod(label, size)
            components.append(component.astype(dtype))
        components.reverse()
        return components


a_and_shoulder_bucketer = Cartesian([(2, bool), (len(AnalogShoulder), np.uint8)])
Z_LABEL = np.uint8(a_and_shoulder_bucketer.num_labels)
Z_A_SHOULDER_SIZE = a_and_shoulder_bucketer.num_labels + 1


def bucket_z_a_shoulder(controller: Controller) -> np.ndarray:
    combined = a_and_shoulder_bucketer.flatten(
        (controller.buttons.A, bucket_analog_shoulder(controller.shoulder)))
    combined[controller.buttons.Z] = Z_LABEL
    return combined


def decode_z_a_shoulder(bucket: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = bucket == Z_LABEL
    no_z_a, no_z_shoulder = a_and_shoulder_bucketer.unflatten(bucket[~z])
    a = np.zeros_like(bucket, dtype=bool)
    a[~z] = no_z_a.astype(bool)
    a[z] = True
    shoulder_label = np.zeros_like(bucket, dtype=np.uint8)
    shoulder_label[~z] = no_z_shoulder.astype(np.uint8)
    shoulder_label[z] = AnalogShoulder.LIGHT
    return z, a, SHOULDER_TABLE[shoulder_label]


def stick_to_raw(value: np.ndarray) -> np.ndarray:
    """[0, 1] -> raw controller units [-80, 80]."""
    return np.rint(value * 160 - 80).astype(np.int16)


def stick_from_raw(value: np.ndarray) -> np.ndarray:
    return ((value + 80) / 160).astype(np.float32)


MIN_NONZERO_RADIUS = 23   # raw units
MAX_RADIUS = 80
min_log_radius = np.log(MIN_NONZERO_RADIUS)
max_log_radius = np.log(MAX_RADIUS)


def build_radius_table(n_radius_buckets: int) -> np.ndarray:
    nonzero = np.exp(np.linspace(min_log_radius, max_log_radius, n_radius_buckets))
    return np.concatenate(([0.0], nonzero)).astype(np.float32)


def build_angle_table(n_angle_buckets: int) -> np.ndarray:
    return (np.arange(n_angle_buckets) / n_angle_buckets * 2 * np.pi - np.pi).astype(np.float32)


class RaggedCartesian:
    """Labels for (outer, inner) where each outer value has its own inner size."""

    def __init__(self, bucket_sizes: tp.Sequence[int]):
        self.radius_label_offset = np.cumsum([0, *bucket_sizes[:-1]])
        self.num_labels = int(sum(bucket_sizes))
        self.label_to_outer = np.concatenate(
            [np.full(size, i, dtype=np.uint16) for i, size in enumerate(bucket_sizes)])
        self.label_to_inner = np.concatenate(
            [np.arange(size, dtype=np.uint16) for size in bucket_sizes])

    def to_label(self, outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
        return self.radius_label_offset[outer] + inner

    def from_label(self, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.label_to_outer[label], self.label_to_inner[label]


class PolarStickBucketer:

    def __init__(self, n_radius_buckets: int, n_angle_buckets: tp.Sequence[int]):
        if len(n_angle_buckets) != n_radius_buckets:
            raise ValueError(f"n_angle_buckets {n_angle_buckets} must match n_radius_buckets {n_radius_buckets}")
        self.n_radius_buckets = n_radius_buckets
        self.radius_table = build_radius_table(n_radius_buckets)
        self.n_angle_buckets_table = np.array([1, *n_angle_buckets])
        self.angle_table = np.concatenate([build_angle_table(n) for n in self.n_angle_buckets_table])
        self.ragged_cartesian = RaggedCartesian(self.n_angle_buckets_table)
        self.num_labels = self.ragged_cartesian.num_labels

    def bucket(self, stick: Stick) -> np.ndarray:
        raw_x, raw_y = stick_to_raw(stick.x), stick_to_raw(stick.y)
        radius = np.sqrt(raw_x.astype(np.float32) ** 2 + raw_y.astype(np.float32) ** 2)
        normalized_radius = (np.log(radius + 1e-3) - min_log_radius) / (max_log_radius - min_log_radius)
        nonzero_radius_bucket = np.rint(normalized_radius * (self.n_radius_buckets - 1)).astype(np.uint16)
        radius_bucket = np.where(radius <= MIN_NONZERO_RADIUS - 1, 0, nonzero_radius_bucket + 1).astype(np.uint16)
        n_angle_buckets = self.n_angle_buckets_table[radius_bucket]
        normalized_angle = (np.arctan2(raw_y, raw_x) + np.pi) / (2 * np.pi)
        angle_bucket = np.mod(np.rint(normalized_angle * n_angle_buckets).astype(np.uint16), n_angle_buckets)
        return self.ragged_cartesian.to_label(radius_bucket, angle_bucket.astype(np.uint16))

    def decode(self, label: np.ndarray) -> Stick:
        radius_bucket, _ = self.ragged_cartesian.from_label(label)
        raw_radius, angle = self.radius_table[radius_bucket], self.angle_table[label]
        return Stick(stick_from_raw(raw_radius * np.cos(angle)), stick_from_raw(raw_radius * np.sin(angle)))


@dataclasses.dataclass
class PolarStickConfig:
    n_radius_buckets: int
    n_angle_buckets: tp.Sequence[int]


class ButtonCombination(tp.NamedTuple):
    b: np.ndarray
    xy: np.ndarray
    lr: np.ndarray
    z_a_shoulder: np.ndarray
    c_stick: np.ndarray


class ControllerV1(tp.NamedTuple):
    buttons: np.ndarray      # [0, 728)
    main_stick: np.ndarray   # [0, 85)


class ControllerBucketer:

    def __init__(self, c_stick: PolarStickConfig, main_stick: PolarStickConfig):
        self.c_stick = PolarStickBucketer(c_stick.n_radius_buckets, c_stick.n_angle_buckets)
        self.main_stick = PolarStickBucketer(main_stick.n_radius_buckets, main_stick.n_angle_buckets)
        self.buttons = Cartesian(ButtonCombination(
            b=(2, bool), xy=(2, bool), lr=(2, bool),
            z_a_shoulder=(Z_A_SHOULDER_SIZE, np.uint8), c_stick=(self.c_stick.num_labels, np.uint16)))
        self.sizes = ControllerV1(self.buttons.num_labels, self.main_stick.num_labels)

    def bucket(self, controller: Controller) -> ControllerV1:
        b = controller.buttons
        combo = ButtonCombination(
            b=b.B, xy=b.X | b.Y, lr=b.L | b.R,
            z_a_shoulder=bucket_z_a_shoulder(controller), c_stick=self.c_stick.bucket(controller.c_stick))
        return ControllerV1(self.buttons.flatten(combo), self.main_stick.bucket(controller.main_stick))

    def decode(self, labels: ControllerV1) -> Controller:
        combo = ButtonCombination(*self.buttons.unflatten(labels.buttons))
        z, a, shoulder = decode_z_a_shoulder(combo.z_a_shoulder)
        no = np.zeros_like(combo.b, dtype=bool)
        return Controller(
            main_stick=self.main_stick.decode(labels.main_stick),
            c_stick=self.c_stick.decode(combo.c_stick),
            shoulder=shoulder,
            buttons=Buttons(A=a, B=combo.b, X=no, Y=combo.xy, Z=z, L=combo.lr, R=no, D_UP=no))


@dataclasses.dataclass
class Config:
    c_stick: PolarStickConfig = dataclasses.field(
        default_factory=lambda: PolarStickConfig(n_radius_buckets=2, n_angle_buckets=[4, 8]))
    main_stick: PolarStickConfig = dataclasses.field(
        default_factory=lambda: PolarStickConfig(n_radius_buckets=3, n_angle_buckets=[4, 16, 64]))

    def create_bucketer(self) -> ControllerBucketer:
        return ControllerBucketer(self.c_stick, self.main_stick)
