"""PyTorch bridge over slippi-ai's (pure-numpy, reused) data pipeline.

slippi-ai's loader already does everything up to numpy batches:
  meta.json -> ReplayInfo list (character/name filters, swap doubling)
  -> train_test_split -> DataSource (multiprocess parquet decode,
  per-manager sequential chunking with `extra_frames` overlap,
  preallocated BatchAccumulator) -> Batch(game, name, is_resetting, reward)

This module adds the only missing pieces:
  * name_map construction (ported from tf/train_lib.create_name_map)
  * numpy -> torch conversion (with copies: the accumulator reuses buffers)
  * a background prefetch thread producing pinned-memory batches
"""

import collections
import dataclasses
import queue
import threading
import typing as tp

import numpy as np
import torch
import tree

from slippi_ai import data as data_lib
from slippi_ai import nametags

from shinebot.configs import DataConfig

# torch has no uint16; actions/percents must widen.
_DTYPE_MAP = {np.dtype(np.uint16): np.int32}


def _to_torch(x: np.ndarray, pin: bool) -> torch.Tensor:
    target = _DTYPE_MAP.get(x.dtype)
    if target is not None:
        x = x.astype(target)
    t = torch.from_numpy(x)
    # pin_memory() copies, which also detaches us from the accumulator's
    # reused buffers. Without pinning we must copy explicitly.
    return t.pin_memory() if pin else t.clone()


def batch_to_torch(batch: data_lib.Batch, pin: bool) -> data_lib.Batch:
    return tree.map_structure(lambda x: _to_torch(x, pin), batch)


def create_name_map(
    replays: list[data_lib.ReplayInfo],
    max_names: int,
) -> dict[str, int]:
    # Ported from slippi_ai/tf/train_lib.py (module imports TF, function doesn't).
    name_map: dict[str, int] = {}
    name_counts = collections.Counter(
        nametags.normalize_name(replay.main_player.name) for replay in replays
    )
    for i, (name, _) in enumerate(name_counts.most_common(max_names)):
        name_map[name] = i

    for first, *rest in nametags.NAME_GROUPS:
        if first not in name_map:
            continue
        for name in rest:
            name_map[name] = name_map[first]

    return name_map


@dataclasses.dataclass
class Sources:
    train: data_lib.DataSource
    test: data_lib.DataSource
    name_map: dict[str, int]


def make_sources(config: DataConfig, extra_frames: int) -> Sources:
    """Build train/test DataSources. extra_frames must be policy.delay + 1."""
    train_replays, test_replays = data_lib.train_test_split(config.dataset)
    name_map = create_name_map(train_replays, config.max_names)

    def make(replays: list[data_lib.ReplayInfo]) -> data_lib.DataSource:
        return data_lib.DataSource(
            replays=replays,
            batch_size=config.batch_size,
            unroll_length=config.unroll_length,
            extra_frames=extra_frames,
            random_offset=config.random_offset,
            damage_ratio=config.damage_ratio,
            balance_characters=config.balance_characters,
            name_map=name_map,
            num_workers=config.num_workers,
        )

    return Sources(train=make(train_replays), test=make(test_replays), name_map=name_map)


def batch_to_frames(batch: data_lib.Batch, network, pin: bool = False):
    """Training-path glue: numpy Batch [B, T] -> encoded, time-major torch Frames.

    Mirrors slippi-ai's TrainManager.produce_frames: the p0 controller becomes
    the action stream, the network's embedding encodes (discretizes) it, and
    everything transposes to time-major.
    """
    from slippi_ai.types import Frames, StateAction

    if np.any(np.asarray(batch.is_resetting)[:, 1:]):
        raise ValueError("Unexpected mid-episode reset.")

    state_action = StateAction(
        state=batch.game, action=batch.game.p0.controller, name=batch.name
    )
    state_action = network.encode(state_action)  # numpy

    frames = data_lib.Frames(
        state_action=state_action,
        is_resetting=batch.is_resetting,
        reward=batch.reward,
    )
    frames = tree.map_structure(lambda x: np.asarray(x).swapaxes(0, 1), frames)
    return tree.map_structure(lambda x: _to_torch(np.ascontiguousarray(x), pin), frames)


class TorchBatchStream:
    """Background thread: pulls numpy batches, converts to (pinned) torch.

    With `encode_network` set, instead yields encoded, time-major Frames ready
    for Policy.imitation_loss (mirrors slippi-ai's TrainManager.produce_frames).
    """

    def __init__(
        self,
        source: data_lib.AbstractDataSource,
        config: DataConfig,
        encode_network=None,
    ):
        self._source = source
        self._network = encode_network
        self._pin = config.pin_memory and torch.cuda.is_available()
        self._queue: queue.Queue = queue.Queue(maxsize=config.prefetch)
        self._stop = threading.Event()
        self._error: tp.Optional[BaseException] = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self) -> None:
        try:
            while not self._stop.is_set():
                batch_with_meta, epoch = next(self._source)
                if self._network is not None:
                    item = batch_to_frames(
                        batch_with_meta.batch, self._network, pin=self._pin
                    )
                else:
                    item = batch_to_torch(batch_with_meta.batch, self._pin)
                while not self._stop.is_set():
                    try:
                        self._queue.put((item, epoch), timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            self._error = e

    def __iter__(self):
        return self

    def __next__(self) -> tuple[data_lib.Batch, float]:
        while True:
            try:
                return self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._error is not None:
                    raise self._error
                if not self._thread.is_alive():
                    raise StopIteration

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._source.shutdown()
