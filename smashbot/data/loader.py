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
  * exact stream position (per-row replay + frame) for checkpoint resume
"""

import collections
import dataclasses
import multiprocessing as mp
import queue
import threading
import typing as tp

import numpy as np
import torch
import tree

from slippi_ai import data as data_lib
from slippi_ai import nametags

from smashbot.configs import DataConfig

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


class _Cursor:
    """Count of replays handed to managers, in the split's cycle order."""

    def __init__(self, count: int = 0):
        self.count = count


class _Feed:
    """One manager's view of the shared decoded-replay iterator.

    `index` is the cycle position of the replay the manager currently holds,
    which is all a resume needs to re-decode it.
    """

    def __init__(self, inner, cursor: _Cursor):
        self.inner = inner
        self.cursor = cursor
        self.index = -1

    def __next__(self):
        self.index = self.cursor.count
        self.cursor.count += 1
        return next(self.inner)


@dataclasses.dataclass
class Split:
    """A DataSource plus what it takes to rebuild it mid-stream.

    state() after a batch is the exact position of the stream: which replay
    each row holds and its frame within it, plus how many replays the cycle
    has handed out. make_sources(..., state=) resumes from it bit-identically.
    """

    source: data_lib.DataSource
    replays: list[data_lib.ReplayInfo]
    cursor: _Cursor
    feeds: list[_Feed]

    def __next__(self):
        return next(self.source)

    def shutdown(self):
        self.source.shutdown()

    def state(self) -> dict:
        rows = [(feed.index, m.frame) for feed, m in zip(self.feeds, self.source.managers)]
        return {"consumed": self.cursor.count, "rows": rows}

    def load_rows(self, rows: list[tuple[int, int]], num_workers: int) -> None:
        infos = [self.replays[index % len(self.replays)] for index, _ in rows]
        if num_workers > 0:
            with mp.get_context("forkserver").Pool(num_workers) as pool:
                decoded = pool.map(data_lib.ReplayInfo.to_replay, infos)
        else:
            decoded = [info.to_replay() for info in infos]
        for manager, feed, replay, (index, frame) in zip(
            self.source.managers, self.feeds, decoded, rows
        ):
            manager.source = iter([replay])
            manager.find_game()
            manager.frame = frame
            manager.source = feed
            feed.index = index


@dataclasses.dataclass
class Sources:
    train: Split
    test: Split
    name_map: dict[str, int]


def _make_split(
    replays: list[data_lib.ReplayInfo],
    config: DataConfig,
    extra_frames: int,
    name_map: dict[str, int],
    state: tp.Optional[dict],
) -> Split:
    consumed = state["consumed"] if state else 0
    if consumed and config.balance_characters:
        raise ValueError("resume needs a plain replay cycle; balance_characters interleaves")
    offset = consumed % len(replays)
    source = data_lib.DataSource(
        replays=replays[offset:] + replays[:offset],
        batch_size=config.batch_size,
        unroll_length=config.unroll_length,
        extra_frames=extra_frames,
        random_offset=config.random_offset,
        damage_ratio=config.damage_ratio,
        balance_characters=config.balance_characters,
        name_map=name_map,
        num_workers=config.num_workers,
    )
    source.replay_counter += consumed
    cursor = _Cursor(consumed)
    feeds = [_Feed(source.replay_ds, cursor) for _ in source.managers]
    for manager, feed in zip(source.managers, feeds):
        manager.source = feed
    split = Split(source=source, replays=replays, cursor=cursor, feeds=feeds)
    if state and state.get("rows"):
        split.load_rows(state["rows"], config.num_workers)
    return split


def make_sources(
    config: DataConfig,
    extra_frames: int,
    name_map: tp.Optional[dict[str, int]] = None,
    train_state: tp.Optional[dict] = None,
    test_state: tp.Optional[dict] = None,
) -> Sources:
    """Build train/test splits. extra_frames must be policy.delay + 1.

    name_map: pass the checkpoint's map when resuming; indices are assigned
    by frequency, so recomputing on changed data would silently permute them.
    train_state / test_state: Split.state() snapshots to resume from. A
    snapshot without rows (checkpoints that predate row tracking) only
    rotates the cycle; every row then restarts at frame 0 of a fresh replay.
    """
    train_replays, test_replays = data_lib.train_test_split(config.dataset)
    if name_map is None:
        name_map = create_name_map(train_replays, config.max_names)
    return Sources(
        train=_make_split(train_replays, config, extra_frames, name_map, train_state),
        test=_make_split(test_replays, config, extra_frames, name_map, test_state),
        name_map=name_map,
    )


def batch_to_frames(batch: data_lib.Batch, network, pin: bool = False):
    """Training-path glue: numpy Batch -> encoded torch Frames, batch-major [B, T].

    Mirrors slippi-ai's TrainManager.produce_frames (minus their time-major
    transpose — all SmashBot tensors are (batch, time, ...)): the p0 controller
    becomes the action stream and the network's embedding encodes it.
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
    return tree.map_structure(lambda x: _to_torch(np.ascontiguousarray(x), pin), frames)


class TorchBatchStream:
    """Background thread: pulls numpy batches, converts to (pinned) torch.

    With `encode_network` set, instead yields encoded, time-major Frames ready
    for Policy.imitation_loss (mirrors slippi-ai's TrainManager.produce_frames).
    Each item carries the Split.state() it was produced from, so a checkpoint
    can record the stream position of the batch the learner actually consumed
    rather than whatever the prefetch thread has run ahead to.
    """

    def __init__(
        self,
        source: Split,
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
                state = self._source.state()
                if self._network is not None:
                    item = batch_to_frames(
                        batch_with_meta.batch, self._network, pin=self._pin
                    )
                else:
                    item = batch_to_torch(batch_with_meta.batch, self._pin)
                while not self._stop.is_set():
                    try:
                        self._queue.put((item, epoch, state), timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:
            self._error = e

    def __iter__(self):
        return self

    def __next__(self) -> tuple[data_lib.Batch, float, dict]:
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
