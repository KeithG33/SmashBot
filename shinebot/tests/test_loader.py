"""Tests for the torch bridge over slippi-ai's data pipeline.

Uses slippi-ai's in-repo toy dataset so tests run anywhere.
"""

import numpy as np
import pytest
import torch
import tree

from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from shinebot.configs import DataConfig
from shinebot.data import loader

DELAY = 3  # small delay for tests
EXTRA_FRAMES = DELAY + 1


def toy_config(**kwargs) -> DataConfig:
    defaults = dict(
        batch_size=2,
        unroll_length=16,
        num_workers=0,
        prefetch=2,
        pin_memory=False,
    )
    defaults.update(kwargs)
    return DataConfig(
        dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET)),
        **defaults,
    )


@pytest.fixture(scope="module")
def sources():
    return loader.make_sources(toy_config(), extra_frames=EXTRA_FRAMES)


def test_batch_structure_and_dtypes(sources):
    batch_with_meta, _ = next(sources.train)
    torch_batch = loader.batch_to_torch(batch_with_meta.batch, pin=False)

    chunk = 16 + EXTRA_FRAMES
    assert torch_batch.game.stage.shape == (2, chunk)
    assert torch_batch.reward.shape == (2, chunk - 1)
    assert torch_batch.is_resetting.dtype == torch.bool
    # uint16 must widen to int32 (torch has no uint16)
    assert torch_batch.game.p0.action.dtype == torch.int32
    assert torch_batch.game.p0.percent.dtype == torch.int32
    assert torch_batch.game.p0.x.dtype == torch.float32

    # Values survive the conversion exactly.
    np_leaves = tree.flatten(batch_with_meta.batch)
    t_leaves = tree.flatten(torch_batch)
    for np_leaf, t_leaf in zip(np_leaves, t_leaves):
        np.testing.assert_array_equal(np.asarray(np_leaf), t_leaf.numpy())


def test_conversion_copies_buffers(sources):
    """The accumulator reuses numpy buffers; torch batches must not alias them."""
    batch_with_meta, _ = next(sources.train)
    torch_batch = loader.batch_to_torch(batch_with_meta.batch, pin=False)
    before = torch_batch.game.p0.x.clone()
    # Trigger buffer reuse.
    next(sources.train)
    next(sources.train)
    torch.testing.assert_close(before, torch_batch.game.p0.x)


def test_chunk_overlap_semantics():
    """Consecutive chunks from one manager overlap by extra_frames."""
    cfg = toy_config(batch_size=1)
    sources = loader.make_sources(cfg, extra_frames=EXTRA_FRAMES)
    b1, _ = next(sources.train)
    b2, _ = next(sources.train)
    (s1, e1), (s2, e2) = (
        (b1.meta.start[0], b1.meta.end[0]),
        (b2.meta.start[0], b2.meta.end[0]),
    )
    if b2.batch.is_resetting[0, 0]:
        pytest.skip("game ended between chunks; overlap not applicable")
    assert s2 == e1 - EXTRA_FRAMES


def test_swap_doubles_replays():
    cfg = toy_config()
    replays = data_lib.replays_from_meta(cfg.dataset)
    cfg_noswap = toy_config()
    cfg_noswap.dataset.swap = False
    replays_noswap = data_lib.replays_from_meta(cfg_noswap.dataset)
    assert len(replays) == 2 * len(replays_noswap)
    swapped = [r for r in replays if r.swap]
    assert len(swapped) == len(replays_noswap)


def test_prefetch_stream():
    cfg = toy_config()
    sources = loader.make_sources(cfg, extra_frames=EXTRA_FRAMES)
    stream = loader.TorchBatchStream(sources.train, cfg)
    try:
        for _ in range(5):
            batch, epoch = next(stream)
            assert batch.game.stage.shape == (2, 16 + EXTRA_FRAMES)
            assert isinstance(epoch, float)
    finally:
        stream.stop()
