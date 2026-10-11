"""ProcessBatchStream is TorchBatchStream with its producer in a process of
its own: the same encoded batches, epochs and stream states, from a fresh
start and from a resume state; a producer failure surfaces in the trainer,
and stop() leaves no producer behind."""
import dataclasses

import pytest
import torch
import tree

from slippi_ai import data as data_lib
from slippi_ai.paths import TOY_DATASET

from smashbot import configs, embed as embed_lib
from smashbot.data import loader
from smashbot.policy import build_policy

DELAY = 3
EXTRA_FRAMES = DELAY + 1
STEPS = 12   # the toy game is 6402 frames: at unroll 1024 the rows roll onto their next replay at step 7


def _config(**kwargs) -> configs.DataConfig:
    return configs.DataConfig(
        dataset=data_lib.DatasetConfig(dataset_path=str(TOY_DATASET), mirror=True),
        batch_size=2, unroll_length=1024, prefetch=2, pin_memory=False, **kwargs)


@pytest.fixture(scope="module")
def network():
    return build_policy(
        embed_config=embed_lib.EmbedConfig(), controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(hidden_size=32, num_layers=1),
        head_config=configs.ControllerHeadConfig(residual_size=16, component_depth=1),
        policy_config=configs.PolicyConfig(delay=DELAY), num_names=configs.DataConfig().max_names).network


def _sources(cfg, **kwargs):
    sources = loader.make_sources(cfg, EXTRA_FRAMES, **kwargs)
    sources.test.shutdown()
    return sources


def _take(stream, n: int) -> list:
    try:
        return [next(stream) for _ in range(n)]
    finally:
        stream.stop()
        if isinstance(stream, loader.ProcessBatchStream):
            assert not stream._process.is_alive()


def _assert_same(a: list, b: list) -> None:
    for (frames_a, epoch_a, state_a), (frames_b, epoch_b, state_b) in zip(a, b, strict=True):
        assert epoch_a == epoch_b and state_a == state_b
        for x, y in zip(tree.flatten(frames_a), tree.flatten(frames_b), strict=True):
            assert x.dtype == y.dtype and torch.equal(x, y)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_process_stream_matches_the_thread_stream(network, num_workers):
    cfg = _config(num_workers=num_workers)
    thread = _take(loader.make_train_stream(_sources(cfg), cfg, EXTRA_FRAMES, None, network), STEPS)
    sources = _sources(cfg, train=False)
    assert sources.train is None
    process = _take(loader.make_train_stream(sources, cfg, EXTRA_FRAMES, None, network), STEPS)
    assert isinstance(process[0][0].state_action, type(thread[0][0].state_action))
    _assert_same(thread, process)


def test_process_stream_resumes_as_the_thread_stream_does(network):
    cfg = _config(num_workers=0)
    first = _take(loader.make_train_stream(_sources(cfg), cfg, EXTRA_FRAMES, None, network), 5)
    state, name_map = first[-1][2], _sources(cfg).name_map
    thread = _take(loader.make_train_stream(_sources(cfg, name_map=name_map, train_state=state),
                                            cfg, EXTRA_FRAMES, state, network), STEPS - 5)
    process = _take(loader.make_train_stream(_sources(cfg, name_map=name_map, train=False),
                                             cfg, EXTRA_FRAMES, state, network), STEPS - 5)
    _assert_same(thread, process)


def test_a_producer_failure_surfaces_in_the_trainer(network, tmp_path):
    missing = dataclasses.replace(_config(num_workers=0),
                                  dataset=data_lib.DatasetConfig(dataset_path=str(tmp_path / "missing")))
    stream = loader.ProcessBatchStream(missing, EXTRA_FRAMES, {}, None, network)
    with pytest.raises(RuntimeError, match="batch producer failed"):
        next(stream)
    stream.stop()
    assert not stream._process.is_alive()
