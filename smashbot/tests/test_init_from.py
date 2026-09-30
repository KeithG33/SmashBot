"""--runtime.init-from starts a fresh run on a checkpoint's policy and value
weights (fresh optimizer, data and step), and refuses weights built for
another model config."""

import dataclasses

import pytest

from smashbot import train_bc
from smashbot.tests.test_resume import _assert_same, _config, _latest


def test_fresh_run_starts_on_the_given_weights(tmp_path):
    train_bc.main(_config(str(tmp_path), "source", 3))
    cfg = _config(str(tmp_path), "warm", 0)
    cfg = dataclasses.replace(cfg, runtime=dataclasses.replace(
        cfg.runtime, init_from=f"{tmp_path}/source/latest.pt"))
    train_bc.main(cfg)
    source, warm = _latest(str(tmp_path), "source"), _latest(str(tmp_path), "warm")
    _assert_same(source["policy"], warm["policy"], "policy")
    _assert_same(source["value"], warm["value"], "value")
    assert warm["step"] == 0


def test_refuses_weights_for_another_model(tmp_path):
    train_bc.main(_config(str(tmp_path), "source", 1))
    cfg = _config(str(tmp_path), "warm", 1)
    cfg = dataclasses.replace(
        cfg, policy=dataclasses.replace(cfg.policy, delay=cfg.policy.delay + 3),
        runtime=dataclasses.replace(cfg.runtime, init_from=f"{tmp_path}/source/latest.pt"))
    with pytest.raises(SystemExit, match="policy"):
        train_bc.main(cfg)
