"""Opponent spec parsing for the Dolphin play harness."""

import pytest

from smashbot.eval.game import Opponent


def test_opponent_spec_parsing():
    o = Opponent.parse("cpu:9")
    assert (o.kind, o.level, o.character) == ("cpu", 9, "MARTH")
    o = Opponent.parse("cpu:3:FALCO")
    assert (o.kind, o.level, o.character) == ("cpu", 3, "FALCO")
    o = Opponent.parse("ckpt:/path/to/best.pt")
    assert (o.kind, o.ckpt_path, o.character) == ("ckpt", "/path/to/best.pt", "FOX")
    o = Opponent.parse("ckpt:/path/best.pt:MARTH")
    assert (o.kind, o.ckpt_path, o.character) == ("ckpt", "/path/best.pt", "MARTH")
    assert Opponent.parse("human").kind == "human"
    with pytest.raises(ValueError):
        Opponent.parse("cpu:0")
    with pytest.raises(ValueError):
        Opponent.parse("ckpt:")
    with pytest.raises(ValueError):
        Opponent.parse("wombo:9")
