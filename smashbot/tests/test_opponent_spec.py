"""Opponent spec parsing for Dolphin play (play.py)."""

import pytest

from smashbot.eval.game import Opponent


def test_opponent_spec_parsing():
    o = Opponent.parse("cpu:9")
    assert (o.kind, o.level, o.character) == ("cpu", 9, "MARTH")
    o = Opponent.parse("cpu:3:FALCO")
    assert (o.kind, o.level, o.character) == ("cpu", 3, "FALCO")
    assert Opponent.parse("human").kind == "human"
    with pytest.raises(ValueError):
        Opponent.parse("cpu:0")
    with pytest.raises(ValueError):
        Opponent.parse("ckpt:/path/to/best.pt")   # checkpoint opponents seat through watch_live
    with pytest.raises(ValueError):
        Opponent.parse("wombo:9")
