"""The eval tools: MatchSet's bookkeeping over a scripted worker (every slate
game counted once, fillers never, the run held until the last game ends); a
full checkpoint loads alone, bare snapshot weights take the config of a full
checkpoint, a refused checkpoint raises its own error; two runs' best.pt stay
two entries."""
import types

import pytest
import torch

from smashbot import train_bc
from smashbot.eval import sim_arena
from smashbot.eval.sim_arena import load_player, unique_labels
from smashbot.tests.test_restart_guard import _config


def test_full_checkpoints_and_bare_snapshots_load(tmp_path):
    train_bc.main(_config(tmp_path))
    full = str(tmp_path / "run" / "latest.pt")
    policy, _, step = load_player(full, "cpu")
    assert step == 2
    bare = tmp_path / "snapshot.pt"
    torch.save(policy.state_dict(), bare)
    snapshot, code, snapshot_step = load_player(str(bare), "cpu", config_from=full)
    assert (code, snapshot_step) == (1, None)
    assert all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(), snapshot.state_dict().values()))
    with pytest.raises(ValueError, match="bare policy weights"):
        load_player(str(bare), "cpu")
    with pytest.raises(ValueError, match="not a full checkpoint"):
        load_player(str(bare), "cpu", config_from=str(bare))
    torch.save({"network.core.blocks.0.uv.weight": torch.zeros(1)}, tmp_path / "pre-paper.pt")
    with pytest.raises(ValueError, match="predates the unconditional paper block"):
        load_player(str(tmp_path / "pre-paper.pt"), "cpu", config_from=full)


def test_labels_keep_same_named_checkpoints_apart():
    assert unique_labels(["/r/a/best.pt", "/r/b/best.pt", "/r/c/latest.pt"]) == \
        ["a/best.pt", "b/best.pt", "c/latest.pt"]
    assert unique_labels(["x/best.pt", "y/latest.pt"]) == ["best.pt", "latest.pt"]
    with pytest.raises(ValueError, match="listed twice"):
        unique_labels(["/r/a/best.pt", "/r/./a/best.pt"])


# slate game -> (frames it lasts, final stocks); a stock lost to 0 is an event on its last frame
SCRIPT = [(10, (0, 2)), (50, (1, 1)), (10, (3, 0)), (10, (2, 1)), (60, (0, 1))]
FILLER = (5, (0, 3))


class _ScriptedWorker:
    """MultiOpponentSimWorker's protocol with scripted games: match_fn at boot
    and at every game end, record_fn at the end, the next game's info
    committed on the frame after (the entry frame)."""

    def __init__(self, student, opponents, n, unroll, data_dir, stage, char_pairs, *,
                 record_fn, event_fn, match_fn, **_):
        self.record_fn, self.event_fn, self.match_fn = record_fn, event_fn, match_fn
        self.game_info = [match_fn(e, "opponent")[1] for e in range(n)]
        self.left = [self._script(e)[0] for e in range(n)]
        self.pending = {}

    def _script(self, env):
        game = self.game_info[env]
        return FILLER if game is None else SCRIPT[game]

    def collect(self, frames):
        for _ in range(frames):
            for env in range(len(self.left)):
                if env in self.pending:
                    self.game_info[env] = self.pending.pop(env)
                    self.left[env] = self._script(env)[0]
                    continue
                self.left[env] -= 1
                if self.left[env] == 0:
                    s0, s1 = self._script(env)[1]
                    if s0 == 0:
                        self.event_fn(env, "opponent", "death", 40.0)
                    if s1 == 0:
                        self.event_fn(env, "opponent", "kill", 90.0)
                    self.record_fn(env, "opponent", s0, s1)
                    self.pending[env] = self.match_fn(env, "opponent")[1]
        return [], []

    def close(self):
        pass


def test_matchset_counts_each_slate_game_once_and_never_a_filler(monkeypatch):
    fake_sim = types.SimpleNamespace(
        Stage=["FD", "BF"], Character={c: c for c in sim_arena.MAIN_12_MSL},
        PlayerConfig=lambda character, controller_port: (character, controller_port),
        MatchConfig=lambda **kw: kw)
    monkeypatch.setitem(__import__("sys").modules, "melee_sim", fake_sim)
    monkeypatch.setattr("smashbot.rl.sim_league.MultiOpponentSimWorker", _ScriptedWorker)
    ms = sim_arena.MatchSet(None, None, sim_arena.stratified(len(SCRIPT)), "", envs=2, unroll=8)
    ms.run()   # env 1 plays fillers from frame ~50 while env 0 plays the last game to ~95
    assert ms.results == [outcome for _, outcome in SCRIPT]
    assert [len(d) for d in ms.death_percents] == [1, 0, 0, 0, 1]
    assert [len(k) for k in ms.kill_percents] == [0, 0, 1, 0, 0]
    st = ms.stats()
    assert (st["games"], st["wins"], st["losses"], st["draws"]) == (5, 2, 2, 1)
    assert (st["avg_percent_at_kill"], st["avg_percent_at_death"]) == (90.0, 40.0)
