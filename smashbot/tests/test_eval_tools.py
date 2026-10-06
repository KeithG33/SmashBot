"""The eval tools: MatchSet's bookkeeping over a scripted worker (every slate
game counted once for its player, fillers never, the run held until the
last game ends, results turned to the players' side); a
full checkpoint loads alone, bare snapshot weights take the config of a full
checkpoint, a refused checkpoint raises its own error; two runs' best.pt stay
two entries."""
import types

import pytest
import torch

from smashbot import train_bc
from smashbot.eval import sim_arena
from smashbot.eval.sim_arena import load_player, unique_labels
from smashbot.rl.rollouts import game_outcome
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


# slate game -> (frames it lasts, final stocks, final percents), seat 0 (the
# opponent) first; a stock lost to 0 is an event on its last frame; the
# level-stocks game goes to the timer. Player 1's games are the mirror.
SCRIPT = [(10, (0, 2), (0, 30)), (50, (1, 1), (85.6, 60.2)), (10, (3, 0), (40, 0)),
          (10, (2, 1), (12, 99)), (60, (0, 1), (0, 70))]
FILLER = (5, (0, 3), (0, 0))


class _ScriptedWorker:
    """MultiOpponentSimWorker's protocol with scripted games: each env's
    player is the grid member seated on it, match_fn at boot and at every
    game end, record_fn at the end, the next game's info committed on the
    frame after (the entry frame)."""

    def __init__(self, opponent, n, unroll, data_dir, stage, char_pairs, *,
                 grids, record_fn, event_fn, match_fn, **_):
        self.record_fn, self.event_fn, self.match_fn = record_fn, event_fn, match_fn
        self.player = [None] * n
        for gr in grids:
            for c in range(gr.S * gr.Nc):
                if gr.valid[c]:
                    self.player[gr.cell_env[c]] = gr.members[c // gr.Nc]
        self.game_info = [match_fn(e, self.player[e])[1] for e in range(n)]
        self.left = [self._script(e)[0] for e in range(n)]
        self.pending = {}

    def _script(self, env):
        game = self.game_info[env]
        frames, stocks, percents = FILLER if game is None else SCRIPT[game]
        if self.player[env] == 1:
            stocks, percents = stocks[::-1], percents[::-1]
        return frames, stocks, percents

    def collect(self, frames):
        for _ in range(frames):
            for env in range(len(self.left)):
                if env in self.pending:
                    self.game_info[env] = self.pending.pop(env)
                    self.left[env] = self._script(env)[0]
                    continue
                self.left[env] -= 1
                if self.left[env] == 0:
                    _, (s0, s1), (p0, p1) = self._script(env)
                    player = self.player[env]
                    if s0 == 0:
                        self.event_fn(env, player, "death", 40.0)
                    if s1 == 0:
                        self.event_fn(env, player, "kill", 90.0)
                    self.record_fn(env, player, s0, s1, game_outcome(s0, s1, p0, p1))
                    self.pending[env] = self.match_fn(env, player)[1]
        return [], []

    def close(self):
        pass


def test_matchset_counts_each_slate_game_once_per_player_and_never_a_filler(monkeypatch):
    from smashbot.tests.test_ppo import _tiny_policy
    fake_sim = types.SimpleNamespace(
        Stage=["FD", "BF"], Character={c: c for c in sim_arena.MAIN_12_MSL},
        PlayerConfig=lambda character, controller_port: (character, controller_port),
        MatchConfig=lambda **kw: kw)
    monkeypatch.setitem(__import__("sys").modules, "melee_sim", fake_sim)
    monkeypatch.setattr("smashbot.rl.sim_league.MultiOpponentSimWorker", _ScriptedWorker)
    players = [(_tiny_policy(0), 1), (_tiny_policy(1), 1)]
    ms = sim_arena.MatchSet(players, _tiny_policy(2), sim_arena.stratified(len(SCRIPT)), "",
                            envs=2, unroll=8)
    ms.run()   # a seat plays fillers from frame ~50 while its sibling plays the last game to ~95
    assert ms.results[0] == [stocks[::-1] for _, stocks, _ in SCRIPT]   # the player is seat 1
    assert ms.outcomes[0] == [1, 1, -1, -1, 1]   # level stocks, opponent at 85% vs 60%: a win
    assert [len(k) for k in ms.kill_percents[0]] == [1, 0, 0, 0, 1]   # the opponent's deaths
    assert [len(d) for d in ms.death_percents[0]] == [0, 0, 1, 0, 0]
    st = ms.stats(0)
    assert (st["games"], st["wins"], st["losses"], st["draws"]) == (5, 3, 2, 0)
    assert (st["avg_percent_at_kill"], st["avg_percent_at_death"]) == (40.0, 90.0)
    assert ms.results[1] == [stocks for _, stocks, _ in SCRIPT]   # player 1 played the mirror
    assert ms.outcomes[1] == [-1, -1, 1, 1, -1]
