"""Sim-backed match engine for evaluation: N parallel games between two
policies on melee-sim-light, deterministic slates, full GameTracker stats.

This is the measurement backend for battery.py and tournament.py (the
Dolphin fleet versions were retired once training and eval both moved to
the sim; Dolphin remains for human play/watching via eval/game.py).

Determinism: pairs/stages come from a seeded RNG, the sim itself is
deterministic, and policy sampling uses torch's global RNG — seed it for
exactly reproducible replays.
"""
from __future__ import annotations

import random
import typing as tp

import numpy as np

from smashbot.rl.rollouts import GameTracker

MAIN_12_MSL = ["FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "FALCON",
               "PEACH", "YOSHI", "ICE_CLIMBERS", "LUIGI", "PIKACHU", "SAMUS"]


def full_grid(seed: int = 3) -> list[tuple[str, str]]:
    """All 144 (student_char, opponent_char) pairs, order shuffled by seed
    (matches the v10-vs-gm baseline slate)."""
    pairs = [(a, b) for a in MAIN_12_MSL for b in MAIN_12_MSL]
    random.Random(seed).shuffle(pairs)
    return pairs


def stratified(n: int, opponent_char: str | None = None, seed: int = 3):
    """n pairs: every student char covered once per 12, opponent uniform
    (or locked)."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        chars = list(MAIN_12_MSL)
        rng.shuffle(chars)
        for a in chars:
            b = opponent_char or rng.choice(MAIN_12_MSL)
            pairs.append((a, b))
    return pairs[:n]


class MatchSet:
    """Play `pairs` between student and one opponent; each env owns one pair
    and keeps replaying it until close(). Results accumulate in a
    GameTracker + a per-pair first-decision map."""

    def __init__(self, student, opponent, pairs, data_dir, device="cpu",
                 student_name_code=1, opp_name_code=1, unroll=240,
                 stage_seed=11, opp_delay_note=True):
        import melee_sim as msl
        from smashbot.rl.sim_league import MultiOpponentSimWorker
        self.msl = msl
        self.N = len(pairs)
        self.pairs = list(pairs)
        rng = random.Random(stage_seed)
        stages = [rng.choice(list(msl.Stage)) for _ in range(self.N)]
        char_pairs = [(msl.Character[a], msl.Character[b]) for a, b in pairs]
        self.tracker = GameTracker()
        self.first: dict[int, tuple[int, int]] = {}   # env -> first decided
        self.games = 0

        def on_game(i, gid, s0, s1):
            self.games += 1
            self.tracker.add_game((s0, s1), pairs[i][1])
            if i not in self.first and s0 != s1:
                self.first[i] = (s0, s1)

        def on_event(i, gid, kind, pct):
            (self.tracker.add_kill if kind == "kill"
             else self.tracker.add_death)(pct)

        self.worker = MultiOpponentSimWorker(
            student, [("opponent", opponent, list(range(self.N)), False,
                       opp_name_code)],
            self.N, unroll, data_dir, stages, char_pairs,
            name_code=student_name_code, device=device,
            record_fn=on_game, event_fn=on_event,
        )
        self._unroll = unroll

    def run(self, min_games: int, max_frames: int = 200_000) -> None:
        frames = 0
        while self.games < min_games and frames < max_frames:
            self.worker.collect(self._unroll)
            frames += self._unroll

    def close(self):
        self.worker.close()

    def stats(self) -> dict:
        s = self.tracker.stats()
        s["games"] = self.games
        s["pairs_decided"] = len(self.first)
        return s
