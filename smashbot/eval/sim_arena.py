"""Sim-backed match engine for evaluation: a fixed slate of games between two
policies on melee-sim-light, every game played to its end and counted once.

This is the measurement backend for battery.py and tournament.py (the
Dolphin fleet versions were retired once training and eval both moved to
the sim; Dolphin remains for human play/watching via eval/game.py).

Determinism: stages, ports and game seeds come from a seeded RNG, the sim
itself is deterministic, and policy sampling uses torch's global RNG — seed
it for exactly reproducible results.
"""
from __future__ import annotations

import os
import random

import torch

MAIN_12_MSL = ["FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "FALCON",
               "PEACH", "YOSHI", "ICE_CLIMBERS", "LUIGI", "PIKACHU", "SAMUS"]
MAX_GAME_FRAMES = 28800   # Melee's 8-minute timer, so every game ends


def full_grid(seed: int = 3) -> list[tuple[str, str]]:
    """All 144 (student_char, opponent_char) pairs, order shuffled by seed
    (matches the v10-vs-gm baseline slate)."""
    pairs = [(a, b) for a in MAIN_12_MSL for b in MAIN_12_MSL]
    random.Random(seed).shuffle(pairs)
    return pairs


def stratified(n: int, seed: int = 3) -> list[tuple[str, str]]:
    """n pairs: every student char covered once per 12, opponent uniform."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        chars = list(MAIN_12_MSL)
        rng.shuffle(chars)
        pairs += [(a, rng.choice(MAIN_12_MSL)) for a in chars]
    return pairs[:n]


def load_player(path: str, device: str, config_from: str = ""):
    """(policy, name code, step) from a full checkpoint, or from bare policy
    weights (a league snapshot) built with the config of the full checkpoint
    `config_from`."""
    from smashbot import saving
    from smashbot.eval.game import resolve_name_code
    from smashbot.networks import check_loadable
    from smashbot.policy import build_policy_from_config

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in ckpt:
        ckpt = saving.upgrade_checkpoint(ckpt)
        config, weights, network_cfg = ckpt["config"], ckpt["state"]["policy"], ckpt["config"]["network"]
        code = resolve_name_code(ckpt["state"].get("name_map", {}), "Master Player", verbose=False)
        step = ckpt["state"].get("step")
    else:
        if not config_from:
            raise ValueError(f"{path} holds bare policy weights: give --config-from a full checkpoint")
        source = torch.load(config_from, map_location="cpu", weights_only=False)
        if "config" not in source:
            raise ValueError(f"--config-from {config_from} is not a full checkpoint")
        config, weights, network_cfg = saving.upgrade_checkpoint(source)["config"], ckpt, {}
        code, step = 1, None
    check_loadable(network_cfg, weights)
    policy = build_policy_from_config(config).to(device)
    policy.load_state_dict(weights)
    policy.eval()
    policy.requires_grad_(False)
    return policy, code, step


def unique_labels(paths: list[str]) -> list[str]:
    """Each path's last components, as few as keep every label distinct
    (two runs' best.pt stay apart as runA/best.pt and runB/best.pt)."""
    parts = [os.path.normpath(os.path.abspath(p)).split(os.sep) for p in paths]
    if len({tuple(p) for p in parts}) < len(parts):
        raise ValueError(f"a checkpoint is listed twice: {paths}")
    for k in range(1, max(map(len, parts)) + 1):
        labels = [os.path.join(*p[-k:]) for p in parts]
        if len(set(labels)) == len(labels):
            return labels


class MatchSet:
    """Plays every (student char, opponent char) game of `slate` once, `envs`
    at a time, and counts exactly those games: an env that finishes a game
    takes the next unplayed one, so quick games can't crowd out long ones.
    Each game draws its own stage, ports and sim seed."""

    def __init__(self, student, opponent, slate, data_dir, envs, device="cpu",
                 student_name_code=1, opp_name_code=1, unroll=240, seed=11):
        import melee_sim as msl
        from smashbot.rl.sim_league import MultiOpponentSimWorker
        rng = random.Random(seed)
        self.slate = list(slate)
        self._configs = []
        for a, b in self.slate:
            port = rng.randrange(2)   # the port-priority edge goes to either side
            self._configs.append(msl.MatchConfig(
                stage=rng.choice(list(msl.Stage)),
                players=(msl.PlayerConfig(msl.Character[a], controller_port=port),
                         msl.PlayerConfig(msl.Character[b], controller_port=1 - port)),
                seed=rng.getrandbits(31), max_frame=MAX_GAME_FRAMES))
        self.results: list = [None] * len(self.slate)   # game -> (student stocks, opponent stocks)
        self.kill_percents: list = [[] for _ in self.slate]    # game -> opponent % at our kills
        self.death_percents: list = [[] for _ in self.slate]   # game -> our % at our deaths
        unplayed = iter(range(len(self.slate)))

        def next_game(env, member):
            game = next(unplayed, None)   # past the slate: a filler game nobody counts
            return self._configs[0 if game is None else game], game

        # game_info[env]: the slate game the env's frames belong to (None for
        # a filler), held until the next game's first frame
        def on_game(env, gid, s0, s1):
            game = self.worker.game_info[env]
            if game is not None:
                self.results[game] = (s0, s1)

        def on_event(env, gid, kind, percent):
            game = self.worker.game_info[env]
            if game is not None:
                (self.kill_percents if kind == "kill" else self.death_percents)[game].append(percent)

        n = min(envs, len(self.slate))
        self.worker = MultiOpponentSimWorker(
            student, [("opponent", opponent, list(range(n)), False, opp_name_code)],
            n, unroll, data_dir, None, None, name_code=student_name_code, device=device,
            record_fn=on_game, event_fn=on_event, match_fn=next_game, max_frame=MAX_GAME_FRAMES)
        self._unroll = unroll

    def run(self) -> None:
        while None in self.results:
            self.worker.collect(self._unroll)

    def close(self):
        self.worker.close()

    def stats(self) -> dict:
        games = len(self.results)
        wins = sum(s0 > s1 for s0, s1 in self.results)
        losses = sum(s0 < s1 for s0, s1 in self.results)
        kills = [p for game in self.kill_percents for p in game]
        deaths = [p for game in self.death_percents for p in game]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        return {
            "games": games, "wins": wins, "losses": losses, "draws": games - wins - losses,
            "win_rate": wins / games,
            "avg_stock_diff": sum(s0 - s1 for s0, s1 in self.results) / games,
            "avg_percent_at_kill": mean(kills),
            "avg_percent_at_death": mean(deaths),
        }
