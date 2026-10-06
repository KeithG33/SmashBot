"""Sim-backed match engine for evaluation (scripts/battery.py and
tournament.py): fixed slates of games on melee-sim-light, every game played
to its end and counted once.

Determinism: stages, ports and game seeds come from a seeded RNG, the sim
itself is deterministic, and policy sampling uses torch's global RNG — seed
it for exactly reproducible results.
"""
from __future__ import annotations

import os
import random
import time

import torch

MAIN_12_MSL = ["FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "FALCON",
               "PEACH", "YOSHI", "ICE_CLIMBERS", "LUIGI", "PIKACHU", "SAMUS"]
MAX_GAME_FRAMES = 28800   # Melee's 8-minute timer, so every game ends


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def full_grid(seed: int = 3) -> list[tuple[str, str]]:
    """All 144 (student_char, opponent_char) pairs, order shuffled by seed."""
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
    from smashbot.networks import check_loadable
    from smashbot.policy import build_policy_from_config

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in ckpt:
        ckpt = saving.upgrade_checkpoint(ckpt)
        config, weights, network_cfg = ckpt["config"], ckpt["state"]["policy"], ckpt["config"]["network"]
        code = saving.resolve_name_code(ckpt["state"].get("name_map", {}), "Master Player", verbose=False)
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


def dittos(per_char: int) -> list[tuple[str, str]]:
    """per_char games of each of the 12 characters against itself."""
    return [(c, c) for _ in range(per_char) for c in MAIN_12_MSL]


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
    """Each of `players` plays every (player char, opponent char) game of
    `slate` once against `opponent`, all players at once, and exactly those
    games are counted. The players are slices of stacked-weight grids (one
    per architecture, rl/sim_league.static_grids: one vmapped forward per
    frame however many players); the opponent is the single forward. Each
    player gets `envs` seats, and a seat that finishes a game takes its
    player's next unplayed one, so quick games can't crowd out long ones.
    Each game draws its own stage, ports and sim seed — the same for every
    player."""

    def __init__(self, players, opponent, slate, data_dir, envs, device="cpu",
                 opp_name_code=1, unroll=240, seed=11):
        """players: [(policy, name_code)], on any device (the grids copy their
        weights). Results are the players' view."""
        import melee_sim as msl
        from smashbot.networks import use_manual_recurrent_step
        from smashbot.rl.sim_league import MultiOpponentSimWorker, static_grids
        rng = random.Random(seed)
        self.slate = list(slate)
        self._configs = []
        for a, b in self.slate:   # the worker's player 0 is the opponent
            port = rng.randrange(2)   # the port-priority edge goes to either side
            self._configs.append(msl.MatchConfig(
                stage=rng.choice(list(msl.Stage)),
                players=(msl.PlayerConfig(msl.Character[b], controller_port=port),
                         msl.PlayerConfig(msl.Character[a], controller_port=1 - port)),
                seed=rng.getrandbits(31), max_frame=MAX_GAME_FRAMES))
        P, G = len(players), len(self.slate)
        self.results: list = [[None] * G for _ in range(P)]   # [player][game] -> (its stocks, opponent's)
        self.outcomes: list = [[None] * G for _ in range(P)]  # rollouts.game_outcome, the player's view
        self.kill_percents: list = [[[] for _ in range(G)] for _ in range(P)]   # opponent % at the player's kills
        self.death_percents: list = [[[] for _ in range(G)] for _ in range(P)]  # player % at its deaths
        unplayed = [iter(range(G)) for _ in range(P)]

        def next_game(env, player):
            game = next(unplayed[player], None)   # past the slate: a filler game nobody counts
            return self._configs[0 if game is None else game], game

        # game_info[env]: the slate game the env's frames belong to (None for
        # a filler), held until the next game's first frame
        def on_game(env, player, s0, s1, outcome):
            game = self.worker.game_info[env]
            if game is not None:
                self.results[player][game] = (s1, s0)
                self.outcomes[player][game] = -outcome

        def on_event(env, player, kind, percent):   # kind is seat 0's: its kill is the player's death
            game = self.worker.game_info[env]
            if game is not None:
                (self.death_percents if kind == "kill" else self.kill_percents)[player][game].append(percent)

        cells = min(envs, G)
        self._envs, self._unroll = P * cells, unroll
        cuda = torch.device(device).type == "cuda"
        opponent = opponent.to(device)
        use_manual_recurrent_step(opponent)   # capturable, fp16-faithful one-frame cells (as RL serves)
        grids = static_grids(dict(enumerate(players)),
                             {p: range(p * cells, (p + 1) * cells) for p in range(P)}, device)
        self.worker = MultiOpponentSimWorker(
            opponent, self._envs, unroll, data_dir, None, None, name_code=opp_name_code,
            device=device, precision="fp16" if cuda else "fp32", capture=cuda, grids=grids,
            record_fn=on_game, event_fn=on_event, match_fn=next_game, max_frame=MAX_GAME_FRAMES,
            shards=-(-self._envs // 320), learn=False)

    def run(self, progress_every: float = 60.0) -> None:
        """Plays the slate out, printing games done, frames per second and an
        ETA every progress_every seconds."""
        start = last = time.perf_counter()
        frames = 0
        total = sum(map(len, self.results))
        left = lambda: sum(r.count(None) for r in self.results)
        while left():
            self.worker.collect(self._unroll)
            frames += self._envs * self._unroll
            now = time.perf_counter()
            if now - last >= progress_every:
                last = now
                done = total - left()
                eta = _clock((now - start) * (total - done) / done) if done else "?"
                print(f"  {done}/{total} games | {frames / (now - start):.0f} fps | "
                      f"{_clock(now - start)} elapsed, eta {eta}", flush=True)

    def close(self):
        self.worker.close()

    def stats(self, player: int = 0) -> dict:
        results, outcomes = self.results[player], self.outcomes[player]
        games = len(results)
        wins = sum(o > 0 for o in outcomes)
        losses = sum(o < 0 for o in outcomes)
        kills = [p for game in self.kill_percents[player] for p in game]
        deaths = [p for game in self.death_percents[player] for p in game]
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        return {
            "games": games, "wins": wins, "losses": losses, "draws": games - wins - losses,
            "win_rate": wins / games,
            "avg_stock_diff": sum(s0 - s1 for s0, s1 in results) / games,
            "avg_percent_at_kill": mean(kills),
            "avg_percent_at_death": mean(deaths),
        }
