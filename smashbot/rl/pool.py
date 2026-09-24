"""Snapshot archive + PFSP draw (SnapshotPool): the student-ghost league.

save() freezes policies as league members; draw_member() picks opponents
weighted by AlphaStar's f_hard/f_var over the decayed-count payoff table
(pfsp.json). The Dolphin env-partition machinery that shared this module
left with the Dolphin fleet."""

from __future__ import annotations

import json
import os
import random
import typing as tp

import torch

# Imported league members (frozen checkpoints from a PREVIOUS run) use
# "import:NAME" keys. Their payoff rows live in pfsp.json like ghost rows live
# under snapshot paths, and are NEVER pruned (neither by thinning, which only
# touches archive paths, nor by the load-time prune) — see _is_import_key.
IMPORT_KEY_PREFIX = "import:"


def _is_import_key(key: str) -> bool:
    """League-member key of an imported frozen checkpoint ("import:NAME").
    Import rows are permanent: never pruned by thinning (which only touches
    archive paths) nor by the load-time prune, flags on or off — the payoff
    row vs an old run's best snapshot is the cross-generation benchmark and
    must survive restarts."""
    return key.startswith(IMPORT_KEY_PREFIX)


def f_hard(x: float, p: float = 1.0) -> float:
    """AlphaStar PFSP hardness weighting: f_hard(x) = (1 - x)^p where x is
    the student's estimated win rate vs the candidate. x=1 (fully beaten)
    => weight 0; low win rates dominate the sampling."""
    return (1.0 - x) ** p


def f_var(x: float, p: float = 1.0) -> float:
    """AlphaStar's catch-up weighting f_var(x) = (x(1 - x))^p: peaked at
    even matchups, zero at both ends (probes come from pfsp_explore)."""
    return (x * (1.0 - x)) ** p


class SnapshotPool:
    """Student snapshots on disk + the league's per-match PFSP draw.

    save() freezes the current policy every snapshot_interval learner steps
    (a new league member); draw_member() picks one opponent for one match
    weighted by AlphaStar's f_hard / f_var over the student's estimated win
    rate per member (payoff table persisted as pfsp.json in the snapshot
    directory)."""

    PRIOR_GAMES = 5  # below this, a snapshot's win rate is the 0.5 prior

    def __init__(
        self,
        directory: str,
        keep: int = 30,
        pfsp_p: float = 2.0,  # exponent on f_hard / f_var (AlphaStar: 2)
        # fraction of weighted draws using f_hard (rest f_var): 1.0 = pure
        # f_hard (AlphaStar mains), 0.0 = pure f_var (their catch-up mode)
        pfsp_hard_frac: float = 1.0,
        # fraction of draws sampled uniformly (AlphaStar's 15% forgotten-
        # players bucket); keeps 0%/100% members from being benched forever
        pfsp_explore: float = 0.0,
        # ~100-game effective memory per ghost (matches AlphaStar's 0.99
        # payoff decay). A serving ghost sees ~200 games/hour, so faster
        # alphas track only the last minutes of a stint; slower ones lag
        # across student versions. Char mixture averages out at this horizon.
        payoff_ema_alpha: float = 0.01,
        # Imported members ("import:NAME") folded into the candidate set for
        # non-latest slots; empty = snapshots only (today's league).
        league_members: tp.Sequence[str] = (),
    ):
        self.dir = directory
        self.keep = keep
        self.pfsp_p = pfsp_p
        assert 0.0 <= pfsp_hard_frac <= 1.0, pfsp_hard_frac
        self.pfsp_hard_frac = pfsp_hard_frac
        self.pfsp_explore = pfsp_explore
        self.payoff_ema_alpha = payoff_ema_alpha
        assert all(_is_import_key(m) for m in league_members), (
            f"unknown league members {list(league_members)}; "
            f"valid: '{IMPORT_KEY_PREFIX}NAME'"
        )
        self.league_members = list(league_members)
        os.makedirs(directory, exist_ok=True)
        # Adopt snapshots already on disk (restarts must not amnesia the
        # league: without this, every resume served only its own boot's
        # saves and orphaned the older ghosts).
        import glob

        self.archive: list[str] = sorted(
            glob.glob(os.path.join(directory, "snapshot-*.pt")),
            key=self._step_of,
        )
        if self.archive:
            print(f"snapshot archive: adopted {len(self.archive)} existing "
                  f"(steps {self._step_of(self.archive[0])}-"
                  f"{self._step_of(self.archive[-1])})", flush=True)
        # Per-snapshot payoff table {path: {wins, games, win_ema}}, persisted
        # across restarts and pruned to snapshots that still exist.
        self._payoff_path = os.path.join(directory, "pfsp.json")
        self.payoff: dict[str, dict] = {}
        self._load_payoff()

    def _load_payoff(self) -> None:
        try:
            with open(self._payoff_path) as f:
                table = json.load(f)
        except (OSError, ValueError):
            return
        # keep rows for surviving snapshots; import rows ("import:NAME") are
        # permanent — the payoff row vs a previous run's checkpoint is a
        # cross-generation record
        existing = set(self.archive)
        self.payoff = {
            path: entry for path, entry in table.items()
            if path in existing or _is_import_key(path)
        }
        if self.payoff:
            print(f"pfsp payoff table: loaded {len(self.payoff)} entries",
                  flush=True)

    def _save_payoff(self) -> None:
        tmp = self._payoff_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.payoff, f, indent=1)
        os.replace(tmp, self._payoff_path)

    def record_result(self, path: str, won: bool) -> None:
        """One decided game vs the member keyed by `path` — a snapshot path
        or a special league member key (won = student won).

        Estimator: DECAYED COUNTS (AlphaStar-payoff style), not an EMA of
        the rate. wins_d/games_d with per-game 0.99 decay equals the exact
        empirical mean at small n and a ~100-game recency window at large n
        — an EMA of the rate mostly reports its own seed below ~1/alpha
        games (live-caught: a 9-of-13 member displayed 0.963)."""
        entry = self.payoff.setdefault(
            path, {"wins": 0, "games": 0, "wins_d": 0.0, "games_d": 0.0}
        )
        entry["games"] += 1
        entry["wins"] += int(won)
        if "wins_d" not in entry:  # legacy row (rate-EMA era): adopt its
            # lifetime record at a capped effective count so old members
            # rejoin with their raw rate, not their seed-polluted EMA
            eff = min(float(entry["games"] - 1), 1.0 / (1 - self.PAYOFF_DECAY))
            rate = entry["wins"] / max(1, entry["games"])
            entry["games_d"] = eff
            entry["wins_d"] = rate * eff
            entry.pop("win_ema", None)
        d = self.PAYOFF_DECAY
        entry["wins_d"] = d * entry["wins_d"] + float(won)
        entry["games_d"] = d * entry["games_d"] + 1.0
        # payoff_autosave=False (sim worker): the caller flushes at
        # checkpoint cadence instead of json-dumping per decided game
        if getattr(self, "payoff_autosave", True):
            self._save_payoff()

    # ~100-game effective recency window at large n; exact mean at small n
    PAYOFF_DECAY = 0.99

    def win_estimate(self, path: str) -> float:
        """Student's estimated win rate vs this snapshot; 0.5 prior below
        PRIOR_GAMES decided games. Legacy rate-EMA rows (no decayed counts)
        fall back to their raw lifetime rate."""
        entry = self.payoff.get(path)
        if entry is None or entry["games"] < self.PRIOR_GAMES:
            return 0.5
        if entry.get("games_d"):
            return float(entry["wins_d"] / entry["games_d"])
        return float(entry["wins"] / entry["games"])

    def save(self, policy, step: int) -> str:
        path = os.path.join(self.dir, f"snapshot-{step:07d}.pt")
        tmp = path + ".tmp"
        torch.save(policy.state_dict(), tmp)
        os.replace(tmp, path)
        if path not in self.archive:  # crash-restore can re-save a step
            self.archive.append(path)
        self._thin()
        return path

    @staticmethod
    def _step_of(path: str) -> int:
        return int(os.path.basename(path).split("-")[1].split(".")[0])

    def _thin(self) -> None:
        """Exponential retention: the newest `recent` snapshots are kept
        densely; beyond the cap, evict from whichever OLD region is densest
        relative to its age (span-covered / age score), so retained old
        snapshots end up roughly exponentially spaced in step-age. Eviction
        is interior-only: making the head evictable degenerates the whole
        scheme to FIFO (measured — span/age always prefers the oldest), so
        the earliest snapshot persists as the log-spacing anchor. It rarely
        actually serves games: PFSP weights keep ancient snapshots to a
        tiny fraction of draws."""
        if self.keep <= 0:  # <=0: immortal archive, never prune
            return
        recent = min(8, self.keep // 2)
        while len(self.archive) > self.keep:
            olds = self.archive[:-recent] if recent else list(self.archive)
            if len(olds) < 3:
                victim = self.archive[0]
            else:
                latest = self._step_of(self.archive[-1])
                victim, best = None, None
                for j in range(1, len(olds) - 1):
                    span = self._step_of(olds[j + 1]) - self._step_of(olds[j - 1])
                    age = latest - self._step_of(olds[j]) + 1
                    score = span / age
                    if best is None or score < best:
                        victim, best = olds[j], score
            self.archive.remove(victim)
            try:
                os.remove(victim)
            except OSError:
                pass
            if victim in self.payoff:  # evicted ghost: drop its payoff row
                del self.payoff[victim]
                self._save_payoff()

    def _draw_weight_fn(self, rng: random.Random):
        """f_hard with prob pfsp_hard_frac, else f_var (one draw)."""
        fn = f_hard if rng.random() < self.pfsp_hard_frac else f_var
        return lambda x: fn(x, self.pfsp_p)

    def draw_member(
        self, rng: random.Random, allowed: tp.Collection[str] | None = None,
    ) -> str | None:
        """ONE per-match PFSP draw over the whole league (AlphaStar draws
        the opponent per match, not per generation).

        Two-stage class weighting (user-chosen to stop ghost-mass swamping:
        ~30 ghosts' collective weight must not outvote one hard external
        member): stage 1 picks a class — "ghosts" (the whole snapshot
        archive, latest included) or a singleton league member — with
        probability ∝ f(class hardness), f = f_hard or f_var per
        pfsp_hard_frac; stage 2 picks a ghost ∝ f(its own win estimate).
        pfsp_explore draws are uniform at both stages. Draws are
        independent (with replacement): a popular member simply wins many
        matches. `allowed` restricts the candidates (the worker's
        resident-only fallback when the grid has no free slice). Returns
        None when nothing is drawable."""
        ghosts = [g for g in self.archive if allowed is None or g in allowed]
        members = [
            m for m in self.league_members if allowed is None or m in allowed
        ]
        classes: dict[str, float] = {}
        if ghosts:
            classes["ghosts"] = sum(self.win_estimate(g) for g in ghosts) / len(ghosts)
        for m in members:
            classes[m] = self.win_estimate(m)
        if not classes:
            return None
        explore = rng.random() < self.pfsp_explore
        wfn = self._draw_weight_fn(rng)
        names = list(classes)
        weights = [1.0] * len(names) if explore else [wfn(classes[c]) for c in names]
        if sum(weights) <= 0.0:  # everyone beaten: uniform fallback
            weights = [1.0] * len(names)
        cls = names[rng.choices(range(len(names)), weights=weights)[0]]
        if cls != "ghosts":
            return cls
        gw = [1.0] * len(ghosts) if explore else [wfn(self.win_estimate(g)) for g in ghosts]
        if sum(gw) <= 0.0:
            gw = [1.0] * len(ghosts)
        return ghosts[rng.choices(range(len(ghosts)), weights=gw)[0]]

    def boot_draws(self, rng: random.Random, n: int) -> list[str]:
        """First opponents for n envs at boot: every drawable member once
        (shuffled) so the payoff table gets a reading on everyone, then
        per-match draws."""
        members = list(self.league_members) + list(self.archive)
        rng.shuffle(members)
        picks = members[:n]
        while len(picks) < n:
            m = self.draw_member(rng, allowed=[m for m in members])
            if m is None:
                break
            picks.append(m)
        return picks
