"""Opponent pool: env partition (CPU / teacher / snapshot slots), character
sampling, random seats, and the student-snapshot lifecycle.

Design (user-decided):
- Student plays characters from a whitelist (default FOX-only: RL round one
  focuses one character; the BC prior retains the rest via the KL leash).
  See student_whitelist() for the legacy bot_char interaction.
- Opponent characters vary. Policy opponents (teacher/snapshots) draw from
  the MAIN-12 roster only — the BC prior never saw off-roster characters and
  would play them at garbage tier (noise, not diversity). CPU opponents draw
  60% main-12 / 40% rest-of-cast: the engine AI genuinely plays anyone.
- Random seats: each env independently seats the student on port 1 or 2,
  cancelling Melee's port-priority edge in aggregate and preventing
  priority-dependent habits.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import typing as tp

import torch

MAIN_12 = [
    "FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "CPTFALCON",
    "PEACH", "YOSHI", "POPO", "LUIGI", "PIKACHU", "SAMUS",
]
# Policy opponents can be any of the 12: Sheik works via the netplay CSS
# Zelda slot (its Sheik/Zelda toggle defaults to Sheik); occasional menu
# races are survived by the env-process retry guard. CPU opponents cannot
# be Sheik (libmelee cannot force a CPU to transform), and Zelda is
# unpickable on the netplay CSS entirely.
OPPONENT_CHARS = list(MAIN_12)
# CPU Sheik is IMPOSSIBLE (tested live: 362/362 attempts spawned Zelda —
# the engine ignores held A on CPU-status ports, so the Zelda->Sheik
# transform never triggers; libmelee's guard was right). Sheik matchup
# coverage flows through the policy-opponent envs instead.
CPU_CHARS = [c for c in MAIN_12 if c != "SHEIK"]
# Rest of the CSS cast reachable by simple menuing (SHEIK reached via ZELDA
# is already in MAIN_12 through the parser's lens; ZELDA herself included).
OFF_ROSTER = [
    "MARIO", "DOC", "LINK", "YLINK", "NESS", "BOWSER", "DK",
    "GANONDORF", "GAMEANDWATCH", "KIRBY", "MEWTWO", "PICHU",
    "ROY",
]


def student_whitelist(
    char_whitelist: tp.Sequence[str], bot_char: str = "FOX"
) -> list[str]:
    """Effective student-character whitelist.

    The default whitelist ["FOX"] defers to the legacy bot_char flag (so
    `--rollouts.bot-char MARTH` keeps working); any non-default whitelist
    wins. len==1 reproduces the fixed-character behavior exactly."""
    wl = [c.upper() for c in char_whitelist]
    if wl == ["FOX"]:
        return [bot_char.upper()]
    return wl


@dataclasses.dataclass
class EnvSpec:
    """Per-env assignment, fixed for the run."""

    kind: str  # "cpu" | "teacher" | "reference" | "self" | "snapshot"
    group: int  # snapshot slot index (0 = freshest); -1 otherwise
    student_port: int  # 1 or 2
    opponent_char: str
    cpu_level: int = 9


def make_partition(
    num_envs: int,
    cpu_envs: int,
    teacher_envs: int,
    snapshot_slots: int,
    main12_prob: float = 0.6,
    seed: int = 0,
    ref_envs: int = 0,
    self_envs: int = 0,
    char_whitelist: tp.Sequence[str] = ("FOX",),
) -> list[EnvSpec]:
    """Fixed env partition. Snapshot envs are split evenly across slots
    (num_envs - cpu - teacher must divide evenly); seats alternate so each
    kind is port-balanced.

    num_envs is the LEARNER trajectory budget, not the Dolphin count: a
    self-play env contributes BOTH seats as PPO trajectories, so it costs 2
    budget units while running one Dolphin. The returned list has
    num_envs - self_envs specs (= Dolphins to boot). Order:
    cpu / teacher / reference / self / snapshot."""
    if teacher_envs < 0:  # default: teacher takes every env not otherwise used
        assert snapshot_slots == 0, "specify teacher_envs explicitly with slots"
        teacher_envs = num_envs - cpu_envs - ref_envs - 2 * self_envs
    snap_envs = num_envs - cpu_envs - teacher_envs - ref_envs - 2 * self_envs
    assert snap_envs >= 0 and (
        snapshot_slots == 0 or snap_envs % snapshot_slots == 0
    ), "snapshot envs must divide evenly across slots"
    assert snapshot_slots > 0 or snap_envs == 0, (
        "leftover envs with no snapshot slots — set teacher_envs/cpu_envs "
        "to cover num_envs"
    )
    rng = random.Random(seed)

    def cpu_char() -> str:
        pool = CPU_CHARS if rng.random() < main12_prob else OFF_ROSTER
        return rng.choice(pool)

    def stratified(n: int, roster: tp.Sequence[str] = OPPONENT_CHARS) -> list[str]:
        """All roster chars guaranteed once (when n >= len(roster)), remainder
        random, order shuffled — pure random draws left holes (live-audited:
        Phillip's 32-env group drew zero PEACH for an entire run)."""
        chars = list(roster) if n >= len(roster) else []
        chars += [rng.choice(roster) for _ in range(n - len(chars))]
        rng.shuffle(chars)
        return chars

    specs: list[EnvSpec] = []
    for i in range(cpu_envs):
        specs.append(EnvSpec("cpu", -1, 1 + (i % 2), cpu_char()))
    for i, ch in enumerate(stratified(teacher_envs)):
        specs.append(EnvSpec("teacher", -1, 1 + (i % 2), ch))
    for i, ch in enumerate(stratified(ref_envs)):
        # reference agent (e.g. medium-v2) plays the main 12 (user-verified)
        specs.append(EnvSpec("reference", -1, 1 + (i % 2), ch))
    # self-play: both seats are the student, so the second seat's boot char
    # draws from the student whitelist (stratified for coverage). NOTE: with
    # self_envs == 0 this consumes ZERO rng draws, keeping the stream (and
    # therefore every downstream char draw) identical to the pre-self code.
    for i, ch in enumerate(stratified(self_envs, list(char_whitelist))):
        specs.append(EnvSpec("self", -1, 1 + (i % 2), ch))
    per_slot = snap_envs // snapshot_slots if snapshot_slots else 0
    snap_chars = stratified(per_slot * snapshot_slots)
    for slot in range(snapshot_slots):
        for i in range(per_slot):
            specs.append(
                EnvSpec("snapshot", slot, 1 + (i % 2), snap_chars.pop())
            )
    assert len(specs) == num_envs - self_envs, (
        "dolphin count must be num_envs - self_envs (memory-neutral batching)"
    )
    return specs


# Special (non-snapshot) league members that can compete for snapshot slots
# when the league_teacher/league_cpu/league_phillip flags fold them into the
# PFSP league. Their payoff rows live in pfsp.json under these string keys,
# exactly like ghost rows live under snapshot paths — and they are NEVER
# pruned (neither by thinning, which only touches archive paths, nor by the
# load-time prune, so toggling the flags across restarts loses no data).
LEAGUE_MEMBER_KEYS = ("teacher", "cpu", "phillip")


def f_hard(x: float, p: float = 1.0) -> float:
    """AlphaStar PFSP hardness weighting: f_hard(x) = (1 - x)^p where x is
    the student's estimated win rate vs the candidate. x=1 (fully beaten)
    => weight 0; low win rates dominate the sampling."""
    return (1.0 - x) ** p


class SnapshotPool:
    """Student snapshots on disk + PFSP (or recency-biased) slot assignments.

    save() freezes the current policy every snapshot_interval learner steps;
    refresh() reassigns serving slots: slot 0 always the latest snapshot,
    remaining slots sampled without replacement. With pfsp=True (default)
    the sampling weight is AlphaStar's f_hard over the student's estimated
    win rate vs each snapshot (payoff table persisted as pfsp.json in the
    snapshot directory); pfsp=False keeps the original exponential recency
    bias exactly."""

    PRIOR_GAMES = 5  # below this, a snapshot's win rate is the 0.5 prior

    def __init__(
        self,
        directory: str,
        slots: int,
        keep: int = 30,
        pfsp: bool = True,
        pfsp_p: float = 1.0,
        # ~100-game effective memory per ghost (matches AlphaStar's 0.99
        # payoff decay). A serving ghost sees ~200 games/hour, so faster
        # alphas track only the last minutes of a stint; slower ones lag
        # across student versions. Char mixture averages out at this horizon.
        payoff_ema_alpha: float = 0.01,
        # Special league members ("teacher"/"cpu") folded into the candidate
        # set for non-latest slots; empty = snapshots only (today's league).
        league_members: tp.Sequence[str] = (),
    ):
        self.dir = directory
        self.slots = slots
        self.keep = keep
        self.pfsp = pfsp
        self.pfsp_p = pfsp_p
        self.payoff_ema_alpha = payoff_ema_alpha
        assert all(m in LEAGUE_MEMBER_KEYS for m in league_members), (
            f"unknown league members {list(league_members)}; "
            f"valid: {LEAGUE_MEMBER_KEYS}"
        )
        assert pfsp or not league_members, (
            "league members (teacher/cpu) need PFSP win-rate prioritization "
            "to earn/lose serving time — enable pfsp=True (the recency "
            "sampler has no notion of them)"
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
        # keep rows for surviving snapshots AND the special league members
        # (special rows persist regardless of the current league flags)
        existing = set(self.archive) | set(LEAGUE_MEMBER_KEYS)
        self.payoff = {
            path: entry for path, entry in table.items() if path in existing
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
        or a special league member key (won = student won)."""
        entry = self.payoff.setdefault(
            path, {"wins": 0, "games": 0, "win_ema": None}
        )
        entry["games"] += 1
        entry["wins"] += int(won)
        outcome = 1.0 if won else 0.0
        a = self.payoff_ema_alpha
        # seed from the 0.5 prior, not the first outcome: at alpha 0.01 a
        # first-game coin flip carries ~26% of the EMA for 100+ games and
        # skews auction weights (live-caught: phillip read 0.60 vs a true
        # ~0.46 because his first league game happened to be a student win)
        prev = 0.5 if entry["win_ema"] is None else entry["win_ema"]
        entry["win_ema"] = (1 - a) * prev + a * outcome
        self._save_payoff()

    def win_estimate(self, path: str) -> float:
        """Student's estimated win rate vs this snapshot; 0.5 prior below
        PRIOR_GAMES decided games."""
        entry = self.payoff.get(path)
        if (
            entry is None
            or entry["games"] < self.PRIOR_GAMES
            or entry["win_ema"] is None
        ):
            return 0.5
        return float(entry["win_ema"])

    def save(self, policy, step: int) -> str:
        path = os.path.join(self.dir, f"snapshot-{step:07d}.pt")
        tmp = path + ".tmp"
        torch.save(policy.state_dict(), tmp)
        os.replace(tmp, path)
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
        actually serves games: assignments() recency bias keeps ancient
        snapshots to a tiny fraction of slot picks."""
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

    def class_hardness(self) -> dict[str, float]:
        """Per-class student win estimate for the two-stage PFSP sampler
        (and wandb): "ghosts" = mean win_ema over the whole archive minus
        the latest snapshot (0.5 prior for unmeasured members), each league
        member ("phillip"/"teacher"/"cpu") a singleton class = its own row.
        Only nonempty/enabled classes appear."""
        out: dict[str, float] = {}
        ghosts = self.archive[:-1]
        if ghosts:
            out["ghosts"] = (
                sum(self.win_estimate(g) for g in ghosts) / len(ghosts)
            )
        for m in self.league_members:
            out[m] = self.win_estimate(m)
        return out

    def _class_weighted_picks(self, rng: random.Random) -> list[str]:
        """Two-stage class-weighted PFSP for the non-latest slots (active
        only when league members exist — user-chosen to stop ghost-mass
        swamping: ~30 ghosts' collective flat weight must not outvote one
        hard external member).

        Stage 1, per slot independently (WITH replacement across slots):
        sample a class with probability ∝ f_hard(class hardness) where the
        hardness is the class's MEAN win_ema (class_hardness()). Singleton
        classes can therefore hold multiple slots at once.
        Stage 2, within "ghosts": the existing per-ghost f_hard, WITHOUT
        replacement across slots — a ghost serves at most one slot."""
        ghosts = list(self.archive[:-1])
        hard = self.class_hardness()
        picks: list[str] = []
        while len(picks) < self.slots - 1:
            classes = [c for c in hard if c != "ghosts" or ghosts]
            weights = [f_hard(hard[c], self.pfsp_p) for c in classes]
            if sum(weights) <= 0.0:  # everyone beaten: uniform fallback
                weights = [1.0] * len(classes)
            cls = classes[rng.choices(range(len(classes)), weights=weights)[0]]
            if cls == "ghosts":
                gw = [
                    f_hard(self.win_estimate(g), self.pfsp_p) for g in ghosts
                ]
                if sum(gw) <= 0.0:
                    gw = [1.0] * len(ghosts)
                j = rng.choices(range(len(ghosts)), weights=gw)[0]
                picks.append(ghosts.pop(j))
            else:
                picks.append(cls)
        return picks

    def assignments(self, rng: random.Random | None = None) -> list[str]:
        """One member key per slot: an archive path, or a special league
        member ("phillip"/"teacher"/"cpu") when league_members is set.
        Slot 0 = ALWAYS the latest snapshot. With league members the rest
        use the two-stage class-weighted PFSP (_class_weighted_picks);
        without them, the flat per-ghost sampling is byte-identical to the
        pre-league code — PFSP f_hard weights by default, the original
        exponential recency bias with pfsp=False. Empty archive -> []
        (league members only start serving once a first snapshot anchors
        slot 0)."""
        if not self.archive:
            return []
        rng = rng or random.Random()
        picks = [self.archive[-1]]
        candidates = list(self.archive[:-1])
        if self.pfsp and self.league_members:
            picks += self._class_weighted_picks(rng)
        elif self.pfsp:
            # PFSP (AlphaStar f_hard): weight by how much the student still
            # struggles vs each snapshot; beaten snapshots fade out.
            while len(picks) < self.slots and candidates:
                weights = [
                    f_hard(self.win_estimate(c), self.pfsp_p)
                    for c in candidates
                ]
                if sum(weights) <= 0.0:  # everyone beaten: uniform fallback
                    weights = [1.0] * len(candidates)
                chosen = rng.choices(range(len(candidates)), weights=weights)[0]
                picks.append(candidates.pop(chosen))
        else:
            # exponential recency bias: newer snapshots ~2x likelier per halving
            while len(picks) < self.slots and candidates:
                weights = [2.0 ** (i / max(1, len(candidates) / 3)) for i in range(len(candidates))]
                chosen = rng.choices(range(len(candidates)), weights=weights)[0]
                picks.append(candidates.pop(chosen))
        while len(picks) < self.slots:
            picks.append(self.archive[-1])  # early training: duplicate latest
        return picks


def apply_assignments(
    assigns: tp.Sequence[str],
    slot_policies: tp.Sequence[tuple[int, tp.Any]],
    teacher_module,
    worker,
    slot_keys: dict[int, str],
    device: str = "cpu",
) -> None:
    """Route one epoch's slot assignments into the serving machinery.

    - snapshot path: load from disk into the slot policy (instant hot-swap,
      today's behavior); the slot serves kind "snapshot".
    - "teacher": copy the LIVE teacher module's CURRENT weights into the slot
      policy (in-memory state_dict copy, no disk). Staleness bound: if the
      teacher watcher hot-swaps the module mid-epoch, this slot keeps serving
      its copy until the next snapshot_interval refresh re-copies it.
    - "phillip": ROUTING only — Phillip's architecture differs from the
      student's, so he can never be loaded into a slot policy. The worker
      sends this slot's rows to Phillip's own BatchedPolicyAgent instead
      (rollouts: _phillip_agent_for). The slot policy module is untouched
      and idle. Hot-swappable like a snapshot: both seats stay standard
      controllers, so no Dolphin reboot is needed.
    - "cpu": only record the desired kind — Dolphin CPU ports cannot hot-swap
      mid-game, so each env adopts (policy|phillip)<->cpu at its NEXT recycle
      boundary (rollouts._env_process_main). slot_keys/slot_serving are
      deliberately left on the PREVIOUS member: attribution must follow what
      each env is ACTUALLY serving, and not-yet-adopted envs still serve the
      old policy.
    """
    for slot, slot_policy in slot_policies:
        if slot >= len(assigns):
            continue
        key = assigns[slot]
        if key == "cpu":
            worker.slot_desired[slot] = "cpu"
            continue
        if key == "phillip":
            worker.slot_serving[slot] = "phillip"
        elif key == "teacher":
            slot_policy.load_state_dict(teacher_module.state_dict())
            worker.slot_serving[slot] = "teacher"
        else:
            slot_policy.load_state_dict(torch.load(key, map_location=device))
            worker.slot_serving[slot] = "snapshot"
        worker.slot_desired[slot] = "policy"
        slot_keys[slot] = key
