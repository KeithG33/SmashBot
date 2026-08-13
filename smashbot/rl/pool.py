"""Opponent pool: env partition (CPU / teacher / snapshot slots), character
sampling, random seats, and the student-snapshot lifecycle.

Design (user-decided):
- Student always plays FOX (RL round one focuses one character; the BC prior
  retains the rest via the KL leash).
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


@dataclasses.dataclass
class EnvSpec:
    """Per-env assignment, fixed for the run."""

    kind: str  # "cpu" | "teacher" | "snapshot"
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
) -> list[EnvSpec]:
    """Fixed env partition. Snapshot envs are split evenly across slots
    (num_envs - cpu - teacher must divide evenly); seats alternate so each
    kind is port-balanced."""
    if teacher_envs < 0:  # default: teacher takes every env not otherwise used
        assert snapshot_slots == 0, "specify teacher_envs explicitly with slots"
        teacher_envs = num_envs - cpu_envs - ref_envs
    snap_envs = num_envs - cpu_envs - teacher_envs - ref_envs
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

    def stratified(n: int) -> list[str]:
        """All 12 guaranteed once (when n >= 12), remainder random, order
        shuffled — pure random draws left holes (live-audited: Phillip's
        32-env group drew zero PEACH for an entire run)."""
        chars = list(OPPONENT_CHARS) if n >= len(OPPONENT_CHARS) else []
        chars += [rng.choice(OPPONENT_CHARS) for _ in range(n - len(chars))]
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
    per_slot = snap_envs // snapshot_slots if snapshot_slots else 0
    snap_chars = stratified(per_slot * snapshot_slots)
    for slot in range(snapshot_slots):
        for i in range(per_slot):
            specs.append(
                EnvSpec("snapshot", slot, 1 + (i % 2), snap_chars.pop())
            )
    return specs


class SnapshotPool:
    """Student snapshots on disk + recency-biased slot assignments.

    save() freezes the current policy every snapshot_interval learner steps;
    refresh() reassigns serving slots: slot 0 always the latest snapshot,
    remaining slots sampled with exponential recency bias over the archive
    (old styles stay in rotation; difficulty tracks the student)."""

    def __init__(self, directory: str, slots: int, keep: int = 20):
        self.dir = directory
        self.slots = slots
        self.keep = keep
        os.makedirs(directory, exist_ok=True)
        self.archive: list[str] = []

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

    def assignments(self, rng: random.Random | None = None) -> list[str]:
        """One archive path per slot (slot 0 = latest; others recency-biased
        without replacement where possible). Empty archive -> []."""
        if not self.archive:
            return []
        rng = rng or random.Random()
        picks = [self.archive[-1]]
        candidates = list(self.archive[:-1])
        # exponential recency bias: newer snapshots ~2x likelier per halving
        while len(picks) < self.slots and candidates:
            weights = [2.0 ** (i / max(1, len(candidates) / 3)) for i in range(len(candidates))]
            chosen = rng.choices(range(len(candidates)), weights=weights)[0]
            picks.append(candidates.pop(chosen))
        while len(picks) < self.slots:
            picks.append(self.archive[-1])  # early training: duplicate latest
        return picks
