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
# Rest of the CSS cast reachable by simple menuing (SHEIK reached via ZELDA
# is already in MAIN_12 through the parser's lens; ZELDA herself included).
OFF_ROSTER = [
    "MARIO", "DOC", "LINK", "YLINK", "NESS", "BOWSER", "DK",
    "GANONDORF", "GAMEANDWATCH", "KIRBY", "MEWTWO", "PICHU",
    "ROY", "ZELDA",
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
) -> list[EnvSpec]:
    """Fixed env partition. Snapshot envs are split evenly across slots
    (num_envs - cpu - teacher must divide evenly); seats alternate so each
    kind is port-balanced."""
    snap_envs = num_envs - cpu_envs - teacher_envs
    assert snap_envs >= 0 and (
        snapshot_slots == 0 or snap_envs % snapshot_slots == 0
    ), "snapshot envs must divide evenly across slots"
    rng = random.Random(seed)

    def cpu_char() -> str:
        pool = MAIN_12 if rng.random() < main12_prob else OFF_ROSTER
        return rng.choice(pool)

    specs: list[EnvSpec] = []
    for i in range(cpu_envs):
        specs.append(EnvSpec("cpu", -1, 1 + (i % 2), cpu_char()))
    for i in range(teacher_envs):
        specs.append(EnvSpec("teacher", -1, 1 + (i % 2), rng.choice(MAIN_12)))
    per_slot = snap_envs // snapshot_slots if snapshot_slots else 0
    for slot in range(snapshot_slots):
        for i in range(per_slot):
            specs.append(
                EnvSpec("snapshot", slot, 1 + (i % 2), rng.choice(MAIN_12))
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
        while len(self.archive) > self.keep:
            old = self.archive.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass
        return path

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
