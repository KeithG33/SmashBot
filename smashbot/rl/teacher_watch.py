"""Hot-swappable frozen teacher: watch a checkpoint path, safely load newer
versions mid-RL-run.

Swap semantics: teacher weights are copied in place (load_state_dict), so the
opponent agent and any compiled functions sharing the module see the new
weights immediately; the caller must zero the carried teacher recurrent state
(stale by one window otherwise). Between same-run BC checkpoints the KL jump
is small, and the learner's revert backstop guards oversized reactions.

Collision safety (reader side): the file is stat'd twice across a settle
delay and skipped while still changing; torn reads fail the torch.load and
retry next poll. Writer side must be atomic — our checkpoint writers use
tmp + os.replace, and scripts/pull_teacher.sh fetches the same way.
"""

from __future__ import annotations

import os
import time

import torch


class TeacherWatcher:
    def __init__(self, path: str, settle_seconds: float = 1.0):
        self.path = path
        self.settle_seconds = settle_seconds
        try:
            self._seen = os.stat(path).st_mtime_ns
        except FileNotFoundError:
            self._seen = 0

    def poll(self) -> dict | None:
        """Returns the new checkpoint's policy state_dict if the file changed
        and is fully written; else None. Never raises on torn/missing files."""
        try:
            st1 = os.stat(self.path)
        except FileNotFoundError:
            return None
        if st1.st_mtime_ns == self._seen:
            return None
        time.sleep(self.settle_seconds)
        try:
            st2 = os.stat(self.path)
        except FileNotFoundError:
            return None
        if (st2.st_mtime_ns, st2.st_size) != (st1.st_mtime_ns, st1.st_size):
            return None  # still being written; retry next poll
        try:
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            state = ckpt["state"]["policy"]
        except Exception:
            return None  # torn write or bad file; retry next poll
        self._seen = st2.st_mtime_ns
        return state
