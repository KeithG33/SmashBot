"""Bridge to a slippi-ai (TensorFlow) reference opponent — e.g. medium-v2 —
running in the venv-ref subprocess (scripts/ref_server.py).

STATUS: scaffolding, not yet live-tested; worker integration (EnvSpec kind
"reference", raw-game payload forwarding) lands with the first live test.
Nothing imports this module unless a pool config sets ref_envs > 0, so it
is inert for the A/B runs.
"""

from __future__ import annotations

import pickle
import struct
import subprocess

REF_PYTHON = "/home/kage/drive2/ShineBot/venv-ref/bin/python"
REF_SERVER = "/home/kage/smashbot_workspace/SmashBot/scripts/ref_server.py"
DEFAULT_REF_CKPT = "/home/kage/drive2/ShineBot/models/medium-v2"


class RefBridge:
    """Owns the ref-server subprocess; one instance per reference env group."""

    def __init__(self, batch_size: int, ckpt: str = DEFAULT_REF_CKPT):
        self.proc = subprocess.Popen(
            [REF_PYTHON, REF_SERVER, "--path", ckpt,
             "--batch-size", str(batch_size)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        hello = self._read()
        assert hello and hello.get("ready"), f"ref server failed: {hello}"
        self.delay = hello["delay"]

    def _read(self):
        header = self.proc.stdout.read(4)
        if len(header) < 4:
            return None
        (n,) = struct.unpack("<I", header)
        return pickle.loads(self.proc.stdout.read(n))

    def _write(self, obj) -> None:
        payload = pickle.dumps(obj, protocol=4)
        self.proc.stdin.write(struct.pack("<I", len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def step(self, games: list, needs_reset: list) -> list:
        """games: RAW parsed slippi-ai Game structs (their encoding happens
        server-side); returns per-env controller states."""
        self._write({"games": games, "needs_reset": needs_reset})
        reply = self._read()
        if reply is None:
            raise RuntimeError("ref server died")
        return reply["controllers"]

    def stop(self) -> None:
        try:
            self._write(None)
        except (BrokenPipeError, OSError):
            pass
        self.proc.terminate()
