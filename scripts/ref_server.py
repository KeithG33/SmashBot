"""Reference-agent server: runs a slippi-ai (TensorFlow) model as an RL
opponent, in the venv-ref environment, speaking length-prefixed pickle over
stdin/stdout.

Run with the REFERENCE venv's python (TF stack), never the training venv:
  /home/kage/drive2/ShineBot/venv-ref/bin/python scripts/ref_server.py \
      --path /home/kage/drive2/ShineBot/models/medium-v2 --batch-size 8

Protocol (binary, length-prefixed pickle both ways):
  -> {"games": [per-env slippi-ai Game struct (single frame)],
      "needs_reset": [bool] * batch}
  <- {"controllers": [per-env melee-compatible controller state]}
A None message shuts the server down. The agent manages its own recurrent
state and delay queue internally (their eval_lib machinery).
"""

import argparse
import pickle
import struct
import sys


def read_msg(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (n,) = struct.unpack("<I", header)
    return pickle.loads(stream.read(n))


def write_msg(stream, obj) -> None:
    payload = pickle.dumps(obj, protocol=4)
    stream.write(struct.pack("<I", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--name", default=None,
                    help="conditioning name (imitation states require one)")
    args = ap.parse_args()

    import os

    # Claim the protocol channel BEFORE any TF import: stray library prints
    # to stdout would corrupt the pickle stream, so fd-1 gets repointed at
    # stderr and the protocol keeps a private dup of the original stdout.
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    # TF imports deferred until after arg parsing for fast failure
    import numpy as np
    import tree

    from slippi_ai import eval_lib, saving

    state = saving.load_state_from_disk(args.path)
    summary = eval_lib.AgentSummary.from_checkpoint(args.path)
    print(f"ref agent: type={summary.type} delay={summary.delay} "
          f"chars={summary.characters}", file=sys.stderr, flush=True)

    agent = eval_lib.build_delayed_agent(
        state=state,
        console_delay=0,
        name=args.name,
        batch_size=args.batch_size,
    )
    if hasattr(agent, "warmup") and callable(agent.warmup):
        agent.warmup()

    stdin = sys.stdin.buffer
    stdout = proto_out
    write_msg(stdout, {"ready": True, "delay": summary.delay})
    while True:
        msg = read_msg(stdin)
        if msg is None:
            return
        games = msg["games"]
        needs_reset = np.asarray(msg["needs_reset"], dtype=bool)
        batched = tree.map_structure(lambda *xs: np.stack(xs), *games)
        sampled = agent.step(batched, needs_reset)
        controllers = agent.decode_controller(sampled.controller_state)
        write_msg(stdout, {"controllers": controllers})


if __name__ == "__main__":
    main()
