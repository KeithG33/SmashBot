"""Golden equivalence test for the ported slippi-ai "medium-v2" checkpoint.

Asserts that /home/kage/drive2/ShineBot/models/medium-v2-torch.pt still
reproduces the original TF policy on the checked-in golden set
(assets/ref_port_golden.npz, produced by scripts/port_ref_model.py):

  - embedding: our fp32 encoded state-action vector matches TF's (< 1e-5);
  - exactness: the ported model run in fp64 matches an independent fp64
    reference (sonnet equations + raw TF weights) to < 1e-6 -- measured
    ~1e-13 -- over 13 controller components x 16 steps incl. recurrent-state
    carry across two T=8 chunks;
  - fp32 end-to-end stays within 1e-3 of the TF fp32 run (the cross-framework
    kernel-rounding floor is ~3e-4; see the port report).

Skips when the ported checkpoint is absent (the golden asset is checked in).
"""

import importlib.util
import os

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(REPO, "scripts", "port_ref_model.py")
ASSET = os.path.join(os.path.dirname(__file__), "assets", "ref_port_golden.npz")


def _port_module():
    spec = importlib.util.spec_from_file_location("port_ref_model", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not os.path.exists(ASSET), reason="golden asset missing")
def test_ref_port_equivalence():
    prm = _port_module()
    if not os.path.exists(prm.TORCH_CKPT):
        pytest.skip(f"ported checkpoint not found: {prm.TORCH_CKPT}")

    from smashbot.eval.game import load_policy

    policy, name_map, _ = load_policy(prm.TORCH_CKPT, "cpu")
    assert name_map.get("Master Player") == 1  # their name_map came along

    diffs = prm.run_report(policy, np.load(ASSET))
    failures = prm.check_report(diffs)
    assert not failures, failures

    # The exactness gate is fp64-tight, not merely under tolerance.
    worst_exact = max(v for k, v in diffs.items() if k.startswith("exact64/"))
    assert worst_exact < 1e-9, worst_exact
