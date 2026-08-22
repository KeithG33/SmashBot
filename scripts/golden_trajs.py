"""Bit-exact regression tool for rollout-loop refactors.

Builds the fake-env worker exactly as the test harness does, drives a
scripted league scenario (boot, ghost->ghost parking, phillip in and out,
imports, self-play rows, harvest), and saves every tensor leaf of the
emitted trajectories. Run it on the reference commit and on the change,
then compare:

    .venv/bin/python scripts/golden_trajs.py /tmp/ref.pt      # on main
    .venv/bin/python scripts/golden_trajs.py /tmp/new.pt      # on branch
    .venv/bin/python scripts/golden_trajs.py --compare /tmp/ref.pt /tmp/new.pt

Add --deterministic (both sides) when the change reorders agent calls:
sampling at temperature 1e-3 makes trajectories independent of RNG order.

A refactor that preserves behavior must report 0 differing leaves.
"""
import sys, random, torch, numpy as np, tree
sys.path.insert(0, "smashbot/tests")
import pytest
from _pytest.monkeypatch import MonkeyPatch
import test_opponent_league as T

if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
    assert a["kinds"] == b["kinds"], (a["kinds"], b["kinds"])
    bad = [i for i, (x, y) in enumerate(zip(a["leaves"], b["leaves"]))
           if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y)]
    for i in bad[:5]:
        print("  differs:", a["names"][i])
    print(f"{len(a['leaves'])} leaves, {len(bad)} differ -> "
          f"{'BIT-IDENTICAL' if not bad else 'MISMATCH'}")
    sys.exit(1 if bad else 0)

torch.manual_seed(0); random.seed(0); np.random.seed(0)
mp = MonkeyPatch()
if "--deterministic" in sys.argv:
    # near-zero temperature: every Bernoulli/multinomial draw saturates, so
    # trajectories no longer depend on the ORDER random numbers are drawn
    # in — the right reference when a refactor reorders agent calls
    sys.argv.remove("--deterministic")
    from smashbot.rl import agent as _agent_mod
    _orig_init = _agent_mod.BatchedPolicyAgent.__init__

    def _det_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.temperature = 1e-6
    mp.setattr(_agent_mod.BatchedPolicyAgent, "__init__", _det_init)
worker, envs = T._make_worker(
    mp, num_envs=8, teacher_envs=2, snapshot_slots=2, self_envs=1,
    league_phillip=True, harvest=True,
    opp_chars={3: "FOX", 4: "FOX", 5: "MARTH", 6: "FOX"},
)
pol = {g: worker.opponents[("slot", g)].policy for g in (0, 1)}
worker.begin_transition(0, "ghostA", None, pol[0])
worker.begin_transition(1, "phillip", None, pol[1])
out = list(worker.collect(2))
worker.begin_transition(0, "ghostB", None, pol[0])   # park ghostA
envs.final_stocks[3] = (4, 0)
out += worker.collect(2)
envs.final_stocks[4] = (0, 4)
worker.begin_transition(1, "ghostC", None, pol[1])   # phillip -> ghost
for i in (5, 6):
    envs.final_stocks[i] = (4, 0)
out += worker.collect(3)
leaves = []; offsets = []; names = []
for ti, t in enumerate(out):
    offsets.append(len(leaves))
    for path, x in tree.flatten_with_path(t):
        if isinstance(x, torch.Tensor):
            leaves.append(x.detach().cpu()); names.append((ti, path))
print(f"{len(out)} trajectories, {len(leaves)} tensor leaves, kinds {[t.kind for t in out]}")
torch.save({"leaves": leaves, "kinds": [t.kind for t in out], "offsets": offsets, "names": names}, sys.argv[1])
