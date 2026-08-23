"""Bit-exact regression tool for rollout-loop refactors.

Builds the fake-env worker exactly as the test harness does, drives a
scripted league scenario over the per-match routing (boot over every
member, scripted game boundaries with draws of ghosts / the teacher /
Phillip / a char-locked import, self-play rows, harvest), and saves every
tensor leaf of the emitted trajectories. Run it on the reference commit
and on the change, then compare:

    .venv/bin/python scripts/golden_trajs.py /tmp/ref.pt      # on main
    .venv/bin/python scripts/golden_trajs.py /tmp/new.pt      # on branch
    .venv/bin/python scripts/golden_trajs.py --compare /tmp/ref.pt /tmp/new.pt

Add --deterministic (both sides) when the change reorders agent calls:
sampling at temperature 1e-6 makes trajectories independent of RNG order.

A refactor that preserves behavior must report 0 differing leaves.
(The league-routing baseline starts at the league-routing branch: the
per-match design is not comparable to the generation/slot design.)
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
import tempfile
from smashbot.rl import agent as _agent_mod
import test_league_routing as R

if "--deterministic" in sys.argv or _agent_mod.BatchedPolicyAgent.__init__.__name__ == "_det_init":
    _orig_league = _agent_mod.LeagueAgent.__init__

    def _det_league(self, *a, **k):
        k["temperature"] = 1e-6
        _orig_league(self, *a, **k)
    mp.setattr(_agent_mod.LeagueAgent, "__init__", _det_league)

pool_dir = tempfile.mkdtemp(prefix="golden-league-")
imp = pool_dir + "/import-v3.pt"
torch.save(T._tiny_policy(seed=9).state_dict(), imp)
worker, envs = T._make_worker(
    mp, num_envs=8, teacher_envs=0, self_envs=1, league_slices=2,
    league_teacher=True, league_phillip=True, harvest=True,
    league_imports=[f"v3={imp}@FOX"], members=[0, 100], pool_dir=pool_dir,
    pfsp_explore=1.0, phillip_capacity=2,
    char_whitelist=["FOX", "MARTH", "FALCO", "PEACH"],
)
envs = R._ProtocolEnvs(worker, seed=1, opp_chars=dict(envs.opp_chars))
envs.install(mp)
lg = worker.league
out = list(worker.collect(2))
# scripted boundaries: every league env ends a game at a different frame,
# so adoptions, locks and fallbacks all happen mid-chunk
for i, st in zip(worker.league_idx, [(4, 0), (0, 4), (4, 1), (2, 4)]):
    envs.end_game(i, st)
    out += worker.collect(1)
for i in worker.league_idx:
    envs.end_game(i)
out += worker.collect(2)
print(f"members now: {[lg.member_now[i] for i in worker.league_idx]} | "
      f"draws {lg.draws} fallbacks {lg.fallbacks} warnings {lg.warnings}")
leaves = []; offsets = []; names = []
for ti, t in enumerate(out):
    offsets.append(len(leaves))
    for path, x in tree.flatten_with_path(t):
        if isinstance(x, torch.Tensor):
            leaves.append(x.detach().cpu()); names.append((ti, path))
print(f"{len(out)} trajectories, {len(leaves)} tensor leaves, kinds {[t.kind for t in out]}")
torch.save({"leaves": leaves, "kinds": [t.kind for t in out], "offsets": offsets, "names": names}, sys.argv[1])
