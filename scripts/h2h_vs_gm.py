"""v10-final vs phillip-gm: FULL 12x12 character-pair coverage on the CPU
sim. 3 phases x 48 envs; each env owns one pair; first decision per pair
counts (env replays its pair afterwards). Timeout-capped per phase."""
import os, random, numpy as np, torch, json
import melee_sim as msl
from smashbot.eval.game import load_policy, resolve_name_code
from smashbot.rl.sim_league import MultiOpponentSimWorker

torch.set_num_threads(16)
DEV = "cpu"
student, snm, _ = load_policy("/home/kage/drive2/ShineBot/runs/rl-pool-v10/latest.pt", DEV)
student.eval()
gm, gnm, _ = load_policy("/home/kage/drive2/ShineBot/models/gm-torch.pt", DEV)
gm.eval(); gm.requires_grad_(False)
sc = resolve_name_code(snm, "Master Player")
gc = resolve_name_code(gnm, "Master Player")

CH = ["FOX","FALCO","MARTH","SHEIK","JIGGLYPUFF","FALCON",
      "PEACH","YOSHI","ICE_CLIMBERS","LUIGI","PIKACHU","SAMUS"]
ALL = [(a, b) for a in CH for b in CH]           # 144 (student, gm)
rng = random.Random(3)
rng.shuffle(ALL)
results = {}                                      # pair -> (s0, s1)

for phase in range(3):
    todo = ALL[phase*48:(phase+1)*48]
    N = len(todo)
    pairs = [(msl.Character[a], msl.Character[b]) for a, b in todo]
    stages = [rng.choice(list(msl.Stage)) for _ in range(N)]
    decided = [False]*N
    def on_game(i, gid, s0, s1, todo=todo, decided=decided):
        if not decided[i] and s0 != s1:
            decided[i] = True
            results[todo[i]] = (s0, s1)
            d = sum(decided)
            print(f"[phase] {todo[i][0][:4]} vs {todo[i][1][:4]}: {s0}-{s1} "
                  f"{'W' if s0>s1 else 'L'} ({len(results)}/144)", flush=True)
    w = MultiOpponentSimWorker(
        student, [("phillip:gm", gm, list(range(N)), False, gc)],
        N, 240, "/home/kage/drive2/ShineBot/msl-data",
        stages, pairs, name_code=sc, device=DEV, record_fn=on_game)
    frames = 0
    while not all(decided) and frames < 30000:   # ~8 game-min cap
        w.collect(240); frames += 240
    w.close()
    print(f"phase {phase} done: {sum(decided)}/{N} decided in {frames} frames", flush=True)

W = sum(1 for s0, s1 in results.values() if s0 > s1)
L = sum(1 for s0, s1 in results.values() if s1 > s0)
print(f"\n==== v10-final vs phillip-gm: {W}-{L} over {len(results)}/144 pairs ====")
rows = {}
for (a, b), (s0, s1) in results.items():
    rows.setdefault(a, []).append((b, s0 > s1))
print("\nper-student-char (wins/games):")
for a in CH:
    r = rows.get(a, [])
    print(f"  {a:13s} {sum(1 for _, w_ in r if w_):2d}/{len(r):2d}  "
          + " ".join(f"{b[:2]}{'+' if w_ else '-'}" for b, w_ in sorted(r)))
json.dump({f"{a}|{b}": [s0, s1] for (a, b), (s0, s1) in results.items()},
          open(os.path.expanduser("~/h2h_v10_gm_matrix.json"), "w"))
print("matrix saved to ~/h2h_v10_gm_matrix.json")
