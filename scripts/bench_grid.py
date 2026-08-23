"""League grid scaling: captured-replay ms/frame and VRAM per (slices x
cells) factorization of the same cell count, fp32 vs fp16 stacked weights,
plus fp16-vs-fp32 controller agreement at saturated sampling.

    .venv/bin/python scripts/bench_grid.py CKPT
"""
import sys, time, numpy as np, torch, tree
sys.path.insert(0, ".")
from smashbot.eval.game import load_policy
from smashbot.rl.agent import LeagueAgent
from smashbot import embed as embed_lib
from smashbot.tests.test_rollouts import _rand_raw_game

ckpt = sys.argv[1]
tmpl, _, _ = load_policy(ckpt, "cpu"); tmpl.train_value_head = False
game = embed_lib.EmbedConfig().make_game_embedding()
rng = np.random.default_rng(0)

def views_for(S, N):
    raw = _rand_raw_game(game, (S, N), rng)
    return tree.map_structure(lambda x: torch.from_numpy(np.ascontiguousarray(
        x.astype(np.int64) if x.dtype.kind in "iu" else x)).cuda(), game.from_state(raw))

def bench(S, N, dtype, frames=60):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    m0 = torch.cuda.memory_allocated()
    g = LeagueAgent(tmpl, S, N, name_code=1, device="cuda", weights_dtype=dtype, temperature=1e-6)
    v = views_for(S, N); r = torch.zeros(S, N, dtype=torch.bool, device="cuda")
    for _ in range(5): g.step(v, r)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(frames): g.step(v, r)
    torch.cuda.synchronize(); ms = (time.perf_counter() - t) / frames * 1e3
    mem = (torch.cuda.memory_allocated() - m0) / 2**30
    return ms, mem, g

print(f"{'grid':>8} {'cells':>5} {'dtype':>5} {'ms/frame':>9} {'VRAM GiB':>9}")
agree = {}
for S, N in [(12, 13), (16, 9), (24, 7), (48, 4)]:
    for dtype in (torch.float32, torch.float16):
        ms, mem, g = bench(S, N, dtype)
        print(f"{S:>3}x{N:<4} {S*N:>5} {str(dtype)[6:]:>5} {ms:>9.2f} {mem:>9.2f}")
        if (S, N) == (12, 13):
            agree[dtype] = g
        else:
            del g
# fp16 vs fp32 agreement on identical inputs / slices (T -> 0)
g32, g16 = agree[torch.float32], agree[torch.float16]
sd = tmpl.state_dict()
for s in range(12):
    g32.load_slice(s, sd); g16.load_slice(s, sd)
same = total = 0
for f in range(30):
    v = views_for(12, 13); r = torch.zeros(12, 13, dtype=torch.bool, device="cuda")
    a, _ = g32.step(v, r); b, _ = g16.step(v, r)
    same += int((a == b).all(axis=1).sum()); total += a.shape[0]
print(f"fp16 vs fp32 identical controller rows (saturated sampling): {same}/{total} = {same/total:.1%}")
