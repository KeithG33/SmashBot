"""End to end on melee-sim-light: a grid seat running at a longer delay than
the student is harvested as what it actually sent to the sim, sliced with
the student's delay; a self-play game's second seat plays but isn't learned
from."""
import sys

import numpy as np
import pytest
import torch
import tree

from smashbot import configs, embed as embed_lib, encode, paths
from smashbot.policy import build_policy

if paths.MELEE_SIM_DIR.is_dir():
    sys.path.insert(0, str(paths.MELEE_SIM_DIR))
msl = pytest.importorskip("melee_sim")
if not paths.MSL_DATA_DIR.is_dir():
    pytest.skip(f"no melee-sim-light data at {paths.MSL_DATA_DIR}", allow_module_level=True)

from smashbot.rl import sim_env, sim_league  # noqa: E402


def _policy(delay, seed):
    torch.manual_seed(seed)
    return build_policy(
        embed_config=embed_lib.EmbedConfig(),
        controller_config=embed_lib.ControllerConfig(),
        network_config=configs.NetworkConfig(name="sgu", num_layers=1, hidden_size=64,
                                             num_heads=1, window=4),
        head_config=configs.ControllerHeadConfig(residual_size=32, component_depth=0),
        policy_config=configs.PolicyConfig(delay=delay),
        num_names=4,
    )


def test_grid_seat_is_harvested_as_what_it_pressed(monkeypatch):
    T, N = 16, 2
    student, opponent = _policy(delay=2, seed=0), _policy(delay=5, seed=1)
    D = student.delay
    sent = []
    write = sim_env.write_controller_rows

    def record(env, rows, player):
        if player == 1:
            sent.append(rows.copy())
        write(env, rows, player)

    monkeypatch.setattr(sim_env, "write_controller_rows", record)
    grid = sim_league.PfspGrid(opponent, 1, N, 2, "cpu")
    grid.load(0, "opponent", lambda key: opponent.state_dict())
    grid.assign_static([list(range(N))])
    worker = sim_league.MultiOpponentSimWorker(
        student, N, T, str(paths.MSL_DATA_DIR), msl.Stage.FINAL_DESTINATION,
        [(msl.Character.FOX, msl.Character.FALCON)] * N, name_code=1, grids=[grid],
        max_frame=50)   # games end on the timer: resets inside the run
    _, chunks = worker.collect(12 * T)
    worker.close()

    sent = np.stack(sent, axis=1)                     # [N, frames, 13]
    embed = student.controller_head.controller_embedding
    assert len(chunks) >= 10
    assert any(bool((~c.valid).any()) for c in chunks)
    for c, chunk in enumerate(chunks):
        t0 = c * T
        encoded = embed.map(lambda e, x: x.numpy().astype(getattr(e, "dtype", x.numpy().dtype)),
                            chunk.actions.controller_state)
        pressed = encode.controller_rows(embed.decode(encoded))      # [N, T+1, 13]
        np.testing.assert_array_equal(pressed, sent[:, t0 + D - 1:t0 + T + D])
        assert (chunk.name == 1).all() and chunk.valid.any()


def test_self_play_second_seat_plays_but_is_not_a_learner_row(monkeypatch):
    """Self-play envs' second seats are the student forward's rows N..: what
    player 1 presses is exactly those rows, while the learner's chunks carry
    one row per game."""
    T, N = 16, 2
    student = _policy(delay=2, seed=0)
    sent = []
    write = sim_env.write_controller_rows

    def record(env, rows, player):
        if player == 1:
            sent.append(rows.copy())
        write(env, rows, player)

    monkeypatch.setattr(sim_env, "write_controller_rows", record)
    worker = sim_league.MultiOpponentSimWorker(
        student, N, T, str(paths.MSL_DATA_DIR), msl.Stage.FINAL_DESTINATION,
        [(msl.Character.FOX, msl.Character.FALCON)] * N, name_code=1,
        self_idx=range(N), max_frame=50)
    executed = []
    execute = worker.student.execute

    def execute_and_record(resets):
        rows = execute(resets)
        executed.append(np.stack(rows))
        return rows

    worker.student.execute = execute_and_record
    chunks, _ = worker.collect(4 * T)
    worker.close()

    assert (worker.student.num_envs, worker.rows) == (2 * N, N)
    assert len(chunks) >= 2
    for chunk in chunks:
        per_row = [chunk.rewards, chunk.name, chunk.is_resetting,
                   *tree.flatten(chunk.states), *tree.flatten(chunk.initial_state)]
        assert all(t.shape[0] == N for t in per_row if isinstance(t, torch.Tensor) and t.dim())
    np.testing.assert_array_equal(np.stack(sent), np.stack(executed)[:, N:])


def test_sharded_sim_steps_as_one_batch():
    """Envs split across sim batches stepped on parallel threads give the same
    chunks as one batch: uneven shards, every env its own matchup and game
    length (so a row routed to the wrong env, or a reset credited to the
    wrong one, shows), games ending and reconfigured mid-run."""
    T, N = 16, 5
    C = msl.Character
    pairs = [(C.FOX, C.FALCON), (C.MARTH, C.SHEIK), (C.FALCO, C.PEACH),
             (C.JIGGLYPUFF, C.FOX), (C.SAMUS, C.MARTH)]

    def match(e, member):
        return msl.MatchConfig(stage=msl.Stage.BATTLEFIELD, seed=e, max_frame=40 + 9 * e,
                               players=tuple(msl.PlayerConfig(c) for c in pairs[e])), None

    def run(shards):
        student = _policy(delay=2, seed=0)
        worker = sim_league.MultiOpponentSimWorker(
            student, N, T, str(paths.MSL_DATA_DIR), None, None, name_code=1,
            self_idx=range(N), match_fn=match, shards=shards)
        torch.manual_seed(1)
        chunks, _ = worker.collect(12 * T)
        worker.close()
        return chunks

    single, sharded = run(1), run(2)
    assert len(single) == len(sharded) >= 10
    resets = torch.cat([c.is_resetting[:, 1:] for c in single], dim=1)
    assert resets.any() and not (resets == resets[:1]).all(), "resets must happen, staggered"
    for a, b in zip(tree.flatten(single), tree.flatten(sharded)):
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a, b)
