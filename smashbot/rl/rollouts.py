"""Shared rollout-collection primitives (backend-agnostic):

- ChunkAssembler: per-frame records + transition rewards -> [N, T+1]
  Trajectory chunks with the delay-shifted reward alignment the learner
  expects (reward slot t = the game transition at sample-time t + delay).
- compute_reward: stock/percent deltas -> reward, zeroed at game boundaries.
- GameTracker: per-opponent-class win/stock/kill-percent statistics.

Consumed by the sim training worker (rl/sim_league.py) and the sim eval
arena (eval/sim_arena.py). The Dolphin RL fleet that lived here was removed
after training AND eval moved to melee-sim-light (git history has it);
Dolphin remains for human play/watching via eval/game.py.
"""

from __future__ import annotations

import typing as tp

import torch
import tree

from smashbot.rl.agent import FrameRecord  # noqa: F401  (typing)
from smashbot.rl.ppo import ActionData, Trajectory, slice_trajectory_rows


class ChunkAssembler:
    """Accumulates FrameRecords + rewards, emits [N, T+1] Trajectory chunks.

    push_frame() every frame (with each env's is_resetting flag and the
    agent's hidden snapshot at chunk starts); push_reward() every transition
    (aligned to real time). A chunk covering sample-times [0, T] emits once
    rewards through real-transition T + delay - 1 have arrived. Chunks
    overlap by one frame, per the Frames convention.
    """

    def __init__(self, unroll_length: int, delay: int):
        self.T = unroll_length
        self.delay = delay
        self._records: list[FrameRecord] = []
        self._resets: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._initial_state: tp.Any = None
        self._next_initial: tp.Any = None

    def push_frame(
        self,
        record: FrameRecord,
        is_resetting: torch.Tensor,
        hidden_snapshot=None,
    ) -> None:
        """hidden_snapshot must be provided whenever this frame starts a chunk
        (every `unroll_length` frames, including the very first): it is the
        agent's recurrent state BEFORE stepping this frame."""
        if hidden_snapshot is not None:
            if not self._records:
                self._initial_state = hidden_snapshot
            else:
                self._next_initial = hidden_snapshot
        self._records.append(record)
        self._resets.append(is_resetting)

    def push_reward(self, reward: torch.Tensor) -> None:  # [N]
        self._rewards.append(reward)

    def ready(self) -> bool:
        return (
            len(self._records) >= self.T + 1
            and len(self._rewards) >= self.T + self.delay
        )

    def emit(self) -> Trajectory:
        assert self.ready()
        T, D = self.T, self.delay
        stack = lambda seq: tree.map_structure(
            lambda *xs: torch.stack(xs, dim=1), *seq
        )
        records = self._records[: T + 1]
        traj = Trajectory(
            states=stack([r.state for r in records]),
            name=torch.stack([r.name for r in records], dim=1),
            actions=ActionData(
                controller_state=stack([r.prev_action for r in records]),
                logits=stack([r.logits for r in records]),
            ),
            # reward slot t <- real transition t + D ("rewards that follow
            # actions"), matching the BC value-training alignment.
            rewards=torch.stack(self._rewards[D : T + D], dim=1),
            is_resetting=torch.stack(self._resets[: T + 1], dim=1),
            initial_state=self._initial_state,
        )
        # Keep the overlap frame and the not-yet-consumed reward tail.
        self._records = self._records[T:]
        self._resets = self._resets[T:]
        self._rewards = self._rewards[T:]
        self._initial_state = self._next_initial
        self._next_initial = None
        return traj


def compute_reward(
    prev_stocks: torch.Tensor,  # [N, 2] (own, opp)
    stocks: torch.Tensor,
    prev_percent: torch.Tensor,  # [N, 2]
    percent: torch.Tensor,
    is_resetting: torch.Tensor,  # [N]
    damage_ratio: float = 0.01,
) -> torch.Tensor:
    """Zero-sum reward from the bot's perspective, zeroed at game boundaries.

    death: stock decrease. damage: positive percent delta (percent resets to
    zero on death; negative deltas are ignored).
    """
    own_death = (stocks[:, 0] < prev_stocks[:, 0]).float()
    opp_death = (stocks[:, 1] < prev_stocks[:, 1]).float()
    # Percent is a raw libmelee read; the state path wraps+clamps it but
    # the reward path would pass garbage straight through. Nothing deals
    # 100% in one frame, so the delta cap keeps |reward| <= 2.
    own_dmg = (percent[:, 0] - prev_percent[:, 0]).clamp(min=0, max=100)
    opp_dmg = (percent[:, 1] - prev_percent[:, 1]).clamp(min=0, max=100)
    reward = (opp_death - own_death) + damage_ratio * (opp_dmg - own_dmg)
    return torch.where(is_resetting, torch.zeros_like(reward), reward)


class GameTracker:
    """Game-outcome metrics vs the CURRENT training opponent (teacher now,
    snapshot pool later); fixed-yardstick evals stay in the M8 batteries.

    Time-free by design: a win is a win at 2 minutes or 7. Tracks rolling
    win rate, average final stock differential (-4..+4 dominance scale),
    average opponent percent at our kills (low = early kills, strong punish
    game), and average own percent at our deaths (high = hard to kill)."""

    # ema_alpha 0.008 ~ a 250-game horizon: several full fleet waves, so
    # the EMA reflects rounds rather than single-batch luck. Restored
    # checkpoints store EMA values only, so alpha changes apply cleanly.
    def __init__(self, window: int = 100, event_window: int = 200,
                 ema_alpha: float = 0.008):
        import collections

        self.diffs = collections.deque(maxlen=window)  # per finished game
        self.kill_percents = collections.deque(maxlen=event_window)
        self.death_percents = collections.deque(maxlen=event_window)
        self.wins = self.losses = self.draws = 0
        # EMA companion to the window: smoother (no window-exit jumps) and
        # persistable across restarts via state()/load_state — the window
        # resets every boot; the EMA rides in the RL checkpoint.
        self.ema_alpha = ema_alpha
        self.win_ema: float | None = None
        self.diff_ema: float | None = None
        self.by_char: dict[str, tuple[int, int]] = {}

    def add_game(self, final_stocks: tuple[int, int],
                 opp_char: str | None = None) -> None:
        bot, opp = final_stocks
        diff = bot - opp
        # Winrate by OPPONENT character (locked members excluded at the
        # call site: their identity would pollute their char's column).
        if opp_char and diff != 0:
            w, g = self.by_char.get(opp_char, (0, 0))
            self.by_char[opp_char] = (w + (1 if diff > 0 else 0), g + 1)
        self.diffs.append(diff)
        if bot > opp:
            self.wins += 1
        elif opp > bot:
            self.losses += 1
        else:
            self.draws += 1
        if diff != 0:  # EMA over decided games, matching win_rate_recent
            outcome = 1.0 if diff > 0 else 0.0
            a = self.ema_alpha
            # seed at the 0.5 prior, not the first outcome: an extreme seed
            # takes ~200 games to wash out at this alpha
            prev = 0.5 if self.win_ema is None else self.win_ema
            self.win_ema = (1 - a) * prev + a * outcome
        a = self.ema_alpha
        prev_d = 0.0 if self.diff_ema is None else self.diff_ema
        self.diff_ema = (1 - a) * prev_d + a * diff

    def state(self) -> dict:
        """Persistable summary state (EMA VALUES + lifetime counters); the
        raw windows are boot-local by design. ema_alpha is deliberately NOT
        persisted: the horizon is a code-level tuning knob, so restored
        checkpoints pick up the current default automatically."""
        return {"win_ema": self.win_ema, "diff_ema": self.diff_ema,
                "wins": self.wins, "losses": self.losses,
                "draws": self.draws}

    def load_state(self, st: dict) -> None:
        self.win_ema = st.get("win_ema")
        self.diff_ema = st.get("diff_ema")
        self.wins = st.get("wins", 0)
        self.losses = st.get("losses", 0)
        self.draws = st.get("draws", 0)

    def add_kill(self, opp_percent: float) -> None:
        self.kill_percents.append(opp_percent)

    def add_death(self, own_percent: float) -> None:
        self.death_percents.append(own_percent)

    def stats(self) -> dict:
        mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0
        decided = [d for d in self.diffs if d != 0]
        return {
            "games_played": self.wins + self.losses + self.draws,
            "win_rate_recent": (
                sum(1 for d in decided if d > 0) / len(decided) if decided else 0.5
            ),
            "avg_stock_diff": mean(self.diffs),
            "avg_percent_at_kill": mean(self.kill_percents),
            "avg_percent_at_death": mean(self.death_percents),
            "win_rate_ema": self.win_ema if self.win_ema is not None else 0.5,
            "stock_diff_ema": self.diff_ema if self.diff_ema is not None else 0.0,
        }


