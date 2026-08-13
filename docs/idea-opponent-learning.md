# Opponent Advantage Imitation

In PPO league training, opponent trajectories can also be used as demonstrations.

For each opponent state-action pair (s, a_opp), estimate how good the opponent
action was and use that to weight an imitation loss, so good opponent actions
increase in probability under our policy while bad ones contribute no gradient.
Opponent wins/losses do not directly decide what gets copied — advantage per
action does: a losing Phillip still executes sequences worth harvesting, and a
winning one commits errors worth ignoring. The critic learns from
returns/rewards; the actor learns through the weighted imitation loss. Normal
PPO continues on the agent's own on-policy data.

## Literature grounding

This is the Self-Imitation Learning / MARWIL family, pointed at opponent data:

- **SIL** (Oh et al., ICML 2018): `-max(A,0) * log pi(a|s)` on the agent's own
  past good transitions, alternated with on-policy A2C/PPO.
- **MARWIL** (Wang et al., NeurIPS 2018): `exp(beta*A)`-weighted imitation of
  *another player's* demonstration data — the closest setting to ours — with a
  proof that the learned policy can exceed the demonstrator (the answer to
  "imitation caps you at Phillip": it doesn't, with advantage weighting).
- **AWR / AWAC** (Peng et al. 2019; Nair et al. 2020): same exponential
  advantage weighting as the general offline/off-policy actor update.
- **Mimicking To Dominate** (2023): a SIBLING branch, not an ancestor — uses
  imitation to PREDICT opponents' actions (opponent modeling as an auxiliary,
  jointly trained with the policy; SOTA claims on SMACv2). Distinct mechanism
  from harvesting opponent actions into our own policy, but suggests a cheap
  complementary experiment: an auxiliary opponent-action-prediction head to
  shape representations, with zero imitation-cap risk.
- Opponent-aware league training (NeurIPS 2023) extends AlphaStar-style
  leagues with opponent-conditioned robustness — league-design branch.

Why not importance-sampled off-policy PG (V-trace-style) instead: IS ratios
truncate exactly where our policy and the opponent's disagree most — which is
precisely the region containing the tricks we don't know. Advantage-weighted
imitation has no actor-side IS ratio, so it learns MOST from disagreement.
The opponent's losses still inform training through the critic's value
targets; they just don't push probability mass directly.

## Estimator: no Q-function needed

Use the trajectory's own n-step / Monte-Carlo return as the sample estimate of
Q(s, a_opp) — "what actually followed from the action the opponent took":

```
A(s_t, a_opp_t) = G_t - V(s_t)        # G_t from the opponent's seat
```

- `G_t`: discounted return along the opponent's trajectory from t (zero-sum
  mirror of our rewards — already computed for every opponent seat).
- `V(s_t)`: our critic evaluated on the opponent-view state (the seat-swapped
  encoding already exists; the critic carries the 12-character BC prior).
- Detach A before the policy loss (actor gradient must not leak into the
  critic estimate).

## Loss (v1 baseline = literature standard)

```
w = clip(exp(A_norm / beta), max=w_cap)   # MARWIL/AWR weighting
L_opp = -w * log pi(a_opp | s)
L_total = L_ppo + lambda_t * L_opp
```

- **Weighting**: exponential (MARWIL/AWR) is the literature baseline for
  demonstration data; hard `max(A,0)` (SIL) is our first ablation, not the
  default. Normalize advantages before exponentiating; cap weights (~20,
  AWR-style) so no single outlier action dominates a batch.
- **Coefficient**: small and decaying — lambda ~= 0.01-0.05 relative to the
  PPO loss, annealed over training. Literature is consistent: these auxiliary
  terms help most early and must never dominate the gradient.

## Scope and phasing

- **Which opponents**: Phillip (medium-v2-torch) and strong snapshots only.
  CPU lvl-9 demonstrations are worthless (food, not teachers). This is a
  strong-opponent mechanism and its budget should concentrate there.
- **Phase 1 (fox-only student, current runs)**: mirror-Fox envs only — the
  critic's value estimates are cleanest where the demonstrated seat matches
  what the student actually plays. Validate via the fixed-yardstick battery
  before widening.
- **Phase 2 (v4: all-character student)**: the full-power version. Fox-only
  wastes 11/12ths of Phillip's demonstrations; an all-character student can
  harvest every seat, matches slippi-ai's own RL scope, and enriches the
  league. The per-game character-mutation machinery built for opponents works
  identically on the student's seat, so this is nearly config-level.
  Experiment: if the all-character model beats the fox specialist (battery
  says), the specialist retires into the league as another ghost opponent.

## Batching (memory-neutral)

GPU budget is fixed (see the OOM history): opponent-trajectory data must not
grow the learner batch. Substitute opponent slots into the existing batch
(e.g., swap k of the N env-trajectories per update), or alternate updates
between own-data and mixed batches. The mix ratio is a tunable; substitution
forces the mechanism to justify each slot it takes from ordinary experience.

## Evaluation

A/B against the plain-PPO baseline via the battery (fixed 50-game evals vs
frozen teacher + Phillip at matched checkpoints). Success = faster R% growth
at equal-or-better teacher/battery numbers; failure mode to watch = style
drift toward Phillip visible as teacher-KL inflation without win-rate gains.

## References

- Oh et al., Self-Imitation Learning, ICML 2018 — proceedings.mlr.press/v80/oh18b.html
- Wang et al., MARWIL: Exponentially Weighted Imitation Learning for Batched Historical Data, NeurIPS 2018
- Peng et al., Advantage-Weighted Regression, 2019; Nair et al., AWAC, 2020
- Mimicking To Dominate: Imitation Learning Strategies for Success in Multiagent Competitive Games, 2023 — arxiv.org/pdf/2308.10188
- A Robust and Opponent-Aware League Training Method for StarCraft II, NeurIPS 2023
- Vinyals et al., AlphaStar / Grandmaster-level StarCraft II, Nature 2019
