"""Tabular TD-control agents (COEN874 Week 3, slides 13-14).

Q-learning (off-policy):
    Q(S_t,A_t) <- Q(S_t,A_t) + alpha [ r_{t+1} + gamma max_a' Q(S_{t+1},a') - Q(S_t,A_t) ]
SARSA (on-policy):
    Q(S_t,A_t) <- Q(S_t,A_t) + alpha [ r_{t+1} + gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t) ]

Terminal transitions (hole or goal) do not bootstrap. Time-limit truncation is
not a true termination, so it bootstraps (for SARSA from an action sampled
from the behaviour policy but not executed).
"""
from __future__ import annotations

import numpy as np

ALGORITHMS = ("qlearning", "sarsa")


def linear_epsilon(t: int, budget: int, eps_start: float, eps_end: float, decay_fraction: float) -> float:
    """GLIE-style linear decay over the first decay_fraction of the step budget."""
    decay_steps = max(1, int(decay_fraction * budget))
    frac = min(1.0, t / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def epsilon_greedy(row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """epsilon-greedy with uniform random tie-breaking among maximal actions.

    Random tie-breaking matters for quantised tables, where ties are common.
    """
    n = row.shape[0]
    if rng.random() < eps:
        return int(rng.integers(n))
    best = np.flatnonzero(row == row.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def td_target(algo: str, r: float, terminated: bool, gamma: float, next_row: np.ndarray | None,
              next_action: int | None) -> float:
    if terminated:
        return float(r)
    if algo == "qlearning":
        return float(r + gamma * next_row.max())
    if algo == "sarsa":
        return float(r + gamma * next_row[next_action])
    raise ValueError(f"unknown algorithm {algo!r}")


def td_update(table, s: int, a: int, target: float, alpha: float) -> float:
    """Apply one TD update; returns the TD error delta_t (computed in float)."""
    old = table.get(s, a)
    delta = target - old
    table.store(s, a, old + alpha * delta)
    return delta
