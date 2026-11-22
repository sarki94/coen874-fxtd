"""Theory used for the a-priori collapse prediction.

1. Dead-zone condition (round-to-nearest). A stored entry Q changes only if
   |alpha * delta| >= Delta / 2. Starting from Q = 0, an update driven by a
   successor value q can only be stored if alpha * gamma * q >= Delta / 2, i.e.

       q >= v_min(b) = Delta_b / (2 * alpha * gamma),   Delta_b = 1 / (2^b - 1).

   A state therefore cannot start learning until one of its successors holds
   a value above v_min(b): value cannot propagate across states whose values
   sit below the threshold.

2. Quantised expected-update ("mean-field") model. Replace the sampled TD
   update by its expectation under the model and apply the same storage
   rounding:

       Q_{k+1}(s,a) = round_b( Q_k(s,a) + alpha * [ (T Q_k)(s,a) - Q_k(s,a) ] ),

   with T the Bellman optimality operator for Q-learning, and for SARSA the
   Bellman expectation operator of the final epsilon-greedy behaviour policy.
   Under stochastic rounding the expected stored value equals the float update
   (E[SR(x)] = x), so the mean-field model of SR is the float model. Under RTN
   the iteration stalls at a dead-zone fixed point; the value of its greedy
   policy is the prediction. Sampling noise can occasionally push individual
   updates across the threshold, so the model is expected to be pessimistic
   (it predicts collapse at, or slightly above, the observed bit-width).
"""
from __future__ import annotations

import numpy as np

from .mdp import TabularModel, greedy_policy


def grid_step(bits: int, v_min: float = 0.0, v_max: float = 1.0) -> float:
    return (v_max - v_min) / ((1 << bits) - 1)


def propagation_threshold(bits: int, alpha: float, gamma: float) -> float:
    return grid_step(bits) / (2.0 * alpha * gamma)


def _rtn(x: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(np.rint(x / delta), 0, round(1 / delta)) * delta


def mean_field_fixed_point(model: TabularModel, algo: str, bits: int | None, alpha: float, gamma: float,
                           eps: float = 0.05, max_iter: int = 200_000, tol: float = 1e-12) -> np.ndarray:
    """Iterate the quantised expected update to its fixed point. bits=None -> float."""
    nS, nA = model.n_states, model.n_actions
    cont = model.T * (1.0 - model.D)
    delta = None if bits is None else grid_step(bits)
    Q = np.zeros((nS, nA))
    for _ in range(max_iter):
        if algo == "qlearning":
            nxt = Q.max(axis=1)
        else:
            best = (Q == Q.max(axis=1, keepdims=True)).astype(float)
            pi = eps / nA + (1 - eps) * best / best.sum(axis=1, keepdims=True)
            nxt = (pi * Q).sum(axis=1)
        target = model.R + gamma * cont @ nxt
        new = Q + alpha * (target - Q)
        if delta is not None:
            new = _rtn(new, delta)
        if np.max(np.abs(new - Q)) < tol:
            Q = new
            break
        Q = new
    return Q


def predict_retained(model: TabularModel, algo: str, bits_list, alpha: float, gamma: float,
                     eps: float = 0.05) -> dict:
    """Predicted V^pi(s0)/V*(s0) of the greedy policy at the RTN dead-zone fixed point."""
    V_star, _ = model.value_iteration(gamma)
    out = {}
    for b in bits_list:
        Q = mean_field_fixed_point(model, algo, b, alpha, gamma, eps)
        v0 = model.policy_value(greedy_policy(Q), gamma)[0]
        out[b] = float(v0 / V_star[0])
    return out


def predicted_collapse_bits(retained: dict, level: float = 0.10) -> int | None:
    """Largest tested bit-width whose predicted retained performance is below `level`."""
    collapsed = [b for b, r in retained.items() if r < level]
    return max(collapsed) if collapsed else None
