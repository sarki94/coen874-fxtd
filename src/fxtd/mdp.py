"""Environment helpers and exact (model-based) reference computations.

The agents never see the model. The model, read from env.unwrapped.P exactly
as in the COEN874 Week 3 lab, is used only for the reference points: the
dynamic-programming optimum V* (value iteration) and the exact value of a
learned greedy policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


def make_env(env_cfg: dict) -> gym.Env:
    return gym.make(
        env_cfg["id"],
        map_name=env_cfg.get("map_name", "8x8"),
        is_slippery=env_cfg.get("is_slippery", True),
        disable_env_checker=True,
    )


@dataclass
class TabularModel:
    T: np.ndarray        # (S, A, S') transition probabilities
    R: np.ndarray        # (S, A) expected immediate reward
    D: np.ndarray        # (S, A, S') 1.0 if the transition terminates
    n_states: int
    n_actions: int

    @classmethod
    def from_env(cls, env: gym.Env) -> "TabularModel":
        P = env.unwrapped.P
        nS, nA = env.observation_space.n, env.action_space.n
        T = np.zeros((nS, nA, nS)); R = np.zeros((nS, nA)); D = np.zeros((nS, nA, nS))
        for s in range(nS):
            for a in range(nA):
                for p, s2, r, done in P[s][a]:
                    T[s, a, s2] += p
                    R[s, a] += p * r
                    if done:
                        D[s, a, s2] = 1.0
        return cls(T, R, D, nS, nA)

    # --- dynamic programming ----------------------------------------------
    def value_iteration(self, gamma: float, theta: float = 1e-12, max_iter: int = 100_000):
        """Bellman optimality backup (Week 3, slide 8). Returns V*, Q*."""
        V = np.zeros(self.n_states)
        cont = self.T * (1.0 - self.D)
        for _ in range(max_iter):
            Q = self.R + gamma * cont @ V
            V_new = Q.max(axis=1)
            if np.max(np.abs(V_new - V)) < theta:
                V = V_new
                break
            V = V_new
        Q = self.R + gamma * cont @ V
        return V, Q

    def policy_value(self, policy: np.ndarray, gamma: float) -> np.ndarray:
        """Exact V^pi for a deterministic policy (solves the Bellman expectation equation)."""
        idx = np.arange(self.n_states)
        cont = (self.T * (1.0 - self.D))[idx, policy]          # (S, S')
        r = self.R[idx, policy]
        return np.linalg.solve(np.eye(self.n_states) - gamma * cont, r)

    def random_policy_value(self, gamma: float) -> np.ndarray:
        """Exact V for the uniform random policy (reference baseline)."""
        cont = (self.T * (1.0 - self.D)).mean(axis=1)
        r = self.R.mean(axis=1)
        return np.linalg.solve(np.eye(self.n_states) - gamma * cont, r)


def greedy_policy(q_values: np.ndarray) -> np.ndarray:
    """Greedy policy with first-index tie-breaking (np.argmax, as in the lab)."""
    return np.argmax(q_values, axis=1)


def rollout_success_rate(env: gym.Env, policy: np.ndarray, n_episodes: int, seed: int) -> tuple[float, float]:
    """Run the greedy policy with exploration off on a dedicated evaluation env.

    Returns (success rate, mean episode length). The episode limit is the
    environment's registered TimeLimit.
    """
    successes = 0
    lengths = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        steps = 0
        while True:
            obs, r, terminated, truncated, _ = env.step(int(policy[obs]))
            steps += 1
            if terminated or truncated:
                successes += int(r > 0)
                break
        lengths += steps
    return successes / n_episodes, lengths / n_episodes
