"""One training run = (algorithm, precision, rounding, seed) under a fixed step budget.

Everything that is logged is returned as arrays so that it can be saved raw:
every episode (end step, return, length) and every evaluation checkpoint.
"""
from __future__ import annotations

import random
import time

import numpy as np

from .agents import epsilon_greedy, linear_epsilon, td_target, td_update
from .mdp import TabularModel, greedy_policy, make_env, rollout_success_rate
from .qtable import make_qtable


def seed_everything(seed: int) -> None:
    """Seed the global Python and NumPy generators (PyTorch is not used in this project)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))


def run(cfg: dict, algo: str, precision: str, rounding: str, seed: int) -> dict:
    env_cfg, tr, ev = cfg["env"], cfg["train"], cfg["eval"]
    gamma = float(cfg["gamma"])
    hp = cfg["hyperparameters"][algo]
    alpha, decay_fraction = float(hp["alpha"]), float(hp["eps_decay_fraction"])
    eps_start, eps_end = float(tr["eps_start"]), float(tr["eps_end"])
    budget = int(tr["budget_steps"])
    eval_every = int(ev["every_steps"])

    seed_everything(seed)
    ss = np.random.SeedSequence([seed, 874])
    rng_behaviour, rng_round = (np.random.default_rng(s) for s in ss.spawn(2))

    env = make_env(env_cfg)                  # training instance
    eval_env = make_env(env_cfg)             # separate evaluation instance
    model = TabularModel.from_env(env)
    nS, nA = model.n_states, model.n_actions
    table = make_qtable(nS, nA, precision, rounding, rng_round,
                        cfg["fixed_point"]["v_min"], cfg["fixed_point"]["v_max"])

    ckpt_steps, ckpt_value, ckpt_success, ckpt_len = [], [], [], []
    ckpt_updates, ckpt_lost, ckpt_eps = [], [], []
    ep_end, ep_return, ep_len = [], [], []

    def evaluate(t: int, eps: float) -> None:
        pi = greedy_policy(table.values())
        v0 = model.policy_value(pi, gamma)[0]
        sr, ln = rollout_success_rate(eval_env, pi, int(ev["rollouts"]), int(ev["seed_base"]))
        ckpt_steps.append(t); ckpt_value.append(v0); ckpt_success.append(sr); ckpt_len.append(ln)
        ckpt_updates.append(table.n_updates); ckpt_lost.append(table.n_lost); ckpt_eps.append(eps)

    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    eps = eps_start
    action = epsilon_greedy(table.row(obs), eps, rng_behaviour)
    g, length = 0.0, 0
    evaluate(0, eps)
    for t in range(1, budget + 1):
        if algo == "qlearning":
            action = epsilon_greedy(table.row(obs), eps, rng_behaviour)
        next_obs, r, terminated, truncated, _ = env.step(action)
        eps = linear_epsilon(t, budget, eps_start, eps_end, decay_fraction)
        next_action = None
        if not terminated and algo == "sarsa":
            next_action = epsilon_greedy(table.row(next_obs), eps, rng_behaviour)
        target = td_target(algo, r, terminated, gamma,
                           None if terminated else table.row(next_obs), next_action)
        td_update(table, obs, action, target, alpha)
        g += r; length += 1
        if terminated or truncated:
            ep_end.append(t); ep_return.append(g); ep_len.append(length)
            g, length = 0.0, 0
            obs, _ = env.reset()
            if algo == "sarsa":
                action = epsilon_greedy(table.row(obs), eps, rng_behaviour)
        else:
            obs = next_obs
            if algo == "sarsa":
                action = next_action
        if t % eval_every == 0:
            evaluate(t, eps)
    wall = time.perf_counter() - t0

    q_final = table.values()
    pi_final = greedy_policy(q_final)
    final_sr, final_len = rollout_success_rate(eval_env, pi_final, int(ev["final_rollouts"]),
                                               int(ev["seed_base"]) + 1_000_000)
    env.close(); eval_env.close()
    return dict(
        algo=algo, precision=str(precision), rounding=rounding, seed=seed,
        ckpt_steps=np.array(ckpt_steps), ckpt_value=np.array(ckpt_value),
        ckpt_success=np.array(ckpt_success), ckpt_len=np.array(ckpt_len),
        ckpt_updates=np.array(ckpt_updates), ckpt_lost=np.array(ckpt_lost), ckpt_eps=np.array(ckpt_eps),
        ep_end=np.array(ep_end), ep_return=np.array(ep_return), ep_len=np.array(ep_len),
        q_final=q_final, final_value=float(model.policy_value(pi_final, gamma)[0]),
        final_success=final_sr, final_len=final_len,
        n_updates=table.n_updates, n_lost=table.n_lost, table_bytes=table.nbytes(),
        wall_seconds=wall,
    )
