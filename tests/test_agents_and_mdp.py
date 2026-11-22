import numpy as np
import pytest

from fxtd.agents import epsilon_greedy, linear_epsilon, td_target, td_update
from fxtd.mdp import TabularModel, greedy_policy, make_env
from fxtd.qtable import FloatQTable

ENV = {"id": "FrozenLake-v1", "map_name": "4x4", "is_slippery": True}


def test_q_learning_update_matches_lecture_worked_example():
    # Week 3, slide 16: alpha=0.5, gamma=0.9, r=+2, Q(S1,right)=1.0, max_a' Q(S2,a')=3.0 -> 2.85
    t = FloatQTable(3, 2)
    t.store(1, 1, 1.0)
    t.store(2, 0, 3.0)
    target = td_target("qlearning", 2.0, False, 0.9, t.row(2), None)
    assert target == pytest.approx(4.7)
    delta = td_update(t, 1, 1, target, alpha=0.5)
    assert delta == pytest.approx(3.7)
    assert t.get(1, 1) == pytest.approx(2.85)


def test_sarsa_uses_the_action_actually_taken():
    row = np.array([0.2, 0.9])
    assert td_target("sarsa", 0.0, False, 1.0, row, next_action=0) == pytest.approx(0.2)
    assert td_target("qlearning", 0.0, False, 1.0, row, None) == pytest.approx(0.9)


def test_terminal_transition_does_not_bootstrap():
    row = np.array([5.0, 5.0])
    assert td_target("qlearning", 1.0, True, 0.99, row, None) == 1.0
    assert td_target("sarsa", 0.0, True, 0.99, row, 1) == 0.0


def test_epsilon_schedule():
    assert linear_epsilon(0, 1000, 1.0, 0.05, 0.6) == 1.0
    assert linear_epsilon(300, 1000, 1.0, 0.05, 0.6) == pytest.approx(0.525)
    assert linear_epsilon(600, 1000, 1.0, 0.05, 0.6) == pytest.approx(0.05)
    assert linear_epsilon(999, 1000, 1.0, 0.05, 0.6) == pytest.approx(0.05)


def test_epsilon_greedy_breaks_ties_randomly_and_exploits():
    rng = np.random.default_rng(0)
    picks = {epsilon_greedy(np.zeros(4), 0.0, rng) for _ in range(200)}
    assert picks == {0, 1, 2, 3}
    assert all(epsilon_greedy(np.array([0, 1.0, 0, 0]), 0.0, rng) == 1 for _ in range(50))


def test_value_iteration_reproduces_week3_lab_values():
    # Week 3 lab output (4x4, gamma=0.99): V*(0)=0.542, V*(14)=0.863
    model = TabularModel.from_env(make_env(ENV))
    V, Q = model.value_iteration(0.99)
    assert V[0] == pytest.approx(0.542, abs=1e-3)
    assert V[14] == pytest.approx(0.863, abs=1e-3)
    # the exact value of the greedy optimal policy equals V*
    assert model.policy_value(greedy_policy(Q), 0.99) == pytest.approx(V, abs=1e-8)


def test_model_rows_are_distributions():
    model = TabularModel.from_env(make_env({"id": "FrozenLake-v1", "map_name": "8x8", "is_slippery": True}))
    assert model.T.shape == (64, 4, 64)
    assert np.allclose(model.T.sum(axis=2), 1.0)
    rnd = model.random_policy_value(0.99)
    V, _ = model.value_iteration(0.99)
    assert 0.0 <= rnd[0] < V[0]
