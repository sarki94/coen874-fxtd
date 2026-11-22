import numpy as np
import pytest

from fxtd.stats import bootstrap_ci, cliffs_delta, hedges_g, holm, kruskal, mann_whitney


def test_holm_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, max(0.06, 0.04*1)=0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_cliffs_delta_extremes_and_ties():
    assert cliffs_delta([2, 3, 4], [0, 0, 0]) == 1.0
    assert cliffs_delta([0, 0], [1, 1]) == -1.0
    assert cliffs_delta([1, 1], [1, 1]) == 0.0


def test_hedges_g_sign_and_zero_variance():
    assert hedges_g([1, 2, 3, 4], [0, 1, 2, 3]) > 0
    assert np.isnan(hedges_g([1, 1, 1], [1, 1, 1]))


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(5, 1, 50)
    m, lo, hi = bootstrap_ci(x)
    assert lo < m < hi and abs(m - 5) < 0.5


def test_tests_handle_identical_constant_groups():
    assert mann_whitney([0, 0, 0], [0, 0, 0]) == 1.0
    assert kruskal([0, 0], [0, 0], [0, 0]) == 1.0
    assert mann_whitney([1, 2, 3, 4, 5], [0, 0, 0, 0, 0], alternative="greater") < 0.01
