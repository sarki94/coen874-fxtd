import numpy as np
import pytest

from fxtd.qtable import FixedPointQTable, FloatQTable, make_qtable


def table(bits=8, rounding="rtn", seed=0):
    return FixedPointQTable(4, 2, bits, rounding, np.random.default_rng(seed))


def test_grid_spacing_and_range():
    t = table(bits=8)
    assert t.delta == pytest.approx(1 / 255)
    assert t.decode(0) == 0.0 and t.decode(255) == pytest.approx(1.0)
    assert t.codes.dtype == np.uint8
    assert table(bits=12).codes.dtype == np.uint16


def test_rtn_rounds_to_nearest_and_clips():
    t = table(bits=4)                     # delta = 1/15
    d = t.delta
    assert t.encode(0.4 * d) == 0
    assert t.encode(0.6 * d) == 1
    assert t.encode(-0.3) == 0 and t.encode(1.7) == 15


def test_rtn_loses_updates_below_half_a_step():
    t = table(bits=8)
    t.store(0, 0, 0.4 * t.delta)          # change smaller than delta/2
    assert t.get(0, 0) == 0.0
    assert t.n_lost == 1 and t.n_updates == 0
    t.store(0, 0, 0.6 * t.delta)
    assert t.get(0, 0) == pytest.approx(t.delta)
    assert t.n_updates == 1


def test_stochastic_rounding_is_unbiased():
    t = table(bits=4, rounding="sr", seed=1)
    x = 0.3 * t.delta + 5 * t.delta
    vals = np.array([t.decode(t.encode(x)) for _ in range(20000)])
    assert set(np.unique(np.round(vals / t.delta))) <= {5.0, 6.0}
    assert vals.mean() == pytest.approx(x, abs=3 * t.delta * 0.5 / np.sqrt(20000) + 1e-4)


def test_stochastic_rounding_exact_values_unchanged():
    t = table(bits=6, rounding="sr", seed=2)
    for c in [0, 7, 63]:
        assert t.encode(t.decode(c)) == c


def test_float_table_and_factory():
    f = make_qtable(3, 2, "fp32", "none", np.random.default_rng(0))
    assert isinstance(f, FloatQTable)
    f.store(1, 1, 0.123)
    assert f.get(1, 1) == pytest.approx(0.123, rel=1e-6)
    assert f.n_lost == 0 and f.n_updates == 1
    q = make_qtable(3, 2, "8", "sr", np.random.default_rng(0))
    assert isinstance(q, FixedPointQTable) and q.bits == 8


def test_memory_footprint():
    assert table(bits=8).nbytes() == 8             # 4x2 entries, 1 byte each
    assert FloatQTable(64, 4).nbytes() == 1024    # 64x4 float32
    assert FixedPointQTable(64, 4, 6, "rtn", np.random.default_rng(0)).nbytes() == 192


def test_invalid_arguments():
    with pytest.raises(ValueError):
        table(bits=1)
    with pytest.raises(ValueError):
        table(rounding="floor")
