"""Q-tables with configurable numeric precision.

A fixed-point Q-table stores each entry as an unsigned integer code c in
{0, ..., 2^b - 1}; the represented value is c * Delta with
Delta = (v_max - v_min) / (2^b - 1) added to v_min. On FrozenLake every Q-value
lies in [0, 1] (single terminal reward of +1, gamma < 1), so the default grid
is [0, 1].

Arithmetic model: the TD update is computed in higher precision (float64,
standing in for a wide accumulator on a microcontroller) and only the
*stored* value is rounded back to b bits. This matches the usual low-precision
training model (Gupta et al., ICML 2015).

Rounding modes
--------------
rtn : round to nearest (ties to even, as numpy.rint).
sr  : stochastic rounding; round up with probability equal to the fractional
      part, so E[SR(x)] = x (unbiased).
"""
from __future__ import annotations

import numpy as np

ROUNDING_MODES = ("rtn", "sr")


class FloatQTable:
    """Reference 32-bit floating-point Q-table (the FP32 baseline)."""

    is_fixed_point = False

    def __init__(self, n_states: int, n_actions: int, rng: np.random.Generator | None = None):
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.n_updates = 0       # updates whose float change was non-zero
        self.n_lost = 0          # always 0 for float
        self.bits = 32
        self.rounding = "none"

    def values(self) -> np.ndarray:
        return self.q.astype(np.float64)

    def get(self, s: int, a: int) -> float:
        return float(self.q[s, a])

    def row(self, s: int) -> np.ndarray:
        return self.q[s].astype(np.float64)

    def store(self, s: int, a: int, new_value: float) -> None:
        old = self.q[s, a]
        self.q[s, a] = new_value
        if new_value != float(old):
            self.n_updates += 1

    def nbytes(self) -> int:
        return int(self.q.nbytes)


class FixedPointQTable:
    """Unsigned b-bit fixed-point Q-table over [v_min, v_max]."""

    is_fixed_point = True

    def __init__(self, n_states: int, n_actions: int, bits: int, rounding: str,
                 rng: np.random.Generator, v_min: float = 0.0, v_max: float = 1.0):
        if not 2 <= bits <= 16:
            raise ValueError("bits must be in [2, 16]")
        if rounding not in ROUNDING_MODES:
            raise ValueError(f"rounding must be one of {ROUNDING_MODES}")
        self.bits = bits
        self.rounding = rounding
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.max_code = (1 << bits) - 1
        self.delta = (self.v_max - self.v_min) / self.max_code
        dtype = np.uint8 if bits <= 8 else np.uint16
        self.codes = np.zeros((n_states, n_actions), dtype=dtype)
        if self.v_min > 0.0:
            raise ValueError("initial value 0 must be representable")
        self._rng = rng
        self.n_updates = 0       # stored value changed
        self.n_lost = 0          # float change non-zero but stored value unchanged

    # --- conversion -------------------------------------------------------
    def encode(self, x: float) -> int:
        """Round a real value onto the grid and return its integer code."""
        y = (x - self.v_min) / self.delta
        if self.rounding == "rtn":
            c = np.rint(y)
        else:
            lo = np.floor(y)
            c = lo + (1.0 if self._rng.random() < (y - lo) else 0.0)
        return int(min(max(c, 0), self.max_code))

    def decode(self, c) -> np.ndarray | float:
        return self.v_min + np.asarray(c, dtype=np.float64) * self.delta

    # --- table interface --------------------------------------------------
    def values(self) -> np.ndarray:
        return self.decode(self.codes)

    def get(self, s: int, a: int) -> float:
        return float(self.v_min + int(self.codes[s, a]) * self.delta)

    def row(self, s: int) -> np.ndarray:
        return self.decode(self.codes[s])

    def store(self, s: int, a: int, new_value: float) -> None:
        old_code = int(self.codes[s, a])
        old_value = self.v_min + old_code * self.delta
        new_code = self.encode(new_value)
        if new_code != old_code:
            self.n_updates += 1
        elif new_value != old_value:
            self.n_lost += 1
        self.codes[s, a] = new_code

    def nbytes(self) -> int:
        """Storage if codes were bit-packed (b bits per entry)."""
        return int(np.ceil(self.codes.size * self.bits / 8))


def make_qtable(n_states: int, n_actions: int, precision: str, rounding: str,
                rng: np.random.Generator, v_min: float = 0.0, v_max: float = 1.0):
    """precision: 'fp32' or an integer bit-width given as a string, e.g. '8'."""
    if str(precision).lower() == "fp32":
        return FloatQTable(n_states, n_actions)
    return FixedPointQTable(n_states, n_actions, int(precision), rounding, rng, v_min, v_max)
