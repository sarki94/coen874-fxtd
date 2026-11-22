"""End-to-end smoke test (about 1 minute): unit tests, a tiny experiment
(2 seeds, 20k steps), and regeneration of every figure and table from it.
Outputs go to results/*/smoke/ and never touch the main results.

python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def step(args: list[str]) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    step(["-m", "pytest", "-q"])
    step(["scripts/run_experiment.py", "--config", "configs/smoke.yaml", "--workers", "2"])
    step(["scripts/make_figures.py", "--exp", "smoke"])
    step(["scripts/make_tables.py", "--exp", "smoke"])
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
