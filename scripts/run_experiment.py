"""Run an experiment grid defined in a config file (resumable).

Examples
--------
python scripts/run_experiment.py --config configs/smoke.yaml
python scripts/run_experiment.py --config configs/main.yaml --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fxtd.config import load_config  # noqa: E402
from fxtd.experiment import run_experiment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop launching new runs after this many minutes (rerun to resume)")
    args = p.parse_args()
    cfg = load_config(args.config)
    remaining = run_experiment(cfg, workers=args.workers, max_minutes=args.max_minutes)
    sys.exit(0 if remaining == 0 else 3)


if __name__ == "__main__":
    main()
