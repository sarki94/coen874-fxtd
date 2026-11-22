"""Compute the a-priori RTN collapse prediction (run BEFORE the main experiment).

Writes results/tables/predictions.json with, per algorithm: the propagation
threshold v_min(b) and the mean-field predicted retained performance per
bit-width, plus the predicted collapse bit-width.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fxtd.config import ROOT, load_config  # noqa: E402
from fxtd.mdp import TabularModel, make_env  # noqa: E402
from fxtd.theory import predict_retained, predicted_collapse_bits, propagation_threshold  # noqa: E402

BITS = [16, 14, 12, 11, 10, 9, 8, 7, 6]


def main() -> None:
    cfg = load_config("configs/main.yaml")
    model = TabularModel.from_env(make_env(cfg["env"]))
    gamma = float(cfg["gamma"])
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "note": "computed before the main experiment; not tuned on its results", "algorithms": {}}
    for algo in cfg["algorithms"]:
        alpha = float(cfg["hyperparameters"][algo]["alpha"])
        ret = predict_retained(model, algo, BITS, alpha, gamma, eps=float(cfg["train"]["eps_end"]))
        out["algorithms"][algo] = {
            "alpha": alpha,
            "v_min": {str(b): propagation_threshold(b, alpha, gamma) for b in BITS},
            "mean_field_retained_rtn": {str(b): r for b, r in ret.items()},
            "predicted_collapse_bits_rtn": predicted_collapse_bits(ret),
        }
        print(algo, "alpha", alpha, {b: round(r, 3) for b, r in ret.items()},
              "-> collapse at <=", predicted_collapse_bits(ret), "bits")
    path = ROOT / "results" / "tables" / "predictions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
