"""Load raw runs into tidy tables and compute per-seed scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .experiment import load_run

PRECISION_ORDER = ["fp32", "16", "12", "10", "8", "6"]


def load_experiment(name: str) -> list[dict]:
    d = ROOT / "results" / "raw" / name
    files = sorted(p for p in d.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not files:
        raise FileNotFoundError(f"no runs found in {d}")
    return [load_run(p) for p in files]


def per_seed_table(runs: list[dict], v_star: float, threshold_frac: float = 0.8) -> pd.DataFrame:
    """One row per run (= per seed): the unit of statistical analysis."""
    rows = []
    for r in runs:
        steps, vals = r["ckpt_steps"], r["ckpt_value"]
        auc = float(np.trapezoid(vals, steps) / (steps[-1] - steps[0]) / v_star)
        hit = np.flatnonzero(vals >= threshold_frac * v_star)
        steps_to = float(steps[hit[0]]) if hit.size else np.nan
        tot = r["n_updates"] + r["n_lost"]
        rows.append(dict(
            algo=r["algo"], precision=r["precision"], rounding=r["rounding"], seed=r["seed"],
            alpha=r.get("alpha"), eps_decay_fraction=r.get("eps_decay_fraction"),
            budget=r.get("budget_steps"),
            final_value=r["final_value"], gap=v_star - r["final_value"],
            retained=r["final_value"] / v_star,
            final_success=r["final_success"], final_len=r["final_len"],
            auc=auc, steps_to_threshold=steps_to,
            lost_frac=(r["n_lost"] / tot) if tot else 0.0,
            table_bytes=r["table_bytes"], wall_seconds=r["wall_seconds"],
        ))
    df = pd.DataFrame(rows)
    df["precision"] = pd.Categorical(df["precision"], [p for p in PRECISION_ORDER if p in set(df["precision"])],
                                     ordered=True)
    return df.sort_values(["algo", "precision", "rounding", "seed"]).reset_index(drop=True)


def curves(runs: list[dict], key: str = "ckpt_value") -> dict:
    """Group learning curves by (algo, precision, rounding) -> (steps, array[n_seeds, n_ckpt])."""
    groups: dict = {}
    for r in runs:
        k = (r["algo"], r["precision"], r["rounding"])
        groups.setdefault(k, []).append(r)
    out = {}
    for k, rs in groups.items():
        rs = sorted(rs, key=lambda x: x["seed"])
        out[k] = (rs[0]["ckpt_steps"], np.vstack([x[key] for x in rs]))
    return out
