"""Regenerate every paper table from raw results (CSV + Markdown in results/tables/).

python scripts/make_tables.py            # experiment "main"
python scripts/make_tables.py --exp smoke

Tables
  table1_mdp            environment and MDP specification
  table2_hyperparams    tuning grid, selected hyperparameters, budgets, seeds
  table3_main_results   per configuration: mean [95% bootstrap CI] of every metric
  table4_stats          H1 (SR vs RTN), H2 (Q-learning vs SARSA), Kruskal-Wallis across bit-widths
  table5_value_bias     mean (max_a Q - V*) over non-terminal states: over/under-estimation
  per_seed_scores       the unit of analysis: one row per run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxtd.analysis import PRECISION_ORDER, load_experiment, per_seed_table  # noqa: E402
from fxtd.config import ROOT, load_config  # noqa: E402
from fxtd.mdp import TabularModel, greedy_policy, make_env, rollout_success_rate  # noqa: E402
from fxtd.stats import (bootstrap_ci, cliffs_delta, cliffs_magnitude, hedges_g, holm,  # noqa: E402
                        kruskal, mann_whitney)

OUT = ROOT / "results" / "tables"


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    with open(OUT / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print("wrote", (OUT / f"{name}.csv").relative_to(ROOT))


def fmt_ci(x, stat=np.mean, digits=3):
    m, lo, hi = bootstrap_ci(np.asarray(x, float), stat=stat)
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table_mdp(cfg, model, v_star, rnd_v, opt_success):
    env = make_env(cfg["env"])
    rows = [
        ("Environment ID", f"{cfg['env']['id']} (map_name={cfg['env']['map_name']}, is_slippery={cfg['env']['is_slippery']})"),
        ("Gymnasium version", gym.__version__),
        ("States S", f"{model.n_states} discrete cells, s = 8*row + col; start s0 = 0; "
                     f"{int((env.unwrapped.desc == b'H').sum())} holes; goal s = 63"),
        ("Actions A", "4: 0 left, 1 down, 2 right, 3 up"),
        ("Transitions P(s'|s,a)", "intended direction 1/3, each perpendicular direction 1/3; walls keep the agent in place"),
        ("Reward R(s,a,s')", "+1 on entering the goal, 0 otherwise"),
        ("Discount gamma", str(cfg["gamma"])),
        ("Termination", "entering a hole or the goal (no bootstrap)"),
        ("Truncation", f"{env.spec.max_episode_steps} steps (registered TimeLimit; bootstraps)"),
        ("DP optimum V*(s0)", f"{v_star:.4f} (value iteration)"),
        ("Optimal-policy success rate", f"{opt_success:.3f} (1,000 greedy rollouts, 100-step limit)"),
        ("Random-policy V(s0)", f"{rnd_v:.4f}"),
    ]
    write(pd.DataFrame(rows, columns=["Element", "Specification"]), "table1_mdp")


def table_hyperparams(main_cfg):
    tune = load_config("configs/tuning.yaml")
    rows = [("Training budget (env steps)", f"{main_cfg['train']['budget_steps']:,} (all methods)"),
            ("Epsilon schedule", f"linear {main_cfg['train']['eps_start']} -> {main_cfg['train']['eps_end']} "
                                 "over eps_decay_fraction of the budget"),
            ("Tuning grid (both algorithms)", f"alpha {tune['tune_grid']['alpha']} x eps_decay_fraction "
                                              f"{tune['tune_grid']['eps_decay_fraction']}; FP32; seeds 1000-1004"),
            ("Selection rule", "highest mean final V^pi(s0); tie-break mean AUC")]
    for algo, hp in main_cfg["hyperparameters"].items():
        rows.append((f"Selected ({algo})", f"alpha = {hp['alpha']}, eps_decay_fraction = {hp['eps_decay_fraction']}"))
    rows += [("Q-table initialisation", "0 for all entries"),
             ("Fixed-point grid", f"unsigned b-bit codes over [{main_cfg['fixed_point']['v_min']}, "
                                  f"{main_cfg['fixed_point']['v_max']}], Delta = 1/(2^b - 1)"),
             ("Bit-widths", ", ".join(str(p) for p in main_cfg["precisions"])),
             ("Rounding modes", "round-to-nearest (RTN), stochastic (SR)"),
             ("Reporting seeds", f"{main_cfg['seeds']['start']}-{main_cfg['seeds']['start'] + main_cfg['seeds']['count'] - 1}"
                                 f" ({main_cfg['seeds']['count']} per configuration)"),
             ("Evaluation", f"every {main_cfg['eval']['every_steps']:,} steps: exact V^pi(s0) + "
                            f"{main_cfg['eval']['rollouts']} greedy rollouts on a separate env; final: "
                            f"{main_cfg['eval']['final_rollouts']} rollouts")]
    write(pd.DataFrame(rows, columns=["Setting", "Value"]), "table2_hyperparams")


def table_main(df):
    rows = []
    for (algo, prec, rnd), g in df.groupby(["algo", "precision", "rounding"], observed=True):
        rows.append(dict(
            algorithm=algo, precision=str(prec), rounding=rnd, n_seeds=len(g),
            optimality_gap=fmt_ci(g["gap"]), retained_median=fmt_ci(g["retained"], np.median),
            success_rate=fmt_ci(g["final_success"]), auc=fmt_ci(g["auc"]),
            reached_80pct=f"{g['steps_to_threshold'].notna().mean():.2f}",
            steps_to_80pct_median=("-" if g["steps_to_threshold"].isna().all()
                                   else f"{g['steps_to_threshold'].median():,.0f}"),
            seed_IQR=f"{g['retained'].quantile(0.75) - g['retained'].quantile(0.25):.3f}",
            updates_not_stored=f"{g['lost_frac'].mean():.3f}",
            table_bytes=int(g["table_bytes"].iloc[0]),
        ))
    out = pd.DataFrame(rows)
    out["precision"] = pd.Categorical(out["precision"], PRECISION_ORDER, ordered=True)
    write(out.sort_values(["algorithm", "precision", "rounding"]), "table3_main_results")


def table_stats(df, metric="final_value"):
    rows = []
    bits = [p for p in PRECISION_ORDER if p != "fp32" and p in set(df["precision"].astype(str))]
    # H1: SR > RTN (one-sided), per algorithm, Holm across bit-widths
    for algo in sorted(df["algo"].unique()):
        block = []
        for b in bits:
            sr = df[(df.algo == algo) & (df.precision.astype(str) == b) & (df.rounding == "sr")][metric].to_numpy()
            rt = df[(df.algo == algo) & (df.precision.astype(str) == b) & (df.rounding == "rtn")][metric].to_numpy()
            if sr.size == 0 or rt.size == 0:
                continue
            block.append(dict(hypothesis="H1: SR > RTN", algorithm=algo, precision=b, rounding="-",
                              comparison="SR vs RTN", n=f"{sr.size}/{rt.size}",
                              median_a=np.median(sr), median_b=np.median(rt),
                              test="Mann-Whitney U (one-sided)", p=mann_whitney(sr, rt, "greater"),
                              cliffs_delta=cliffs_delta(sr, rt), hedges_g=hedges_g(sr, rt)))
        adj = holm([r["p"] for r in block])
        for r, a in zip(block, adj):
            r["p_holm"] = a
        rows += block
    # H2: retained fraction differs between algorithms (two-sided), per (bits, rounding), Holm across all
    block = []
    for b in bits:
        for rnd in ("rtn", "sr"):
            q = df[(df.algo == "qlearning") & (df.precision.astype(str) == b) & (df.rounding == rnd)]["retained"].to_numpy()
            s = df[(df.algo == "sarsa") & (df.precision.astype(str) == b) & (df.rounding == rnd)]["retained"].to_numpy()
            if q.size == 0 or s.size == 0:
                continue
            block.append(dict(hypothesis="H2: retained(Q) != retained(SARSA)", algorithm="Q vs SARSA",
                              precision=b, rounding=rnd, comparison="Q-learning vs SARSA",
                              n=f"{q.size}/{s.size}", median_a=np.median(q), median_b=np.median(s),
                              test="Mann-Whitney U (two-sided)", p=mann_whitney(q, s, "two-sided"),
                              cliffs_delta=cliffs_delta(q, s), hedges_g=hedges_g(q, s)))
    adj = holm([r["p"] for r in block])
    for r, a in zip(block, adj):
        r["p_holm"] = a
    rows += block
    # Omnibus: does precision matter? Kruskal-Wallis across bit-widths (incl. FP32), per (algo, rounding)
    for algo in sorted(df["algo"].unique()):
        for rnd in ("rtn", "sr"):
            groups = [df[(df.algo == algo) & (df.precision.astype(str) == "fp32")][metric].to_numpy()]
            groups += [df[(df.algo == algo) & (df.precision.astype(str) == b) & (df.rounding == rnd)][metric].to_numpy()
                       for b in bits]
            groups = [g for g in groups if g.size]
            if len(groups) >= 3:
                rows.append(dict(hypothesis="Omnibus: precision effect", algorithm=algo, precision="all",
                                 rounding=rnd, comparison=f"FP32 + {len(groups) - 1} bit-widths",
                                 n=str(sum(g.size for g in groups)), median_a=np.nan, median_b=np.nan,
                                 test="Kruskal-Wallis", p=kruskal(*groups), p_holm=np.nan,
                                 cliffs_delta=np.nan, hedges_g=np.nan))
    out = pd.DataFrame(rows)
    for c in ("median_a", "median_b", "cliffs_delta", "hedges_g"):
        out[c] = out[c].astype(float).round(3)
    out["magnitude"] = out["cliffs_delta"].apply(lambda d: "" if pd.isna(d) else cliffs_magnitude(d))
    out["p"] = out["p"].map(lambda v: f"{v:.2e}")
    out["p_holm"] = out["p_holm"].map(lambda v: "" if pd.isna(v) else f"{v:.2e}")
    write(out, "table4_stats")


def table_value_bias(runs, V_star, env_cfg):
    """Mean over non-terminal states of max_a Q(s,a) - V*(s): sign shows over/under-estimation."""
    desc = make_env(env_cfg).unwrapped.desc.astype(str).ravel()
    nt = np.array([c in "SF" for c in desc])
    rows = {}
    for r in runs:
        k = (r["algo"], r["precision"], r["rounding"])
        rows.setdefault(k, []).append(float(np.mean(r["q_final"].max(1)[nt] - V_star[nt])))
    out = pd.DataFrame([dict(algorithm=k[0], precision=k[1], rounding=k[2], n_seeds=len(v),
                             value_bias=fmt_ci(v, digits=4)) for k, v in rows.items()])
    out["precision"] = pd.Categorical(out["precision"], PRECISION_ORDER, ordered=True)
    write(out.sort_values(["algorithm", "precision", "rounding"]), "table5_value_bias")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    args = ap.parse_args()
    global OUT
    if args.exp != "main":
        OUT = ROOT / "results" / "tables" / args.exp
    main_cfg = load_config("configs/main.yaml")
    model = TabularModel.from_env(make_env(main_cfg["env"]))
    gamma = float(main_cfg["gamma"])
    V_star, Q_star = model.value_iteration(gamma)
    opt_success, _ = rollout_success_rate(make_env(main_cfg["env"]), greedy_policy(Q_star), 1000, 20_000_000)
    rnd_v = model.random_policy_value(gamma)[0]
    runs = load_experiment(args.exp)
    df = per_seed_table(runs, V_star[0])
    write(df.round(5), "per_seed_scores")
    table_mdp(main_cfg, model, V_star[0], rnd_v, opt_success)
    table_hyperparams(main_cfg)
    table_main(df)
    table_stats(df)
    table_value_bias(runs, V_star, main_cfg["env"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "reference_points.json", "w", encoding="utf-8") as f:
        json.dump({"V_star_s0": V_star[0], "optimal_policy_success_100_steps": opt_success,
                   "random_policy_V_s0": rnd_v}, f, indent=2)


if __name__ == "__main__":
    main()
