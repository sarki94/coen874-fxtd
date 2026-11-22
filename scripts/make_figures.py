"""Regenerate every paper figure from raw results (no manual editing).

python scripts/make_figures.py                 # all figures for experiment "main"
python scripts/make_figures.py --only sensitivity --exp smoke

Outputs vector PDF (for the paper) and PNG (for quick viewing) in results/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fxtd.analysis import PRECISION_ORDER, curves, load_experiment, per_seed_table  # noqa: E402
from fxtd.config import ROOT, load_config  # noqa: E402
from fxtd.mdp import TabularModel, greedy_policy, make_env  # noqa: E402
from fxtd.stats import bootstrap_ci, bootstrap_curve_ci  # noqa: E402

# Palette: categorical slots 1-2 of the validated reference palette (blue, orange),
# identity also carried by line style (solid = SR, dashed = RTN) and the legend.
COL = {"qlearning": "#2a78d6", "sarsa": "#eb6834"}
NAME = {"qlearning": "Q-learning", "sarsa": "SARSA"}
LS = {"sr": "-", "rtn": "--", "none": ":"}
RND_NAME = {"sr": "stochastic", "rtn": "nearest", "none": "FP32"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIG = ROOT / "results" / "figures"   # experiment "main"; other experiments get a subfolder

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", (FIG / f"{name}.pdf").relative_to(ROOT))


def fig_learning_curves(runs, v_star, rnd_value):
    cv = curves(runs, "ckpt_value")
    precs = [p for p in PRECISION_ORDER if p != "fp32" and any(k[1] == p for k in cv)]
    fig, axes = plt.subplots(1, len(precs), figsize=(7.16, 2.1), sharey=True, squeeze=False)
    for ax, prec in zip(axes[0], precs):
        for algo in ("qlearning", "sarsa"):
            if (algo, "fp32", "none") in cv:
                steps, arr = cv[(algo, "fp32", "none")]
                ax.plot(steps / 1e5, arr.mean(0) / v_star, color=COL[algo], ls=":", lw=1.0, alpha=0.8)
            for rnd in ("rtn", "sr"):
                k = (algo, prec, rnd)
                if k not in cv:
                    continue
                steps, arr = cv[k]
                m, lo, hi = bootstrap_curve_ci(arr / v_star)
                ax.fill_between(steps / 1e5, lo, hi, color=COL[algo], alpha=0.15, lw=0)
                ax.plot(steps / 1e5, m, color=COL[algo], ls=LS[rnd],
                        label=f"{NAME[algo]}, {RND_NAME[rnd]}")
        ax.axhline(rnd_value / v_star, color=INK2, lw=0.8, ls="-.")
        ax.set_title(f"{prec}-bit Q-table")
        ax.set_ylim(-0.05, 1.1)
    axes[0][0].set_ylabel(r"$V^{\pi}(s_0)\,/\,V^*(s_0)$")
    h, l = axes[0][0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color=INK2, ls=":", lw=1.0), plt.Line2D([], [], color=INK2, ls="-.", lw=0.8)]
    l += ["FP32 reference (same colour)", "random policy"]
    fig.supxlabel(r"Environment steps ($\times 10^5$)", fontsize=8, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    save(fig, "fig1_learning_curves")


def fig_final_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.0), sharey=True)
    precs = [p for p in PRECISION_ORDER if p in set(df["precision"].astype(str))]
    rng = np.random.default_rng(0)
    for ax, algo in zip(axes, ("qlearning", "sarsa")):
        for i, prec in enumerate(precs):
            for j, rnd in enumerate(["none"] if prec == "fp32" else ["rtn", "sr"]):
                sel = df[(df.algo == algo) & (df.precision.astype(str) == prec) & (df.rounding == rnd)]
                if sel.empty:
                    continue
                x = i + (0 if prec == "fp32" else (-0.18 if rnd == "rtn" else 0.18))
                y = sel["retained"].to_numpy()
                ax.boxplot(y, positions=[x], widths=0.28, showfliers=False,
                           medianprops=dict(color=INK, lw=1.2), boxprops=dict(color=INK2, lw=0.7),
                           whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
                face = COL[algo] if rnd != "rtn" else "white"
                ax.scatter(x + rng.uniform(-0.07, 0.07, y.size), y, s=9, facecolor=face,
                           edgecolor=COL[algo], lw=0.7, zorder=3)
        ax.set_xticks(range(len(precs)), [p.upper() if p == "fp32" else f"{p}-bit" for p in precs])
        ax.set_title(f"{NAME[algo]}  (hollow = nearest, filled = stochastic)")
        ax.set_ylim(-0.05, 1.1)
    axes[0].set_ylabel(r"Final $V^{\pi}(s_0)\,/\,V^*(s_0)$ per seed")
    save(fig, "fig2_final_distribution")


def fig_sensitivity(df, predictions):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bits = [int(p) for p in PRECISION_ORDER if p != "fp32" and p in set(df["precision"].astype(str))]
    for algo in ("qlearning", "sarsa"):
        fp = df[(df.algo == algo) & (df.precision.astype(str) == "fp32")]["retained"]
        if len(fp):
            ax.axhline(fp.median(), color=COL[algo], ls=":", lw=1.0)
        for rnd, off in (("rtn", -0.12), ("sr", 0.12)):
            med, lo, hi = [], [], []
            for b in bits:
                y = df[(df.algo == algo) & (df.precision.astype(str) == str(b)) & (df.rounding == rnd)]["retained"]
                m, l_, h_ = bootstrap_ci(y.to_numpy(), stat=np.median)
                med.append(m); lo.append(m - l_); hi.append(h_ - m)
            x = np.array(bits) + (off if algo == "qlearning" else -off) * 0.5
            ax.errorbar(x, med, yerr=[lo, hi], color=COL[algo], ls=LS[rnd], marker="o" if rnd == "sr" else "s",
                        ms=4, mfc=COL[algo] if rnd == "sr" else "white", capsize=2, lw=1.3,
                        label=f"{NAME[algo]}, {RND_NAME[rnd]}")
    if predictions:
        algs = [a for a in ("qlearning", "sarsa") if a in predictions["algorithms"]]
        prs = [predictions["algorithms"][a]["mean_field_retained_rtn"] for a in algs]
        same = all(all(abs(pr[b] - prs[0][b]) < 1e-3 for b in pr) for pr in prs)
        for a, pr in zip(algs if not same else algs[:1], prs if not same else prs[:1]):
            pb = sorted(int(b) for b in pr if int(b) >= min(bits))
            lab = "a-priori prediction, nearest (both algorithms)" if same else f"{NAME[a]}, a-priori prediction"
            ax.plot(pb, [pr[str(b)] for b in pb], color=INK2 if same else COL[a], lw=0.9, ls="-",
                    marker="x", ms=4, alpha=0.8, label=lab)
    ax.set_xlabel("Q-table bit-width $b$ (fixed point on [0, 1])")
    ax.set_ylabel(r"Median final $V^{\pi}(s_0)/V^*(s_0)$")
    ax.invert_xaxis()
    ax.set_ylim(-0.05, 1.12)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    save(fig, "fig3_sensitivity")


def fig_policy_maps(runs, df, model, v_star):
    """Greedy policy and value maps for the median-performing seed of three configurations."""
    configs = [("qlearning", "fp32", "none"), ("qlearning", "8", "rtn"), ("qlearning", "6", "rtn"),
               ("qlearning", "6", "sr")]
    configs = [c for c in configs if any((r["algo"], r["precision"], r["rounding"]) == c for r in runs)]
    if not configs:
        return
    desc = model_desc = None
    env = make_env({"id": "FrozenLake-v1", "map_name": "8x8", "is_slippery": True})
    desc = env.unwrapped.desc.astype(str)
    n = desc.shape[0]
    arrows = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    fig, axes = plt.subplots(1, len(configs), figsize=(7.16, 2.2))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, configs):
        rs = [r for r in runs if (r["algo"], r["precision"], r["rounding"]) == c]
        vals = np.array([r["final_value"] for r in rs])
        r = rs[int(np.argsort(vals)[len(vals) // 2])]
        Q = r["q_final"]; V = Q.max(1).reshape(n, n); pi = greedy_policy(Q)
        im = ax.imshow(V, cmap="Blues", vmin=0, vmax=1)
        for s in range(n * n):
            y, x = divmod(s, n)
            ch = desc[y, x]
            if ch in "HG":
                ax.text(x, y, ch, ha="center", va="center", fontsize=7,
                        color="#e34948" if ch == "H" else "#008300", fontweight="bold")
                continue
            if np.all(Q[s] == Q[s][0]):
                ax.plot(x, y, marker=".", color=INK2, ms=3)
            else:
                dx, dy = arrows[int(pi[s])]
                ax.annotate("", xy=(x + 0.32 * dx, y + 0.32 * dy), xytext=(x - 0.32 * dx, y - 0.32 * dy),
                            arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
        title = "FP32" if c[1] == "fp32" else f"{c[1]}-bit, {RND_NAME[c[2]]}"
        ax.set_title(f"{title}\n$V^\\pi(s_0)/V^*$ = {r['final_value'] / v_star:.2f}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label=r"$\max_a Q(s,a)$")
    fig.suptitle("Q-learning, median seed of each configuration (arrows: greedy action; dots: all actions tied)",
                 fontsize=8, y=1.04)
    save(fig, "fig4_policy_maps")


def fig_lost_updates(df):
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    bits = [int(p) for p in PRECISION_ORDER if p != "fp32" and p in set(df["precision"].astype(str))]
    for algo in ("qlearning", "sarsa"):
        for rnd in ("rtn", "sr"):
            y = [df[(df.algo == algo) & (df.precision.astype(str) == str(b)) & (df.rounding == rnd)]["lost_frac"].mean()
                 for b in bits]
            ax.plot(bits, y, color=COL[algo], ls=LS[rnd], marker="o" if rnd == "sr" else "s", ms=4,
                    mfc=COL[algo] if rnd == "sr" else "white", label=f"{NAME[algo]}, {RND_NAME[rnd]}")
    ax.set_xlabel("Q-table bit-width $b$"); ax.set_ylabel("Fraction of updates\nnot stored")
    ax.invert_xaxis(); ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=6)
    save(fig, "fig5_lost_updates")


FIGS = ["learning_curves", "final_distribution", "sensitivity", "policy_maps", "lost_updates"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="main")
    p.add_argument("--only", choices=FIGS, default=None)
    args = p.parse_args()
    global FIG
    if args.exp != "main":
        FIG = ROOT / "results" / "figures" / args.exp
    cfg = load_config("configs/base.yaml")
    model = TabularModel.from_env(make_env(cfg["env"]))
    gamma = float(cfg["gamma"])
    v_star = model.value_iteration(gamma)[0][0]
    rnd_value = model.random_policy_value(gamma)[0]
    runs = load_experiment(args.exp)
    df = per_seed_table(runs, v_star)
    pred_path = ROOT / "results" / "tables" / "predictions.json"
    predictions = json.loads(pred_path.read_text()) if pred_path.exists() else None
    todo = [args.only] if args.only else FIGS
    if "learning_curves" in todo: fig_learning_curves(runs, v_star, rnd_value)
    if "final_distribution" in todo: fig_final_distribution(df)
    if "sensitivity" in todo: fig_sensitivity(df, predictions)
    if "policy_maps" in todo: fig_policy_maps(runs, df, model, v_star)
    if "lost_updates" in todo: fig_lost_updates(df)


if __name__ == "__main__":
    main()
