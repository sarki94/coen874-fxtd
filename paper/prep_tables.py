"""Read the generated result tables and emit tables.json for the paper builder.
No number here is typed by hand: everything comes from results/tables/*.csv."""
import json
import pandas as pd

T = "/home/claude/coen874-fxtd/results/tables/"
ps = pd.read_csv(T + "per_seed_scores.csv")
t3 = pd.read_csv(T + "table3_main_results.csv")
t4 = pd.read_csv(T + "table4_stats.csv")
t5 = pd.read_csv(T + "table5_value_bias.csv")
t1 = pd.read_csv(T + "table1_mdp.csv")
ps["precision"] = ps["precision"].astype(str)
t3["precision"] = t3["precision"].astype(str)
t4["precision"] = t4["precision"].astype(str)
t5["precision"] = t5["precision"].astype(str)

ORDER = [("fp32", "none"), ("16", "rtn"), ("16", "sr"), ("12", "rtn"), ("12", "sr"), ("10", "rtn"),
         ("10", "sr"), ("8", "rtn"), ("8", "sr"), ("6", "rtn"), ("6", "sr")]
RN = {"none": "–", "rtn": "RTN", "sr": "SR"}
PN = lambda p: "FP32" if p == "fp32" else p


def ci(s):  # "0.959 [0.931, 0.973]" -> "0.959 [0.931, 0.973]" (2 decimals)
    m, rest = s.split(" [")
    lo, hi = rest.rstrip("]").split(", ")
    return f"{float(m):.2f} [{float(lo):.2f}, {float(hi):.2f}]"


def steps(v):
    return "never" if v in ("-", "nan") or pd.isna(v) else f"{int(str(v).replace(',', '')) // 1000}k"


main = []
for prec, rnd in ORDER:
    row = [PN(prec), RN[rnd]]
    for algo in ("qlearning", "sarsa"):
        r = t3[(t3.algorithm == algo) & (t3.precision == prec) & (t3.rounding == rnd)].iloc[0]
        row += [ci(r.retained_median), f"{float(r.success_rate.split(' ')[0]):.2f}",
                steps(r.steps_to_80pct_median), f"{100 * r.updates_not_stored:.0f}%"]
    row.append(str(int(r.table_bytes)))
    main.append(row)

retained = ps.groupby(["algo", "precision", "rounding"])["retained"].median()


def fmtp(p):
    p = float(p)
    if p >= 0.01:
        return f"{p:.2f}" if p < 0.995 else "1.00"
    m, e = f"{p:.1e}".split("e")
    return f"{m}×10^{{{int(e)}}}"


stats = []
for _, r in t4.iterrows():
    if r.hypothesis.startswith("Omnibus"):
        continue
    b = r.precision
    if r.hypothesis.startswith("H1"):
        a = "Q" if r.algorithm == "qlearning" else "SARSA"
        ma, mb = retained[(r.algorithm, b, "sr")], retained[(r.algorithm, b, "rtn")]
        stats.append(["H1", a, b, "SR vs RTN", f"{ma:.2f} / {mb:.2f}", f"{r.cliffs_delta:+.2f}", fmtp(r.p_holm)])
    else:
        ma, mb = retained[("qlearning", b, r.rounding)], retained[("sarsa", b, r.rounding)]
        stats.append(["H2", "Q vs SARSA", b, RN[r.rounding], f"{ma:.2f} / {mb:.2f}",
                      f"{r.cliffs_delta:+.2f}", fmtp(r.p_holm)])

bias = []
for prec, rnd in ORDER:
    row = [PN(prec), RN[rnd]]
    for algo in ("qlearning", "sarsa"):
        r = t5[(t5.algorithm == algo) & (t5.precision == prec) & (t5.rounding == rnd)].iloc[0]
        m, rest = r.value_bias.split(" [")
        lo, hi = rest.rstrip("]").split(", ")
        row.append(f"{float(m):+.3f} [{float(lo):+.3f}, {float(hi):+.3f}]")
    bias.append(row)

mdp = [[e, s] for e, s in zip(t1.Element, t1.Specification)]
json.dump({"main": main, "stats": stats, "bias": bias, "mdp": mdp}, open("/home/claude/paper/tables.json", "w"),
          ensure_ascii=False, indent=1)
print(len(main), len(stats), len(bias), len(mdp))
print(main[0], stats[0], stats[14], sep="\n")
