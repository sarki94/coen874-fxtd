# When Rounding Erases Learning: Fixed-Point Q-Tables and TD Control

COEN874 (Reinforcement Learning) term project, Ahmadu Bello University, Zaria.
Author: Aliyu Sarki Iliyasu.

**Research question.** How does Q-learning compare with SARSA in terms of optimality gap on
`FrozenLake-v1` (8x8, slippery) when Q-table precision varies (FP32, then 16, 12, 10, 8 and
6-bit fixed point)? Ablation: round-to-nearest (RTN) vs stochastic rounding (SR) when storing
each updated Q-value.

The agents learn only from interaction through the Gymnasium API. The environment model
(`env.unwrapped.P`) is used only for reference points: the dynamic-programming optimum V*
(value iteration) and the exact value of each learned greedy policy.

## Setup

Tested with Python 3.13.18 on Linux. Windows with Anaconda:

```bash
conda create -n coen874 python=3.13 -y
conda activate coen874
pip install -r requirements.txt
```

Pinned versions (`requirements.txt`): gymnasium 1.4.0, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2, pandas 3.0.5, PyYAML 6.0.3, tabulate 0.10.0, pytest 9.1.1.
PyTorch and Stable-Baselines3 are not used, because the project uses tabular methods only.

## Quick check (about 1 minute)

```bash
python scripts/smoke_test.py
```

This runs the unit tests, a 2-seed, 20k-step experiment, and every figure and table script on
it. Outputs go to `results/{raw,figures,tables}/smoke/`.

## Full reproduction (in order)

| Step | Command | Output | Time (2 cores) |
|---|---|---|---|
| 1. Budget check | `python scripts/run_experiment.py --config configs/budget_check.yaml` | `results/raw/budget_check/` | ~3 min |
| 2. Tuning (FP32, seeds 1000-1004) | `python scripts/run_experiment.py --config configs/tuning.yaml` | `results/raw/tuning/` | ~12 min |
| 3. Select hyperparameters | `python scripts/select_hyperparameters.py` | `configs/tuned.yaml`, `results/tables/tuning_summary.csv` | seconds |
| 4. A-priori prediction | `python scripts/make_predictions.py` | `results/tables/predictions.json` | seconds |
| 5. Main experiment (seeds 0-19) | `python scripts/run_experiment.py --config configs/main.yaml` | `results/raw/main/` (440 runs) | ~45 min |
| 6. All figures | `python scripts/make_figures.py` | `results/figures/*.pdf/.png` | ~1 min |
| 7. All tables | `python scripts/make_tables.py` | `results/tables/*.csv/.md` | ~1 min |

Runs are resumable: an interrupted experiment continues where it stopped when you rerun the
same command. Use `--workers N` to set parallelism.

### One command per figure and table

| Paper item | Command |
|---|---|
| Fig. 1 Learning curves (mean +/- 95% bootstrap CI, all methods) | `python scripts/make_figures.py --only learning_curves` |
| Fig. 2 Final-performance distribution per seed | `python scripts/make_figures.py --only final_distribution` |
| Fig. 3 Sensitivity to bit-width, with the a-priori prediction | `python scripts/make_figures.py --only sensitivity` |
| Fig. 4 Behaviour: greedy policy and value maps | `python scripts/make_figures.py --only policy_maps` |
| Fig. 5 Fraction of updates not stored | `python scripts/make_figures.py --only lost_updates` |
| Table I Environment and MDP | `python scripts/make_tables.py` -> `table1_mdp` |
| Table II Hyperparameters | `python scripts/make_tables.py` -> `table2_hyperparams` |
| Table III Main results | `python scripts/make_tables.py` -> `table3_main_results` |
| Table IV Statistical tests | `python scripts/make_tables.py` -> `table4_stats` |
| Table V Value bias (over/under-estimation) | `python scripts/make_tables.py` -> `table5_value_bias` |

All tables are produced by one script; each writes its own `.csv` and `.md` file.

## Experimental protocol

- **Equal budget:** 1,000,000 environment steps for every method, chosen by `budget_check` as
  the smallest tested budget at which both FP32 baselines retain at least 90% of V*.
- **Fair tuning:** the same 12-point grid (alpha x epsilon-decay fraction) and the same 5
  tuning seeds for both algorithms, at FP32 only. The selected values are then frozen for every
  bit-width, so precision is the only thing that changes.
- **Seeds:** tuning 1000-1004; reporting 0-19 (20 per configuration); evaluation rollouts use
  seeds from 10,000,000 upwards. Python and NumPy are seeded per run; the training env is seeded
  at its first reset.
- **Separate evaluation:** every 10,000 steps the greedy policy (exploration off) is scored
  (a) exactly from the model, V^pi(s0), and (b) by 100 rollouts on a separate env instance.
  The final evaluation uses 1,000 rollouts.
- **Reference points:** random policy, DP optimum V*(s0) = 0.4146, and the optimal policy's
  rollout success rate.
- **Everything logged:** every episode and every checkpoint of every run is saved raw in
  `results/raw/<experiment>/*.npz`; no seed is discarded.
- **Statistics:** one score per seed; mean or median with 95% percentile-bootstrap CIs
  (10,000 resamples); Mann-Whitney U (one-sided for H1, two-sided for H2) with Holm-Bonferroni
  correction; Kruskal-Wallis across bit-widths; effect sizes Cliff's delta (and Hedges' g where
  the variance is non-zero).
- **Pre-registration:** the collapse prediction (`results/tables/predictions.json`) and the
  tuned hyperparameters were committed before the main experiment was run (see git history).

## Repository layout

```
configs/          all hyperparameters (base, budget_check, tuning, tuned [generated], main, smoke)
src/fxtd/
  qtable.py       FP32 and b-bit fixed-point Q-tables (RTN / SR storage rounding)
  agents.py       Q-learning and SARSA updates, epsilon-greedy, GLIE-style epsilon schedule
  mdp.py          env factory, tabular model from env.unwrapped.P, value iteration, exact policy value, rollouts
  train.py        one run: training loop, periodic evaluation, raw logging
  experiment.py   grid expansion, parallel resumable execution, raw result I/O
  theory.py       dead-zone threshold and quantised expected-update collapse predictor
  analysis.py     per-seed score table, learning-curve grouping
  stats.py        bootstrap CIs, Mann-Whitney, Welch, Kruskal-Wallis, Holm, Cliff's delta, Hedges' g
scripts/          run_experiment, select_hyperparameters, make_predictions, make_figures, make_tables, smoke_test
tests/            unit tests (pytest)
results/          raw/, figures/, tables/
```

## Tests

```bash
python -m pytest -q
```

The tests check the fixed-point grid, RTN dead-zone behaviour, unbiasedness of stochastic
rounding, the Q-learning update against the COEN874 Week 3 worked example (Q = 2.85), SARSA vs
Q-learning targets, terminal handling, the epsilon schedule, and value iteration against the
Week 3 lab's V* values. They also check the statistics helpers (Holm, Cliff's delta, bootstrap).

## Credits

- Environment: Gymnasium `FrozenLake-v1` (Farama Foundation, MIT licence).
- The algorithms follow the COEN874 lecture notes (Dr. Yusuf Ibrahim, Weeks 2-3) and
  Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Stochastic rounding follows Gupta et al., "Deep Learning with Limited Numerical Precision",
  ICML 2015.
- All code in this repository was written for this project; no third-party RL code is included.
- AI-use disclosure: code and text were drafted with the assistance of Claude (Anthropic) and
  reviewed, run and verified by the author.
