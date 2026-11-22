"""Expand an experiment config into individual runs, execute them, save raw results.

Each run is saved as one compressed .npz file in results/raw/<experiment name>/,
so experiments are resumable: runs whose file already exists are skipped.
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import ROOT
from .train import run


def expand(cfg: dict) -> list[dict]:
    """Return one job dict per (hyperparameters, algorithm, precision, rounding, seed)."""
    seeds = range(int(cfg["seeds"]["start"]), int(cfg["seeds"]["start"]) + int(cfg["seeds"]["count"]))
    grid = cfg.get("tune_grid")
    hp_sets = [None]
    if grid:
        keys = sorted(grid)
        hp_sets = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    budgets = cfg.get("budget_grid") or [cfg["train"]["budget_steps"]]
    jobs = []
    for budget in budgets:
        for hp in hp_sets:
            for algo in cfg["algorithms"]:
                run_cfg = copy.deepcopy(cfg)
                run_cfg["train"]["budget_steps"] = int(budget)
                if hp:
                    run_cfg["hyperparameters"][algo].update(hp)
                a = run_cfg["hyperparameters"][algo]
                for prec in cfg["precisions"]:
                    prec = str(prec)
                    roundings = ["none"] if prec == "fp32" else list(cfg["roundings"])
                    for rnd in roundings:
                        for seed in seeds:
                            tag = (f"{algo}_{prec}_{rnd}_a{a['alpha']}_d{a['eps_decay_fraction']}"
                                   f"_b{int(budget)}_s{seed}")
                            jobs.append(dict(cfg=run_cfg, algo=algo, precision=prec, rounding=rnd,
                                             seed=seed, tag=tag))
    return jobs


def out_dir(cfg: dict) -> Path:
    return ROOT / "results" / "raw" / cfg["name"]


def _execute(job: dict) -> tuple[str, dict]:
    res = run(job["cfg"], job["algo"], job["precision"], job["rounding"], job["seed"])
    hp = job["cfg"]["hyperparameters"][job["algo"]]
    res["alpha"] = float(hp["alpha"]); res["eps_decay_fraction"] = float(hp["eps_decay_fraction"])
    res["budget_steps"] = int(job["cfg"]["train"]["budget_steps"])
    return job["tag"], res


def save_run(path: Path, res: dict) -> None:
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
    tmp.replace(path)


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        res = json.loads(str(z["meta"]))
        for k in z.files:
            if k != "meta":
                res[k] = z[k]
    return res


def run_experiment(cfg: dict, workers: int = 1, max_minutes: float | None = None, verbose: bool = True) -> int:
    """Run every pending job. Returns the number of jobs still pending afterwards."""
    d = out_dir(cfg); d.mkdir(parents=True, exist_ok=True)
    with open(d / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    jobs = [j for j in expand(cfg) if not (d / f"{j['tag']}.npz").exists()]
    total = len(expand(cfg))
    if verbose:
        print(f"[{cfg['name']}] {total} runs in grid, {len(jobs)} pending, workers={workers}", flush=True)
    if not jobs:
        return 0
    deadline = None if max_minutes is None else time.time() + 60 * max_minutes
    done = 0
    if workers <= 1:
        for j in jobs:
            if deadline and time.time() > deadline:
                break
            tag, res = _execute(j)
            save_run(d / f"{tag}.npz", res); done += 1
            if verbose:
                print(f"  done {tag}  final V={res['final_value']:.3f}  ({res['wall_seconds']:.1f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            pending = iter(jobs)
            futs = {}
            for j in itertools.islice(pending, workers * 2):
                futs[ex.submit(_execute, j)] = j
            while futs:
                fut = next(as_completed(futs))
                futs.pop(fut)
                tag, res = fut.result()
                save_run(d / f"{tag}.npz", res); done += 1
                if verbose:
                    print(f"  done {tag}  final V={res['final_value']:.3f}  ({res['wall_seconds']:.1f}s)", flush=True)
                if deadline is None or time.time() < deadline:
                    nxt = next(pending, None)
                    if nxt is not None:
                        futs[ex.submit(_execute, nxt)] = nxt
    remaining = len(jobs) - done
    if verbose:
        print(f"[{cfg['name']}] finished {done}, remaining {remaining}", flush=True)
    return remaining
