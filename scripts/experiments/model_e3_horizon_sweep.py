#!/usr/bin/env python3
"""
MODEL_E3 — Horizon × Terminal-Smoothing sweep (pre-registered)
===============================================================
Implements docs/MODEL_E3_PREREGISTRATION.md EXACTLY.

Reads the two rebuilt panels (produced by rebuild_targets_e3.py):
  e3_targets_w1.pkl  (TWAP terminal window = 1, single-close exit)
  e3_targets_w5.pkl  (TWAP terminal window = 5, shock-robust exit)

Features (fixed = E2 KERNEL+S): 3m/6m/12m formation returns skip-21d
(computed in-script from close) + features_return_1d/5d/20d/60d (from panel).
Training label: cs_rank_composite (unchanged). Horizon is a GRADING dimension:
train once per fold, grade IC at every horizon in {20,40,60,80,100,120}.

Gate discipline (pre-registered §4):
  PRIMARY cell = (TWAP=5, horizon=60): standard gate IC>=0.03, t>=2, >=4/6 folds.
    The ONLY cell that can claim a pass.
  EXPLORATORY = the other 11 cells: reported as a map; flagged only if they
    clear a Bonferroni bar (family-wise one-sided 0.05 over 12 tests). With
    N=6 folds this bar (~t>=4) is near-unreachable by design — the grid is
    hypothesis-generating, not confirmatory.

Run on Hetzner (after both rebuilds; nothing else panel-sized concurrently):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_e3_horizon_sweep.py \
        > /tmp/model_e3.log 2>&1 &
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Reuse E2's audited, truncation-safe formation-feature builder (DRY).
from model_e2_formation_momentum import build_e2_features, KERNEL, SHORT_B7

PANELS = {
    1: "/mnt/data/artefacts/experiments/e3_targets_w1.pkl",
    5: "/mnt/data/artefacts/experiments/e3_targets_w5.pkl",
}
OUT_PATH = "/mnt/data/artefacts/experiments/model_e3_results.json"

HORIZONS  = [20, 40, 60, 80, 100, 120]
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
FEATURES  = KERNEL + SHORT_B7                 # KERNEL+S (E2's best config)

PRIMARY = (5, 60)                             # (twap_window, horizon) — pre-declared
N_CELLS = len(PANELS) * len(HORIZONS)         # 12 — for the Bonferroni bar

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def run_twap_build(panel: pd.DataFrame, date_level: str) -> dict:
    """Train once per fold on cs_rank_composite; grade IC at every horizon."""
    import lightgbm as lgb
    from scipy.stats import spearmanr

    fold_ics = {h: {} for h in HORIZONS}      # horizon -> {year: mean IC}
    for test_year in FOLD_YEARS:
        dates = panel.index.get_level_values(date_level)
        tr = panel[dates.year < test_year].dropna(subset=FEATURES + ["cs_rank_composite"])
        # test rows need at least the shortest-horizon label present
        te = panel[dates.year == test_year].dropna(subset=FEATURES + ["future_20d_excess_return"])
        if len(tr) < 5000 or len(te) < 500:
            continue

        if not tr.index.get_level_values(date_level).is_monotonic_increasing:
            raise RuntimeError("train slice not date-major — lambdarank groups misalign")

        model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(
                tr[FEATURES], label=cs_rank_to_label(tr["cs_rank_composite"]),
                group=tr.groupby(level=date_level).size().sort_index().values,
                free_raw_data=False,
            ),
            num_boost_round=LGBM_PARAMS["n_estimators"],
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        scores = pd.Series(model.predict(te[FEATURES]), index=te.index)

        date_ics = {h: [] for h in HORIZONS}
        for dt in te.index.get_level_values(date_level).unique():
            grp = te.xs(dt, level=date_level)
            sc  = scores.xs(dt, level=date_level)
            for h in HORIZONS:
                col = f"future_{h}d_excess_return"
                if col not in grp.columns:
                    continue
                m = pd.DataFrame({"s": sc, "r": grp[col]}).dropna()
                if len(m) < 20 or m["s"].std() < 1e-9:
                    continue
                ic, _ = spearmanr(m["s"], m["r"])
                date_ics[h].append(ic)
        for h in HORIZONS:
            if date_ics[h]:
                fold_ics[h][test_year] = float(np.mean(date_ics[h]))
        print(f"    {test_year}: " + "  ".join(
            f"IC{h}={fold_ics[h].get(test_year, float('nan')):+.4f}" for h in HORIZONS))
        gc.collect()

    def agg(vals: dict):
        v = list(vals.values()); n = len(v)
        if n == 0:
            return {"mean_ic": float("nan"), "t_stat": 0.0, "n_folds": 0,
                    "n_folds_positive": 0, "min_fold_ic": float("nan"), "fold_ics": {}}
        m = float(np.mean(v))
        s = float(np.std(v, ddof=1)) if n > 1 else float("nan")   # ddof=1
        return {"mean_ic": m, "std_ic": s,
                "t_stat": float(m / (s / np.sqrt(n))) if n > 1 and s > 0 else 0.0,
                "n_folds": n, "n_folds_positive": sum(1 for x in v if x > 0),
                "min_fold_ic": float(min(v)),
                "fold_ics": {str(y): round(x, 4) for y, x in vals.items()}}

    return {h: agg(fold_ics[h]) for h in HORIZONS}


def main():
    from scipy.stats import t as tdist
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Bonferroni one-sided bar at df = folds-1 = 5 (pre-registered §4)
    bonf_t = float(tdist.ppf(1 - 0.05 / N_CELLS, df=len(FOLD_YEARS) - 1))

    print("=" * 70)
    print("  MODEL_E3 — horizon × TWAP sweep (KERNEL+S)")
    print(f"  PRIMARY cell = TWAP{PRIMARY[0]} × {PRIMARY[1]}d @ standard gate "
          f"(IC>=0.03, t>=2, >=4/6)")
    print(f"  EXPLORATORY bar = Bonferroni t>={bonf_t:.2f} "
          f"(one-sided 0.05/{N_CELLS}, df={len(FOLD_YEARS)-1})")
    print("=" * 70)

    results = {}
    for w, path in PANELS.items():
        if not os.path.exists(path):
            sys.exit(f"FATAL: panel missing: {path} — run rebuild_targets_e3.py first.")
        print(f"\nLoading TWAP={w} panel: {path}")
        panel = pd.read_pickle(path)
        date_level = "date" if "date" in panel.index.names else panel.index.names[0]

        miss = [c for c in SHORT_B7 + ["cs_rank_composite"] if c not in panel.columns]
        if miss:
            sys.exit(f"FATAL: TWAP={w} panel missing columns: {miss}")
        for h in HORIZONS:
            if f"future_{h}d_excess_return" not in panel.columns:
                sys.exit(f"FATAL: TWAP={w} panel missing future_{h}d_excess_return — "
                         f"was it rebuilt with TARGET_HORIZONS extended?")

        keep = SHORT_B7 + ["close", "cs_rank_composite"] + \
               [f"future_{h}d_excess_return" for h in HORIZONS]
        panel = panel[list(dict.fromkeys(keep))].copy()
        panel = panel.sort_index()
        panel = build_e2_features(panel, date_level)
        panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
        gc.collect()
        print(f"  panel {panel.shape[0]:,} rows × {panel.shape[1]} cols; training ...")

        results[f"twap{w}"] = run_twap_build(panel, date_level)
        del panel; gc.collect()

    # ── Grid summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GRID  (rows = TWAP window, cols = horizon; IC / t)   [ddof=1]")
    print("=" * 70)
    header = "  TWAP  " + "".join(f"{h:>13}d" for h in HORIZONS)
    print(header)
    for w in PANELS:
        cells = []
        for h in HORIZONS:
            a = results[f"twap{w}"][h]
            tag = "*" if (w, h) == PRIMARY else (" B" if a["t_stat"] >= bonf_t else "")
            cells.append(f"{a['mean_ic']:+.4f}/{a['t_stat']:+.1f}{tag}")
        print(f"  w={w:<3}" + "".join(f"{c:>14}" for c in cells))

    print("\n  * = PRIMARY cell   B = clears Bonferroni exploratory bar")
    prim = results[f"twap{PRIMARY[0]}"][PRIMARY[1]]
    prim_pass = (prim["mean_ic"] >= 0.03 and prim["t_stat"] >= 2.0
                 and prim["n_folds_positive"] >= 4)
    print("\n" + "─" * 70)
    print(f"  PRIMARY (TWAP{PRIMARY[0]} × {PRIMARY[1]}d): IC={prim['mean_ic']:+.4f}  "
          f"t={prim['t_stat']:+.2f}  minIC={prim['min_fold_ic']:+.4f}  "
          f"folds+={prim['n_folds_positive']}/{prim['n_folds']}  → "
          f"{'GATE PASS' if prim_pass else 'GATE FAIL'}")
    bonf_hits = [(w, h) for w in PANELS for h in HORIZONS
                 if results[f"twap{w}"][h]["t_stat"] >= bonf_t]
    print(f"  Exploratory cells clearing Bonferroni t>={bonf_t:.2f}: "
          f"{bonf_hits if bonf_hits else 'none (expected — see §4 power caveat)'}")

    out = {"primary_cell": {"twap": PRIMARY[0], "horizon": PRIMARY[1],
                            "gate_pass": prim_pass, **prim},
           "bonferroni_t": bonf_t, "n_cells": N_CELLS,
           "grid": results,
           "_meta": {"features": FEATURES, "note": "training label cs_rank_composite; "
                     "horizon is a grading dimension; ddof=1 t-stat; date-major groups"}}
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
