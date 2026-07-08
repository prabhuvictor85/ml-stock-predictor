#!/usr/bin/env python3
"""
MODEL_E2 — Formation-Window Momentum (pre-registered)
======================================================
Implements docs/MODEL_E2_PREREGISTRATION.md EXACTLY. One run, as specified.

Kernel: 3/6/12-month trailing returns, each skipping the most recent 21
trading days (Jegadeesh-Titman style), computed IN-SCRIPT from panel close —
pure trailing arithmetic, truncation-safe by construction.
Buckets: V = vol-scaled kernel (÷ trailing 126d daily-return std),
         S = MODEL_E's short windows (features_return_1d/5d/20d/60d).
Configs: KERNEL, KERNEL+V, KERNEL+S, KERNEL+V+S.
Gate:    mean IC20 >= 0.03 AND t >= 2.0 AND >= 4/6 folds positive
         (full cross-section). 40d/60d + PIT-subset ICs informational only.

Run on Hetzner (only when nothing else panel-sized is running):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_e2_formation_momentum.py \
        > /tmp/model_e2.log 2>&1 &
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

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
OUT_PATH      = "/mnt/data/artefacts/experiments/model_e2_results.json"

SKIP_D    = 21                       # skip most recent month (literature default)
WINDOWS   = {"e2_mom_3m": 63, "e2_mom_6m": 126, "e2_mom_12m": 252}
VOL_WIN   = 126                      # trailing daily-return std window for Bucket V
SHORT_B7  = ["features_return_1d", "features_return_5d",
             "features_return_20d", "features_return_60d"]

KERNEL = list(WINDOWS.keys())
VOLSC  = [f"{k}_volsc" for k in KERNEL]

# Pre-registered thresholds — see MODEL_E2_PREREGISTRATION.md §4.
KEEP_DELTA = 0.005
T_FLOOR    = 2.0     # absolute floor: guard cannot go toothless (MODEL_E lesson)

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
HORIZONS   = [20, 40, 60]


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def build_e2_features(panel: pd.DataFrame, date_level: str) -> pd.DataFrame:
    """Kernel + vol-scaled features from close, per ticker, trailing-only."""
    tkr_level = "ticker"
    out = {}
    close = panel["close"].astype(float)
    grp = close.groupby(level=tkr_level)
    near = grp.shift(SKIP_D)
    for name, far_d in WINDOWS.items():
        far = grp.shift(far_d)
        out[name] = (near / far - 1.0).astype(np.float32)
    ret1 = grp.pct_change()
    vol = (ret1.groupby(level=tkr_level)
               .rolling(VOL_WIN, min_periods=int(VOL_WIN * 0.8)).std()
               .reset_index(level=0, drop=True))
    # guard: sub-1bp daily vol is flat/untraded — floor to avoid exploding ratios
    vol = vol.where(vol > 1e-4, np.nan)
    for name in WINDOWS:
        out[f"{name}_volsc"] = (out[name] / vol).astype(np.float32)
    for k, v in out.items():
        panel[k] = v
    return panel


def run_cv(panel: pd.DataFrame, features: list, date_level: str) -> dict:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    fold_ics = {h: {} for h in HORIZONS}
    fold_pit = {}                     # year -> mean IC20 within in_universe
    for test_year in FOLD_YEARS:
        dates = panel.index.get_level_values(date_level)
        tr = panel[dates.year < test_year].dropna(subset=features + ["cs_rank_composite"])
        te = panel[dates.year == test_year].dropna(subset=features + ["future_20d_excess_return"])
        if len(tr) < 5000 or len(te) < 500:
            continue

        # Group-contiguity invariant (bug class found 2026-07-08): lambdarank
        # group arrays require date-major contiguous rows. Fail loudly.
        if not tr.index.get_level_values(date_level).is_monotonic_increasing:
            raise RuntimeError("train slice not date-major — sort before training")

        model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(
                tr[features], label=cs_rank_to_label(tr["cs_rank_composite"]),
                group=tr.groupby(level=date_level).size().sort_index().values,
                free_raw_data=False,
            ),
            num_boost_round=LGBM_PARAMS["n_estimators"],
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        scores = pd.Series(model.predict(te[features]), index=te.index)

        date_ics = {h: [] for h in HORIZONS}
        pit_ics  = []
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
            if "in_universe" in grp.columns:
                pit = grp["in_universe"] == True
                m = pd.DataFrame({"s": sc[pit], "r": grp.loc[pit, "future_20d_excess_return"]}).dropna()
                if len(m) >= 20 and m["s"].std() > 1e-9:
                    ic, _ = spearmanr(m["s"], m["r"])
                    pit_ics.append(ic)

        for h in HORIZONS:
            if date_ics[h]:
                fold_ics[h][test_year] = float(np.mean(date_ics[h]))
        if pit_ics:
            fold_pit[test_year] = float(np.mean(pit_ics))
        print(f"    {test_year}: IC20={fold_ics[20].get(test_year, float('nan')):+.4f}  "
              f"IC40={fold_ics[40].get(test_year, float('nan')):+.4f}  "
              f"IC60={fold_ics[60].get(test_year, float('nan')):+.4f}  "
              f"PIT20={fold_pit.get(test_year, float('nan')):+.4f}")
        gc.collect()

    def agg(vals: dict):
        v = list(vals.values())
        n = len(v)
        if n == 0:
            return {"mean_ic": float("nan"), "t_stat": 0.0, "n_folds": 0,
                    "n_folds_positive": 0, "min_fold_ic": float("nan"), "fold_ics": {}}
        m, s = float(np.mean(v)), float(np.std(v))
        return {"mean_ic": m, "std_ic": s,
                "t_stat": float(m / (s / np.sqrt(n))) if n > 1 and s > 0 else 0.0,
                "n_folds": n, "n_folds_positive": sum(1 for x in v if x > 0),
                "min_fold_ic": float(min(v)),
                "fold_ics": {str(y): round(x, 4) for y, x in vals.items()}}

    out = {f"h{h}": agg(fold_ics[h]) for h in HORIZONS}
    out["pit20"] = agg(fold_pit)
    out["n_features"] = len(features)
    return out


def gate_pass(r) -> bool:
    a = r["h20"]
    return (a["mean_ic"] >= 0.03) and (a["t_stat"] >= 2.0) and (a["n_folds_positive"] >= 4)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print("=" * 66)
    print("  MODEL_E2 — Formation-window momentum (gate = 20d IC, full universe)")
    print("=" * 66)

    print(f"\nLoading panel: {DEFAULT_PANEL}")
    panel = pd.read_pickle(DEFAULT_PANEL)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    if "ticker" not in panel.index.names:
        sys.exit("FATAL: panel index has no 'ticker' level.")

    label_cols = ["cs_rank_composite"] + [f"future_{h}d_excess_return" for h in HORIZONS]
    required = ["close", "in_universe"] + label_cols + SHORT_B7
    missing = [c for c in required if c not in panel.columns]
    if missing:
        sys.exit(f"FATAL: panel missing required columns: {missing}")

    panel = panel[required].copy()
    gc.collect()

    print("Building E2 formation features (3m/6m/12m skip-1m + vol-scaled) ...")
    panel = panel.sort_index()
    panel = build_e2_features(panel, date_level)
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    gc.collect()
    for k in KERNEL + VOLSC:
        nn = panel[k].notna().mean()
        print(f"  {k:<18} non-null {nn:.1%}")
        if nn < 0.5:
            sys.exit(f"FATAL: {k} mostly NaN — feature build wrong; aborting.")
    print(f"Fenced panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols  "
          f"({panel.memory_usage(deep=True).sum()/1e9:.2f} GB)\n")

    configs = {
        "KERNEL":     KERNEL,
        "KERNEL+V":   KERNEL + VOLSC,
        "KERNEL+S":   KERNEL + SHORT_B7,
        "KERNEL+V+S": KERNEL + VOLSC + SHORT_B7,
    }

    results = {}
    base = None
    for name, feats in configs.items():
        print("─" * 48)
        print(f"{name} — {len(feats)} features")
        r = run_cv(panel, feats, date_level)
        r["label"] = name
        if base is None:
            base = r
        r["delta_ic"] = round(r["h20"]["mean_ic"] - base["h20"]["mean_ic"], 4)
        results[name] = r
        a = r["h20"]
        print(f"  → IC20={a['mean_ic']:+.4f}  t={a['t_stat']:+.2f}  "
              f"IC40={r['h40']['mean_ic']:+.4f}  IC60={r['h60']['mean_ic']:+.4f}  "
              f"PIT20={r['pit20']['mean_ic']:+.4f}  "
              f"gate={'PASS' if gate_pass(r) else 'FAIL'}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    base_t = base["h20"]["t_stat"]
    t_guard = max(T_FLOOR, base_t - 1.0)
    print("=" * 66)
    print("  SUMMARY  (gate = 20d full-universe; 40/60d & PIT informational)")
    print("=" * 66)
    print(f"  {'Config':<12} {'n':>3} {'IC20':>8} {'t':>6} {'minIC':>8} {'f+':>3} "
          f"{'IC40':>8} {'IC60':>8} {'PIT20':>8} {'delta':>7}  verdict")
    print("  " + "-" * 92)
    for name, r in results.items():
        a = r["h20"]
        if name == "KERNEL":
            verdict = "GATE PASS" if gate_pass(r) else "GATE FAIL"
        else:
            keepit = r["delta_ic"] >= KEEP_DELTA and a["t_stat"] >= t_guard
            verdict = ("KEEP" if keepit else ("MARG" if r["delta_ic"] > 0 else "DROP"))
            verdict += " | " + ("GATE PASS" if gate_pass(r) else "GATE FAIL")
        print(f"  {name:<12} {r['n_features']:>3} {a['mean_ic']:>+8.4f} {a['t_stat']:>+6.2f} "
              f"{a['min_fold_ic']:>+8.4f} {a['n_folds_positive']:>3} "
              f"{r['h40']['mean_ic']:>+8.4f} {r['h60']['mean_ic']:>+8.4f} "
              f"{r['pit20']['mean_ic']:>+8.4f} {r['delta_ic']:>+7.4f}  {verdict}")

    print(f"\n  KEEP rule: delta >= +{KEEP_DELTA} AND t >= max({T_FLOOR}, baseline_t-1) "
          f"= {t_guard:.2f}")
    print("  Survivorship note: dead tickers absent from panel — measured momentum "
          "is a FLOOR (see pre-registration §5).")
    print("  References: ICT −0.000 | pivots +0.0092 | zones(causal) +0.0069 | "
          "short-momentum −0.0017")

    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
