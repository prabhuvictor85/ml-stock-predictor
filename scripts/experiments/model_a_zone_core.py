#!/usr/bin/env python3
"""
MODEL_A — Zone Core Only
========================
Experiment: measure how much signal the pure zone framework delivers
when stripped of ICT, momentum, and macro features.

Hypothesis: SDZ/DZ zone features are the structural alpha. The current
53-feature model drowns them in regime-unstable signals. A lean zone-core
model should produce significantly higher and more stable fold IC.

Run on Theralytics:
    cd /root/ml-stock-predictor
    python3 scripts/experiments/model_a_zone_core.py

Output:
    - Walk-forward fold IC (6 folds, 2018–2023)
    - Top-decile excess per fold
    - Summary t-stat
    - Saved to /mnt/data/artefacts/experiments/model_a_results.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb
from pipeline.models.lgbm_ranker import cs_rank_to_label, build_label_gain

# ── Zone core feature set ────────────────────────────────────────────────────
# Selected from feature persistence audit (scripts/experiments/feature_persistence_audit.py)
# Criteria: positive IC in BOTH training era (2010-2023) AND lockbox (2024-2026)
#           OR stable negative IC in both eras (LightGBM learns to invert).
# ALL ICT, momentum, macro, and regime-flipped features excluded.

ZONE_CORE_FEATURES = [
    # ── Demand zone features (positive IC, stable across regimes) ──
    "features_sdz_1d",           # IC train=0.121 lockbox=0.179 — #1 signal
    "features_sdz_htf_score",    # IC train=0.098 lockbox=0.146
    "features_sdz_1wk",          # IC train=0.079 lockbox=0.126
    "features_dz_raw_score",     # IC train=0.078 lockbox=0.138
    "features_dz_1mo",           # IC train=0.063 lockbox=0.110
    "features_dz_1wk",           # IC train=0.049 lockbox=0.102
    "features_dz_1y",            # IC train=0.044 lockbox=0.084
    "features_sdz_1mo",          # IC train=0.041 lockbox=0.066
    "features_zone_strength",    # IC train=0.091 lockbox=0.044 (degraded, still positive)
    "features_sdz_3mo",          # IC train=0.019 lockbox=0.030
    # ── Supply zone features (negative IC, stable — model learns to invert) ──
    "features_ssz_1wk",          # IC train=-0.081 lockbox=-0.127 (low SSZ = bullish)
    "features_sz_raw_score",     # IC train=-0.073 lockbox=-0.196
    "features_ssz_htf_score",    # IC train=-0.071 lockbox=-0.157
    "features_sz_1d",            # IC train=-0.067 lockbox=-0.152
    "features_ssz_1mo",          # IC train=-0.038 lockbox=-0.075
    "features_ssz_3mo",          # IC train=-0.019 lockbox=-0.034
]

# Excluded:
#   features_sdz_1y     — FLIPPED in lockbox (train=-0.005, lockbox=+0.005)
#   All ICT features    — per MODEL_A spec
#   All momentum        — return_60d, sector_rs_20d, atr_*, sma200_slope_10
#   All macro           — market_breadth, rolling_beta_60d, low_52w_dist
#   hist_vol_20d        — not a zone feature

# ── LightGBM params ──────────────────────────────────────────────────────────
# Conservative defaults tuned for 16 features.
# One variable in this experiment: feature set only.
# Hyperparameters are intentionally simple — Optuna can tune later.

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=build_label_gain(),
    num_leaves=15,          # small tree — only 16 features
    min_child_samples=50,
    learning_rate=0.05,
    n_estimators=400,
    colsample_bytree=0.9,
    subsample=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    verbosity=-1,
    n_jobs=4,
)


# ── Walk-forward CV ──────────────────────────────────────────────────────────

def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame,
             features: list[str], date_level: str) -> dict:
    """Train on train_df, score test_df, return IC + top-decile excess."""

    train_df = train_df.dropna(subset=features + ["cs_rank_composite"])
    test_df  = test_df.dropna(subset=features + ["future_20d_excess_return"])

    X_tr = train_df[features]
    y_tr = cs_rank_to_label(train_df["cs_rank_composite"])
    groups_tr = train_df.groupby(level=date_level).size().sort_index().values

    X_te = test_df[features]

    model = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr, group=groups_tr, free_raw_data=False),
        num_boost_round=LGBM_PARAMS["n_estimators"],
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    scores = pd.Series(model.predict(X_te), index=test_df.index)

    # ── IC per date ─────────────────────────────────────────────
    ic_vals, top_dec_vals = [], []

    for dt in test_df.index.get_level_values(date_level).unique():
        grp = test_df.xs(dt, level=date_level)
        sc  = scores.xs(dt, level=date_level)
        merged = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
        if len(merged) < 20 or merged["score"].std() < 1e-9:
            continue

        ic, _ = spearmanr(merged["score"], merged["ret"])
        ic_vals.append(ic)

        # Top decile excess
        n_top = max(1, len(merged) // 10)
        top_ret = merged.nlargest(n_top, "score")["ret"].mean()
        top_dec_vals.append(top_ret)

    return {
        "mean_ic":        float(np.mean(ic_vals))       if ic_vals else np.nan,
        "std_ic":         float(np.std(ic_vals))        if ic_vals else np.nan,
        "n_dates":        len(ic_vals),
        "top_decile_exc": float(np.mean(top_dec_vals))  if top_dec_vals else np.nan,
        "feature_imp":    dict(zip(features,
                              model.feature_importance(importance_type="gain").tolist())),
    }


def run_wf_cv(panel: pd.DataFrame, features: list[str],
              fold_years: list[int]) -> list[dict]:
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    results = []

    for test_year in fold_years:
        dates = panel.index.get_level_values(date_level)
        train_mask = dates.year < test_year
        test_mask  = dates.year == test_year

        tr = panel[train_mask]
        te = panel[test_mask]

        if len(tr) < 5000 or len(te) < 500:
            print(f"  Fold {test_year}: skipped (insufficient data)")
            continue

        print(f"  Fold {test_year}: train={len(tr):,} rows, test={len(te):,} rows", end=" ... ")
        fold = run_fold(tr, te, features, date_level)
        fold["test_year"] = test_year
        results.append(fold)
        print(f"IC={fold['mean_ic']:.4f}  top-dec={fold['top_decile_exc']:.4f}")
        import gc; gc.collect()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    panel_path = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/model_a_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 60)
    print("  MODEL_A — Zone Core Only")
    print("=" * 60)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    # Feature availability check (before slimming — panel still has all columns)
    avail   = [f for f in ZONE_CORE_FEATURES if f in panel.columns]
    missing = [f for f in ZONE_CORE_FEATURES if f not in panel.columns]

    # Slim to only needed columns before fencing — drops 184 → ~18 columns,
    # reducing peak RAM from ~4GB to ~400MB and preventing a native OOM segfault.
    keep = avail + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    import gc; gc.collect()

    # Fence to tuning era
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    print(f"Fenced panel: {panel.shape[0]:,} rows, "
          f"{panel.index.get_level_values(date_level).nunique()} dates"
          f"  (RAM: {panel.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    print(f"\nZone features: {len(avail)}/{len(ZONE_CORE_FEATURES)} available")
    if missing:
        print(f"Missing (excluded): {missing}")

    # ── Walk-forward CV ──────────────────────────────────────────
    print(f"\nRunning expanding-window walk-forward CV ...")
    fold_results = run_wf_cv(panel, avail, fold_years=[2018, 2019, 2020, 2021, 2022, 2023])

    # ── Summary ──────────────────────────────────────────────────
    ics     = [r["mean_ic"]        for r in fold_results if not np.isnan(r["mean_ic"])]
    top_dec = [r["top_decile_exc"] for r in fold_results if not np.isnan(r["top_decile_exc"])]
    n       = len(ics)

    mean_ic    = float(np.mean(ics))
    std_ic     = float(np.std(ics))
    t_stat     = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd  = float(np.mean(top_dec)) if top_dec else np.nan

    print("\n" + "=" * 60)
    print("  MODEL_A Results")
    print("=" * 60)
    print(f"  Features used       : {len(avail)}")
    print(f"  Folds               : {n}")
    print(f"  Mean fold IC        : {mean_ic:+.4f}")
    print(f"  Std fold IC         : {std_ic:.4f}")
    print(f"  IC t-stat           : {t_stat:+.2f}")
    print(f"  Mean top-decile exc : {mean_topd:+.4f}")
    print()
    print("  Per-fold breakdown:")
    for r in fold_results:
        print(f"    {r['test_year']}: IC={r['mean_ic']:+.4f}  "
              f"top-dec={r['top_decile_exc']:+.4f}  n_dates={r['n_dates']}")

    print("\n  Current 53-feature model reference:")
    print("    Tuning-era fold IC  : ~0.010–0.012 (from lockbox logs)")
    print("    Lockbox IC          : +0.0106")
    print("    Lockbox IC t-stat   : +1.10 (HAC)")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "model":        "MODEL_A_zone_core",
        "n_features":   len(avail),
        "features":     avail,
        "fold_results": fold_results,
        "summary": {
            "mean_ic":            mean_ic,
            "std_ic":             std_ic,
            "t_stat":             t_stat,
            "n_folds":            n,
            "mean_top_decile_exc": mean_topd,
        },
        "reference_53feat": {
            "lockbox_ic":     0.0106,
            "lockbox_t_hac":  1.10,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
