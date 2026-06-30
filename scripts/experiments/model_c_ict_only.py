#!/usr/bin/env python3
"""
MODEL_C — ICT Only
===================
Companion experiment to MODEL_A (zone-core). Same walk-forward CV, same
hyperparameters — only the feature set changes: all ICT/ADX columns
instead of the 16 zone features. Isolates whether ICT's poor lockbox
showing is multivariate noise (features fighting each other inside the
tree) or simply weak individual signal that no amount of combination
fixes.

ICT feature list is auto-discovered from the panel (prefix match), same
as the feature-corruption audit, so it always matches whatever ICT
columns actually exist — no hand-typed list to drift out of sync.

Run on Theralytics:
    cd /root/ml-stock-predictor
    python3 scripts/experiments/model_c_ict_only.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

# pandas only at module scope — numpy/scipy/lightgbm imported lazily after
# the panel pickle is loaded (see MODEL_A for why: Python 3.14 + numpy
# 2.4.6 segfaults in _multiarray_umath when those are imported first).
import pandas as pd


def build_label_gain(n_bins: int = 100) -> list:
    return list(range(n_bins))


def cs_rank_to_label(cs_rank, n_bins: int = 100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=build_label_gain(),
    num_leaves=31,          # wider than MODEL_A — 66 features need more splits to use
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


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame,
             features: list, date_level: str) -> dict:
    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr

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

    ic_vals, top_dec_vals = [], []
    for dt in test_df.index.get_level_values(date_level).unique():
        grp = test_df.xs(dt, level=date_level)
        sc  = scores.xs(dt, level=date_level)
        merged = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
        if len(merged) < 20 or merged["score"].std() < 1e-9:
            continue
        ic, _ = spearmanr(merged["score"], merged["ret"])
        ic_vals.append(ic)
        n_top = max(1, len(merged) // 10)
        top_ret = merged.nlargest(n_top, "score")["ret"].mean()
        top_dec_vals.append(top_ret)

    imp = model.feature_importance(importance_type="gain").tolist()
    top_feats = sorted(zip(features, imp), key=lambda x: -x[1])[:10]

    return {
        "mean_ic":        float(np.mean(ic_vals))       if ic_vals else float("nan"),
        "std_ic":         float(np.std(ic_vals))        if ic_vals else float("nan"),
        "n_dates":        len(ic_vals),
        "top_decile_exc": float(np.mean(top_dec_vals))  if top_dec_vals else float("nan"),
        "top10_features": top_feats,
    }


def run_wf_cv(panel: pd.DataFrame, features: list, fold_years: list) -> list:
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
        gc.collect()
    return results


def main() -> None:
    panel_path = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/model_c_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 60)
    print("  MODEL_C — ICT Only")
    print("=" * 60)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    ict_cols = [c for c in panel.columns if c.startswith("features_ict_")
                or c.startswith("features_adx") or c.startswith("features_plus_di")
                or c.startswith("features_minus_di")]
    print(f"Found {len(ict_cols)} ICT/ADX columns")

    keep = ict_cols + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()

    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    print(f"Fenced panel: {panel.shape[0]:,} rows, "
          f"{panel.index.get_level_values(date_level).nunique()} dates"
          f"  (RAM: {panel.memory_usage(deep=True).sum() / 1e9:.2f} GB)")

    print(f"\nRunning expanding-window walk-forward CV ...")
    fold_results = run_wf_cv(panel, ict_cols, fold_years=[2018, 2019, 2020, 2021, 2022, 2023])

    import numpy as np
    ics     = [r["mean_ic"]        for r in fold_results if not np.isnan(r["mean_ic"])]
    top_dec = [r["top_decile_exc"] for r in fold_results if not np.isnan(r["top_decile_exc"])]
    n       = len(ics)

    mean_ic    = float(np.mean(ics))
    std_ic     = float(np.std(ics))
    t_stat     = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd  = float(np.mean(top_dec)) if top_dec else float("nan")

    print("\n" + "=" * 60)
    print("  MODEL_C Results")
    print("=" * 60)
    print(f"  Features used       : {len(ict_cols)}")
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
        print(f"      top features: {[f for f,_ in r['top10_features'][:5]]}")

    print("\n  MODEL_A (zone-core) reference:")
    print("    Mean fold IC        : +0.1441")
    print("    IC t-stat           : +8.53")
    print("    Mean top-decile exc : +0.0345")

    output = {
        "model":        "MODEL_C_ict_only",
        "n_features":   len(ict_cols),
        "features":     ict_cols,
        "fold_results": fold_results,
        "summary": {
            "mean_ic":            mean_ic,
            "std_ic":             std_ic,
            "t_stat":             t_stat,
            "n_folds":            n,
            "mean_top_decile_exc": mean_topd,
        },
        "model_a_reference": {
            "mean_ic": 0.1441,
            "t_stat":  8.53,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
