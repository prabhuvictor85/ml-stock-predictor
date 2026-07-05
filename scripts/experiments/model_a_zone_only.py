#!/usr/bin/env python3
"""
MODEL_A — Zone Only (apple-to-apple companion to MODEL_D)
==========================================================
Identical harness to model_d_pivot_only.py — same panel, same 6-fold yearly
expanding-window CV (2018-2023), same LGBM params, same gate criteria.
Only the feature set changes: all zone-family columns
(features_sdz_* / features_ssz_* / features_dz_* / features_sz_* /
features_zone_*) instead of features_pivot_*.

Reads the SAME panel as MODEL_D (us_pivot_v1, built with --pit_universe
PIVOT_FEATURES=1) so the universe, label, and fence are bit-identical.
This makes the IC numbers directly comparable: any difference is features,
not methodology.

Run on Hetzner:
    cd /root/ml-stock-predictor
    python3 scripts/experiments/model_a_zone_only.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import pandas as pd

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
DEFAULT_OUT   = "/mnt/data/artefacts/experiments/model_a_results.json"

ZONE_PREFIXES = (
    "features_sdz_",
    "features_ssz_",
    "features_dz_",
    "features_sz_",
    "features_zone_",
)


def build_label_gain(n_bins: int = 100) -> list:
    return list(range(n_bins))


def cs_rank_to_label(cs_rank, n_bins: int = 100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=build_label_gain(),
    num_leaves=31,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODEL_A zone-only walk-forward CV")
    p.add_argument("--panel", default=DEFAULT_PANEL,
                   help="Path to panel_targets.pkl (same panel as MODEL_D for apple-to-apple).")
    p.add_argument("--out", default=DEFAULT_OUT, help="Results JSON output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel_path, out_path = args.panel, args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 60)
    print("  MODEL_A — Zone Only (apple-to-apple vs MODEL_D)")
    print("=" * 60)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    zone_cols = [c for c in panel.columns if c.startswith(ZONE_PREFIXES)]
    print(f"Found {len(zone_cols)} zone columns")
    if not zone_cols:
        print("\nFATAL: no zone feature columns found in panel.")
        sys.exit(2)

    keep = zone_cols + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()

    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    print(f"Fenced panel: {panel.shape[0]:,} rows, "
          f"{panel.index.get_level_values(date_level).nunique()} dates"
          f"  (RAM: {panel.memory_usage(deep=True).sum() / 1e9:.2f} GB)")

    print("\nRunning expanding-window walk-forward CV ...")
    fold_results = run_wf_cv(panel, zone_cols, fold_years=[2018, 2019, 2020, 2021, 2022, 2023])

    import numpy as np
    ics     = [r["mean_ic"]        for r in fold_results if not np.isnan(r["mean_ic"])]
    top_dec = [r["top_decile_exc"] for r in fold_results if not np.isnan(r["top_decile_exc"])]
    n       = len(ics)

    mean_ic   = float(np.mean(ics))                                    if n     else float("nan")
    std_ic    = float(np.std(ics))                                     if n     else float("nan")
    t_stat    = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd = float(np.mean(top_dec))                                if top_dec else float("nan")
    n_pos     = int(sum(1 for x in ics if x > 0))

    gate_pass = (mean_ic >= 0.03) and (t_stat >= 2.0) and (n_pos >= 4)

    print("\n" + "=" * 60)
    print("  MODEL_A Results")
    print("=" * 60)
    print(f"  Features used       : {len(zone_cols)}")
    print(f"  Folds               : {n}  ({n_pos} positive)")
    print(f"  Mean fold IC        : {mean_ic:+.4f}")
    print(f"  Std fold IC         : {std_ic:.4f}")
    print(f"  IC t-stat           : {t_stat:+.2f}")
    print(f"  Mean top-decile exc : {mean_topd:+.4f}")
    print()
    print("  Per-fold breakdown:")
    for r in fold_results:
        print(f"    {r['test_year']}: IC={r['mean_ic']:+.4f}  "
              f"top-dec={r['top_decile_exc']:+.4f}  n_dates={r['n_dates']}")
        print(f"      top features: {[f for f, _ in r['top10_features'][:5]]}")

    print("\n  References (same panel, same CV harness):")
    print("    MODEL_D (pivot-only) : IC=+0.0092  t=+1.10  folds_positive=4/6")
    print("    MODEL_C (ICT-only)   : IC=-0.00002 t=-0.01")
    print()
    print(f"  Gate (IC>=0.03 AND t>=2 AND >=4/6 folds +ve): "
          f"{'PASS' if gate_pass else 'FAIL'}")

    output = {
        "model":        "MODEL_A_zone_only",
        "n_features":   len(zone_cols),
        "features":     zone_cols,
        "fold_results": fold_results,
        "summary": {
            "mean_ic":             mean_ic,
            "std_ic":              std_ic,
            "t_stat":              t_stat,
            "n_folds":             n,
            "n_folds_positive":    n_pos,
            "mean_top_decile_exc": mean_topd,
        },
        "preregistered_gate": {
            "criteria":    "mean_ic>=0.03 AND t_stat>=2.0 AND n_folds_positive>=4",
            "gate_pass":   gate_pass,
        },
        "references": {
            "model_d_pivot_only": {"mean_ic": 0.0092, "t_stat": 1.10},
            "model_c_ict_only":   {"mean_ic": -0.00002, "t_stat": -0.01},
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
