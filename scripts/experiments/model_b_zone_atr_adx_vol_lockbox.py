#!/usr/bin/env python3
"""
MODEL_B — All non-ICT features, walk-forward lockbox protocol
=============================================================
Mirrors the production lockbox v2 procedure exactly:
  - Retrain every 14 days with an expanding window (all data up to that date)
  - Score the cross-section on each retrain date
  - IC = Spearman(scores, future_20d_excess_return) per scoring date
  - Report mean IC and t-stat across all ~65 scoring dates

This is apples-to-apples vs the production lockbox v2 result (+0.0106).
The only difference: this uses ALL non-ICT features (~68 cols) instead of
the production 53-feature HPO-selected set.

Feature set: every features_* column in the panel except features_ict_*.
Also excluded: return_1d and return_5d (microstructure noise, near-zero
predictive power for a 20d forward target).

Run on Theralytics:
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_b_zone_atr_adx_vol_lockbox.py \\
        > /tmp/model_b_lockbox.log 2>&1 &
    tail -f /tmp/model_b_lockbox.log

Output:
    /mnt/data/artefacts/experiments/model_b_lockbox_results.json

Expected runtime: 2-4 hours (~65 retrains x ~2 min each).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

# pandas only at module scope — numpy/scipy/lightgbm imported lazily AFTER the
# panel pickle is loaded (Python 3.14 + numpy 2.4.6 segfaults if numpy is
# imported before pd.read_pickle — see model_a_zone_core.py).
import pandas as pd

LOCKBOX_START  = "2024-01-12"
LOCKBOX_END    = "2026-05-04"
INITIAL_TRAIN_END = "2023-12-31"   # model must not see lockbox data before this
CADENCE_DAYS   = 14                # retrain every 2 weeks, matching production

# Explicitly excluded: sub-week returns are noise for a 20d forward target
_EXCLUDE = {"features_return_1d", "features_return_5d"}

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=list(range(100)),
    num_leaves=63,
    min_child_samples=50,
    learning_rate=0.05,
    n_estimators=400,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.05,
    reg_lambda=0.1,
    verbosity=-1,
    n_jobs=4,
)


def cs_rank_to_label(cs_rank, n_bins: int = 100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def main() -> None:
    panel_path = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/model_b_lockbox_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 64)
    print("  MODEL_B — All non-ICT features — Walk-forward lockbox")
    print(f"  cadence={CADENCE_DAYS}d  lockbox=[{LOCKBOX_START}, {LOCKBOX_END}]")
    print("=" * 64)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    # Auto-discover all non-ICT feature columns
    avail = sorted([
        c for c in panel.columns
        if c.startswith("features_")
        and not c.startswith("features_ict_")
        and c not in _EXCLUDE
    ])
    ict_excluded = [c for c in panel.columns if c.startswith("features_ict_")]

    print(f"\nFeatures included : {len(avail)}")
    print(f"ICT cols excluded : {len(ict_excluded)}")
    print(f"Also excluded     : {sorted(_EXCLUDE)}")

    keep = avail + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()
    print(f"Panel slimmed: {panel.shape[0]:,} rows, "
          f"{panel.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # All unique dates in the panel
    all_dates = pd.DatetimeIndex(
        panel.index.get_level_values(date_level).unique()
    ).sort_values()

    # Scoring dates: every CADENCE_DAYS within the lockbox window
    lockbox_dates = all_dates[
        (all_dates >= pd.Timestamp(LOCKBOX_START)) &
        (all_dates <= pd.Timestamp(LOCKBOX_END))
    ]
    scoring_dates = lockbox_dates[::CADENCE_DAYS]
    print(f"\nScoring dates     : {len(scoring_dates)} "
          f"({LOCKBOX_START} → {LOCKBOX_END}, every {CADENCE_DAYS} trading days)")

    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr

    ic_vals, top_dec_vals, ic_dates = [], [], []
    feat_imp_acc = np.zeros(len(avail), dtype=np.float64)

    for step, score_dt in enumerate(scoring_dates):
        # Expanding train window: all data strictly before this scoring date
        train_mask = panel.index.get_level_values(date_level) < score_dt
        train_df   = panel[train_mask].dropna(subset=avail + ["cs_rank_composite"])

        # Cross-section on the scoring date
        test_mask = panel.index.get_level_values(date_level) == score_dt
        test_df   = panel[test_mask].dropna(subset=avail + ["future_20d_excess_return"])

        if len(train_df) < 5000 or len(test_df) < 20:
            print(f"  [{step+1:02d}/{len(scoring_dates)}] {score_dt.date()}  "
                  f"SKIPPED (train={len(train_df)}, test={len(test_df)})")
            continue

        X_tr      = train_df[avail]
        y_tr      = cs_rank_to_label(train_df["cs_rank_composite"])
        groups_tr = train_df.groupby(level=date_level).size().sort_index().values

        model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr, label=y_tr, group=groups_tr, free_raw_data=False),
            num_boost_round=LGBM_PARAMS["n_estimators"],
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        scores = pd.Series(model.predict(test_df[avail]), index=test_df.index)
        merged = pd.DataFrame({
            "score": scores,
            "ret":   test_df["future_20d_excess_return"],
        }).dropna()

        if len(merged) >= 20 and merged["score"].std() > 1e-9:
            ic, _ = spearmanr(merged["score"], merged["ret"])
            n_top  = max(1, len(merged) // 10)
            top_r  = merged.nlargest(n_top, "score")["ret"].mean()
            ic_vals.append(ic)
            top_dec_vals.append(top_r)
            ic_dates.append(str(score_dt.date()))
            feat_imp_acc += np.array(model.feature_importance(importance_type="gain"),
                                     dtype=np.float64)
            print(f"  [{step+1:02d}/{len(scoring_dates)}] {score_dt.date()}  "
                  f"n={len(merged)}  IC={ic:+.4f}  top-dec={top_r:+.4f}")
        else:
            print(f"  [{step+1:02d}/{len(scoring_dates)}] {score_dt.date()}  "
                  f"SKIPPED (degenerate cross-section)")

        del train_df, test_df, X_tr, model
        gc.collect()

    ic_arr    = np.array(ic_vals, dtype=float)
    n         = len(ic_arr)
    mean_ic   = float(np.mean(ic_arr))          if n    else float("nan")
    std_ic    = float(np.std(ic_arr, ddof=1))   if n > 1 else 0.0
    t_stat    = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd = float(np.mean(top_dec_vals))    if top_dec_vals else float("nan")

    avg_imp = feat_imp_acc / max(n, 1)
    feat_imp = dict(zip(avail, avg_imp.tolist()))
    top10    = sorted(feat_imp.items(), key=lambda x: -x[1])[:10]

    print("\n" + "=" * 64)
    print("  MODEL_B LOCKBOX Results (all non-ICT, walk-forward)")
    print("=" * 64)
    print(f"  Features used        : {len(avail)}")
    print(f"  Scoring dates        : {n}")
    print(f"  Mean lockbox IC      : {mean_ic:+.4f}")
    print(f"  Std IC               : {std_ic:.4f}")
    print(f"  IC t-stat            : {t_stat:+.2f}")
    print(f"  Mean top-decile exc  : {mean_topd:+.4f}")
    print()
    print("  Top-10 features by avg gain:")
    for feat, gain in top10:
        print(f"    {feat:<50}  {gain:>8.0f}")
    print()
    print("  Reference benchmarks (walk-forward, same cadence):")
    print("    Production 53-feat  lockbox IC : +0.0106  t=+1.10 (HAC)")

    output = {
        "model": "MODEL_B_no_ict_walkforward_LOCKBOX",
        "protocol": {
            "lockbox_start":     LOCKBOX_START,
            "lockbox_end":       LOCKBOX_END,
            "initial_train_end": INITIAL_TRAIN_END,
            "cadence_days":      CADENCE_DAYS,
            "retraining":        "expanding window, every 14 trading days",
        },
        "n_features":              len(avail),
        "n_ict_excluded":          len(ict_excluded),
        "features":                avail,
        "feature_importance_gain": feat_imp,
        "n_score_dates":           n,
        "ic_dates":                ic_dates,
        "ic_values":               ic_vals,
        "top_decile_values":       top_dec_vals,
        "summary": {
            "mean_ic":             mean_ic,
            "std_ic":              std_ic,
            "t_stat":              t_stat,
            "mean_top_decile_exc": mean_topd,
        },
        "reference": {
            "production_lockbox_ic":    0.0106,
            "production_lockbox_t_hac": 1.10,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
