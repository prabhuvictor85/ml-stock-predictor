#!/usr/bin/env python3
"""
MODEL_B — All non-ICT features, LOCKBOX protocol
=================================================
Uses EVERY features_* column in the panel EXCEPT features_ict_* columns.
This is the "ICT disabled" baseline: zones, ATR, ADX, volume, returns, SMAs,
breakouts, regime, market breadth, sector RS, beta, trend — all included.
ICT columns (OB, FVG, BOS, sweeps, zone priority, HTF scores) are excluded.

Same static-split lockbox protocol as model_a_zone_core_lockbox.py:
  - Train on everything up to 2023-12-31
  - Score the full lockbox window (2024-01-01 -> 2026-05-13)
  - One shot, no periodic retraining

Feature count is auto-discovered from the panel at runtime so it stays in sync
with any future panel rebuilds. Explicitly excluded: return_1d and return_5d
(near-zero predictive power for a 20d target — microstructure noise, not
momentum signal; meaningful lookbacks are return_20d and return_60d).

Run on Theralytics:
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_b_zone_atr_adx_vol_lockbox.py \\
        > /tmp/model_b_lockbox.log 2>&1 &
    tail -f /tmp/model_b_lockbox.log

Output:
    /mnt/data/artefacts/experiments/model_b_lockbox_results.json
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
# imported before pd.read_pickle on this server — see model_a_zone_core.py).
import pandas as pd

LOCKBOX_START = "2024-01-01"
LOCKBOX_END   = "2026-05-13"
TRAIN_END     = "2023-12-31"

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=list(range(100)),
    num_leaves=63,           # more capacity for the full non-ICT feature set
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


def _family(col: str) -> str:
    """Classify a features_* column into a family for the summary printout."""
    c = col.removeprefix("features_")
    if c.startswith(("sdz", "ssz", "dz_", "sz_", "zone")):
        return "zone"
    if c.startswith(("atr", "vol_contraction", "compression", "hist_vol")):
        return "atr/vol_regime"
    if c.startswith(("adx", "plus_di", "minus_di")):
        return "adx"
    if c.startswith("vol_ratio"):
        return "volume"
    if c.startswith("return_"):
        return "returns"
    if c.startswith(("sma", "price_vs_sma", "20d_break", "50d_break", "high_52", "low_52")):
        return "sma/breakout"
    if c.startswith("regime") or c in ("market_breadth", "weekly_trend", "monthly_trend",
                                        "quarterly_trend", "yearly_trend"):
        return "regime/trend"
    if c.startswith(("sector_rs", "rolling_beta")):
        return "sector/beta"
    return "other"


def main() -> None:
    panel_path = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/model_b_lockbox_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 64)
    print("  MODEL_B — All non-ICT features — LOCKBOX protocol")
    print(f"  train <= {TRAIN_END}   test = [{LOCKBOX_START}, {LOCKBOX_END}]")
    print("=" * 64)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    # Excluded explicitly: short-horizon returns have near-zero predictive power
    # for a 20d forward target (microstructure noise, not momentum signal).
    # Meaningful momentum lookbacks are return_20d and return_60d.
    _EXCLUDE = {"features_return_1d", "features_return_5d"}

    # Auto-discover: all features_* columns EXCEPT features_ict_* and _EXCLUDE
    avail = sorted([
        c for c in panel.columns
        if c.startswith("features_")
        and not c.startswith("features_ict_")
        and c not in _EXCLUDE
    ])
    ict_excluded = [c for c in panel.columns if c.startswith("features_ict_")]

    print(f"\nFeatures included : {len(avail)} (all features_* excluding ict)")
    print(f"ICT cols excluded : {len(ict_excluded)}")

    # Family breakdown
    from collections import Counter
    counts = Counter(_family(c) for c in avail)
    for fam, n in sorted(counts.items()):
        print(f"  {fam:<20} {n}")

    keep = avail + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()
    print(f"\nPanel slimmed to {panel.shape[1]} cols, "
          f"{panel.memory_usage(deep=True).sum()/1e9:.2f} GB")

    dates    = panel.index.get_level_values(date_level)
    train_df = panel[dates <= pd.Timestamp(TRAIN_END)]
    test_df  = panel[(dates >= pd.Timestamp(LOCKBOX_START)) &
                     (dates <= pd.Timestamp(LOCKBOX_END))]
    del panel
    gc.collect()

    print(f"\nTrain : {len(train_df):,} rows, "
          f"{train_df.index.get_level_values(date_level).nunique()} dates")
    print(f"Test  : {len(test_df):,} rows, "
          f"{test_df.index.get_level_values(date_level).nunique()} dates (lockbox)")

    train_df = train_df.dropna(subset=avail + ["cs_rank_composite"])
    test_df  = test_df.dropna(subset=avail + ["future_20d_excess_return"])
    print(f"After dropna — train: {len(train_df):,}  test: {len(test_df):,}")

    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr

    X_tr      = train_df[avail]
    y_tr      = cs_rank_to_label(train_df["cs_rank_composite"])
    groups_tr = train_df.groupby(level=date_level).size().sort_index().values

    print(f"\nTraining on {len(X_tr):,} rows ({len(avail)} features) ...")
    model = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr, group=groups_tr, free_raw_data=False),
        num_boost_round=LGBM_PARAMS["n_estimators"],
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    print("Training done. Scoring lockbox window ...")

    X_te   = test_df[avail]
    scores = pd.Series(model.predict(X_te), index=test_df.index)

    ic_vals, top_dec_vals, ic_dates = [], [], []
    for dt in test_df.index.get_level_values(date_level).unique():
        grp    = test_df.xs(dt, level=date_level)
        sc     = scores.xs(dt, level=date_level)
        merged = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
        if len(merged) < 20 or merged["score"].std() < 1e-9:
            continue
        ic, _ = spearmanr(merged["score"], merged["ret"])
        ic_vals.append(ic)
        ic_dates.append(str(dt.date()) if hasattr(dt, "date") else str(dt))
        n_top   = max(1, len(merged) // 10)
        top_ret = merged.nlargest(n_top, "score")["ret"].mean()
        top_dec_vals.append(top_ret)

    ic_arr    = np.array(ic_vals, dtype=float)
    n         = len(ic_arr)
    mean_ic   = float(np.mean(ic_arr))          if n    else float("nan")
    std_ic    = float(np.std(ic_arr, ddof=1))   if n > 1 else 0.0
    t_naive   = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd = float(np.mean(top_dec_vals))    if top_dec_vals else float("nan")

    feat_imp = dict(zip(avail, model.feature_importance(importance_type="gain").tolist()))
    top10    = sorted(feat_imp.items(), key=lambda x: -x[1])[:10]

    print("\n" + "=" * 64)
    print("  MODEL_B LOCKBOX Results (all non-ICT features)")
    print("=" * 64)
    print(f"  Features used        : {len(avail)}")
    print(f"  ICT cols excluded    : {len(ict_excluded)}")
    print(f"  n score dates        : {n}")
    print(f"  Mean lockbox IC      : {mean_ic:+.4f}")
    print(f"  Std IC               : {std_ic:.4f}")
    print(f"  t-stat (naive)       : {t_naive:+.2f}   <- inflated by overlap, sanity only")
    print(f"  Mean top-decile exc  : {mean_topd:+.4f}")
    print()
    print("  Top-10 features by gain:")
    for feat, gain in top10:
        print(f"    {feat:<50}  {gain:>8.0f}")
    print()
    print("  Reference benchmarks:")
    print("    MODEL_A zone-only   in-sample  CV (2018-2023) : +0.1441  t=+8.53")
    print("    Production 53-feat  lockbox IC (walk-fwd HAC) : +0.0106  t=+1.10")

    output = {
        "model": "MODEL_B_no_ict_LOCKBOX",
        "protocol": {
            "train_end":     TRAIN_END,
            "lockbox_start": LOCKBOX_START,
            "lockbox_end":   LOCKBOX_END,
            "retraining":    "none (single static split)",
        },
        "n_features":              len(avail),
        "n_ict_excluded":          len(ict_excluded),
        "ict_excluded":            ict_excluded,
        "features":                avail,
        "feature_importance_gain": feat_imp,
        "n_score_dates":           n,
        "ic_dates":                ic_dates,
        "ic_values":               ic_vals,
        "top_decile_values":       top_dec_vals,
        "summary": {
            "mean_ic":             mean_ic,
            "std_ic":              std_ic,
            "t_stat_naive":        t_naive,
            "mean_top_decile_exc": mean_topd,
        },
        "reference": {
            "production_lockbox_ic":    0.0106,
            "production_lockbox_t_hac": 1.10,
            "model_a_insample_ic":      0.1441,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
