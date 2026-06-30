#!/usr/bin/env python3
"""
MODEL_A — Zone Core Only, LOCKBOX protocol
===========================================
model_a_zone_core.py measured walk-forward CV IC *inside* 2018-2023 — the
same window the zone thresholds (ssz/sdz cutoffs) were calibrated on. That
in-sample number (+0.14 mean fold IC) is not comparable to the production
lockbox number (+0.0106), which is scored on genuinely unseen 2024-2026 data.

This script closes that gap: ONE static split — train on everything up to
2023-12-31, score the entire lockbox window (2024-01-01 .. 2026-05-13) — using
ONLY the 16 zone-core features. No periodic retraining (unlike the real
production walk-forward that produced +0.0106), so treat this as a fast
upper-bound proxy, not an exact replication.

Run on Hetzner:
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_a_zone_core_lockbox.py \
        > /tmp/model_a_lockbox.log 2>&1 &
    tail -f /tmp/model_a_lockbox.log

Output:
    /mnt/data/artefacts/experiments/model_a_lockbox_results.json
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
# panel pickle is loaded and slimmed (numpy/lightgbm import-before-read_pickle
# segfaults on this server: Python 3.14 + numpy 2.4.6 ABI issue).
import pandas as pd

LOCKBOX_START = "2024-01-01"
LOCKBOX_END   = "2026-05-13"   # matches validate_lockbox.py example window
TRAIN_END     = "2023-12-31"

ZONE_CORE_FEATURES = [
    "features_sdz_1d", "features_sdz_htf_score", "features_sdz_1wk",
    "features_dz_raw_score", "features_dz_1mo", "features_dz_1wk",
    "features_dz_1y", "features_sdz_1mo", "features_zone_strength",
    "features_sdz_3mo",
    "features_ssz_1wk", "features_sz_raw_score", "features_ssz_htf_score",
    "features_sz_1d", "features_ssz_1mo", "features_ssz_3mo",
]

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=list(range(100)),
    num_leaves=15,
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


def cs_rank_to_label(cs_rank, n_bins: int = 100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def main() -> None:
    panel_path = "/mnt/data/artefacts/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/model_a_lockbox_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 64)
    print("  MODEL_A — Zone Core Only — LOCKBOX protocol")
    print(f"  train <= {TRAIN_END}   test = [{LOCKBOX_START}, {LOCKBOX_END}]")
    print("=" * 64)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    avail   = [f for f in ZONE_CORE_FEATURES if f in panel.columns]
    missing = [f for f in ZONE_CORE_FEATURES if f not in panel.columns]
    print(f"\nZone features: {len(avail)}/{len(ZONE_CORE_FEATURES)} available")
    if missing:
        print(f"Missing (excluded): {missing}")

    keep = avail + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()

    dates = panel.index.get_level_values(date_level)
    train_df = panel[dates <= pd.Timestamp(TRAIN_END)]
    test_df  = panel[(dates >= pd.Timestamp(LOCKBOX_START)) & (dates <= pd.Timestamp(LOCKBOX_END))]
    del panel
    gc.collect()

    print(f"\nTrain: {len(train_df):,} rows, "
          f"{train_df.index.get_level_values(date_level).nunique()} dates")
    print(f"Test  (lockbox): {len(test_df):,} rows, "
          f"{test_df.index.get_level_values(date_level).nunique()} dates")

    train_df = train_df.dropna(subset=avail + ["cs_rank_composite"])
    test_df  = test_df.dropna(subset=avail + ["future_20d_excess_return"])

    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr

    X_tr = train_df[avail]
    y_tr = cs_rank_to_label(train_df["cs_rank_composite"])
    groups_tr = train_df.groupby(level=date_level).size().sort_index().values

    print(f"\nTraining single static model on {len(X_tr):,} rows ...")
    model = lgb.train(
        LGBM_PARAMS,
        lgb.Dataset(X_tr, label=y_tr, group=groups_tr, free_raw_data=False),
        num_boost_round=LGBM_PARAMS["n_estimators"],
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    print("Training done. Scoring lockbox window ...")

    X_te = test_df[avail]
    scores = pd.Series(model.predict(X_te), index=test_df.index)

    ic_vals, top_dec_vals, ic_dates = [], [], []
    for dt in test_df.index.get_level_values(date_level).unique():
        grp = test_df.xs(dt, level=date_level)
        sc  = scores.xs(dt, level=date_level)
        merged = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
        if len(merged) < 20 or merged["score"].std() < 1e-9:
            continue
        ic, _ = spearmanr(merged["score"], merged["ret"])
        ic_vals.append(ic)
        ic_dates.append(str(dt.date()) if hasattr(dt, "date") else str(dt))
        n_top = max(1, len(merged) // 10)
        top_ret = merged.nlargest(n_top, "score")["ret"].mean()
        top_dec_vals.append(top_ret)

    ic_arr = np.array(ic_vals, dtype=float)
    n = len(ic_arr)
    mean_ic = float(np.mean(ic_arr)) if n else float("nan")
    std_ic  = float(np.std(ic_arr, ddof=1)) if n > 1 else 0.0
    t_naive = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd = float(np.mean(top_dec_vals)) if top_dec_vals else float("nan")

    print("\n" + "=" * 64)
    print("  MODEL_A LOCKBOX Results (zone-core, 16 features)")
    print("=" * 64)
    print(f"  n score dates        : {n}")
    print(f"  Mean lockbox IC      : {mean_ic:+.4f}")
    print(f"  Std IC               : {std_ic:.4f}")
    print(f"  t-stat (naive)       : {t_naive:+.2f}   <- inflated by overlap, sanity only")
    print(f"  Mean top-decile exc  : {mean_topd:+.4f}")
    print()
    print("  Reference — production 53-feature lockbox:")
    print("    Lockbox IC          : +0.0106")
    print("    Lockbox IC t-stat   : +1.10 (HAC)")
    print()
    print("  Reference — MODEL_A in-sample walk-forward CV (2018-2023):")
    print("    Mean fold IC        : +0.1441  <- NOT comparable, same window as calibration")

    feat_imp = dict(zip(avail, model.feature_importance(importance_type="gain").tolist()))

    output = {
        "model": "MODEL_A_zone_core_LOCKBOX",
        "protocol": {
            "train_end": TRAIN_END,
            "lockbox_start": LOCKBOX_START,
            "lockbox_end": LOCKBOX_END,
            "retraining": "none (single static split — NOT periodic like production)",
        },
        "n_features": len(avail),
        "features": avail,
        "feature_importance_gain": feat_imp,
        "n_score_dates": n,
        "ic_dates": ic_dates,
        "ic_values": ic_vals,
        "summary": {
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "t_stat_naive": t_naive,
            "mean_top_decile_exc": mean_topd,
        },
        "reference": {
            "production_lockbox_ic": 0.0106,
            "production_lockbox_t_hac": 1.10,
            "model_a_insample_wfcv_ic": 0.1441,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
