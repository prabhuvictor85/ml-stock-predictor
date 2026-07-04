#!/usr/bin/env python3
"""
MODEL_D — Pivot Only, LOCKBOX protocol
======================================
model_d_pivot_only.py measured walk-forward CV IC *inside* 2018-2023 — the same
window every pivot parameter default lives in spirit. This script closes the
gap with ONE static split: train on everything up to 2023-12-31, score the
entire lockbox window (2024-01-01 .. last date with realized 20d forward
returns) using ONLY the `features_pivot_*` family. No periodic retraining
(unlike production walk-forward), so treat it as a fast upper-bound proxy.

ONE-SHOT RULE (PROTOCOL.md §6): run this ONLY after the CV gate in
model_d_pivot_only.py passes (mean IC >= +0.03, t >= 2, >= 4/6 folds positive).
It consumes one look at the 2024-26 lockbox window — do not re-run it to chase a
disappointing number.

Run on Hetzner (only if CV passed):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_d_pivot_only_lockbox.py \
        > /tmp/model_d_lockbox.log 2>&1 &
    tail -f /tmp/model_d_lockbox.log

Output:
    /mnt/data/artefacts/experiments/model_d_lockbox_results.json
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

# pandas only at module scope — numpy/lightgbm imported lazily AFTER the panel
# pickle is loaded and slimmed (numpy/lightgbm import-before-read_pickle
# segfaults on this server: Python 3.14 + numpy 2.4.6 ABI issue).
import pandas as pd

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
DEFAULT_OUT   = "/mnt/data/artefacts/experiments/model_d_lockbox_results.json"

LOCKBOX_START = "2024-01-01"
TRAIN_END     = "2023-12-31"

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=list(range(100)),
    num_leaves=31,          # matches MODEL_D CV
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MODEL_D pivot-only lockbox static split")
    p.add_argument("--panel", default=DEFAULT_PANEL)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    panel_path, out_path = args.panel, args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 64)
    print("  MODEL_D — Pivot Only — LOCKBOX protocol")
    print("=" * 64)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    pivot_cols = [c for c in panel.columns if c.startswith("features_pivot_")]
    print(f"Found {len(pivot_cols)} pivot columns")
    if not pivot_cols:
        print("\nFATAL: no features_pivot_* columns — panel built without PIVOT_FEATURES=1.")
        sys.exit(2)

    # Determine the last date with a realized 20d forward return, from the panel.
    with_ret = panel.dropna(subset=["future_20d_excess_return"])
    lockbox_end = with_ret.index.get_level_values(date_level).max()
    lockbox_end_str = str(lockbox_end.date()) if hasattr(lockbox_end, "date") else str(lockbox_end)
    print(f"  train <= {TRAIN_END}   test = [{LOCKBOX_START}, {lockbox_end_str}]")

    keep = pivot_cols + ["cs_rank_composite", "future_20d_excess_return"]
    panel = panel[[c for c in keep if c in panel.columns]].copy()
    gc.collect()

    dates = panel.index.get_level_values(date_level)
    train_df = panel[dates <= pd.Timestamp(TRAIN_END)]
    test_df  = panel[(dates >= pd.Timestamp(LOCKBOX_START)) & (dates <= lockbox_end)]
    del panel
    gc.collect()

    print(f"\nTrain: {len(train_df):,} rows, "
          f"{train_df.index.get_level_values(date_level).nunique()} dates")
    print(f"Test  (lockbox): {len(test_df):,} rows, "
          f"{test_df.index.get_level_values(date_level).nunique()} dates")

    train_df = train_df.dropna(subset=pivot_cols + ["cs_rank_composite"])
    test_df  = test_df.dropna(subset=pivot_cols + ["future_20d_excess_return"])

    import numpy as np
    import lightgbm as lgb
    from scipy.stats import spearmanr

    X_tr = train_df[pivot_cols]
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

    X_te = test_df[pivot_cols]
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
        top_dec_vals.append(merged.nlargest(n_top, "score")["ret"].mean())

    ic_arr = np.array(ic_vals, dtype=float)
    n = len(ic_arr)
    mean_ic = float(np.mean(ic_arr)) if n else float("nan")
    std_ic  = float(np.std(ic_arr, ddof=1)) if n > 1 else 0.0
    t_naive = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    mean_topd = float(np.mean(top_dec_vals)) if top_dec_vals else float("nan")

    print("\n" + "=" * 64)
    print("  MODEL_D LOCKBOX Results (pivot-only)")
    print("=" * 64)
    print(f"  n score dates        : {n}")
    print(f"  Mean lockbox IC      : {mean_ic:+.4f}")
    print(f"  Std IC               : {std_ic:.4f}")
    print(f"  t-stat (naive)       : {t_naive:+.2f}   <- inflated by overlap, sanity only")
    print(f"  Mean top-decile exc  : {mean_topd:+.4f}")
    print()
    print("  References:")
    print("    Production 53-feature lockbox IC : +0.0106 (t=+1.10 HAC)")
    print("    MODEL_A zone-core in-sample CV   : +0.1441 (NOT comparable)")

    feat_imp = dict(zip(pivot_cols, model.feature_importance(importance_type="gain").tolist()))

    output = {
        "model": "MODEL_D_pivot_only_LOCKBOX",
        "protocol": {
            "train_end": TRAIN_END,
            "lockbox_start": LOCKBOX_START,
            "lockbox_end": lockbox_end_str,
            "retraining": "none (single static split — NOT periodic like production)",
        },
        "n_features": len(pivot_cols),
        "features": pivot_cols,
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
            "model_d_insample_wfcv_note": "see model_d_results.json",
        },
    }
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
