#!/usr/bin/env python3
"""
Feature Degradation Audit
==========================
Model-free diagnostic: for every features_* column in the panel, compute its
own univariate Spearman IC against future_20d_excess_return, separately on
the TRAIN window (<=2023) and the LOCKBOX window (2024-01-01..2026-05-13).

No model training involved -- this isolates which individual features carry
signal that doesn't survive into the unseen period, independent of how the
ranker happens to weight them. A feature that's strongly positive in train
and flips negative (or collapses toward 0) in the lockbox is a direct
suspect for degrading the full model's lockbox IC.

Run on Hetzner:
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/feature_degradation_audit.py \
        > /tmp/feat_audit.log 2>&1 &
    tail -f /tmp/feat_audit.log

Output:
    /mnt/data/artefacts/experiments/feature_degradation_audit.json
    -- sorted by (train_ic - lockbox_ic), worst offenders first
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import pandas as pd

TRAIN_END     = "2023-12-31"
LOCKBOX_START = "2024-01-01"
LOCKBOX_END   = "2026-05-13"

TARGET_COL = "future_20d_excess_return"

# Minimum cross-sectional names per date to keep that date's IC (avoid noisy
# thin-universe days dominating the mean).
MIN_NAMES_PER_DATE = 20


def per_period_ic(panel: pd.DataFrame, feature: str, date_level: str) -> dict:
    """Mean of per-date Spearman IC(feature, target) across all dates present."""
    import numpy as np
    from scipy.stats import spearmanr

    sub = panel[[feature, TARGET_COL]].dropna()
    if sub.empty:
        return {"mean_ic": float("nan"), "n_dates": 0, "n_rows": 0}

    ics = []
    for dt, grp in sub.groupby(level=date_level):
        if len(grp) < MIN_NAMES_PER_DATE:
            continue
        if grp[feature].std() < 1e-12:
            continue
        ic, _ = spearmanr(grp[feature], grp[TARGET_COL])
        if np.isfinite(ic):
            ics.append(ic)

    if not ics:
        return {"mean_ic": float("nan"), "n_dates": 0, "n_rows": len(sub)}
    return {
        "mean_ic": float(np.mean(ics)),
        "std_ic": float(np.std(ics)),
        "n_dates": len(ics),
        "n_rows": int(len(sub)),
    }


def main() -> None:
    panel_path = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
    out_path   = "/mnt/data/artefacts/experiments/feature_degradation_audit.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 64)
    print("  Feature Degradation Audit")
    print(f"  train <= {TRAIN_END}   lockbox = [{LOCKBOX_START}, {LOCKBOX_END}]")
    print("=" * 64)

    print(f"\nLoading panel: {panel_path}")
    panel = pd.read_pickle(panel_path)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    feature_cols = [c for c in panel.columns if c.startswith("features_")]
    if TARGET_COL not in panel.columns:
        print(f"FATAL: {TARGET_COL} not in panel columns. Available targets: "
              f"{[c for c in panel.columns if 'return' in c.lower() or 'target' in c.lower()]}")
        sys.exit(1)

    panel = panel[feature_cols + [TARGET_COL]].copy()
    gc.collect()

    dates = panel.index.get_level_values(date_level)
    train_df   = panel[dates <= pd.Timestamp(TRAIN_END)]
    lockbox_df = panel[(dates >= pd.Timestamp(LOCKBOX_START)) & (dates <= pd.Timestamp(LOCKBOX_END))]
    del panel
    gc.collect()

    print(f"\n{len(feature_cols)} feature columns found.")
    print(f"Train rows: {len(train_df):,}   Lockbox rows: {len(lockbox_df):,}")
    print("\nComputing per-feature univariate IC (train vs lockbox) ...\n")

    results = []
    for i, feat in enumerate(feature_cols):
        tr  = per_period_ic(train_df, feat, date_level)
        lb  = per_period_ic(lockbox_df, feat, date_level)
        tic = tr["mean_ic"]
        lic = lb["mean_ic"]
        degrade = (tic - lic) if (tic == tic and lic == lic) else float("nan")  # NaN-safe
        sign_flip = (tic == tic and lic == lic and tic * lic < 0 and abs(tic) > 0.01)
        results.append({
            "feature": feat,
            "train_ic": tic,
            "train_n_dates": tr["n_dates"],
            "lockbox_ic": lic,
            "lockbox_n_dates": lb["n_dates"],
            "degrade": degrade,
            "sign_flip": bool(sign_flip),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(feature_cols)} features done ...")

    results_sorted = sorted(
        [r for r in results if r["degrade"] == r["degrade"]],  # drop NaN degrade
        key=lambda r: r["degrade"], reverse=True,
    )

    print("\n" + "=" * 64)
    print("  TOP 20 WORST DEGRADERS (train_ic - lockbox_ic, highest first)")
    print("=" * 64)
    print(f"  {'feature':<40} {'train_ic':>9} {'lockbox_ic':>11} {'degrade':>9} {'flip':>5}")
    for r in results_sorted[:20]:
        print(f"  {r['feature']:<40} {r['train_ic']:>+9.4f} {r['lockbox_ic']:>+11.4f} "
              f"{r['degrade']:>+9.4f} {'YES' if r['sign_flip'] else '':>5}")

    n_flips = sum(1 for r in results if r["sign_flip"])
    print(f"\n  Sign flips (|train_ic|>0.01 and opposite sign in lockbox): {n_flips}/{len(results)}")

    out = {
        "config": {
            "train_end": TRAIN_END,
            "lockbox_start": LOCKBOX_START,
            "lockbox_end": LOCKBOX_END,
            "target_col": TARGET_COL,
            "min_names_per_date": MIN_NAMES_PER_DATE,
        },
        "n_features": len(feature_cols),
        "n_sign_flips": n_flips,
        "results_sorted_by_degrade": results_sorted,
        "all_results": results,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  Results saved -> {out_path}")


if __name__ == "__main__":
    main()
