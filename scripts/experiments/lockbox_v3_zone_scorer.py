#!/usr/bin/env python3
"""
Lockbox v3 — zone-only scorer (Option C: single panel load, no per-step FE).

Loads the full pre-built panel once and scores each cadence date's cross-section
using the frozen step-1 ensemble. Skips the per-step FE re-run that makes the
production walk-forward take ~40 hours.

Caveat: zone "active" flags are computed with full history (2010-2026).
A zone that is filled in Mar 2024 will show as active on Feb 2024's row in
this panel (no look-ahead guard). This is acceptable for a first-pass signal
check; the production walk-forward is the ground truth.

Output: scores_detail_momentum_{date}.json per cadence date, in the same
format validate_lockbox.py expects. Run validate_lockbox.py (step 3) unchanged.

Usage (server):
    cd /root/ml-stock-predictor
    export ML_ARTEFACTS_ROOT=/mnt/data/artefacts/us_lockbox_v3
    python3 -u scripts/experiments/lockbox_v3_zone_scorer.py
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np

ROOT         = Path(os.environ.get("ML_ARTEFACTS_ROOT",
                                   "/mnt/data/artefacts/us_lockbox_v3"))
PANEL_PATH   = ROOT / "us_local/checkpoints/panel_targets.pkl"
ART_DIR      = ROOT / "us_local/momentum"
ENSEMBLE_PKL = ART_DIR / "ensemble.pkl"
FEATURES_TXT = ART_DIR / "selected_features.txt"
OUTPUT_DIR   = ROOT / "us_local/output"

LOCKBOX_START = "2024-01-12"
LOCKBOX_END   = "2026-05-04"
CADENCE_DAYS  = 14

ZONE_PREFIXES = ("features_sdz_", "features_ssz_", "features_dz_",
                 "features_sz_",  "features_zone_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Lockbox v3 — zone scorer (single panel load)")
    print(f"  root={ROOT}")
    print("=" * 64)

    print(f"\nLoading panel: {PANEL_PATH}")
    panel = pd.read_pickle(PANEL_PATH)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    print(f"  Panel shape: {panel.shape}  "
          f"dates: {panel.index.get_level_values(date_level).min().date()} → "
          f"{panel.index.get_level_values(date_level).max().date()}")

    print(f"\nLoading ensemble: {ENSEMBLE_PKL}")
    with open(ENSEMBLE_PKL, "rb") as fh:
        ensemble = pickle.load(fh)

    print(f"Loading selected features: {FEATURES_TXT}")
    selected = FEATURES_TXT.read_text().strip().split("\n")
    feat_cols = [f for f in selected if f.startswith(ZONE_PREFIXES)]
    print(f"  Zone features from step-1 selection: {len(feat_cols)}")
    if not feat_cols:
        # Fall back: all zone columns in panel
        feat_cols = [c for c in panel.columns if c.startswith(ZONE_PREFIXES)]
        print(f"  (fallback to all zone cols in panel: {len(feat_cols)})")

    # Identify historical-vol feature for the vol tilt (10% ensemble weight)
    _hv_feat = next(
        (c for c in panel.columns if "hist_vol" in c and "20" in c), None
    )

    all_dates = pd.DatetimeIndex(
        panel.index.get_level_values(date_level).unique()
    ).sort_values()

    lockbox_dates = all_dates[
        (all_dates >= pd.Timestamp(LOCKBOX_START)) &
        (all_dates <= pd.Timestamp(LOCKBOX_END))
    ]
    scoring_dates = lockbox_dates[::CADENCE_DAYS]
    print(f"\nScoring {len(scoring_dates)} cadence dates "
          f"({LOCKBOX_START} → {LOCKBOX_END}, every {CADENCE_DAYS} trading days)\n")

    n_written = 0
    for i, score_dt in enumerate(scoring_dates):
        date_str = score_dt.strftime("%Y-%m-%d")
        out_path = OUTPUT_DIR / f"scores_detail_momentum_{date_str}.json"

        if out_path.exists():
            print(f"  [{i+1:02d}/{len(scoring_dates)}] {date_str}  SKIP (already exists)")
            n_written += 1
            continue

        cross = panel[
            panel.index.get_level_values(date_level) == score_dt
        ].copy()

        if len(cross) < 10:
            print(f"  [{i+1:02d}/{len(scoring_dates)}] {date_str}  SKIP (n={len(cross)})")
            continue

        avail = [f for f in feat_cols if f in cross.columns]
        if not avail:
            print(f"  [{i+1:02d}/{len(scoring_dates)}] {date_str}  SKIP (no zone cols)")
            continue

        X        = cross[avail].fillna(0.0)
        vol_col  = cross[_hv_feat] if _hv_feat and _hv_feat in cross.columns else None
        scores   = ensemble.score(X, vol_col)
        score_s  = pd.Series(scores, index=cross.index)

        # Normalize to [0, 1] within the cross-section
        mn, mx = score_s.min(), score_s.max()
        if mx > mn:
            score_norm = (score_s - mn) / (mx - mn)
        else:
            score_norm = pd.Series(0.5, index=score_s.index)

        # Build scores_detail JSON — minimal fields validate_lockbox.py needs
        tickers = cross.index.get_level_values("ticker") \
            if "ticker" in cross.index.names else cross.index
        detail: dict = {}
        for ticker, raw_sc, norm_sc in zip(
            tickers, score_s.values, score_norm.values
        ):
            detail[str(ticker)] = {
                "bull": {
                    "model_score":      float(norm_sc),
                    "composite_score":  float(norm_sc),
                },
                "bear": {
                    "model_score":      float(1.0 - norm_sc),
                    "composite_score":  float(1.0 - norm_sc),
                },
            }

        with open(out_path, "w") as fh:
            json.dump(detail, fh)

        print(f"  [{i+1:02d}/{len(scoring_dates)}] {date_str}  "
              f"n={len(detail)}  score_range=[{mn:.3f}, {mx:.3f}]  → {out_path.name}")
        n_written += 1

    del panel
    gc.collect()

    print(f"\nDone. {n_written}/{len(scoring_dates)} score files written to {OUTPUT_DIR}")
    print("\nNext: run step 3 (validate_lockbox.py)")
    print(f"  python3 scripts/tools/validate_lockbox.py \\")
    print(f"    --scores_dir {ROOT}/us_local/output \\")
    print(f"    --data_dir /mnt/data/Learning_charts/stock_data/us_stocks \\")
    print(f"    --mode momentum --side bull --score_field model_score \\")
    print(f"    --start 2024-01-01 --end 2026-05-06 \\")
    print(f"    --out {ROOT}/lockbox_verdict.json")


if __name__ == "__main__":
    main()
