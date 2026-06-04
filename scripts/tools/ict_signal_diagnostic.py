#!/usr/bin/env python
"""
ict_signal_diagnostic.py — figure out WHY ICT features barely move the model.

"ICT not driving SHAP" has two very different causes:
  (A) GENUINE-BUT-WEAK : the signal exists but is low-variance (sparse binary
      flags that rarely fire / narrow normalized band) so trees extract little
      gain. Expected, not a bug.
  (B) DEAD/BROKEN       : the column is near-constant at inference time
      (all-zero, all-NaN->0, collapsed by normalization, or simply not computed
      on the inference snapshot). Then it is invisible to SHAP regardless of
      its true predictive value. That IS a bug.

This script profiles the ACTUAL feature VALUES (not the attributions) to tell
them apart, plus checks collinearity with the zone track (which can starve ICT
of splits even when ICT carries signal).

Usage
-----
    python scripts/tools/ict_signal_diagnostic.py \
        --features data/panel/panel_features.parquet \
        --selected artefacts/us_local/selected_features.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def is_ict(c: str) -> bool:
    return c.replace("features_", "").startswith("ict_")


def is_zone(c: str) -> bool:
    return bool(re.match(r"features_(sdz|ssz|dz|sz|zone)", c))


def load_features(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".pkl", ".pickle"):
        return pd.read_pickle(p)
    return pd.read_csv(p)


def profile(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in cols:
        s = df[c]
        vals = pd.to_numeric(s, errors="coerce")
        nan = float(s.isna().mean())
        zero = float((vals.fillna(0) == 0).mean())
        nun = int(s.nunique(dropna=True))
        var = float(np.nanvar(vals.values)) if nun > 1 else 0.0
        nzmean = float(vals[vals != 0].mean()) if (vals != 0).any() else 0.0
        rows.append({
            "feature": c.replace("features_", ""),
            "pct_nan": round(100 * nan, 1),
            "pct_zero": round(100 * zero, 1),
            "nuniq": nun,
            "variance": var,
            "nonzero_mean": round(nzmean, 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                    help="panel feature matrix (parquet/pkl/csv) — same one used for inference")
    ap.add_argument("--selected", default=None,
                    help="optional selected_features.txt — restrict to model's input columns")
    ap.add_argument("--dead_zero", type=float, default=99.0,
                    help="flag DEAD if pct_zero >= this (default 99.0)")
    ap.add_argument("--corr_top", type=int, default=8,
                    help="show top-N zone correlations per composite ICT score (default 8)")
    args = ap.parse_args()

    df = load_features(args.features)
    print(f"Loaded {args.features}: {df.shape[0]} rows x {df.shape[1]} cols")

    feat_cols = [c for c in df.columns if c.startswith("features_")]
    if args.selected and Path(args.selected).exists():
        sel = set(Path(args.selected).read_text().splitlines())
        feat_cols = [c for c in feat_cols if c in sel]
        print(f"Restricted to {len(feat_cols)} model-selected feature columns")

    ict_cols = [c for c in feat_cols if is_ict(c)]
    zone_cols = [c for c in feat_cols if is_zone(c)]
    print(f"ICT cols: {len(ict_cols)}   Zone cols: {len(zone_cols)}")
    if not ict_cols:
        print("No ICT columns present — they were dropped before/at selection. "
              "That alone explains zero SHAP.")
        return

    # ── 1. value profile ──────────────────────────────────────────────────
    prof = profile(df, ict_cols).sort_values("variance")
    print("\n=== ICT feature value profile (sorted by variance, low->high) ===")
    print(prof.to_string(index=False))

    dead = prof[(prof.pct_zero >= args.dead_zero) | (prof.nuniq <= 1)]
    print(f"\nDEAD/near-constant ICT features (pct_zero>={args.dead_zero} or nuniq<=1): "
          f"{len(dead)}/{len(prof)}")
    if len(dead):
        for f in dead.feature:
            print(f"   - {f}")

    # ── 2. zone profile for scale comparison ─────────────────────────────
    if zone_cols:
        zp = profile(df, zone_cols)
        print("\n=== Scale comparison: median variance ICT vs ZONE ===")
        print(f"   ICT  median variance: {prof.variance.median():.3e}   "
              f"median pct_zero: {prof.pct_zero.median():.1f}%")
        print(f"   ZONE median variance: {zp.variance.median():.3e}   "
              f"median pct_zero: {zp.pct_zero.median():.1f}%")
        print("   (If ICT variance << ZONE variance, trees prefer ZONE splits -> "
              "low ICT gain even when ICT carries signal.)")

    # ── 3. collinearity: composite ICT score vs zone scores ──────────────
    comp_ict = [c for c in ict_cols if "htf" in c]
    if comp_ict and zone_cols:
        num = df[comp_ict + zone_cols].apply(pd.to_numeric, errors="coerce")
        corr = num.corr()
        print("\n=== Collinearity: composite ICT score vs zone features ===")
        for ci in comp_ict:
            top = (corr.loc[ci, zone_cols].abs().sort_values(ascending=False)
                   .head(args.corr_top))
            print(f"\n  {ci.replace('features_','')}:")
            for zc, r in top.items():
                signed = corr.loc[ci, zc]
                print(f"     {zc.replace('features_',''):28s} r={signed:+.3f}")
        print("\n  (|r|>0.7 with a higher-variance zone feature means LightGBM can "
              "route the same signal through the zone column and starve ICT of gain.)")

    # ── 4. verdict ───────────────────────────────────────────────────────
    print("\n=== VERDICT ===")
    frac_dead = len(dead) / len(prof)
    if frac_dead >= 0.5:
        print(f"  LIKELY (B) DEAD/BROKEN: {100*frac_dead:.0f}% of ICT features are "
              "near-constant at inference. Check ICT computation on the inference "
              "snapshot (HTF resample gaps, NaN->0 fill, normalization collapse).")
    elif zone_cols and prof.variance.median() < 0.1 * profile(df, zone_cols).variance.median():
        print("  LIKELY (A) GENUINE-BUT-WEAK + collinearity: ICT carries real but "
              "low-variance signal, much of it duplicated by higher-variance zone "
              "features. Fix is to amplify ICT (confluence diff + trend mult), not a bug.")
    else:
        print("  ICT has comparable variance to zones yet low SHAP -> investigate "
              "monotone/selection or genuine non-predictiveness. Inspect per-TF rows above.")


if __name__ == "__main__":
    main()
