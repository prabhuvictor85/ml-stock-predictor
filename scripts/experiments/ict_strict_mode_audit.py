#!/usr/bin/env python3
"""
ICT Strict-Mode Audit — legacy vs strict OB/FVG, recomputed from raw price CSVs
==================================================================================
Tests whether the full reference "Strict ICT Mode" (BOS gate + premium/discount
alignment + FVG sweep confirmation — see ict_features.py implementation_mode=
"strict") produces a more stable train-era vs lockbox-era IC than the legacy
(ungated) OB/FVG triggers.

Recomputes OB/FVG features directly from per-ticker price CSVs (not from the
production panel — those columns don't exist there yet), then joins onto the
existing panel's `future_20d_excess_return` target by (ticker, date).

Safe-load discipline: the panel pickle is loaded with pandas ALONE before
anything imports numpy (pd.read_pickle segfaults on this server's numpy 2.4.6
+ Python 3.14 build if numpy/scipy/lightgbm are already imported — see
model_a_zone_core.py for the original diagnosis). Price CSVs are plain text
and safe to read in any import order.

Run on Theralytics:
    cd /root/ml-stock-predictor
    python3 -u scripts/experiments/ict_strict_mode_audit.py
"""
from __future__ import annotations

import gc
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import pandas as pd

PANEL_PATH = "/mnt/data/artefacts/us_lockbox_v2/us_local/checkpoints/panel_targets.pkl"
DATA_DIR   = "/mnt/data/Learning_charts/stock_data/us_stocks"
FENCE      = pd.Timestamp("2023-12-31")
TARGET_COL = "future_20d_excess_return"
OUT_CSV    = "/tmp/ict_strict_vs_legacy_audit.csv"

print("=" * 64)
print("  ICT Strict-Mode Audit")
print("=" * 64)

# ── Step 1: panel target, pandas-only, before any numpy import ─────────────
print(f"\nLoading panel (target only): {PANEL_PATH}")
panel = pd.read_pickle(PANEL_PATH)
date_level = "date" if "date" in panel.index.names else panel.index.names[0]
ticker_level = [n for n in panel.index.names if n != date_level][0]

target = panel[[TARGET_COL]].copy()
tickers = sorted(panel.index.get_level_values(ticker_level).unique())
del panel
gc.collect()
print(f"Target loaded: {target.shape[0]:,} rows, {len(tickers)} tickers")

# ── Step 2: now safe to import numpy-dependent code ─────────────────────────
import numpy as np
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr

engine = ICTFeatureEngine()

FEATURE_COLS = ["ict_bob_active", "ict_sob_active", "ict_bullfvg_active", "ict_bearfvg_active"]

def compute_for_ticker(ticker: str) -> dict | None:
    path = os.path.join(DATA_DIR, f"{ticker}-1d.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    df.columns = [c.strip().lower() for c in df.columns]
    if not {"date", "open", "high", "low", "close"}.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 100:
        return None

    df["atr_14"] = _wilder_atr(df["high"].values, df["low"].values, df["close"].values, period=14)

    out = {"date": df["date"].values}
    for mode, suffix in [("legacy", "legacy"), ("strict", "strict")]:
        try:
            res = engine.compute(df.copy(), implementation_mode=mode)
        except Exception:
            return None
        for col in FEATURE_COLS:
            out[f"{col}_{suffix}"] = res[col].values
    return out


print(f"\nRecomputing legacy + strict OB/FVG for {len(tickers)} tickers...")
frames = []
n_ok, n_skip = 0, 0
for i, t in enumerate(tickers):
    r = compute_for_ticker(t)
    if r is None:
        n_skip += 1
        continue
    f = pd.DataFrame(r)
    f["ticker"] = t
    frames.append(f)
    n_ok += 1
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(tickers)} processed ({n_ok} ok, {n_skip} skipped)")

print(f"Done: {n_ok} tickers computed, {n_skip} skipped (missing/short data)")

print("\nConcatenating and joining onto target...")
combined = pd.concat(frames, ignore_index=True)
del frames
gc.collect()

combined = combined.set_index(["ticker", "date"])
combined.index = combined.index.set_names([ticker_level, date_level])

joined = combined.join(target, how="inner")
del combined, target
gc.collect()
print(f"Joined panel: {joined.shape[0]:,} rows")

dates = joined.index.get_level_values(date_level)
train = joined[dates <= FENCE]
lockbox = joined[dates > FENCE]
del joined
gc.collect()
print(f"Train: {train.shape[0]:,} rows  Lockbox: {lockbox.shape[0]:,} rows")


def feature_ic(df: pd.DataFrame, feats: list[str], target_col: str, date_level: str) -> dict:
    ranked = df.groupby(level=date_level)[feats + [target_col]].rank(pct=True)
    g = ranked.groupby(level=date_level)
    out = {}
    for f in feats:
        mx = g[f].transform("mean")
        my = g[target_col].transform("mean")
        cov = ((ranked[f] - mx) * (ranked[target_col] - my)).groupby(level=date_level).mean()
        vx = ((ranked[f] - mx) ** 2).groupby(level=date_level).mean()
        vy = ((ranked[target_col] - my) ** 2).groupby(level=date_level).mean()
        ic_per_date = cov / (vx * vy) ** 0.5
        ic_per_date = ic_per_date.replace([np.inf, -np.inf], np.nan).dropna()
        out[f] = (float(ic_per_date.mean()), int(len(ic_per_date))) if len(ic_per_date) > 10 else (None, 0)
    return out


all_feats = [f"{c}_{m}" for c in FEATURE_COLS for m in ("legacy", "strict")]
print("\nComputing training-era IC...")
train_ic = feature_ic(train, all_feats, TARGET_COL, date_level)
print("Computing lockbox-era IC...")
lockbox_ic = feature_ic(lockbox, all_feats, TARGET_COL, date_level)

rows = []
for f in all_feats:
    t, tn = train_ic.get(f, (None, 0))
    l, ln = lockbox_ic.get(f, (None, 0))
    if t is None or l is None:
        continue
    rows.append({
        "feature": f, "train_ic": round(t, 4), "lockbox_ic": round(l, 4),
        "flipped": (t > 0) != (l > 0), "n_train": tn, "n_lockbox": ln,
    })

result = pd.DataFrame(rows).sort_values("feature")
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 150)
print("\n" + result.to_string(index=False))

print("\n=== Legacy vs Strict summary ===")
for base in FEATURE_COLS:
    leg = result[result.feature == f"{base}_legacy"]
    strict = result[result.feature == f"{base}_strict"]
    if len(leg) and len(strict):
        print(f"{base}:")
        print(f"  legacy  train={leg.train_ic.values[0]:+.4f}  lockbox={leg.lockbox_ic.values[0]:+.4f}  flipped={bool(leg.flipped.values[0])}")
        print(f"  strict  train={strict.train_ic.values[0]:+.4f}  lockbox={strict.lockbox_ic.values[0]:+.4f}  flipped={bool(strict.flipped.values[0])}")

result.to_csv(OUT_CSV, index=False)
print(f"\nSaved -> {OUT_CSV}")
