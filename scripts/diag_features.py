"""
Short local feature-recompute diagnostic.

Loads a handful of local NSE CSVs and runs the REAL feature functions
(compute_zone_features + ICTFeatureEngine.compute) to inspect:
  - zone_dist_atr_1d explosion (divide-by-near-zero ATR)
  - ict zone-priority columns: constant (dead) vs live
  - effect of the cutoff guard on HTF zone features

Run:
  ./.venv/Scripts/python.exe scripts/diag_features.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import pandas as pd

from pipeline.features.zone_features import compute_zone_features
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr

DATA_DIR = "C:/Victor/Learning_charts/stock_data/nse_local"
TICKERS = ["CCCL", "ADANIENT", "ABB", "3MINDIA", "ABBOTINDIA"]


def load(tkr: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, f"{tkr}-1d.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def main():
    eng = ICTFeatureEngine()
    rows = []
    for tkr in TICKERS:
        df = load(tkr)
        if df is None or len(df) < 60:
            print(f"skip {tkr} (missing/short)")
            continue

        # ── Zone features (per-fold causal path: cutoff = 30 bars before end) ──
        cutoff = df.index[-30]
        z = compute_zone_features(df, cutoff_date=cutoff)
        zd = z["zone_dist_atr_1d"]

        # ── ICT features ──
        g = df.copy()
        g["atr_14"] = _wilder_atr(g["high"].values, g["low"].values, g["close"].values, 14)
        ict = eng.compute(g, disp_mult=3.0)

        rows.append({
            "ticker": tkr,
            "n": len(df),
            "atr_min": float(np.nanmin(g["atr_14"].values[14:])),
            "zone_dist_max_abs": float(zd.abs().max()),
            "zone_dist_p99": float(zd.abs().quantile(0.99)),
            "bull_prio_nuniq": int(ict["ict_bull_zone_priority"].nunique()),
            "bull_prio_max": float(ict["ict_bull_zone_priority"].max()),
            "bullbb_active_sum": float(ict["ict_bullbb_active"].sum()),
            "bob_active_sum": float(ict["ict_bob_active"].sum()),
        })

    out = pd.DataFrame(rows).set_index("ticker")
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n=== Feature health (local recompute, post-fix code) ===")
    print(out.to_string())
    print("\nINTERPRET:")
    print(" - zone_dist_max_abs >> p99  -> divide-by-tiny-ATR blow-up still present")
    print(" - bull_prio_nuniq == 1       -> priority column DEAD (constant)")
    print(" - bull_prio_nuniq  > 1       -> persistent-priority fix is producing live signal")


if __name__ == "__main__":
    main()
