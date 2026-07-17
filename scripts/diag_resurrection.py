"""
Does the persistent-priority fix actually resurrect the ICT zone-priority
feature on REAL data?

The server panel's last-date cross-section had:
    features_ict_bull_zone_priority : nunique=1, %zero=100, std=0  (DEAD)

That is a vector of ONE value per ticker (the last bar). We reconstruct the
SAME vector locally across every available ticker and check if it now varies.
"""
import glob, os
import numpy as np, pandas as pd
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr

DATA_DIR = "C:/Victor/Learning_charts/stock_data/nse_local"
eng = ICTFeatureEngine()

last_bull, last_bear = [], []
per_ticker_nonzero = []
n = 0
for path in glob.glob(os.path.join(DATA_DIR, "*-1d.csv")):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if len(df) < 60:
        continue
    g = df[["open","high","low","close"]].astype(float).copy()
    g["atr_14"] = _wilder_atr(g["high"].values, g["low"].values, g["close"].values, 14)
    out = eng.compute(g, disp_mult=3.0)
    bp = out["ict_bull_zone_priority"]
    last_bull.append(float(bp.iloc[-1]))            # value on the LAST bar (= panel cross-section)
    last_bear.append(float(out["ict_bear_zone_priority"].iloc[-1]))
    per_ticker_nonzero.append(float((bp > 0).mean()))  # fraction of bars where zone is live
    n += 1

lb = np.array(last_bull)
print(f"tickers scanned: {n}")
print("\n── LAST-BAR cross-section of ict_bull_zone_priority (mirrors the panel) ──")
print(f"  nunique : {len(np.unique(lb))}   (server pre-fix panel = 1, DEAD)")
print(f"  %zero   : {(lb==0).mean()*100:.1f}%   (server pre-fix panel = 100%)")
print(f"  std     : {lb.std():.3f}   (server pre-fix panel = 0.000)")
print(f"  value counts: {dict(zip(*np.unique(lb, return_counts=True)))}")
print("\n── Per-ticker: fraction of bars where a bull zone is LIVE ──")
pn = np.array(per_ticker_nonzero)
print(f"  mean live-fraction across tickers: {pn.mean()*100:.1f}%")
print(f"  tickers with ANY live zone history: {(pn>0).sum()}/{n}")
