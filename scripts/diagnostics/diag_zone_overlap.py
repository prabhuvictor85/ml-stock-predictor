"""A1/A2 diagnostic: how much do the two zone engines overlap?

Builds features (with the B2/B3/B5 fixes live) on a random sample of local US
tickers and measures correlation between the engines' summary scores:
    ict_bull_htf_score  vs  sdz_htf_score
    ict_bear_htf_score  vs  ssz_htf_score
High correlation (>~0.8) => ICT engine is mostly an echo of ZoneAnalyzer.
Moderate => genuinely complementary; harmonize and keep both.
"""
import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.config import get_config
from pipeline.config.paths import PATHS
from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX

DATA_DIR = PATHS.stock_data.us
N_TICKERS = 80
N_DAYS = 750          # ~3 years
SEED = 42

rng = np.random.default_rng(SEED)
files = sorted(DATA_DIR.glob("*-1d.csv"))
rng.shuffle(files)

frames = []
for f in files:
    if len(frames) >= N_TICKERS:
        break
    tk = f.name.replace("-1d.csv", "")
    try:
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
    except Exception:
        continue
    if len(df) < 400:
        continue
    df = df.tail(N_DAYS)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df["in_universe"] = True
    df["sector"] = "X"  # sector RS irrelevant to zone scores
    df["ticker"] = tk
    frames.append(df)

panel = pd.concat(frames)
panel = panel.set_index("ticker", append=True)
panel.index.names = ["date", "ticker"]
panel = panel.sort_index()
print(f"Panel: {panel.index.get_level_values('ticker').nunique()} tickers, "
      f"{panel.index.get_level_values('date').nunique()} dates, {len(panel):,} rows")

# Benchmark only feeds beta/regime — irrelevant to the two zone scores.
dates = panel.index.get_level_values("date").unique().sort_values()
bm = pd.Series(np.linspace(100, 130, len(dates)), index=dates)

cfg = get_config("sp500")
out = FeatureEngineer(cfg, bm).build(panel)
P = FEATURE_PREFIX

pairs = [
    ("ict_bull_htf_score", "sdz_htf_score"),
    ("ict_bear_htf_score", "ssz_htf_score"),
]
print()
for ict_col, zone_col in pairs:
    a = out[P + ict_col].astype(float)
    b = out[P + zone_col].astype(float)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    pear = a.corr(b)
    spear = a.corr(b, method="spearman")
    # presence overlap: both engines say "something bullish/bearish here"
    a_on, b_on = a > 0, b > 0
    jacc = (a_on & b_on).sum() / max((a_on | b_on).sum(), 1)
    print(f"{ict_col:24} vs {zone_col:16} | pearson={pear:+.3f}  "
          f"spearman={spear:+.3f}  | active: ict={a_on.mean():.1%} "
          f"zone={b_on.mean():.1%}  jaccard-overlap={jacc:.1%}")

# Disagreement snapshot: ICT strongly bullish while zone engine sees nothing
ict_b = out[P + "ict_bull_htf_score"]
sdz   = out[P + "sdz_htf_score"]
contra = ((ict_b > 0.3) & (sdz == 0)).mean()
contra2 = ((sdz > 0.3) & (ict_b == 0)).mean()
print(f"\nICT bullish (>0.3) while zone says nothing: {contra:.1%} of stock-days")
print(f"Zone bullish (>0.3) while ICT says nothing: {contra2:.1%} of stock-days")
