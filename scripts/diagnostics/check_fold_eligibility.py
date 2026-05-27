"""Quick script to show which tickers are excluded from early folds due to insufficient history."""
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
DATA_DIR   = PATHS.stock_data.nse_local
STOCK_LIST = PATHS.stock_lists.nse_local

tickers_df = pd.read_csv(STOCK_LIST)
ticker_col = "Symbol"  # constituentsi.csv has Symbol (with .NS suffix) and Symbol1
tickers = tickers_df[ticker_col].dropna().str.strip().tolist()

def get_first_date(ticker):
    path = DATA_DIR / f"{ticker}-1d.csv"
    if not path.exists():
        return ticker, None
    try:
        df = pd.read_csv(path, nrows=2, header=0)
        date_col = next(c for c in df.columns if "date" in c.lower() or "time" in c.lower())
        return ticker, pd.to_datetime(df[date_col].iloc[0])
    except Exception:
        return ticker, None

with ThreadPoolExecutor(max_workers=20) as ex:
    results = list(ex.map(get_first_date, tickers))

first_dates = pd.Series({t: d for t, d in results if d is not None}).sort_values()

# Fold 3 train_end = 2014-12-24, need 504 trading days before that.
# 504 tdays ~ 2 calendar years back => approx sufficient_start = 2012-12-07.
# Stocks first listed AFTER that date are excluded from fold 3.
FOLD3_CUTOFF = pd.Timestamp("2013-01-01")   # conservative approximate boundary

excluded = first_dates[first_dates > FOLD3_CUTOFF]
eligible  = first_dates[first_dates <= FOLD3_CUTOFF]

print(f"\n{'='*60}")
print(f"Fold 3 eligibility (train_end=2014-12-24, need 504 tdays history)")
print(f"  Eligible : {len(eligible)} stocks (listed before ~{FOLD3_CUTOFF.date()})")
print(f"  Excluded : {len(excluded)} stocks (listed after ~{FOLD3_CUTOFF.date()})")
print(f"{'='*60}")

# Group excluded by listing year
print("\nExcluded tickers by listing year:")
for yr, grp in excluded.groupby(pd.DatetimeIndex(excluded).year):
    names = grp.index.tolist()
    print(f"\n  {yr} ({len(names)} stocks):")
    for i in range(0, len(names), 5):
        print("    " + "  ".join(f"{n:<22}" for n in names[i:i+5]))

print(f"\nAll {len(first_dates)} tickers with data range:")
print(f"  Earliest listing: {first_dates.min().date()}  ({first_dates.idxmin()})")
print(f"  Latest listing  : {first_dates.max().date()}  ({first_dates.idxmax()})")
