"""
Generate constituents_nse_tradingv.csv from TV_data_status.xlsx.
Filters: Exch==NSE, Status==ok, Adj rows >= 252 (1 year minimum).
Also resolves the actual filename ticker (TV Symbol if present, else Symbol).
"""
import pandas as pd
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS

SRC    = PATHS.stock_lists.lists_dir / "TV_data_status.xlsx"
OUT    = PATHS.stock_lists.nse_tv
TV_DIR = PATHS.stock_data.nse_tv

# Index/benchmark tickers to exclude
EXCLUDE = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50",
           "NIFTY_MID_SELECT", "NIFTYJR", "CNXFINANCE"}

df = pd.read_excel(SRC)
print(f"Loaded {len(df)} rows from {SRC.name}")
print(f"Columns: {list(df.columns)}")
print(f"Exchanges: {df['Exch'].value_counts().to_dict()}")
print(f"Statuses:  {df['Status'].value_counts().to_dict()}")

# Filter NSE only, status ok
nse = df[(df["Exch"] == "NSE") & (df["Status"] == "ok")].copy()
print(f"\nAfter NSE+ok filter: {len(nse)} tickers")

# Resolve the actual filename ticker
# TV Symbol column: '—' means use Symbol, else use TV Symbol
DASH_VALUES = {"—", "-", "", "nan"}
nse["TV_Symbol"] = nse["TV Symbol"].apply(
    lambda v: None if pd.isna(v) or str(v).strip() in DASH_VALUES else str(v).strip()
)
# pandas stores None as NaN in mixed columns — must use pd.isna()
nse["file_ticker"] = nse.apply(
    lambda r: str(r["Symbol"]).strip() if pd.isna(r["TV_Symbol"]) else str(r["TV_Symbol"]).strip(), axis=1
)

# Check which files actually exist on disk
def file_exists(file_ticker):
    return (TV_DIR / f"NSE_{file_ticker}_1D_TV_div_adj.csv").exists()

nse["file_exists"] = nse["file_ticker"].apply(file_exists)
missing = nse[~nse["file_exists"]]
if len(missing) > 0:
    print(f"\nWARNING: {len(missing)} tickers have no file on disk:")
    print(missing[["Symbol", "file_ticker"]].head(10).to_string())

nse = nse[nse["file_exists"]].copy()
print(f"After file-exists filter: {len(nse)} tickers")

# Filter minimum history (252 trading days = ~1 year)
nse["Adj rows"] = pd.to_numeric(nse["Adj rows"], errors="coerce").fillna(0)
nse_full = nse[nse["Adj rows"] >= 252].copy()
print(f"After min 252 rows filter: {len(nse_full)} tickers")

# Exclude index tickers
nse_full = nse_full[~nse_full["file_ticker"].isin(EXCLUDE)]
print(f"After index exclusion: {len(nse_full)} tickers")

# Build output CSV
out_df = nse_full[["Symbol", "file_ticker", "Adj rows"]].copy()
out_df.columns = ["Symbol", "TV_ticker", "rows"]
out_df = out_df.sort_values("Symbol").reset_index(drop=True)

out_df.to_csv(OUT, index=False)
print(f"\nSaved {len(out_df)} tickers -> {OUT}")
print(out_df.head(10).to_string())
