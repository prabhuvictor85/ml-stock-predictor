"""
Download delisted + missing stock price data from Norgate Data.

Targets:
  1. Tickers in membership_sp1500_norgate.csv with no local CSV
  2. Skips symbol-reuse tickers (different company reused the ticker)

Requirements:
  - Norgate Data Updater (NDU) running
  - pip install norgatedata
  - Run extract_norgate_constituents.py first

Output:
  Price CSVs → C:/Victor/Learning_charts/stock_data/us_stocks/{TICKER}-1d.csv
  Format: Date, Open, High, Low, Close, Volume  (same as yfinance CSVs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from datetime import date

try:
    import norgatedata as ng
except ImportError:
    print("ERROR: Run: pip install norgatedata")
    sys.exit(1)

MEMBERSHIP_CSV = Path(r"C:/Victor/Learning_charts/stock_lists/membership_sp1500_norgate.csv")
DATA_DIR       = Path(r"C:/Victor/Learning_charts/stock_data/us_stocks")
START_DATE     = "2005-01-01"   # trial = last 2 years only; paid = full history
TODAY          = date.today().isoformat()

# Database name confirmed via ng.databases() probe
DELISTED_DB    = "US Equities Delisted"
LIVE_DB        = "US Equities"

# Tickers where the symbol was recycled to a DIFFERENT company — never download
SYMBOL_REUSE = {
    'AIV','APC','BBBY','BEAM','CAM','CPWR','EMC','EP','FB','FOSL',
    'GENZ','INFO','JAVA','KG','LB','LIFE','MHS','MI','NBR','NKTR',
    'PCL','POM','RIG','S','SBNY','SE','SHLD','SII','SOLS','SPLS',
    'STI','SUN','TE','XRX',
}


def download_ticker(ticker: str) -> bool:
    """Download price history for one ticker. Returns True on success."""
    out_file = DATA_DIR / f"{ticker}-1d.csv"

    try:
        priceadjust = ng.StockPriceAdjustmentType.TOTALRETURN
        padding     = ng.PaddingType.NONE

        df = ng.price_timeseries(
            ticker,
            stock_price_adjustment_setting=priceadjust,
            padding_setting=padding,
            start_date=START_DATE,
            end_date=TODAY,
            timeseriesformat='pandas-dataframe',
        )

        if df is None or len(df) == 0:
            return False

        # Normalise columns to match existing yfinance CSV format
        df.index.name = "Date"
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Volume': 'Volume',
            'Unadjusted Close': 'Unadj_Close',
        })
        keep = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        df = df[keep]
        df = df[~df.index.duplicated(keep='last')].sort_index()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file)
        return True

    except Exception as e:
        print(f"  ERROR {ticker}: {e}")
        return False


def main():
    if not MEMBERSHIP_CSV.exists():
        print(f"ERROR: Run extract_norgate_constituents.py first")
        sys.exit(1)

    iv = pd.read_csv(MEMBERSHIP_CSV)
    all_tickers = sorted(iv['ticker'].unique())
    print(f"Total tickers in membership file: {len(all_tickers)}")

    # Which ones are missing locally
    missing = [t for t in all_tickers
               if not (DATA_DIR / f"{t}-1d.csv").exists()
               and t not in SYMBOL_REUSE]

    skipped_reuse = [t for t in all_tickers if t in SYMBOL_REUSE]

    print(f"Already have local CSV:  {len(all_tickers) - len(missing) - len(skipped_reuse)}")
    print(f"Need to download:        {len(missing)}")
    print(f"Skipped (symbol reuse):  {len(skipped_reuse)}")
    print()

    ok = 0
    fail = 0
    fail_list = []

    for i, ticker in enumerate(missing):
        if i % 25 == 0:
            print(f"Progress: {i}/{len(missing)}  ok={ok}  fail={fail}")
        success = download_ticker(ticker)
        if success:
            ok += 1
        else:
            fail += 1
            fail_list.append(ticker)

    print(f"\nDone: {ok} downloaded, {fail} failed")
    if fail_list:
        print(f"Failed tickers ({len(fail_list)}):")
        print("  " + ", ".join(fail_list))
        # Save fail list
        fail_path = Path(r"C:/Victor/Learning_charts/stock_lists/norgate_download_failures.txt")
        fail_path.write_text("\n".join(fail_list))
        print(f"  Saved to: {fail_path}")


if __name__ == "__main__":
    main()
