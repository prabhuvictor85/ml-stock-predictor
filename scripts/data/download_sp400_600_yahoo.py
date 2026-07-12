"""
Download Yahoo price history for all SP400/SP600 ever-members.

Source ticker list: sp400_600_ever_tickers.csv (from Norgate trial)
Output: C:/Victor/Learning_charts/stock_data/us_stocks/{TICKER}-1d.csv

Skips:
  - Tickers already present locally
  - Known symbol-reuse tickers (different company reused the ticker)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import yfinance as yf
import time

TICKERS_CSV = Path(r"C:/Victor/Learning_charts/stock_lists/sp400_600_ever_tickers.csv")
DATA_DIR    = Path(r"C:/Victor/Learning_charts/stock_data/us_stocks")
START_DATE  = "2005-01-01"

import re

def base_ticker(t: str) -> str:
    """Strip Norgate's delisting date suffix: 'CVGW-202605' -> 'CVGW'.
    Norgate appends -YYYYMM to disambiguate delisted tickers.
    The CSV filename always uses the base ticker.
    """
    return re.sub(r'-\d{6}$', '', t)

# Tickers where symbol was recycled to a DIFFERENT company — skip entirely
SYMBOL_REUSE = {
    'AIV','APC','BBBY','BEAM','CAM','CPWR','EMC','EP','FB','FOSL',
    'GENZ','INFO','JAVA','KG','LB','LIFE','MHS','MI','NBR','NKTR',
    'PCL','POM','RIG','S','SBNY','SE','SHLD','SII','SOLS','SPLS',
    'STI','SUN','TE','XRX',
}

def save_ticker(ticker: str, df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    out = DATA_DIR / f"{ticker}-1d.csv"
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(out)
    return True


def main():
    tickers = pd.read_csv(TICKERS_CSV)['ticker'].tolist()
    print(f"Total SP400/600 tickers: {len(tickers)}")

    # Build mapping: base_ticker -> norgate_ticker (for filename saving)
    # Multiple norgate tickers can share the same base (e.g., X-202506 and X are same base)
    # In that case we only need one download
    base_map = {}  # base_ticker -> norgate_ticker (keep first seen)
    for t in tickers:
        b = base_ticker(t)
        if b not in base_map:
            base_map[b] = t

    already  = [b for b in base_map if (DATA_DIR / f"{b}-1d.csv").exists()]
    skipped  = [b for b in base_map if b in SYMBOL_REUSE]
    to_fetch = [b for b in base_map
                if b not in SYMBOL_REUSE
                and not (DATA_DIR / f"{b}-1d.csv").exists()]

    print(f"Total unique base tickers: {len(base_map)}")
    print(f"Already local:    {len(already)}")
    print(f"Symbol reuse skip:{len(skipped)}")
    print(f"To download:      {len(to_fetch)}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0; fail = 0; fail_list = []
    batch_size = 50

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_fetch) - 1) // batch_size + 1
        print(f"Batch {batch_num}/{total_batches}: {batch[0]} .. {batch[-1]}", flush=True)

        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                fail += len(batch)
                fail_list.extend(batch)
                continue

            # MultiIndex columns: (field, ticker)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw['Close']
            else:
                close = raw[['Close']]
                close.columns = [batch[0]]

            for ticker in batch:
                if ticker not in close.columns:
                    fail += 1; fail_list.append(ticker)
                    continue
                s = close[ticker].dropna()
                if len(s) < 50:
                    fail += 1; fail_list.append(ticker)
                    continue
                # Rebuild OHLCV per ticker
                if isinstance(raw.columns, pd.MultiIndex):
                    cols = {}
                    for field in ['Open','High','Low','Close','Volume']:
                        if field in raw and ticker in raw[field].columns:
                            cols[field] = raw[field][ticker]
                    tdf = pd.DataFrame(cols).dropna(how='all')
                else:
                    tdf = raw[['Open','High','Low','Close','Volume']].dropna(how='all')

                if save_ticker(ticker, tdf):
                    ok += 1
                else:
                    fail += 1; fail_list.append(ticker)

        except Exception as e:
            print(f"  Batch error: {e}")
            fail += len(batch)
            fail_list.extend(batch)

        # Brief pause to be polite to Yahoo
        time.sleep(0.5)

    print()
    print(f"=== DONE ===")
    print(f"Downloaded:  {ok}")
    print(f"Failed:      {fail}")
    print(f"Already had: {len(already)}")
    print(f"Skipped:     {len(skipped)}")
    print(f"Total local: {len(already) + ok}")

    if fail_list:
        out = Path(r"C:/Victor/Learning_charts/stock_lists/sp400_600_download_failures.txt")
        out.write_text("\n".join(fail_list))
        print(f"Failures saved: {out}")


if __name__ == "__main__":
    main()
