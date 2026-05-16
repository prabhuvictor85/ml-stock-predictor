"""
download_us_data.py — Download S&P 500 + NASDAQ 100 historical data via yfinance.

What it does:
  1. Scrapes Wikipedia for current S&P 500 and NASDAQ 100 constituents
  2. Merges and deduplicates into a single universe (~540-560 unique tickers)
  3. Saves constituents_us.csv with Symbol, Name, Sector, Exchange columns
  4. Downloads daily OHLCV for every ticker and saves as {TICKER}-1d.csv
  5. Downloads benchmark files: ^GSPC-1d.csv and ^NDX-1d.csv

Usage:
    python download_us_data.py                        # full download
    python download_us_data.py --refresh_after 7      # re-download files older than 7 days
    python download_us_data.py --tickers AAPL MSFT    # download specific tickers only
    python download_us_data.py --benchmarks_only      # only refresh benchmark files

Output directory: C:/Victor/Learning_charts/us_data/
Constituent list: C:/Victor/Learning_charts/stock_lists/constituents_us.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path(r"C:\Victor\Learning_charts\us_data")
LISTS_DIR      = Path(r"C:\Victor\Learning_charts\stock_lists")
CONSTITUENT_CSV = LISTS_DIR / "constituents_us.csv"

BENCHMARK_TICKERS = ["^GSPC", "^NDX"]   # S&P 500 + NASDAQ 100 indices
START_DATE        = "2010-01-01"         # history start
MAX_WORKERS       = 8                    # parallel download threads
RATE_LIMIT_SLEEP  = 0.25                 # seconds between batches (be nice to yfinance)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download S&P 500 + NASDAQ 100 data")
    p.add_argument("--refresh_after", type=int, default=0,
                   help="Re-download files older than N days (0 = skip existing files)")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Download only these specific tickers (bypasses constituent fetch)")
    p.add_argument("--benchmarks_only", action="store_true",
                   help="Only download/refresh benchmark index files")
    p.add_argument("--start", default=START_DATE,
                   help=f"History start date (default: {START_DATE})")
    p.add_argument("--no_constituents", action="store_true",
                   help="Skip constituent CSV refresh (use existing)")
    return p.parse_args()


# ── Constituent scraping ────────────────────────────────────────────────────────

def fetch_sp500_constituents() -> pd.DataFrame:
    """Scrape S&P 500 constituents from Wikipedia."""
    print("  Fetching S&P 500 constituents from Wikipedia...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        df = df[["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["Symbol", "Name", "Sector"]
        df["Exchange"] = "NYSE/NASDAQ"
        # Clean up symbol (some have dots, Wikipedia uses . not -)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False).str.strip()
        print(f"    Found {len(df)} S&P 500 stocks")
        return df
    except Exception as e:
        print(f"  WARNING: Could not fetch S&P 500 from Wikipedia: {e}")
        return pd.DataFrame(columns=["Symbol", "Name", "Sector", "Exchange"])


def fetch_nasdaq100_constituents() -> pd.DataFrame:
    """Scrape NASDAQ 100 constituents from Wikipedia."""
    print("  Fetching NASDAQ 100 constituents from Wikipedia...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # Find the table with ticker symbols
        df = None
        for t in tables:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                sym_col = "Ticker" if "Ticker" in t.columns else "Symbol"
                if len(t) > 50:
                    df = t
                    break
        if df is None:
            # Try a broader search
            for t in tables:
                cols_lower = [c.lower() for c in t.columns]
                if any("tick" in c or "symbol" in c for c in cols_lower) and len(t) > 50:
                    df = t
                    break
        if df is None:
            raise ValueError("Could not find NASDAQ 100 table on Wikipedia page")

        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        sym_col = next(c for c in df.columns if c.lower() in ("ticker", "symbol"))
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)
        sector_col = next((c for c in df.columns
                           if "sector" in c.lower() or "industry" in c.lower()), None)

        result = pd.DataFrame()
        result["Symbol"] = df[sym_col].str.strip()
        result["Name"]   = df[name_col].str.strip() if name_col else ""
        result["Sector"] = df[sector_col].str.strip() if sector_col else "Technology"
        result["Exchange"] = "NASDAQ"
        result = result[result["Symbol"].str.len() > 0]
        print(f"    Found {len(result)} NASDAQ 100 stocks")
        return result
    except Exception as e:
        print(f"  WARNING: Could not fetch NASDAQ 100 from Wikipedia: {e}")
        return pd.DataFrame(columns=["Symbol", "Name", "Sector", "Exchange"])


def build_universe() -> pd.DataFrame:
    """Merge S&P 500 + NASDAQ 100, deduplicate, return combined universe."""
    sp500  = fetch_sp500_constituents()
    ndx100 = fetch_nasdaq100_constituents()

    if sp500.empty and ndx100.empty:
        raise RuntimeError("Could not fetch any constituent data — check internet connection")

    # Mark NASDAQ 100 exchange for stocks already in S&P 500
    # (NASDAQ 100 stocks that also appear in S&P 500 get Exchange = "NASDAQ")
    combined = pd.concat([sp500, ndx100], ignore_index=True)
    # Deduplicate: keep NASDAQ entry if a ticker appears in both
    # (preserves the more specific exchange label)
    combined = combined.sort_values("Exchange", ascending=False)   # NASDAQ sorts before NYSE/NASDAQ
    combined = combined.drop_duplicates(subset="Symbol", keep="first").reset_index(drop=True)
    combined = combined.sort_values("Symbol").reset_index(drop=True)

    print(f"\n  Combined universe: {len(combined)} unique tickers "
          f"({len(sp500)} S&P500 + {len(ndx100)} NASDAQ100, deduplicated)")
    return combined


# ── Download logic ───────────────────────────────────────────────────────────────

def file_needs_update(path: Path, refresh_after_days: int) -> bool:
    """Return True if file doesn't exist or is older than refresh_after_days."""
    if not path.exists():
        return True
    if refresh_after_days <= 0:
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=refresh_after_days)


def download_ticker(ticker: str, data_dir: Path, start: str,
                    refresh_after_days: int) -> tuple[str, bool, str]:
    """
    Download daily OHLCV for a single ticker and save as {ticker}-1d.csv.
    Returns (ticker, success, message).
    """
    # yfinance uses - for BRK.B → BRK-B etc. (Wikipedia already normalised above)
    path = data_dir / f"{ticker}-1d.csv"

    if not file_needs_update(path, refresh_after_days):
        return ticker, True, "skipped (up to date)"

    try:
        df = yf.download(ticker, start=start, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            return ticker, False, "empty response from yfinance"

        # Normalise columns
        df.columns = [c.strip() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure standard columns exist
        col_map = {c.lower(): c for c in df.columns}
        rename = {}
        for std in ("open", "high", "low", "close", "volume"):
            match = next((c for c in df.columns if c.lower() == std), None)
            if match:
                rename[match] = std.capitalize() if std != "volume" else "Volume"
        df = df.rename(columns=rename)

        for req in ("Open", "High", "Low", "Close", "Volume"):
            if req not in df.columns:
                return ticker, False, f"missing column {req}"

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df = df[df["Close"].notna() & (df["Close"] > 0)]

        if len(df) < 50:
            return ticker, False, f"only {len(df)} rows — likely delisted or new"

        df.to_csv(path)
        return ticker, True, f"{len(df)} rows saved"

    except Exception as e:
        return ticker, False, str(e)[:80]


def download_all(tickers: List[str], data_dir: Path, start: str,
                 refresh_after_days: int) -> None:
    """Download all tickers in parallel with progress reporting."""
    data_dir.mkdir(parents=True, exist_ok=True)

    total     = len(tickers)
    succeeded = 0
    skipped   = 0
    failed: List[str] = []

    print(f"\nDownloading {total} tickers to {data_dir} ...")
    print(f"  (existing files {'will be refreshed if older than ' + str(refresh_after_days) + ' days' if refresh_after_days > 0 else 'will be skipped'})\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_ticker, t, data_dir, start, refresh_after_days): t
            for t in tickers
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            ticker, ok, msg = fut.result()
            if msg.startswith("skipped"):
                skipped += 1
            elif ok:
                succeeded += 1
            else:
                failed.append(ticker)
                if len(failed) <= 10:   # only print first 10 failures inline
                    print(f"  FAIL [{ticker}]: {msg}")

            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total}  "
                      f"(ok={succeeded}, skipped={skipped}, failed={len(failed)})")

            # Light rate-limiting — avoid hammering yfinance
            if done % MAX_WORKERS == 0:
                time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nDownload complete:")
    print(f"  Succeeded : {succeeded}")
    print(f"  Skipped   : {skipped} (already up to date)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        fail_path = data_dir / "_failed_tickers.txt"
        fail_path.write_text("\n".join(failed))
        print(f"  Failed list saved to: {fail_path}")


def download_benchmarks(data_dir: Path, start: str, refresh_after_days: int) -> None:
    """Download ^GSPC and ^NDX benchmark files."""
    print(f"\nDownloading benchmark indices ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for ticker in BENCHMARK_TICKERS:
        t, ok, msg = download_ticker(ticker, data_dir, start, refresh_after_days)
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {ticker}: {msg}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    LISTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  US Market Data Downloader")
    print("  Universe: S&P 500 + NASDAQ 100")
    print(f"  Start   : {args.start}")
    print(f"  Output  : {DATA_DIR}")
    print("=" * 60)

    # ── Benchmarks only ────────────────────────────────────────────────────
    if args.benchmarks_only:
        download_benchmarks(DATA_DIR, args.start, max(1, args.refresh_after))
        return

    # ── Specific tickers ───────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers]
        print(f"\nDownloading {len(tickers)} specified ticker(s)...")
        download_all(tickers, DATA_DIR, args.start, args.refresh_after)
        download_benchmarks(DATA_DIR, args.start, args.refresh_after)
        return

    # ── Full run: fetch constituents + download ────────────────────────────
    if not args.no_constituents or not CONSTITUENT_CSV.exists():
        print("\n[1/3] Fetching constituent lists...")
        universe = build_universe()
        universe.to_csv(CONSTITUENT_CSV, index=False)
        print(f"  Saved: {CONSTITUENT_CSV}")
    else:
        print(f"\n[1/3] Using existing constituent list: {CONSTITUENT_CSV}")
        universe = pd.read_csv(CONSTITUENT_CSV)
        print(f"  Loaded {len(universe)} tickers")

    tickers = universe["Symbol"].dropna().str.strip().tolist()

    print(f"\n[2/3] Downloading {len(tickers)} stock files...")
    download_all(tickers, DATA_DIR, args.start, args.refresh_after)

    print(f"\n[3/3] Downloading benchmark indices...")
    download_benchmarks(DATA_DIR, args.start, max(1, args.refresh_after))

    print(f"\n{'='*60}")
    print("  Done! Next step:")
    print(f"  python run_sp500_local.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
