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
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
DATA_DIR        = PATHS.stock_data.us          # stock_data/us_stocks — same path run_sp500_local.py reads
CONSTITUENT_CSV = PATHS.stock_lists.us_combined  # constituents_us_combined.csv

BENCHMARK_TICKERS = ["^GSPC", "^NDX"]   # S&P 500 + NASDAQ 100 indices
START_DATE        = "2010-01-01"         # history start
MAX_WORKERS       = 1                    # sequential — avoids rate-limit errors
RATE_LIMIT_SLEEP  = 0.5                  # 500ms between tickers (base delay)
MAX_RETRIES       = 3                    # retries per ticker on rate-limit / transient error
RETRY_BACKOFF     = 2.0                  # multiply sleep by this factor on each retry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download US stock data via yfinance")
    p.add_argument("--refresh_after", type=int, default=0,
                   help="Re-download files older than N days (0 = skip existing files)")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Download only these specific tickers (bypasses constituent list)")
    p.add_argument("--benchmarks_only", action="store_true",
                   help="Only download/refresh benchmark index files")
    p.add_argument("--start", default=START_DATE,
                   help=f"History start date (default: {START_DATE})")
    p.add_argument("--end", default=None,
                   help="History end date e.g. 2023-12-31 (default: today)")
    p.add_argument("--delay", type=float, default=RATE_LIMIT_SLEEP,
                   help=f"Seconds to sleep between ticker downloads (default: {RATE_LIMIT_SLEEP}).")
    p.add_argument("--retries", type=int, default=MAX_RETRIES,
                   help=f"Max retries per ticker on transient failure (default: {MAX_RETRIES}).")
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


def _normalise_us(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns and rename to capitalised OHLCV (Open/High/Low/Close/Volume)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for std in ("open", "high", "low", "close", "volume"):
        match = next((c for c in df.columns if c.lower() == std), None)
        if match:
            rename[match] = std.capitalize() if std != "volume" else "Volume"
    df = df.rename(columns=rename)
    df.index.name = "Date"
    return df


def _yf_download_with_retry(ticker: str, max_retries: int, **kwargs) -> pd.DataFrame:
    """
    Wrapper around yf.download with exponential backoff on transient failures.

    Yahoo Finance rate-limits aggressive downloaders and returns empty DataFrames
    or raises exceptions. We retry up to `max_retries` times with increasing
    sleep intervals (base_delay * 2^attempt) before giving up.
    """
    base_delay = RATE_LIMIT_SLEEP
    for attempt in range(max_retries + 1):
        try:
            df = yf.download(ticker, progress=False, **kwargs)
            if not df.empty:
                return df
            # Empty could be rate-limit or genuinely no data — retry if we have attempts left
            if attempt < max_retries:
                sleep_secs = base_delay * (RETRY_BACKOFF ** attempt)
                time.sleep(sleep_secs)
            else:
                return df   # genuinely empty after all retries
        except Exception as exc:
            if attempt < max_retries:
                sleep_secs = base_delay * (RETRY_BACKOFF ** attempt)
                time.sleep(sleep_secs)
            else:
                raise exc
    return pd.DataFrame()


def download_ticker(ticker: str, data_dir: Path, start: str,
                    refresh_after_days: int,
                    end: str = None,
                    max_retries: int = MAX_RETRIES) -> tuple[str, bool, str]:
    """
    Download daily OHLCV for a single ticker and save as {ticker}-1d.csv.
    Returns (ticker, success, message).

    Incremental mode: if the file already exists (and no forced refresh), only
    bars after the last stored date up to `end` are fetched and appended. This
    makes repeated `--end <date>` calls cheap and date-based (not mtime-based),
    which is required for walk-forward backtests that advance the as_of date.

    Rate-limit resilience: uses _yf_download_with_retry with exponential backoff.
    """
    # yfinance uses - for BRK.B → BRK-B etc. (Wikipedia already normalised above)
    path = data_dir / f"{ticker}-1d.csv"

    try:
        # ── Incremental update if file exists ─────────────────────────────
        if path.exists() and refresh_after_days <= 0:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            if not existing.empty:
                last_date = existing.index.max()
                new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                # Nothing to fetch if already at/past the requested end date
                if end and new_start >= end:
                    return ticker, True, "skipped (up to date)"
                # Skip if the window contains no trading days (e.g. as_of is a weekend)
                if end:
                    bdays = pd.bdate_range(new_start,
                                           pd.Timestamp(end) - pd.Timedelta(days=1))
                    if len(bdays) == 0:
                        return ticker, True, f"skipped (no trading days {new_start} → {end})"
                df_new = _yf_download_with_retry(
                    ticker, max_retries,
                    start=new_start, end=end,
                    auto_adjust=True, multi_level_index=False,
                )
                if df_new.empty:
                    return ticker, True, f"skipped (no new bars after {last_date.date()})"
                df_new = _normalise_us(df_new)
                for req in ("Open", "High", "Low", "Close", "Volume"):
                    if req not in df_new.columns:
                        return ticker, False, f"missing column {req} in new data"
                df_new = df_new[["Open", "High", "Low", "Close", "Volume"]]
                df_new = df_new[df_new["Close"].notna() & (df_new["Close"] > 0)]
                # Merge and deduplicate
                df = pd.concat([existing, df_new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index.name = "Date"
                df.to_csv(path)
                return ticker, True, f"+{len(df_new)} new rows (total {len(df)})"

        # ── Full download (new file or forced refresh) ─────────────────────
        if not file_needs_update(path, refresh_after_days):
            return ticker, True, "skipped (up to date)"

        df = _yf_download_with_retry(
            ticker, max_retries,
            start=start, end=end,
            auto_adjust=True, multi_level_index=False,
        )
        if df.empty:
            return ticker, False, "empty response from yfinance"

        df = _normalise_us(df)

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
                 refresh_after_days: int, end: str = None,
                 delay: float = RATE_LIMIT_SLEEP,
                 max_retries: int = MAX_RETRIES) -> None:
    """Download all tickers sequentially with progress reporting.

    Parameters
    ----------
    delay       : seconds to sleep between tickers (rate-limit guard)
    max_retries : per-ticker retries with exponential backoff on empty/error
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    total     = len(tickers)
    succeeded = 0
    skipped   = 0
    failed: List[str] = []

    print(f"\nDownloading {total} tickers to {data_dir} ...")
    print(f"  delay={delay}s/ticker  retries={max_retries}")
    print(f"  (existing files {'will be refreshed if older than ' + str(refresh_after_days) + ' days' if refresh_after_days > 0 else 'will be skipped'})\n")

    # MAX_WORKERS=1 keeps downloads sequential, which is the safest approach for
    # Yahoo Finance's rate limiter. The delay is applied after every ticker.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_ticker, t, data_dir, start, refresh_after_days, end, max_retries): t
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

            time.sleep(delay)

    print(f"\nDownload complete:")
    print(f"  Succeeded : {succeeded}")
    print(f"  Skipped   : {skipped} (already up to date)")
    print(f"  Failed    : {len(failed)}")
    if failed:
        fail_path = data_dir / "_failed_tickers.txt"
        fail_path.write_text("\n".join(failed))
        print(f"  Failed list saved to: {fail_path}")


def download_benchmarks(data_dir: Path, start: str, refresh_after_days: int,
                        end: str = None, max_retries: int = MAX_RETRIES) -> None:
    """Download ^GSPC and ^NDX benchmark files."""
    print(f"\nDownloading benchmark indices ...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for ticker in BENCHMARK_TICKERS:
        t, ok, msg = download_ticker(ticker, data_dir, start, refresh_after_days, end,
                                     max_retries=max_retries)
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {ticker}: {msg}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  US Market Data Downloader")
    print(f"  Start   : {args.start}")
    print(f"  End     : {args.end or 'today'}")
    print(f"  Output  : {DATA_DIR}")
    print("=" * 60)

    # ── Benchmarks only ────────────────────────────────────────────────────
    if args.benchmarks_only:
        download_benchmarks(DATA_DIR, args.start, max(1, args.refresh_after), args.end)
        return

    # ── Specific tickers ───────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers]
        print(f"\nDownloading {len(tickers)} specified ticker(s)...")
        download_all(tickers, DATA_DIR, args.start, args.refresh_after, args.end,
                     delay=args.delay, max_retries=args.retries)
        download_benchmarks(DATA_DIR, args.start, args.refresh_after, args.end)
        return

    # ── Full run: load constituent list + download ─────────────────────────
    if not CONSTITUENT_CSV.exists():
        print(f"ERROR: Constituent list not found: {CONSTITUENT_CSV}")
        print("  Copy constituents_us_combined.csv to the stock_lists directory.")
        raise SystemExit(1)

    print(f"\n[1/2] Loading constituent list: {CONSTITUENT_CSV.name}")
    universe = pd.read_csv(CONSTITUENT_CSV)
    col = next((c for c in universe.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        raise ValueError(f"No 'Symbol' column in {CONSTITUENT_CSV}")
    tickers = universe[col].dropna().str.strip().tolist()
    print(f"  Loaded {len(tickers)} tickers")

    print(f"\n[2/2] Downloading {len(tickers)} stock files...")
    download_all(tickers, DATA_DIR, args.start, args.refresh_after, args.end,
                 delay=args.delay, max_retries=args.retries)

    print(f"\nDownloading benchmark indices...")
    download_benchmarks(DATA_DIR, args.start, max(1, args.refresh_after), args.end)

    print(f"\n{'='*60}")
    print("  Done! Next step:")
    print(f"  python run_sp500_local.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
