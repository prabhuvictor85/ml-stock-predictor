"""
download_nse_data.py — Download NSE stock data via yfinance.

Reads tickers from constituentsi.csv (Symbol column, e.g. RELIANCE.NS),
downloads daily OHLCV for each, saves as {TICKER}-1d.csv in stock_data/.
Also refreshes the Nifty 50 benchmark (^NSEI-1d.csv).

Usage:
    python download_nse_data.py                    # skip files that already exist
    python download_nse_data.py --refresh_after 7  # re-download if older than 7 days
    python download_nse_data.py --tickers RELIANCE.NS TCS.NS   # specific tickers only
    python download_nse_data.py --benchmark_only   # only refresh ^NSEI

Output : C:/Victor/Learning_charts/stock_data/
Tickers: C:/Victor/Learning_charts/stock_lists/constituentsi.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

# Point yfinance tz-cache to a real temp dir.
# set_tz_cache_location(None) sets the internal path to None, which causes
# a TypeError('stat: ... NoneType') when yfinance tries to check the cache file.
import tempfile as _tempfile
_YF_CACHE_DIR = _tempfile.gettempdir()
try:
    yf.set_tz_cache_location(_YF_CACHE_DIR)
except Exception:
    pass

# ── Config ─────────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
STOCK_DATA_DIR  = PATHS.stock_data.nse_local
STOCK_LIST_CSV  = PATHS.stock_lists.nse_local
BENCHMARK_TICKER = "^NSEI"
START_DATE      = "2010-01-01"
MAX_WORKERS     = 1   # sequential — avoids rate-limit and SQLite cache conflicts
RATE_LIMIT_SLEEP = 0.3  # 300ms between tickers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NSE stock data via yfinance")
    p.add_argument("--refresh_after", type=int, default=0,
                   help="Re-download files older than N days (0 = skip existing)")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Download only these tickers (bypasses constituent list)")
    p.add_argument("--benchmark_only", action="store_true",
                   help="Only refresh ^NSEI benchmark file")
    p.add_argument("--start", default=START_DATE,
                   help=f"History start date (default: {START_DATE})")
    p.add_argument("--end", default=None,
                   help="History end date e.g. 2024-12-31 (default: today)")
    return p.parse_args()


def load_tickers() -> List[str]:
    """Read Symbol column from constituentsi.csv (e.g. RELIANCE.NS)."""
    df = pd.read_csv(STOCK_LIST_CSV)
    col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        raise ValueError(f"No 'Symbol' column in {STOCK_LIST_CSV}. Columns: {list(df.columns)}")
    tickers = df[col].dropna().str.strip().tolist()
    tickers = [t for t in tickers if t]
    # Append .NS suffix if not already present (yfinance requires it for NSE)
    tickers = [t if t.endswith(".NS") else f"{t}.NS" for t in tickers]
    print(f"  Loaded {len(tickers)} tickers from {STOCK_LIST_CSV.name}")
    return tickers


def file_needs_update(path: Path, refresh_after_days: int) -> bool:
    if not path.exists():
        return True
    if refresh_after_days <= 0:
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age > timedelta(days=refresh_after_days)


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns, rename to standard lower-case OHLCV."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for std in ("open", "high", "low", "close", "volume"):
        match = next((c for c in df.columns if c.lower() == std), None)
        if match:
            rename[match] = std
    df = df.rename(columns=rename)
    df.index.name = "Date"
    return df


def download_ticker(ticker: str, data_dir: Path, start: str,
                    refresh_after_days: int, end: str = None) -> tuple[str, bool, str]:
    """Download daily OHLCV for one ticker. Returns (ticker, success, message).

    Incremental mode: if the file already exists, only downloads data from
    the day after the last known bar up to `end` — avoids re-downloading
    full history on each monthly backtest cycle.
    """
    # Strip .NS suffix for filename — pipeline expects RELIANCE-1d.csv not RELIANCE.NS-1d.csv
    file_ticker = ticker.replace(".NS", "").replace(".BO", "")
    path = data_dir / f"{file_ticker}-1d.csv"

    try:
        # ── Incremental update if file exists ─────────────────────────────
        if path.exists() and refresh_after_days <= 0:
            existing = pd.read_csv(path, index_col=0, parse_dates=True)
            if not existing.empty:
                last_date = existing.index.max()
                new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                # Nothing to download if already at or past end date
                if end and new_start >= end:
                    return ticker, True, "skipped (up to date)"
                df_new = yf.download(ticker, start=new_start, end=end,
                                     auto_adjust=True, progress=False,
                                     multi_level_index=False)
                if df_new.empty:
                    return ticker, True, f"skipped (no new bars after {last_date.date()})"
                df_new = _normalise_df(df_new)
                for req in ("open", "high", "low", "close", "volume"):
                    if req not in df_new.columns:
                        return ticker, False, f"missing column '{req}' in new data"
                df_new = df_new[["open", "high", "low", "close", "volume"]]
                df_new = df_new[df_new["close"].notna() & (df_new["close"] > 0)]
                # Merge and deduplicate
                df = pd.concat([existing, df_new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index.name = "Date"
                df.to_csv(path)
                return ticker, True, f"+{len(df_new)} new rows (total {len(df)})"

        # ── Full download (new file or forced refresh) ─────────────────────
        if not file_needs_update(path, refresh_after_days):
            return ticker, True, "skipped (up to date)"

        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False, multi_level_index=False)
        if df.empty:
            return ticker, False, "empty response from yfinance"

        df = _normalise_df(df)

        for req in ("open", "high", "low", "close", "volume"):
            if req not in df.columns:
                return ticker, False, f"missing column '{req}'"

        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "Date"
        df = df[df["close"].notna() & (df["close"] > 0)]

        if len(df) < 50:
            return ticker, False, f"only {len(df)} rows — likely delisted or new"

        df.to_csv(path)
        return ticker, True, f"{len(df)} rows saved"

    except Exception as e:
        return ticker, False, str(e)[:100]


def download_all(tickers: List[str], data_dir: Path, start: str,
                 refresh_after_days: int, end: str = None) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    total     = len(tickers)
    succeeded = 0
    skipped   = 0
    failed: List[str] = []

    refresh_msg = (
        f"older than {refresh_after_days} days" if refresh_after_days > 0
        else "skipping existing files"
    )
    print(f"\nDownloading {total} tickers to {data_dir}")
    print(f"  ({refresh_msg})\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(download_ticker, t, data_dir, start, refresh_after_days, end): t
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
                if len(failed) <= 10:
                    print(f"  FAIL [{ticker}]: {msg}")

            if done % 50 == 0 or done == total:
                print(f"  Progress: {done}/{total}  "
                      f"(ok={succeeded}, skipped={skipped}, failed={len(failed)})")

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


def download_benchmark(data_dir: Path, start: str, refresh_after_days: int, end: str = None) -> None:
    print(f"\nDownloading benchmark {BENCHMARK_TICKER} ...")
    t, ok, msg = download_ticker(BENCHMARK_TICKER, data_dir, start,
                                  max(1, refresh_after_days), end)
    print(f"  [{'OK' if ok else 'FAILED'}] {BENCHMARK_TICKER}: {msg}")


def main() -> None:
    args = parse_args()
    STOCK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NSE Market Data Downloader")
    print(f"  Start  : {args.start}")
    print(f"  End    : {args.end or 'today'}")
    print(f"  Output : {STOCK_DATA_DIR}")
    print("=" * 60)

    if args.benchmark_only:
        download_benchmark(STOCK_DATA_DIR, args.start, max(1, args.refresh_after), args.end)
        return

    if args.tickers:
        tickers = [t.strip() for t in args.tickers]
        print(f"\nDownloading {len(tickers)} specified ticker(s) ...")
        download_all(tickers, STOCK_DATA_DIR, args.start, args.refresh_after, args.end)
        download_benchmark(STOCK_DATA_DIR, args.start, args.refresh_after, args.end)
        return

    print("\n[1/2] Loading constituent list ...")
    tickers = load_tickers()

    print(f"\n[2/2] Downloading {len(tickers)} NSE stocks ...")
    download_all(tickers, STOCK_DATA_DIR, args.start, args.refresh_after, args.end)
    download_benchmark(STOCK_DATA_DIR, args.start, max(1, args.refresh_after), args.end)

    print(f"\n{'='*60}")
    print("  Done! Next step:")
    print("  python run_nse_local.py --skip_train")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
