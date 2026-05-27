"""
sync_stock_data.py - Download & incrementally update stock data for NSE or SP500.

For each ticker in the constituent list:
  - If {TICKER}-1d.csv does not exist -> full download from --start
  - If {TICKER}-1d.csv exists and is stale -> fetch only the delta, merge & save
  - If {TICKER}-1d.csv is already up to date -> skip

Usage:
    python sync_stock_data.py --market nse
    python sync_stock_data.py --market sp500
    python sync_stock_data.py --market nse   --start 2010-01-01 --end 2024-01-01
    python sync_stock_data.py --market sp500 --start 2015-01-01
    python sync_stock_data.py --market nse   --tickers RELIANCE.NS TCS.NS
    python sync_stock_data.py --market sp500 --tickers AAPL MSFT NVDA

NSE output  : C:/Victor/Learning_charts/stock_data/
SP500 output: C:/Victor/Learning_charts/stock_data/us_stocks/
"""
from __future__ import annotations

import argparse
import datetime
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

# -- Logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- Market config --------------------------------------------------------------
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS

MARKET_CONFIG = {
    "nse": {
        "data_dir":       PATHS.stock_data.nse_local,
        "list_file":      PATHS.stock_lists.nse_local,
        "symbol_col":     "Symbol",          # column holding the ticker (e.g. RELIANCE.NS)
        "benchmarks":     ["^NSEI"],
        "label":          "NSE",
    },
    "sp500": {
        "data_dir":       PATHS.stock_data.us,
        "list_file":      PATHS.stock_lists.us_combined,
        "symbol_col":     "Symbol",          # column holding the ticker (e.g. AAPL)
        "benchmarks":     ["^GSPC", "^NDX"],
        "label":          "SP500 + NASDAQ",
    },
}

DEFAULT_START = "2010-01-01"
MAX_WORKERS   = 4      # keep low - Yahoo rate-limits aggressive parallel requests
RATE_SLEEP    = 1.0    # seconds between batches - be polite to yfinance
WRITE_RETRIES = 3      # retry CSV write on PermissionError (file lock from prev run)

# -- Thread-safe per-file locks -------------------------------------------------
_path_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _get_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _locks_guard:
        if key not in _path_locks:
            _path_locks[key] = threading.Lock()
        return _path_locks[key]


# -- Helpers --------------------------------------------------------------------

def _last_trading_day(end_date: Optional[datetime.date] = None) -> datetime.date:
    """Last completed trading day - last weekday on or before reference date."""
    ref = end_date if end_date else datetime.date.today() - datetime.timedelta(days=1)
    while ref.weekday() >= 5:   # Saturday=5, Sunday=6
        ref -= datetime.timedelta(days=1)
    return ref


def _csv_last_date(df: pd.DataFrame) -> Optional[datetime.date]:
    """Return the latest date in the DataFrame, or None."""
    col = None
    if "Date" in df.columns:
        col = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        col = df.index.to_series()
    if col is None or len(col) == 0:
        return None
    ts = col.max()
    return ts.date() if not pd.isna(ts) else None


def _fetch_from_yfinance(
    ticker: str,
    start: datetime.date,
    end: Optional[datetime.date],
) -> pd.DataFrame:
    """Download OHLCV from yfinance. Returns empty DataFrame on failure."""
    try:
        end_str = str(end + datetime.timedelta(days=1)) if end else None
        df = yf.download(
            ticker,
            start=str(start),
            end=end_str,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if df.empty:
            return pd.DataFrame()

        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Normalise column names to title case
        rename = {}
        for std in ("open", "high", "low", "close", "volume"):
            match = next((c for c in df.columns if c.lower() == std), None)
            if match:
                rename[match] = std.capitalize() if std != "volume" else "Volume"
        df = df.rename(columns=rename)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        df = df[required].copy()
        df = df[df["Close"].notna() & (df["Close"] > 0)]

        # Reset index: Date becomes a column
        df.index.name = "Date"
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        return df

    except Exception as e:
        log.warning("%s - yfinance error: %s", ticker, str(e)[:120])
        return pd.DataFrame()


def _invalidate_drv_files(ticker: str, data_dir: Path) -> None:
    """Delete stale -Drv.csv files when fresh daily data is written."""
    for tf in ["1d", "1wk", "1mo", "3mo", "1y"]:
        drv = data_dir / f"{ticker}-{tf}-Drv.csv"
        if drv.exists():
            try:
                drv.unlink()
                log.debug("Invalidated: %s", drv.name)
            except Exception:
                pass


def _csv_write(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV with retries on PermissionError (file lock)."""
    for attempt in range(WRITE_RETRIES):
        try:
            df.to_csv(path, index=False)
            return
        except PermissionError:
            if attempt < WRITE_RETRIES - 1:
                log.warning("PermissionError writing %s - retrying in 2s (attempt %d/%d)",
                            path.name, attempt + 1, WRITE_RETRIES)
                time.sleep(2)
            else:
                raise


# -- Core sync function ---------------------------------------------------------

def sync_ticker(
    ticker: str,
    data_dir: Path,
    start: datetime.date,
    end: Optional[datetime.date],
) -> tuple[str, str]:
    """
    Sync one ticker's daily CSV.
    Returns (ticker, status_message).
    """
    file_path = data_dir / f"{ticker}-1d.csv"
    last_td   = _last_trading_day(end)

    with _get_lock(file_path):
        # -- File exists: check if stale ----------------------------------
        if file_path.exists():
            try:
                existing  = pd.read_csv(file_path)
                last_date = _csv_last_date(existing)
            except Exception as e:
                return ticker, f"READ ERROR: {e}"

            if last_date is not None and last_date >= last_td:
                return ticker, f"up to date ({last_date})"

            # Fetch delta only
            fetch_from = (
                last_date + datetime.timedelta(days=1)
                if last_date else start
            )
            log.info("%s - fetching delta %s -> %s", ticker, fetch_from, last_td)
            delta = _fetch_from_yfinance(ticker, fetch_from, end)

            if delta.empty:
                return ticker, f"no new data (last={last_date})"

            # Merge, deduplicate, sort, save
            combined = pd.concat([existing, delta], ignore_index=True)
            combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce", utc=True).dt.tz_localize(None)
            combined = (
                combined
                .drop_duplicates(subset=["Date"])
                .sort_values("Date")
                .reset_index(drop=True)
            )
            _csv_write(combined, file_path)
            _invalidate_drv_files(ticker, data_dir)
            return ticker, f"delta +{len(delta)} rows -> {len(combined)} total"

        # -- File does not exist: full download ---------------------------
        log.info("%s - full download from %s", ticker, start)
        df = _fetch_from_yfinance(ticker, start, end)

        if df.empty:
            return ticker, "FAILED - empty response"

        if len(df) < 50:
            return ticker, f"SKIPPED - only {len(df)} rows (likely new/delisted)"

        data_dir.mkdir(parents=True, exist_ok=True)
        _csv_write(df, file_path)
        return ticker, f"new - {len(df)} rows saved"


# -- Batch runner ---------------------------------------------------------------

def sync_all(
    tickers: List[str],
    data_dir: Path,
    start: datetime.date,
    end: Optional[datetime.date],
    label: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    total     = len(tickers)
    succeeded = 0
    skipped   = 0
    delta_d   = 0
    failed: List[str] = []

    print(f"\nSyncing {total} {label} tickers -> {data_dir}")
    end_str = str(end) if end else "today"
    print(f"  Date range: {start} -> {end_str}\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(sync_ticker, t, data_dir, start, end): t
            for t in tickers
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            ticker, msg = fut.result()

            if "up to date" in msg:
                skipped += 1
            elif msg.startswith("FAILED") or msg.startswith("READ ERROR"):
                failed.append(ticker)
                if len(failed) <= 10:
                    print(f"  x [{ticker}]: {msg}")
            elif "delta" in msg:
                delta_d += 1
                succeeded += 1
            else:
                succeeded += 1

            if done % 50 == 0 or done == total:
                print(
                    f"  Progress: {done}/{total}  "
                    f"(new/updated={succeeded}, skipped={skipped}, "
                    f"delta={delta_d}, failed={len(failed)})"
                )

            if done % MAX_WORKERS == 0:
                time.sleep(RATE_SLEEP)

    print(f"\n{'='*60}")
    print(f"  {label} sync complete")
    print(f"  New / full download : {succeeded - delta_d}")
    print(f"  Delta updated       : {delta_d}")
    print(f"  Already up to date  : {skipped}")
    print(f"  Failed              : {len(failed)}")
    if failed:
        fail_path = data_dir / "_failed_tickers.txt"
        fail_path.write_text("\n".join(failed))
        print(f"  Failed list saved   : {fail_path}")
    print(f"{'='*60}\n")


# -- Args -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync stock data (NSE or SP500)")
    p.add_argument(
        "--market", required=True, choices=["nse", "sp500"],
        help="Which market to sync"
    )
    p.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date for full download (default: {DEFAULT_START})"
    )
    p.add_argument(
        "--end", default=None,
        help="End date cap - leave blank for today"
    )
    p.add_argument(
        "--tickers", nargs="+", default=None,
        help="Sync only these specific tickers (bypasses constituent list)"
    )
    p.add_argument(
        "--benchmarks_only", action="store_true",
        help="Only sync the benchmark index file(s)"
    )
    return p.parse_args()


# -- Main -----------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    cfg    = MARKET_CONFIG[args.market]
    start  = datetime.date.fromisoformat(args.start)
    end    = datetime.date.fromisoformat(args.end) if args.end else None

    print("=" * 60)
    print(f"  Stock Data Sync - {cfg['label']}")
    print(f"  Start : {start}")
    print(f"  End   : {end or 'today'}")
    print(f"  Output: {cfg['data_dir']}")
    print("=" * 60)

    data_dir = cfg["data_dir"]

    # -- Benchmarks ---------------------------------------------------------
    print("\nSyncing benchmark(s) ...")
    for bm in cfg["benchmarks"]:
        _, msg = sync_ticker(bm, data_dir, start, end)
        print(f"  [{bm}]: {msg}")

    if args.benchmarks_only:
        return

    # -- Specific tickers ---------------------------------------------------
    if args.tickers:
        tickers = [t.strip() for t in args.tickers]
        print(f"\nSyncing {len(tickers)} specified ticker(s) ...")
        sync_all(tickers, data_dir, start, end, cfg["label"])
        return

    # -- Full constituent list ----------------------------------------------
    list_file = cfg["list_file"]
    if not list_file.exists():
        print(f"\nERROR: Constituent list not found: {list_file}")
        print("  Run download_us_constituents.py first (for SP500)")
        print("  or ensure constituentsi.csv is present (for NSE)")
        return

    df_list = pd.read_csv(list_file)
    sym_col = cfg["symbol_col"]

    # Filter out benchmark rows (^GSPC, ^NDX, etc.)
    df_list = df_list[~df_list[sym_col].astype(str).str.startswith("^")]
    tickers = df_list[sym_col].dropna().str.strip().tolist()
    tickers = [t for t in tickers if t]

    print(f"\nLoaded {len(tickers)} tickers from {list_file.name}")
    sync_all(tickers, data_dir, start, end, cfg["label"])


if __name__ == "__main__":
    main()
