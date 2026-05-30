"""
add_nse_sectors.py
------------------
Builds a sector map for NSE tickers using two sources:

  Step 1 -- NSE sector index archive CSVs (fast, covers ~190 Nifty stocks)
  Step 2 -- yfinance fallback for remaining 'Others' tickers (covers small/micro caps)

Usage:
    python scripts/data/add_nse_sectors.py

Input  : {stock_lists}/constituentsi.csv   (Symbol1, Symbol)
Output : {stock_lists}/constituentsi.csv   (Symbol1, Symbol, Sector)  <- updated in-place
Backup : {stock_lists}/constituentsi_backup.csv
"""

from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
}

# NSE archive CSV URLs for each sector index (public, no auth needed)
SECTOR_ARCHIVE_URLS = [
    ("https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",                "IT"),
    ("https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",              "Banking"),
    ("https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv",           "Financial Services"),
    ("https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",              "Auto"),
    ("https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv",              "FMCG"),
    ("https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",            "Pharma"),
    ("https://archives.nseindia.com/content/indices/ind_niftymetallist.csv",             "Metal"),
    ("https://archives.nseindia.com/content/indices/ind_niftymedialist.csv",             "Media"),
    ("https://archives.nseindia.com/content/indices/ind_niftyrealtylist.csv",            "Realty"),
    ("https://archives.nseindia.com/content/indices/ind_niftypsubanklist.csv",           "PSU Bank"),
    ("https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv",        "Healthcare"),
    ("https://archives.nseindia.com/content/indices/ind_niftyoilgaslist.csv",            "Oil & Gas"),
    ("https://archives.nseindia.com/content/indices/ind_niftyconsumerdurableslist.csv",  "Consumer Durables"),
    ("https://archives.nseindia.com/content/indices/ind_niftyenergylist.csv",            "Energy"),
    ("https://archives.nseindia.com/content/indices/ind_niftycommoditieslist.csv",       "Commodities"),
    ("https://archives.nseindia.com/content/indices/ind_niftyinfrastructurelist.csv",    "Infrastructure"),
    ("https://archives.nseindia.com/content/indices/ind_niftycapitalmarketlist.csv",     "Capital Markets"),
]


def fetch_sector_constituents(url: str, sector_label: str) -> list[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col is None:
            return []
        return df[sym_col].str.strip().dropna().tolist()
    except Exception as e:
        print(f"    Error: {e}")
        return []


def fetch_yfinance_sectors(symbols: list[str]) -> dict[str, str]:
    """
    Fetch sector for each NSE bare symbol via yfinance (e.g. RELIANCE -> RELIANCE.NS).
    Returns dict: bare_symbol -> sector string.
    Skips symbols that fail or return no sector.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed -- skipping fallback (pip install yfinance)")
        return {}

    sector_map: dict[str, str] = {}
    total = len(symbols)
    print(f"\n  Step 2: yfinance fallback for {total} unmatched symbols ...")

    for i, sym in enumerate(symbols, 1):
        ticker_str = f"{sym}.NS"
        try:
            info = yf.Ticker(ticker_str).info
            sector = info.get("sector") or info.get("industry") or ""
            if sector:
                sector_map[sym] = sector.strip()
        except Exception:
            pass

        if i % 50 == 0 or i == total:
            print(f"    {i}/{total} done, {len(sector_map)} sectors found so far ...")
        time.sleep(0.1)  # polite delay

    return sector_map


def main():
    print("=" * 60)
    print("  NSE Sector Merger")
    print("=" * 60)

    constituents_path: Path = PATHS.stock_lists.nse_local
    if not constituents_path.exists():
        print(f"ERROR: {constituents_path} not found.")
        sys.exit(1)

    # Load only clean columns — drop any Unnamed garbage
    const_df = pd.read_csv(constituents_path)
    clean_cols = [c for c in const_df.columns if not str(c).startswith("Unnamed")]
    # Also drop any duplicate/stale Sector column so we rebuild cleanly
    clean_cols = [c for c in clean_cols if c != "Sector"]
    const_df = const_df[clean_cols].copy()

    print(f"\n  Loaded {len(const_df)} rows, columns: {list(const_df.columns)}")

    sym_col = "Symbol1" if "Symbol1" in const_df.columns else "Symbol"
    print(f"  Using '{sym_col}' as NSE bare symbol column")

    # ── Step 1: NSE sector index archive CSVs ─────────────────────────────────
    print("\n  Step 1: NSE sector index archive CSVs ...")
    sector_map: dict[str, str] = {}

    for url, sector_label in SECTOR_ARCHIVE_URLS:
        print(f"  [{sector_label}] ...", end=" ", flush=True)
        symbols = fetch_sector_constituents(url, sector_label)
        if symbols:
            print(f"{len(symbols)} symbols")
            for sym in symbols:
                if sym not in sector_map:
                    sector_map[sym] = sector_label
        else:
            print("0 -- skipped")
        time.sleep(0.3)

    print(f"\n  Step 1 result: {len(sector_map)} symbols mapped")

    # ── Step 2: yfinance fallback for unmatched symbols ───────────────────────
    bare_symbols = const_df[sym_col].str.strip().tolist()
    unmatched = [s for s in bare_symbols if s not in sector_map]

    yf_map = fetch_yfinance_sectors(unmatched)
    sector_map.update(yf_map)
    print(f"  Step 2 result: {len(yf_map)} additional symbols mapped via yfinance")

    # ── Backup ────────────────────────────────────────────────────────────────
    backup_path = constituents_path.parent / "constituentsi_backup.csv"
    const_df.to_csv(backup_path, index=False)
    print(f"\n  Backup saved -> {backup_path.name}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    const_df["Sector"] = const_df[sym_col].str.strip().map(sector_map).fillna("Others")

    matched   = (const_df["Sector"] != "Others").sum()
    unmatched_count = (const_df["Sector"] == "Others").sum()
    print(f"  Matched  : {matched} / {len(const_df)}")
    print(f"  Unmatched: {unmatched_count} (assigned 'Others')")

    # ── Save ──────────────────────────────────────────────────────────────────
    try:
        const_df.to_csv(constituents_path, index=False)
        print(f"\n  Saved -> {constituents_path.name}")
    except PermissionError:
        alt_path = constituents_path.parent / "constituentsi_with_sectors.csv"
        const_df.to_csv(alt_path, index=False)
        print(f"\n  File is open (close Excel). Saved to -> {alt_path.name}")
        print(f"  Copy it manually: copy constituentsi_with_sectors.csv constituentsi.csv")

    print(f"\n  Sector distribution:")
    print(const_df["Sector"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
