"""
add_nse_sectors.py
──────────────────
Builds a sector map by fetching constituents of NSE sector indices
(Nifty IT, Nifty Bank, Nifty Auto, etc.) and merges 'Sector' column
into constituentsi.csv.

Strategy:
  - Uses the same NSE equity-stockIndices API as download_nse_index_constituents.py
  - Each stock is assigned the sector index it belongs to
  - Stocks not in any sector index fall back to 'Others'

Usage:
    python scripts/data/add_nse_sectors.py

Input  : {stock_lists}/constituentsi.csv   (Symbol1, Symbol)
Output : {stock_lists}/constituentsi.csv   (Symbol1, Symbol, Sector)  ← updated in-place
Backup : {stock_lists}/constituentsi_backup.csv
"""

from __future__ import annotations

import sys
import time
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
    ("https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",               "IT"),
    ("https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",             "Banking"),
    ("https://archives.nseindia.com/content/indices/ind_niftyfinancelist.csv",          "Financial Services"),
    ("https://archives.nseindia.com/content/indices/ind_niftyautolist.csv",             "Auto"),
    ("https://archives.nseindia.com/content/indices/ind_niftyfmcglist.csv",             "FMCG"),
    ("https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv",           "Pharma"),
    ("https://archives.nseindia.com/content/indices/ind_niftymetallist.csv",            "Metal"),
    ("https://archives.nseindia.com/content/indices/ind_niftymedialist.csv",            "Media"),
    ("https://archives.nseindia.com/content/indices/ind_niftyrealtylist.csv",           "Realty"),
    ("https://archives.nseindia.com/content/indices/ind_niftypsubanklist.csv",          "PSU Bank"),
    ("https://archives.nseindia.com/content/indices/ind_niftyhealthcarelist.csv",       "Healthcare"),
    ("https://archives.nseindia.com/content/indices/ind_niftyoilgaslist.csv",           "Oil & Gas"),
    ("https://archives.nseindia.com/content/indices/ind_niftyconsumerdurables list.csv", "Consumer Durables"),
    ("https://archives.nseindia.com/content/indices/ind_niftyconsumerdurableslist.csv", "Consumer Durables"),
    ("https://archives.nseindia.com/content/indices/ind_niftyenergylist.csv",           "Energy"),
    ("https://archives.nseindia.com/content/indices/ind_niftycommoditieslist.csv",      "Commodities"),
    ("https://archives.nseindia.com/content/indices/ind_niftyinfrastructurelist.csv",   "Infrastructure"),
    ("https://archives.nseindia.com/content/indices/ind_nifty_infrastructure_list.csv", "Infrastructure"),
    ("https://archives.nseindia.com/content/indices/ind_niftycapitalmarketlist.csv",    "Capital Markets"),
    ("https://archives.nseindia.com/content/indices/ind_niftycapmarketslist.csv",       "Capital Markets"),
]


def fetch_sector_constituents(url: str, sector_label: str) -> list[str]:
    from io import StringIO
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col is None:
            print(f"    No symbol column found. Columns: {list(df.columns)}")
            return []
        return df[sym_col].str.strip().dropna().tolist()
    except Exception as e:
        print(f"    Error: {e}")
        return []


def main():
    print("=" * 60)
    print("  NSE Sector Merger (via sector indices)")
    print("=" * 60)

    constituents_path: Path = PATHS.stock_lists.nse_local
    if not constituents_path.exists():
        print(f"ERROR: {constituents_path} not found.")
        sys.exit(1)

    const_df = pd.read_csv(constituents_path)
    print(f"\n  Loaded {len(const_df)} rows from {constituents_path.name}")
    print(f"  Columns: {list(const_df.columns)}")

    sym_col = "Symbol1" if "Symbol1" in const_df.columns else "Symbol"
    print(f"  Using '{sym_col}' as NSE bare symbol column")

    # ── Fetch sector constituents from archive CSVs ───────────────────────────
    sector_map: dict[str, str] = {}  # NSE bare symbol -> sector name

    for url, sector_label in SECTOR_ARCHIVE_URLS:
        print(f"  [{sector_label}] ...", end=" ", flush=True)
        symbols = fetch_sector_constituents(url, sector_label)
        if symbols:
            print(f"{len(symbols)} symbols")
            for sym in symbols:
                if sym not in sector_map:  # first match wins
                    sector_map[sym] = sector_label
        else:
            print("0 -- skipped")
        time.sleep(0.5)

    print(f"\n  Total sector map: {len(sector_map)} unique symbols")

    # ── Backup ────────────────────────────────────────────────────────────────
    backup_path = constituents_path.parent / "constituentsi_backup.csv"
    const_df.to_csv(backup_path, index=False)
    print(f"  Backup saved -> {backup_path.name}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    bare_symbols = const_df[sym_col].str.strip()
    const_df["Sector"] = bare_symbols.map(sector_map).fillna("Others")

    matched   = (const_df["Sector"] != "Others").sum()
    unmatched = (const_df["Sector"] == "Others").sum()
    print(f"\n  Matched  : {matched} / {len(const_df)}")
    print(f"  Unmatched: {unmatched} (assigned 'Others' - small/micro caps not in sector indices)")

    # ── Save ──────────────────────────────────────────────────────────────────
    try:
        const_df.to_csv(constituents_path, index=False)
        print(f"\n  Saved -> {constituents_path.name}")
    except PermissionError:
        alt_path = constituents_path.parent / "constituentsi_with_sectors.csv"
        const_df.to_csv(alt_path, index=False)
        print(f"\n  PermissionError on original file (close it in Excel if open).")
        print(f"  Saved to alternate -> {alt_path.name}")
        print(f"  Manually replace {constituents_path.name} with {alt_path.name} when ready.")
    print(f"\n  Sector distribution:")
    print(const_df["Sector"].value_counts().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
