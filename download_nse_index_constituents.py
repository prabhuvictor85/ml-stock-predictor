"""
Download NSE index constituent lists and build a cap-tier mapping CSV.

Cap tiers (SEBI official definition):
  Large Cap : Nifty 50 + Nifty Next 50   (top 100 by market cap)
  Mid Cap   : Nifty Midcap 150           (rank 101-250)
  Small Cap : Nifty Smallcap 250         (rank 251-500)

Output:
  C:/Victor/Learning_charts/stock_lists/nse_cap_tiers.csv
  Columns: Symbol, cap_tier, indices

Usage (no prompts):
  python download_nse_index_constituents.py
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import requests

OUT_FILE = Path(r"C:\Victor\Learning_charts\stock_lists\nse_cap_tiers.csv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── NSE index definitions ──────────────────────────────────────────────────────
# (index_api_name, tier, label)
INDEX_DEFS = [
    ("NIFTY 50",          "large", "Nifty50"),
    ("NIFTY NEXT 50",     "large", "NiftyNext50"),
    ("NIFTY MIDCAP 150",  "mid",   "NiftyMidcap150"),
    ("NIFTY SMALLCAP 250","small", "NiftySmallcap250"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

# ── Session with cookies ───────────────────────────────────────────────────────
def get_nse_session() -> requests.Session:
    """Open NSE homepage once to obtain required session cookies."""
    session = requests.Session()
    session.headers.update(HEADERS)
    print("  Fetching NSE homepage for session cookies...", end=" ", flush=True)
    try:
        r = session.get("https://www.nseindia.com/", timeout=20)
        r.raise_for_status()
        print(f"OK  (cookies: {list(session.cookies.keys())})")
    except Exception as e:
        print(f"WARNING: {e}")
    time.sleep(1)
    return session


def fetch_index_constituents(session: requests.Session, index_name: str) -> list[str]:
    """Return list of Symbol strings for the given index via NSE equity-stockIndices API."""
    url = "https://www.nseindia.com/api/equity-stockIndices"
    params = {"index": index_name}
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        # Response: {"data": [{"symbol": "RELIANCE", ...}, ...]}
        symbols = [row["symbol"].strip() for row in data.get("data", [])
                   if row.get("symbol") and row["symbol"] != index_name.replace(" ", "_")]
        return symbols
    except Exception as e:
        print(f"    API error for '{index_name}': {e}")
        return []


# ── Fallback: direct CSV download from NSE archives ───────────────────────────
ARCHIVE_URLS = {
    "NIFTY 50":           "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY NEXT 50":      "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY MIDCAP 150":   "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

def fetch_via_archive_csv(session: requests.Session, index_name: str) -> list[str]:
    url = ARCHIVE_URLS.get(index_name, "")
    if not url:
        return []
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        # Column is typically 'Symbol'
        sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        if sym_col:
            return df[sym_col].str.strip().dropna().tolist()
    except Exception as e:
        print(f"    Archive CSV error for '{index_name}': {e}")
    return []


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  NSE Index Constituent Downloader")
    print("=" * 60)

    session = get_nse_session()

    rows = []
    for index_name, tier, label in INDEX_DEFS:
        print(f"\n  [{label}] Fetching '{index_name}'...")
        symbols = fetch_index_constituents(session, index_name)
        if not symbols:
            print(f"    API returned 0 — trying archive CSV fallback...")
            symbols = fetch_via_archive_csv(session, index_name)
        if symbols:
            print(f"    {len(symbols)} symbols fetched")
            for sym in symbols:
                rows.append({"Symbol": sym, "cap_tier": tier, "index": label})
        else:
            print(f"    WARNING: could not fetch '{index_name}'")
        time.sleep(1.2)   # polite delay between requests

    if not rows:
        print("\nERROR: No data fetched from any index. Check network/NSE access.")
        sys.exit(1)

    # Build dataframe — a stock can appear in multiple indices (e.g. Nifty50 + NiftyNext50).
    # Keep one row per Symbol with the highest tier (large > mid > small)
    # and concatenate all index labels.
    TIER_RANK = {"large": 0, "mid": 1, "small": 2}
    df_all = pd.DataFrame(rows)

    # Aggregate: best tier + all index labels
    def agg_sym(grp):
        best = min(grp["cap_tier"].tolist(), key=lambda t: TIER_RANK[t])
        indices = "|".join(sorted(grp["index"].unique().tolist()))
        return pd.Series({"cap_tier": best, "indices": indices})

    df_out = df_all.groupby("Symbol").apply(agg_sym, include_groups=False).reset_index()
    df_out = df_out.sort_values("Symbol").reset_index(drop=True)

    df_out.to_csv(OUT_FILE, index=False)
    print(f"\n{'=' * 60}")
    print(f"  Saved {len(df_out)} tickers -> {OUT_FILE}")
    print(f"\n  Cap tier summary:")
    for tier, label in [("large","Large Cap"), ("mid","Mid Cap"), ("small","Small Cap")]:
        n = (df_out["cap_tier"] == tier).sum()
        print(f"    {label:12s}: {n}")
    print(f"\n  Sample (first 10 rows):")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
