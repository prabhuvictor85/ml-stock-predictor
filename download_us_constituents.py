"""
download_us_constituents.py — Download constituent lists for 6 US indices.

Indices covered:
  SPX   — S&P 500          (~500 large-cap stocks)       via Wikipedia
  MID   — S&P MidCap 400   (~400 mid-cap stocks)         via Wikipedia
  SML   — S&P SmallCap 600 (~600 small-cap stocks)       via Wikipedia
  NDX   — NASDAQ 100       (~100 tech-heavy stocks)      via Wikipedia
  NGX   — NASDAQ Next Gen 100 (top 100 outside NDX)      via NASDAQ index API
  NQUSS — NASDAQ US Small Cap (thousands of stocks)      via NASDAQ index API

Output: C:/Victor/Learning_charts/stock_lists/
  constituents_spx.csv
  constituents_mid.csv
  constituents_sml.csv
  constituents_ndx.csv
  constituents_ngx.csv
  constituents_nquss.csv
  constituents_us_combined.csv   ← full deduplicated universe

Usage:
    python download_us_constituents.py
    python download_us_constituents.py --indices SPX MID NDX   (specific only)
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
LISTS_DIR = Path(r"C:\Victor\Learning_charts\stock_lists")
HEADERS   = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

INDICES = {
    "SPX":   "S&P 500",
    "MID":   "S&P MidCap 400",
    "SML":   "S&P SmallCap 600",
    "NDX":   "NASDAQ 100",
    "NGX":   "NASDAQ Next Generation 100",
    "NQUSS": "NASDAQ US Small Cap",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--indices", nargs="+", default=list(INDICES.keys()),
                   choices=list(INDICES.keys()),
                   help="Which indices to download (default: all)")
    return p.parse_args()


# ── Wikipedia scrapers ─────────────────────────────────────────────────────────

def fetch_sp500() -> pd.DataFrame:
    print("  Fetching S&P 500 from Wikipedia ...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                              storage_options={"User-Agent": HEADERS["User-Agent"]})
        df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
        df.columns = ["Symbol", "Name", "Sector"]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False).str.strip()
        df["Index"] = "SPX"
        print(f"    {len(df)} stocks")
        return df
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


def fetch_sp400() -> pd.DataFrame:
    print("  Fetching S&P MidCap 400 from Wikipedia ...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                              storage_options={"User-Agent": HEADERS["User-Agent"]})
        df = None
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols) and len(t) > 50:
                df = t
                break
        if df is None:
            raise ValueError("Table not found")
        sym_col  = next(c for c in df.columns if c.lower() in ("ticker", "symbol"))
        name_col = next((c for c in df.columns if "company" in c.lower() or "security" in c.lower()), None)
        sec_col  = next((c for c in df.columns if "sector" in c.lower()), None)
        result = pd.DataFrame()
        result["Symbol"] = df[sym_col].str.strip().str.replace(".", "-", regex=False)
        result["Name"]   = df[name_col].str.strip() if name_col else ""
        result["Sector"] = df[sec_col].str.strip()  if sec_col  else ""
        result["Index"]  = "MID"
        print(f"    {len(result)} stocks")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


def fetch_sp600() -> pd.DataFrame:
    print("  Fetching S&P SmallCap 600 from Wikipedia ...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
                              storage_options={"User-Agent": HEADERS["User-Agent"]})
        df = None
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols) and len(t) > 100:
                df = t
                break
        if df is None:
            raise ValueError("Table not found")
        sym_col  = next(c for c in df.columns if c.lower() in ("ticker", "symbol"))
        name_col = next((c for c in df.columns if "company" in c.lower() or "security" in c.lower()), None)
        sec_col  = next((c for c in df.columns if "sector" in c.lower()), None)
        result = pd.DataFrame()
        result["Symbol"] = df[sym_col].str.strip().str.replace(".", "-", regex=False)
        result["Name"]   = df[name_col].str.strip() if name_col else ""
        result["Sector"] = df[sec_col].str.strip()  if sec_col  else ""
        result["Index"]  = "SML"
        print(f"    {len(result)} stocks")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


def fetch_nasdaq100() -> pd.DataFrame:
    print("  Fetching NASDAQ 100 from Wikipedia ...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100",
                              storage_options={"User-Agent": HEADERS["User-Agent"]})
        df = None
        for t in tables:
            # Flatten MultiIndex columns if present
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [" ".join(str(s) for s in col).strip() for col in t.columns]
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols) and len(t) > 50:
                df = t
                break
        if df is None:
            raise ValueError("Table not found")
        sym_col  = next(c for c in df.columns if str(c).lower() in ("ticker", "symbol"))
        name_col = next((c for c in df.columns if "company" in str(c).lower()), None)
        sec_col  = next((c for c in df.columns if "sector" in str(c).lower() or "industry" in str(c).lower()), None)
        result = pd.DataFrame()
        result["Symbol"] = df[sym_col].astype(str).str.strip()
        result["Name"]   = df[name_col].astype(str).str.strip() if name_col else ""
        result["Sector"] = df[sec_col].astype(str).str.strip()  if sec_col  else "Technology"
        result["Index"]  = "NDX"
        print(f"    {len(result)} stocks")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


# ── NASDAQ index API scrapers ──────────────────────────────────────────────────

def fetch_nasdaq_index_api(index_code: str, label: str) -> pd.DataFrame:
    """
    Download constituent CSV from NASDAQ's official index export endpoint.
    URL: https://indexes.nasdaqomx.com/Index/ExportWeightings/{index_code}
    """
    url = f"https://indexes.nasdaqomx.com/Index/ExportWeightings/{index_code}"
    print(f"  Fetching {label} from NASDAQ export ({url}) ...")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        content = resp.text.strip()

        # NASDAQ export CSV has a metadata header block before the actual data
        # Find the line that starts with the column headers
        lines = content.splitlines()
        header_idx = next(
            (i for i, l in enumerate(lines) if "Symbol" in l or "Ticker" in l),
            None
        )
        if header_idx is None:
            raise ValueError("Could not find header row in CSV")

        csv_text = "\n".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]

        sym_col  = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), None)
        name_col = next((c for c in df.columns if "security" in c.lower() or "name" in c.lower()), None)

        if sym_col is None:
            raise ValueError(f"No symbol column. Columns: {list(df.columns)}")

        result = pd.DataFrame()
        result["Symbol"] = df[sym_col].astype(str).str.strip()
        result["Name"]   = df[name_col].astype(str).str.strip() if name_col else ""
        result["Sector"] = ""
        result["Index"]  = index_code
        result = result[result["Symbol"].notna() & (result["Symbol"] != "") & (result["Symbol"] != "nan")]
        print(f"    {len(result)} stocks")
        return result
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


# ── Main ───────────────────────────────────────────────────────────────────────

FETCHERS = {
    "SPX":   fetch_sp500,
    "MID":   fetch_sp400,
    "SML":   fetch_sp600,
    "NDX":   fetch_nasdaq100,
    "NGX":   lambda: fetch_nasdaq_index_api("NGX",   "NASDAQ Next Generation 100"),
    "NQUSS": lambda: fetch_nasdaq_index_api("NQUSS", "NASDAQ US Small Cap"),
}


def main():
    args = parse_args()
    LISTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  US Index Constituent Downloader")
    print(f"  Indices: {', '.join(args.indices)}")
    print(f"  Output : {LISTS_DIR}")
    print("=" * 60)

    all_frames = []

    for idx in args.indices:
        print(f"\n[{idx}] {INDICES[idx]}")
        df = FETCHERS[idx]()
        if df.empty:
            print(f"  WARNING: No data for {idx} — skipping")
            continue

        # Save individual index file
        out_path = LISTS_DIR / f"constituents_{idx.lower()}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name}")
        all_frames.append(df)
        time.sleep(0.5)  # be polite to servers

    # ── Combined universe ──────────────────────────────────────────────────────
    if len(all_frames) > 1:
        combined = pd.concat(all_frames, ignore_index=True)
        # Keep all index memberships per symbol in a combined column
        index_membership = (
            combined.groupby("Symbol")["Index"]
            .apply(lambda x: "|".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"Index": "Indices"})
        )
        # Deduplicate: keep first occurrence (priority: SPX > MID > SML > NDX > NGX > NQUSS)
        priority = {k: i for i, k in enumerate(["SPX", "MID", "SML", "NDX", "NGX", "NQUSS"])}
        combined["_pri"] = combined["Index"].map(priority)
        combined = combined.sort_values("_pri").drop_duplicates(subset="Symbol", keep="first")
        combined = combined.drop(columns=["_pri", "Index"])
        combined = combined.merge(index_membership, on="Symbol", how="left")
        combined = combined.sort_values("Symbol").reset_index(drop=True)

        out_path = LISTS_DIR / "constituents_us_combined.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n{'='*60}")
        print(f"  Combined universe: {len(combined)} unique stocks")
        print(f"  Saved: {out_path}")

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"  Next: place stock CSVs in C:\\Victor\\Learning_charts\\stock_data\\us_stocks\\")
    print(f"  Then: python run_sp500_local.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
