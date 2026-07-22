"""
Extract point-in-time shares-outstanding history from SEC EDGAR's bulk
companyfacts archive, for every ticker in the project's US universe.

Source: SEC EDGAR bulk XBRL data (data.sec.gov / sec.gov), free, public.
  https://www.sec.gov/edgar/sec-api-documentation  (bulk file locations)
Downloaded once via curl with a declared User-Agent (SEC requires this —
see https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data,
rate limit 10 req/s, no auth needed):
  C:/Victor/Learning_charts/edgar_pit/companyfacts.zip      (~1.39 GB, all filers)
  C:/Victor/Learning_charts/edgar_pit/company_tickers.json  (ticker -> CIK map)

WHY THIS TAG SPECIFICALLY (dei:EntityCommonStockSharesOutstanding):
It is the one clean, unambiguous, genuinely point-in-time input needed for a
causal Market Cap factor (Market Cap = shares_outstanding x price). Unlike
income-statement concepts (Revenue, NetIncome, ...), companies do not use
inconsistent GAAP tag variants for this — it is a single `dei` (Document and
Entity Information) cover-page fact present whenever a company files XBRL at
all. Broader fundamentals (margins, ROE, EBITDA, ...) need a tag-fallback
dictionary per concept and are deliberately OUT of scope for this pass.

CAUSALITY — the whole point of doing this instead of a live snapshot API:
Each fact carries BOTH `end` (the date the share count is "as of", printed on
the filing's cover page) AND `filed` (the date the filing actually became
public). A naive join on `end` leaks the ~3-8 week gap between a fiscal
period ending and the filing that discloses it. This script's output is
indexed by `filed` — the only date a factor built from it can honestly claim
to have "known" the value.

RESTATEMENT HANDLING: some facts get reported more than once for the same
`end` date (a 10-K/A amendment, for instance) at a LATER `filed` date. Point-
in-time means using what was known FIRST, not what was later corrected/
restated — so duplicates on (cik, end) are deduped to the earliest `filed`.

Output:
  C:/Victor/Learning_charts/edgar_pit/shares_outstanding_pit.parquet
  Columns: ticker, cik, end, filed, shares_outstanding, form
  One row per (ticker, end) — the first-reported share count for that period.

Run:
  python scripts/data/extract_edgar_shares_outstanding.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from pipeline.config.paths import PATHS

EDGAR_DIR       = PATHS.edgar_pit
COMPANYFACTS    = EDGAR_DIR / "companyfacts.zip"
TICKER_MAP_JSON = EDGAR_DIR / "company_tickers.json"
OUT_PATH        = EDGAR_DIR / "shares_outstanding_pit.parquet"

TAG_NAMESPACE = "dei"
TAG_NAME      = "EntityCommonStockSharesOutstanding"

# SEC's own company_tickers.json is not 100% reliable even for major names —
# same class of problem as the Yahoo/Norgate ticker-reuse traps this project
# already maintains a manual list for. Verified case: "XOM" in company_
# tickers.json resolves to CIK 2115436 ("ExxonMobil Holdings Corp"), a near-
# empty related entity (0 us-gaap tags) — NOT the real Exxon Mobil Corporation
# (CIK 34088, 438 us-gaap tags, has our target field). Add confirmed
# corrections here as they're found; do not assume the live SEC map is ground
# truth without checking.
CIK_OVERRIDES: dict[str, int] = {
    "XOM": 34088,
}


def load_universe_symbols() -> list[str]:
    uni = pd.read_csv(PATHS.stock_lists.us_combined)
    return sorted(uni["Symbol"].dropna().str.strip().str.upper().unique())


def build_ticker_to_cik() -> dict[str, int]:
    tickers = json.load(open(TICKER_MAP_JSON, encoding="utf-8"))
    mapping = {v["ticker"].upper(): int(v["cik_str"]) for v in tickers.values()}
    mapping.update(CIK_OVERRIDES)
    return mapping


def extract_one(zf: zipfile.ZipFile, ticker: str, cik: int) -> pd.DataFrame | None:
    fname = f"CIK{cik:010d}.json"
    try:
        with zf.open(fname) as f:
            data = json.load(f)
    except KeyError:
        return None  # resolved a CIK but the archive has no facts file for it

    tag = data.get("facts", {}).get(TAG_NAMESPACE, {}).get(TAG_NAME)
    if not tag:
        return None  # company files XBRL but never tagged this specific fact

    entries = tag.get("units", {}).get("shares", [])
    if not entries:
        return None

    df = pd.DataFrame(entries)
    if df.empty or "end" not in df.columns or "filed" not in df.columns:
        return None

    df["ticker"] = ticker
    df["cik"] = cik
    return df[["ticker", "cik", "end", "filed", "val", "form"]]


def main() -> None:
    if not COMPANYFACTS.exists():
        sys.exit(f"Missing {COMPANYFACTS} — run the bulk download first.")
    if not TICKER_MAP_JSON.exists():
        sys.exit(f"Missing {TICKER_MAP_JSON} — run the bulk download first.")

    symbols = load_universe_symbols()
    tick_to_cik = build_ticker_to_cik()
    print(f"Universe: {len(symbols)} tickers")

    frames: list[pd.DataFrame] = []
    unresolved: list[str] = []
    no_tag: list[str] = []

    with zipfile.ZipFile(COMPANYFACTS) as zf:
        for i, sym in enumerate(symbols, 1):
            cik = tick_to_cik.get(sym)
            if cik is None:
                unresolved.append(sym)
                continue
            df = extract_one(zf, sym, cik)
            if df is None:
                no_tag.append(sym)
                continue
            frames.append(df)
            if i % 250 == 0:
                print(f"  {i}/{len(symbols)} tickers processed...")

    if not frames:
        sys.exit("No data extracted — something is wrong upstream.")

    combined = pd.concat(frames, ignore_index=True)
    combined["end"] = pd.to_datetime(combined["end"])
    combined["filed"] = pd.to_datetime(combined["filed"])
    combined = combined.rename(columns={"val": "shares_outstanding"})

    before = len(combined)
    # Point-in-time = first known, not later-restated. Sort by filed date
    # ascending, then keep the FIRST row per (cik, end) — i.e. the earliest
    # filing that ever reported that period's share count. A later 10-K/A
    # amending the same period is correctly discarded here, even if its
    # value differs from the original.
    combined = combined.sort_values(["cik", "end", "filed"])
    deduped = combined.drop_duplicates(subset=["cik", "end"], keep="first")
    n_amended = before - len(deduped)

    deduped = deduped.sort_values(["ticker", "filed"]).reset_index(drop=True)
    deduped.to_parquet(OUT_PATH, index=False)

    tickers_with_data = deduped["ticker"].nunique()
    print()
    print("=" * 62)
    print("  EDGAR shares-outstanding PIT extraction — summary")
    print("=" * 62)
    print(f"  universe tickers            : {len(symbols)}")
    print(f"  resolved to a CIK           : {len(symbols) - len(unresolved)}")
    print(f"  unresolved (no CIK match)   : {len(unresolved)}  {unresolved[:10]}")
    print(f"  resolved but tag missing    : {len(no_tag)}  {no_tag[:10]}")
    print(f"  tickers with >=1 data row   : {tickers_with_data}")
    print(f"  total rows (pre-dedup)      : {before:,}")
    print(f"  dropped as later amendments : {n_amended:,}")
    print(f"  total rows (final)         : {len(deduped):,}")
    print(f"  filed-date range            : {deduped['filed'].min().date()} -> {deduped['filed'].max().date()}")
    print(f"  output                      : {OUT_PATH}")
    print("=" * 62)


if __name__ == "__main__":
    main()
