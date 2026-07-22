"""
Extract point-in-time fundamentals from SEC EDGAR's bulk companyfacts
archive, for every ticker in the project's US universe. Sibling script to
extract_edgar_shares_outstanding.py — same archive, same causality rules,
same CIK-override list — extended to the small set of "hub" concepts that
compose into most standard value/quality ratios (P/B, P/S, D/E, margins,
ROE, EV, FCF) without needing a separate pull per ratio.

CONCEPTS PULLED (see CONCEPTS dict below for the exact tag fallback lists):
  revenue, operating_income, net_income   — income-statement (duration facts)
  stockholders_equity, cash, assets_current,
  liabilities_current, long_term_debt,
  current_debt                            — balance-sheet (instant facts)
  operating_cash_flow, capex              — cash-flow (duration facts)

WHY A TAG FALLBACK LIST PER CONCEPT: unlike shares outstanding, there is no
single consistent XBRL tag for "Revenue" across companies and years. Verified
on AAPL's own filings: SalesRevenueNet was used 2009-2018, then briefly
Revenues (2018-11-05 10-K only, comparative-year figures), then
RevenueFromContractWithCustomerExcludingAssessedTax from 2019 onward — the
ASC 606 revenue-recognition standard transition. Each concept below is
extracted from ALL its candidate tags; when two tags report the same (cik,
end) period, the earlier entry in the list wins (it names the tag that was
the reporting standard closest to "now" for that concept).

INSTANT vs DURATION FACTS: balance-sheet concepts (equity, cash, debt,
current assets/liabilities) are "instant" — a snapshot, only an `end` date,
same shape as shares outstanding. Income-statement and cash-flow concepts
are "duration" — they carry both `start` and `end`, and a single company's
feed mixes quarterly (~90-day), half-year (~180-day), 9-month (~270-day),
and annual (~365-day) windows for the SAME tag, because 10-Qs report both
the discrete quarter and the year-to-date cumulative. Mixing these blindly
would blend a 3-month figure with a 9-month one. This script tags each
duration by length — Q (~90d), YTD2 (~180d), YTD3 (~270d), FY (~365d) — and
keeps all four. Income statements carry a discrete Q column, but CASH-FLOW
statements report ONLY the YTD ladder (all rungs sharing the fiscal-year
start); the feature builder differences that ladder (Q2 = YTD2 - YTD1,
Q3 = YTD3 - YTD2, Q4 = FY - YTD3) to recover discrete quarters. Keeping the
YTD rungs is what makes OCF/capex (and thus FCF and accruals) buildable.

KNOWN LIMITATION — no standalone Q4: most companies never file a discrete
"Q4" duration fact (the 10-K reports the full fiscal year only). The feature
builder derives it (FY - Q1 - Q2 - Q3, or FY - YTD3); this script just
preserves every rung it needs to.

CAUSALITY AND DEDUP: identical rule to shares outstanding. Indexed by
`filed`, not `end`. Duplicate (cik, end, period_type) rows — a later
restatement, or a lower-priority fallback tag re-reporting the same period —
are deduped to the earliest `filed`, after first sorting so the
higher-priority tag wins on a same-day tie.

Output:
  C:/Victor/Learning_charts/edgar_pit/fundamentals_pit.parquet
  Long format. Columns: ticker, cik, concept, period_type, tag, start, end,
  filed, val, form. One row per (ticker, concept, period_type, end) — the
  first-reported value for that period from the highest-priority tag that
  covers it.

Run:
  python scripts/data/extract_edgar_fundamentals.py
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from scripts.data.extract_edgar_shares_outstanding import (
    COMPANYFACTS,
    EDGAR_DIR,
    TICKER_MAP_JSON,
    build_ticker_to_cik,
    load_universe_symbols,
)

OUT_PATH = EDGAR_DIR / "fundamentals_pit.parquet"

# concept -> (kind, [tags in priority order — first match for a given period wins])
CONCEPTS: dict[str, tuple[str, list[str]]] = {
    "revenue": ("duration", [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ]),
    "operating_income": ("duration", ["OperatingIncomeLoss"]),
    "net_income": ("duration", ["NetIncomeLoss", "ProfitLoss"]),
    "stockholders_equity": ("instant", [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    # Total assets — the single highest-value balance-sheet snapshot. Present
    # for every sector including banks (verified). Denominator for asset
    # growth (investment factor), accruals (Sloan earnings quality), and
    # asset turnover (DuPont efficiency).
    "total_assets": ("instant", ["Assets"]),
    "cash": ("instant", [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ]),
    "assets_current": ("instant", ["AssetsCurrent"]),
    "liabilities_current": ("instant", ["LiabilitiesCurrent"]),
    "long_term_debt": ("instant", ["LongTermDebtNoncurrent", "LongTermDebt"]),
    "current_debt": ("instant", ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"]),
    "operating_cash_flow": ("duration", ["NetCashProvidedByUsedInOperatingActivities"]),
    "capex": ("duration", [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
    ]),
}

# Duration-fact period lengths. Income statements report a discrete "three
# months ended" column (Q), but CASH-FLOW statements report only YTD
# cumulatives (Q1, then 6-mo, 9-mo, FY — all sharing the fiscal-year start).
# We therefore keep the intermediate YTD rungs (YTD2 ~180d, YTD3 ~270d) so the
# feature builder can difference the ladder back into discrete quarters
# (Q2 = YTD2 - YTD1, ...). Without these, OCF/capex TTM can never be built.
QUARTER_DAYS = (80, 100)
YTD2_DAYS = (170, 195)
YTD3_DAYS = (260, 285)
ANNUAL_DAYS = (350, 380)


def _period_type(dur_days: int) -> str | None:
    if QUARTER_DAYS[0] <= dur_days <= QUARTER_DAYS[1]:
        return "Q"
    if YTD2_DAYS[0] <= dur_days <= YTD2_DAYS[1]:
        return "YTD2"
    if YTD3_DAYS[0] <= dur_days <= YTD3_DAYS[1]:
        return "YTD3"
    if ANNUAL_DAYS[0] <= dur_days <= ANNUAL_DAYS[1]:
        return "FY"
    return None


def extract_concept(data: dict, concept: str, kind: str, tags: list[str]) -> pd.DataFrame | None:
    us_gaap = data.get("facts", {}).get("us-gaap", {})
    rows = []
    for rank, tag in enumerate(tags):
        entry = us_gaap.get(tag)
        if not entry:
            continue
        for rec in entry.get("units", {}).get("USD", []):
            if "end" not in rec or "filed" not in rec or "val" not in rec:
                continue
            row = {
                "tag": tag,
                "tag_rank": rank,
                "end": rec["end"],
                "filed": rec["filed"],
                "val": rec["val"],
                "form": rec.get("form", ""),
            }
            if kind == "duration":
                if "start" not in rec:
                    continue
                dur_days = (pd.Timestamp(rec["end"]) - pd.Timestamp(rec["start"])).days
                ptype = _period_type(dur_days)
                if ptype is None:
                    continue
                row["start"] = rec["start"]
                row["period_type"] = ptype
            else:
                row["start"] = None
                row["period_type"] = "PIT"
            rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["concept"] = concept
    return df


def main() -> None:
    if not COMPANYFACTS.exists():
        sys.exit(f"Missing {COMPANYFACTS} — run the bulk download first.")
    if not TICKER_MAP_JSON.exists():
        sys.exit(f"Missing {TICKER_MAP_JSON} — run the bulk download first.")

    symbols = load_universe_symbols()
    tick_to_cik = build_ticker_to_cik()
    print(f"Universe: {len(symbols)} tickers, {len(CONCEPTS)} concepts")

    frames: list[pd.DataFrame] = []
    unresolved: list[str] = []
    no_data: list[str] = []

    with zipfile.ZipFile(COMPANYFACTS) as zf:
        for i, sym in enumerate(symbols, 1):
            cik = tick_to_cik.get(sym)
            if cik is None:
                unresolved.append(sym)
                continue

            fname = f"CIK{cik:010d}.json"
            try:
                with zf.open(fname) as f:
                    data = json.load(f)
            except KeyError:
                no_data.append(sym)
                continue

            ticker_frames = []
            for concept, (kind, tags) in CONCEPTS.items():
                cdf = extract_concept(data, concept, kind, tags)
                if cdf is not None:
                    ticker_frames.append(cdf)

            if not ticker_frames:
                no_data.append(sym)
                continue

            combined = pd.concat(ticker_frames, ignore_index=True)
            combined["ticker"] = sym
            combined["cik"] = cik
            frames.append(combined)

            if i % 250 == 0:
                print(f"  {i}/{len(symbols)} tickers processed...")

    if not frames:
        sys.exit("No data extracted — something is wrong upstream.")

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["end"] = pd.to_datetime(all_rows["end"])
    all_rows["filed"] = pd.to_datetime(all_rows["filed"])
    all_rows = all_rows.rename(columns={"val": "value"})

    before = len(all_rows)
    # Priority-then-causality dedup: sort so the higher-priority tag (lower
    # tag_rank) AND the earliest filing sort first, then keep first per
    # (cik, concept, period_type, end).
    all_rows = all_rows.sort_values(["cik", "concept", "period_type", "end", "tag_rank", "filed"])
    deduped = all_rows.drop_duplicates(
        subset=["cik", "concept", "period_type", "end"], keep="first"
    )
    n_dropped = before - len(deduped)

    deduped = deduped.drop(columns=["tag_rank"])
    deduped = deduped.sort_values(["ticker", "concept", "filed"]).reset_index(drop=True)
    deduped.to_parquet(OUT_PATH, index=False)

    tickers_with_data = deduped["ticker"].nunique()
    print()
    print("=" * 62)
    print("  EDGAR fundamentals PIT extraction — summary")
    print("=" * 62)
    print(f"  universe tickers            : {len(symbols)}")
    print(f"  resolved to a CIK           : {len(symbols) - len(unresolved)}")
    print(f"  unresolved (no CIK match)   : {len(unresolved)}  {unresolved[:10]}")
    print(f"  resolved but no concepts    : {len(no_data)}  {no_data[:10]}")
    print(f"  tickers with >=1 data row   : {tickers_with_data}")
    print(f"  total rows (pre-dedup)      : {before:,}")
    print(f"  dropped (dup tag/restatement): {n_dropped:,}")
    print(f"  total rows (final)         : {len(deduped):,}")
    print()
    print("  rows per concept:")
    print(deduped["concept"].value_counts().to_string())
    print(f"  output                      : {OUT_PATH}")
    print("=" * 62)


if __name__ == "__main__":
    main()
