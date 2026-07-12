"""
Extract SP500 / SP400 / SP600 historical constituent intervals from Norgate Data.

Requirements:
  - Norgate Data Updater (NDU) running + initial sync complete
  - pip install norgatedata

Output:
  C:/Victor/Learning_charts/stock_lists/membership_sp1500_norgate.csv
  Columns: ticker, index, start_date, end_date   (end_date blank = still member)

Run:
  python scripts/data/extract_norgate_constituents.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from datetime import date

try:
    import norgatedata as ng
except ImportError:
    print("ERROR: norgatedata not installed. Run: pip install norgatedata")
    sys.exit(1)

OUT_PATH   = Path(r"C:/Victor/Learning_charts/stock_lists/membership_sp1500_norgate.csv")
START_DATE = "2000-01-01"
TODAY      = date.today().isoformat()

# "Current & Past" variants include all historical members, not just today's
INDICES = {
    "SP500": "S&P 500 Current & Past",
    "SP400": "S&P MidCap 400 Current & Past",
    "SP600": "S&P SmallCap 600 Current & Past",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def bool_series_to_intervals(series: pd.Series) -> list[tuple]:
    """Convert a daily True/False membership series into (start, end) date pairs.
    end=None means still a member on the last available date.
    """
    series = series.sort_index().astype(bool)
    if series.empty:
        return []

    intervals = []
    in_block  = False
    start     = None

    for dt, val in series.items():
        if val and not in_block:
            start    = dt
            in_block = True
        elif not val and in_block:
            intervals.append((start, dt))   # end = first day OUT (exclusive)
            in_block = False

    if in_block:
        intervals.append((start, None))     # still member

    return intervals


def get_all_tickers_for_index(norgate_name: str) -> list[str]:
    """Get every ticker that was ever in the index (current + historical)."""
    # Primary: watchlist_symbols returns all-time members for index watchlists
    try:
        syms = ng.watchlist_symbols(norgate_name)
        if syms and len(syms) > 0:
            return list(syms)
    except Exception:
        pass

    # Fallback: try common variant names
    variants = [
        norgate_name,
        norgate_name.replace('S&P ', 'S&P '),
        f"{norgate_name} (Current & Past)",
        f"{norgate_name} Current & Past",
        f"S&P 500 (Current & Past)",
    ]
    for v in variants:
        try:
            syms = ng.watchlist_symbols(v)
            if syms and len(syms) > 0:
                print(f"  Found under variant name: '{v}'")
                return list(syms)
        except Exception:
            pass

    return []


def extract_index(norgate_name: str, short_name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Extracting: {norgate_name}  ({short_name})")
    print(f"{'='*60}")

    tickers = get_all_tickers_for_index(norgate_name)

    if not tickers:
        print(f"  WARNING: No tickers found via watchlist_symbols.")
        print(f"  Trying: listing all watchlists to find correct name...")
        try:
            all_wl = ng.watchlists()
            sp_wl = [w for w in all_wl if '500' in str(w) or '400' in str(w) or '600' in str(w) or 'S&P' in str(w)]
            print(f"  Available S&P watchlists: {sp_wl}")
        except Exception as e:
            print(f"  Could not list watchlists: {e}")
        return pd.DataFrame()

    print(f"  Total tickers in index (all time): {len(tickers)}")

    # For each ticker get daily membership boolean → convert to intervals
    rows   = []
    errors = []

    for i, ticker in enumerate(tickers):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(tickers)}  ({i*100//len(tickers)}%)", flush=True)
        try:
            ts = ng.index_constituent_timeseries(
                ticker,
                norgate_name,
                padding_setting=ng.PaddingType.NONE,
                start_date=START_DATE,
                end_date=TODAY,
                timeseriesformat='pandas-dataframe',
                datetimeformat='iso',
            )

            if ts is None or (hasattr(ts, '__len__') and len(ts) == 0):
                errors.append((ticker, "empty"))
                continue

            if isinstance(ts, pd.DataFrame):
                # typically single column with True/False
                col = ts.iloc[:, 0]
            else:
                col = pd.Series(ts)

            col.index = pd.to_datetime(col.index)
            intervals = bool_series_to_intervals(col)

            for start, end in intervals:
                rows.append({
                    "ticker":     ticker,
                    "index":      short_name,
                    "start_date": start.date() if hasattr(start, 'date') else start,
                    "end_date":   end.date()   if (end is not None and hasattr(end, 'date')) else end,
                })

        except Exception as e:
            errors.append((ticker, str(e)))

    print(f"  Intervals extracted: {len(rows)}")
    print(f"  Errors: {len(errors)}")
    if errors[:5]:
        for t, e in errors[:5]:
            print(f"    {t}: {e}")

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Norgate constituent extraction")
    print(f"norgatedata version: {ng.__version__}")

    # Quick connectivity check — use recent dates (trial = last 2 years only)
    print("\nConnectivity check: fetching AAPL price (recent)...")
    try:
        test = ng.price_timeseries(
            'AAPL',
            start_date='2026-01-01', end_date=TODAY,
            timeseriesformat='pandas-dataframe',
        )
        if test is not None and len(test) > 0:
            print(f"  OK — AAPL data found ({len(test)} bars, last: {test.index[-1].date()})")
        else:
            print("  WARNING: AAPL price empty — but constituent data may still work (no 2yr limit)")
            print("  Continuing with constituent extraction...")
    except Exception as e:
        print(f"  WARNING: {e} — continuing anyway")

    # List available watchlists so user can verify index names
    print("\nAvailable watchlists (S&P related):")
    try:
        all_wl = ng.watchlists()
        sp_wl  = [w for w in all_wl if any(x in str(w) for x in ['S&P','500','400','600','MidCap','SmallCap'])]
        for w in sp_wl:
            print(f"  '{w}'")
        if not sp_wl:
            print("  (none found — printing first 20)")
            for w in list(all_wl)[:20]:
                print(f"  '{w}'")
    except Exception as e:
        print(f"  Could not list watchlists: {e}")

    all_frames = []
    for short_name, norgate_name in INDICES.items():
        df = extract_index(norgate_name, short_name)
        if len(df):
            all_frames.append(df)
            print(f"  {short_name}: {len(df)} rows, {df['ticker'].nunique()} unique tickers")

    if not all_frames:
        print("\nERROR: No data extracted. Check watchlist names above and update INDICES dict.")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(['index', 'ticker', 'start_date']).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for idx_name, grp in combined.groupby('index'):
        current = grp['end_date'].isna().sum()
        total   = grp['ticker'].nunique()
        print(f"  {idx_name}: {total} unique tickers, {current} current members")

    # Spot checks
    for ticker, idx, check_date, desc in [
        ("TSLA", "SP500", "2020-12-21", "TSLA added SP500 Dec 2020"),
        ("AAPL", "SP500", "2015-01-01", "AAPL in SP500 throughout 2015"),
    ]:
        row = combined[(combined.ticker == ticker) & (combined['index'] == idx)]
        if len(row):
            r = row.iloc[0]
            ok = pd.Timestamp(check_date) >= pd.Timestamp(str(r.start_date))
            print(f"  {'PASS' if ok else 'FAIL'}: {desc}")
        else:
            print(f"  MISSING: {ticker} not in {idx}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    print(f"  {len(combined)} rows | {combined['ticker'].nunique()} unique tickers")


if __name__ == "__main__":
    main()
