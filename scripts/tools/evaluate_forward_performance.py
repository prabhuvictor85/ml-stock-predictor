"""
evaluate_forward_performance.py
--------------------------------
Evaluates model predictions by comparing OHLC at the model's scoring date
vs a forward date (default: 6 months later). Fetches forward prices live
from yfinance — this data is NEVER written to the training CSVs.

Output: Excel file with one sheet per market containing:
  - OHLC at base date (from local CSV)
  - OHLC at forward date (live yfinance fetch)
  - % change in Close
  - Model rank and score (from watchlist)

Usage:
    python evaluate_forward_performance.py --market sp500
    python evaluate_forward_performance.py --market nse
    python evaluate_forward_performance.py --market sp500 --base_date 2024-03-01 --months 6
    python evaluate_forward_performance.py --market sp500 --base_date 2024-03-01 --forward_date 2024-09-01
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Config ─────────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
PROJECT_DIR = PATHS.project_root

MARKET_CONFIG = {
    "sp500": {
        "data_dir":    PATHS.stock_data.us,
        "output_dir":  PROJECT_DIR / "output" / "us_local",
        "list_file":   PATHS.stock_lists.us_combined,
        "label":       "SP500 + NASDAQ",
    },
    "nse": {
        "data_dir":    PATHS.stock_data.nse_local,
        "output_dir":  PROJECT_DIR / "output" / "nse_local",
        "list_file":   PATHS.stock_lists.nse_local,
        "label":       "NSE",
    },
}

EVAL_DIR = PROJECT_DIR / "output" / "evaluation"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise CSV columns to standard Date/Open/High/Low/Close/Volume."""
    # Rename to standard names (case-insensitive)
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "date":        rename[col] = "Date"
        elif cl == "open":      rename[col] = "Open"
        elif cl == "high":      rename[col] = "High"
        elif cl == "low":       rename[col] = "Low"
        elif cl == "close":     rename[col] = "Close"
        elif cl == "volume":    rename[col] = "Volume"
    df = df.rename(columns=rename)
    # TradingView format: ts (unix epoch), o/h/l/c/v
    if "Date" not in df.columns and "ts" in df.columns:
        df["Date"] = pd.to_datetime(df["ts"], unit="s").dt.date.astype(str)
    if "Open"  not in df.columns and "o" in df.columns: df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    return df


def nearest_trading_date(df: pd.DataFrame, target: datetime.date) -> datetime.date | None:
    """Return the nearest available date in df on or before target."""
    dates = pd.to_datetime(df["Date"]).dt.date
    candidates = dates[dates <= target]
    return candidates.max() if not candidates.empty else None


def get_ohlc_on_date(df: pd.DataFrame, target: datetime.date) -> dict | None:
    """Return OHLC row nearest to target date from local CSV."""
    df = _normalise_csv(df)
    if "Date" not in df.columns:
        return None
    nearest = nearest_trading_date(df, target)
    if nearest is None:
        return None
    row = df[pd.to_datetime(df["Date"]).dt.date == nearest].iloc[0]
    return {
        "date":   str(nearest),
        "open":   round(float(row.get("Open",  0)), 2),
        "high":   round(float(row.get("High",  0)), 2),
        "low":    round(float(row.get("Low",   0)), 2),
        "close":  round(float(row.get("Close", 0)), 2),
        "volume": int(row["Volume"]) if "Volume" in row.index else None,
    }


def fetch_forward_ohlc(ticker: str, target: datetime.date) -> dict | None:
    """Fetch OHLC for a single ticker/date from yfinance (fallback for stragglers)."""
    try:
        start = target - datetime.timedelta(days=5)
        end   = target + datetime.timedelta(days=5)
        df    = yf.download(
            ticker, start=str(start), end=str(end),
            auto_adjust=True, progress=False, multi_level_index=False
        )
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).date
        candidates = [d for d in df.index if d <= target]
        if not candidates:
            return None
        nearest = max(candidates)
        row = df.loc[nearest]
        return {
            "date":   str(nearest),
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if "Volume" in row.index else None,
        }
    except Exception:
        return None


def _read_local_ohlc(ticker: str, target: datetime.date, data_dir: Path) -> dict | None:
    """Read OHLC for ticker/date from the local stock_data CSV (no network call).

    Returns None if:
      - the file doesn't exist, or
      - the CSV's last bar is more than 7 days before target (stale — caller
        falls back to yfinance so we don't silently return the wrong date's price).

    The 7-day tolerance handles weekends and public holidays for the base date.
    For a forward date that's genuinely beyond the CSV's history (e.g. Nov 2024
    when CSVs only run to May 2024) this returns None and forces a live fetch.
    """
    # Strip .NS / .BO suffix for filename lookup (NSE tickers)
    file_ticker = ticker.replace(".NS", "").replace(".BO", "")
    path = data_dir / f"{file_ticker}-1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        df = df.reset_index().rename(columns={"index": "Date", df.index.name or "index": "Date"})
        if "Date" not in df.columns:
            df.columns.values[0] = "Date"

        # Staleness guard: if the CSV ends more than 7 days before the target,
        # the data doesn't cover the requested date — return None for yfinance fallback.
        df_norm = _normalise_csv(df.copy())
        if "Date" in df_norm.columns:
            last_date = pd.to_datetime(df_norm["Date"]).dt.date.max()
            if last_date < target - datetime.timedelta(days=7):
                return None  # stale CSV — let caller fetch from yfinance

        return get_ohlc_on_date(df, target)
    except Exception:
        return None


def fetch_batch_ohlc(
    tickers: list,
    target: datetime.date,
    data_dir: Path | None = None,
    batch_size: int = 100,
) -> dict:
    """Fetch OHLC for many tickers on one date — local CSV first, then one batch yfinance call.

    Strategy
    --------
    1. For every ticker, check the local stock_data CSV (instant, no network).
       This covers Hetzner and local runs where CSVs are pre-populated.
    2. Any ticker not found locally is fetched in a SINGLE batched yf.download()
       call rather than N individual calls.  One request avoids the rate-limit
       that kills HF Spaces when 151 separate downloads fire in quick succession.

    Returns
    -------
    {ticker: ohlc_dict_or_None}
    """
    results: dict = {}
    need_yf: list = []

    # ── Step 1: local CSV lookup ───────────────────────────────────────────────
    if data_dir and data_dir.exists():
        for ticker in tickers:
            ohlc = _read_local_ohlc(ticker, target, data_dir)
            results[ticker] = ohlc
            if ohlc is None:
                need_yf.append(ticker)
    else:
        need_yf = list(tickers)

    if not need_yf:
        return results

    # ── Step 2: batch yfinance for anything not in local CSVs ─────────────────
    start = str(target - datetime.timedelta(days=5))
    end   = str(target + datetime.timedelta(days=5))

    for i in range(0, len(need_yf), batch_size):
        batch = need_yf[i: i + batch_size]
        if not batch:
            continue

        if len(batch) == 1:
            # yfinance returns different structure for single-ticker — use scalar path
            results[batch[0]] = fetch_forward_ohlc(batch[0], target)
            continue

        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False,
                # multi_level_index=True is default for multi-ticker: columns are (Field, Ticker)
            )
            if raw.empty:
                for t in batch:
                    results[t] = None
                continue

            raw.index = pd.to_datetime(raw.index).date
            candidates = [d for d in raw.index if d <= target]
            if not candidates:
                for t in batch:
                    results[t] = None
                continue
            nearest = max(candidates)
            row = raw.loc[nearest]

            for ticker in batch:
                try:
                    close_val = row.get(("Close", ticker), float("nan"))
                    if pd.isna(close_val):
                        results[ticker] = None
                        continue
                    results[ticker] = {
                        "date":   str(nearest),
                        "open":   round(float(row.get(("Open",   ticker), 0)), 2),
                        "high":   round(float(row.get(("High",   ticker), 0)), 2),
                        "low":    round(float(row.get(("Low",    ticker), 0)), 2),
                        "close":  round(float(close_val), 2),
                        "volume": (int(row[("Volume", ticker)])
                                   if not pd.isna(row.get(("Volume", ticker), float("nan")))
                                   else None),
                    }
                except Exception:
                    results[ticker] = None

        except Exception:
            # Batch failed entirely — fall back to per-ticker
            for ticker in batch:
                if results.get(ticker) is None:
                    results[ticker] = fetch_forward_ohlc(ticker, target)

    return results


def _watchlist_search_dirs(output_dir: Path, date_str: str) -> list:
    """Return candidate directories to search for watchlist files.

    Supports two layouts:
      - Flat  : output/us_local/watchlist_momentum_bull_2024-05-03.csv
      - Dated : output/us_local/2024-05-03/watchlist_momentum_*_bull_2024-05-03.csv
    """
    dirs = [output_dir]
    dated = output_dir / date_str
    if dated.is_dir():
        dirs.append(dated)
    return dirs


def load_watchlist_scores(output_dir: Path, base_date: datetime.date) -> pd.DataFrame:
    """Load model scores from watchlist CSVs for the base date.

    Handles both naming patterns:
      - Old: watchlist_{model}_{side}_{date}.csv
      - New: watchlist_{model}_{variant}_{side}_{date}.csv  (composite / pureml)
    And both layouts (flat output_dir or date subdirectory).
    """
    date_str = str(base_date)
    rows = []
    search_dirs = _watchlist_search_dirs(output_dir, date_str)

    for model in ["momentum", "reversal"]:
        for side in ["bull", "bear"]:
            for d in search_dirs:
                # Match both old and new naming: any variant between model and side
                matched = list(d.glob(f"watchlist_{model}_*{side}_{date_str}.csv"))
                # Exclude cap-tier files (large/mid/small) — use combined lists only
                matched = [f for f in matched
                           if not any(t in f.stem for t in ("_large_", "_mid_", "_small_"))]
                for f in matched:
                    df = pd.read_csv(f)
                    df["model"] = model
                    df["side"]  = side
                    rows.append(df)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    # Normalise ticker column name
    for col in ["ticker", "Ticker", "symbol", "Symbol"]:
        if col in combined.columns:
            combined = combined.rename(columns={col: "ticker"})
            break
    combined = combined.drop_duplicates(subset=["ticker"])
    return combined


# ── Main ───────────────────────────────────────────────────────────────────────

def run(market: str, base_date: datetime.date, forward_date: datetime.date,
        custom_tickers: list[str] | None = None):
    cfg        = MARKET_CONFIG[market]
    data_dir   = cfg["data_dir"]
    output_dir = cfg["output_dir"]

    print("=" * 60)
    print(f"  Forward Performance Evaluation — {cfg['label']}")
    print(f"  Base date    : {base_date}")
    print(f"  Forward date : {forward_date}")
    print(f"  Window       : {(forward_date - base_date).days} days")
    print("=" * 60)

    # ── Load ticker list ───────────────────────────────────────────────────
    if custom_tickers:
        tickers = custom_tickers
        print(f"\nUsing {len(tickers)} tickers from --tickers argument")
    else:
        list_file = cfg["list_file"]
        if not list_file.exists():
            print(f"ERROR: {list_file} not found")
            return
        df_list  = pd.read_csv(list_file)
        sym_col  = next((c for c in df_list.columns if c.lower() in ("symbol", "ticker")), None)
        tickers  = df_list[sym_col].dropna().str.strip().tolist()
        tickers  = [t for t in tickers if t and not t.startswith("^")]
        print(f"\nLoaded {len(tickers)} tickers from {list_file.name}")

    # ── Load watchlist scores for ranking context ──────────────────────────
    scores_df = load_watchlist_scores(output_dir, base_date)
    if scores_df.empty:
        print(f"  No watchlist files found for {base_date} — scores will be blank")
    else:
        print(f"  Loaded {len(scores_df)} watchlist entries for {base_date}")

    # ── Build evaluation rows ──────────────────────────────────────────────
    # Prices are fetched in two bulk calls (local CSV first, then one batched
    # yfinance request for anything not available locally).  This replaces the
    # old per-ticker yf.download() loop that triggered rate-limits on HF Spaces.
    total = len(tickers)
    print(f"\nFetching base-date prices  ({base_date})  for {total} tickers ...")
    base_ohlc_map = fetch_batch_ohlc(tickers, base_date,    data_dir)
    base_hit = sum(1 for v in base_ohlc_map.values() if v)
    print(f"  -> {base_hit}/{total} prices found")

    print(f"Fetching forward-date prices ({forward_date}) for {total} tickers ...")
    fwd_ohlc_map  = fetch_batch_ohlc(tickers, forward_date, data_dir)
    fwd_hit = sum(1 for v in fwd_ohlc_map.values() if v)
    print(f"  -> {fwd_hit}/{total} prices found\n")

    results = []
    for i, ticker in enumerate(tickers, 1):
        base_ohlc = base_ohlc_map.get(ticker)
        fwd_ohlc  = fwd_ohlc_map.get(ticker)

        pct_change = None
        if (base_ohlc and fwd_ohlc
                and base_ohlc["close"] and base_ohlc["close"] != 0):
            pct_change = round(
                (fwd_ohlc["close"] - base_ohlc["close"]) / base_ohlc["close"] * 100, 2
            )

        row = {
            "ticker":              ticker,
            # Base date OHLC
            "base_date":           base_ohlc["date"]   if base_ohlc else None,
            "base_open":           base_ohlc["open"]   if base_ohlc else None,
            "base_high":           base_ohlc["high"]   if base_ohlc else None,
            "base_low":            base_ohlc["low"]    if base_ohlc else None,
            "base_close":          base_ohlc["close"]  if base_ohlc else None,
            "base_volume":         base_ohlc["volume"] if base_ohlc else None,
            # Forward date OHLC
            "fwd_date":            fwd_ohlc["date"]   if fwd_ohlc else None,
            "fwd_open":            fwd_ohlc["open"]   if fwd_ohlc else None,
            "fwd_high":            fwd_ohlc["high"]   if fwd_ohlc else None,
            "fwd_low":             fwd_ohlc["low"]    if fwd_ohlc else None,
            "fwd_close":           fwd_ohlc["close"]  if fwd_ohlc else None,
            "fwd_volume":          fwd_ohlc["volume"] if fwd_ohlc else None,
            # Performance
            "close_pct_change":    pct_change,
        }
        results.append(row)

        status = f"{pct_change:+.1f}%" if pct_change is not None else "no data"
        print(f"  [{i:>3}/{total}] {ticker:<20} {status}")

    if not results:
        print("No results generated.")
        return

    df_result = pd.DataFrame(results)

    # ── Merge watchlist scores ─────────────────────────────────────────────
    if not scores_df.empty:
        score_cols = ["ticker", "rank", "score", "model", "side"]
        score_cols = [c for c in score_cols if c in scores_df.columns]
        df_result  = df_result.merge(
            scores_df[score_cols], on="ticker", how="left"
        )

    # Sort: highest % gain first
    if "close_pct_change" in df_result.columns:
        df_result = df_result.sort_values("close_pct_change", ascending=False)

    # ── Save to Excel ──────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix   = f"_{custom_tickers[0]}" if custom_tickers and len(custom_tickers) <= 10 else ""
    out_file = EVAL_DIR / f"forward_eval_{market}_{base_date}_{forward_date}{suffix}.xlsx"
    # Avoid collision with an already-open file
    if out_file.exists():
        import time as _time
        out_file = EVAL_DIR / f"forward_eval_{market}_{base_date}_{forward_date}_{int(_time.time())}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="All Tickers", index=False)

        # Summary stats sheet
        stats = {
            "Metric": [
                "Total tickers evaluated",
                "Forward price fetched",
                "Avg % change (all)",
                "Median % change",
                "% positive (gainers)",
                "% negative (losers)",
                "Best performer",
                "Best % change",
                "Worst performer",
                "Worst % change",
            ],
            "Value": [
                len(df_result),
                df_result["fwd_close"].notna().sum(),
                round(df_result["close_pct_change"].mean(), 2),
                round(df_result["close_pct_change"].median(), 2),
                round((df_result["close_pct_change"] > 0).mean() * 100, 1),
                round((df_result["close_pct_change"] < 0).mean() * 100, 1),
                df_result.loc[df_result["close_pct_change"].idxmax(), "ticker"] if df_result["close_pct_change"].notna().any() else "N/A",
                round(df_result["close_pct_change"].max(), 2),
                df_result.loc[df_result["close_pct_change"].idxmin(), "ticker"] if df_result["close_pct_change"].notna().any() else "N/A",
                round(df_result["close_pct_change"].min(), 2),
            ]
        }
        pd.DataFrame(stats).to_excel(writer, sheet_name="Summary", index=False)

        # Top 50 gainers
        top50 = df_result.nlargest(50, "close_pct_change")
        top50.to_excel(writer, sheet_name="Top 50 Gainers", index=False)

        # Top 50 losers
        bot50 = df_result.nsmallest(50, "close_pct_change")
        bot50.to_excel(writer, sheet_name="Top 50 Losers", index=False)

        # Watchlist stocks only (model predicted)
        if "rank" in df_result.columns:
            wl = df_result[df_result["rank"].notna()].sort_values("rank")
            wl.to_excel(writer, sheet_name="Watchlist Only", index=False)

    print(f"\nSaved: {out_file}")
    print(f"\nSummary:")
    print(f"  Tickers evaluated  : {len(df_result)}")
    print(f"  Forward fetched    : {df_result['fwd_close'].notna().sum()}")
    print(f"  Avg % change       : {df_result['close_pct_change'].mean():.2f}%")
    print(f"  Median % change    : {df_result['close_pct_change'].median():.2f}%")
    print(f"  Gainers            : {(df_result['close_pct_change'] > 0).sum()}")
    print(f"  Losers             : {(df_result['close_pct_change'] < 0).sum()}")
    print("=" * 60)


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Forward performance evaluation")
    p.add_argument("--market",       required=True, choices=["sp500", "nse"])
    p.add_argument("--base_date",    default=None,
                   help="Model scoring date (default: auto-detect from latest watchlist)")
    p.add_argument("--forward_date", default=None,
                   help="Forward evaluation date (overrides --months)")
    p.add_argument("--months",       type=int, default=6,
                   help="Months ahead for forward date (default: 6)")
    p.add_argument("--tickers",      default=None,
                   help="Comma-separated ticker list, e.g. ABBOTINDIA.NS,GLAXO.NS  "
                        "(overrides the full market list)")
    p.add_argument("--watchlist_only", action="store_true",
                   help="Only evaluate tickers that appear in the watchlist for this date "
                        "(much faster — skips full 1600-ticker universe fetch).")
    p.add_argument("--output_dir", default=None,
                   help="Override watchlist output directory "
                        "(e.g. /mnt/data/artefacts/us_local/output on Hetzner).")
    return p.parse_args()


def detect_latest_base_date(output_dir: Path) -> datetime.date | None:
    """Auto-detect base date from latest watchlist file.

    Handles both flat layout (watchlist_momentum_bull_DATE.csv directly in
    output_dir) and dated subdirectory layout (output_dir/DATE/watchlist_...).
    """
    import re
    DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

    # Flat layout
    files = list(output_dir.glob("watchlist_momentum_*bull_*.csv"))
    # Dated subdirectory layout
    files += list(output_dir.glob("*/watchlist_momentum_*bull_*.csv"))

    dates = []
    for f in files:
        m = DATE_RE.search(f.stem)
        if m:
            try:
                dates.append(datetime.date.fromisoformat(m.group(1)))
            except Exception:
                pass
    return max(dates) if dates else None


if __name__ == "__main__":
    args = parse_args()
    cfg  = MARKET_CONFIG[args.market]

    # Allow --output_dir to override where watchlist CSVs are read from
    if args.output_dir:
        cfg = dict(cfg)   # shallow copy so we don't mutate the module-level dict
        cfg["output_dir"] = Path(args.output_dir)

    # Resolve base date
    if args.base_date:
        base_date = datetime.date.fromisoformat(args.base_date)
    else:
        base_date = detect_latest_base_date(cfg["output_dir"])
        if base_date is None:
            print("ERROR: Could not auto-detect base date. Use --base_date YYYY-MM-DD")
            exit(1)
        print(f"Auto-detected base date: {base_date}")

    # Resolve forward date
    if args.forward_date:
        forward_date = datetime.date.fromisoformat(args.forward_date)
    else:
        # Add N months dynamically
        month = base_date.month + args.months
        year  = base_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        forward_date = base_date.replace(year=year, month=month)
        print(f"Forward date (+{args.months} months): {forward_date}")

    custom_tickers = None
    if args.tickers:
        custom_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        print(f"Custom ticker list: {len(custom_tickers)} tickers")

    # --watchlist_only: pull tickers from watchlist files for this base_date
    if args.watchlist_only and custom_tickers is None:
        scores_df = load_watchlist_scores(cfg["output_dir"], base_date)
        if scores_df.empty:
            print(f"ERROR: No watchlist files found for {base_date}. "
                  f"Cannot use --watchlist_only without watchlist files.")
            exit(1)
        ticker_col = next((c for c in scores_df.columns
                           if c.lower() in ("ticker", "symbol")), None)
        if ticker_col:
            custom_tickers = scores_df[ticker_col].dropna().unique().tolist()
            print(f"--watchlist_only: evaluating {len(custom_tickers)} unique watchlist tickers")

    run(args.market, base_date, forward_date, custom_tickers=custom_tickers)
