"""
evaluate_forward_performance.py
--------------------------------
Evaluates model predictions by comparing OHLC at the model's scoring date
vs a forward date (default: 6 months later). Fetches forward prices live
from yfinance — this data is NEVER written to the training CSVs.

Enrichment (via --hf_data_dir):
  - HF JSON watchlist files: rank, score, model_type, mode, cap_tier
  - scores_detail JSON: sector, cap_tier fallback
  - Benchmark (SPY / ^NSEI) 6M return -> per-stock excess return
  - Multi-period returns: 4-week, 3-month, 6-month
  - Peak return date within 6M window (month 1-2 vs 3-4 vs 5-6)
  - Rank bucket analysis: Rank 1-5, 6-15, 16-30, 31+
  - Mode comparison: momentum vs reversal
  - Model type: PureML vs Composite
  - Cap tier: large / mid / small
  - SDZ zone quality analysis
  - Random baseline (5 x 30-stock samples)
  - Failed pick flags: LOW-SCORE, NO-SDZ

Output: Excel file with sheets:
  All Tickers | Summary | Top 50 Gainers | Top 50 Losers |
  Watchlist Detail | Rank Analysis | Mode Analysis |
  Model Type | Cap Tier | SDZ Analysis | Timing Analysis |
  Peak Detail | Random Baseline | Failed Picks

Usage:
    python evaluate_forward_performance.py --market sp500
    python evaluate_forward_performance.py --market sp500 --base_date 2024-01-12 --months 6
    python evaluate_forward_performance.py --market sp500 --base_date 2024-01-12 \\
        --hf_data_dir C:/Victor/Projects/ml-stock-dashboard/public/data/us_local
    python evaluate_forward_performance.py --market nse --base_date 2024-01-12 --months 6
    python evaluate_forward_performance.py --market sp500 --watchlist_only \\
        --base_date 2024-01-12 --months 6 \\
        --hf_data_dir C:/Victor/Projects/ml-stock-dashboard/public/data/us_local
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
PROJECT_DIR = PATHS.project_root

_STOCK_LISTS = Path(r"C:\Victor\Learning_charts\stock_lists")

# Per-cap-tier universe files (SP500=large, SP400=mid, SP600=small)
CAP_TIER_LISTS = {
    "large": _STOCK_LISTS / "constituents_spx.csv",
    "mid":   _STOCK_LISTS / "constituents_mid.csv",
    "small": _STOCK_LISTS / "constituents_sml.csv",
}

MARKET_CONFIG = {
    "sp500": {
        "data_dir":   PATHS.stock_data.us,
        "output_dir": PROJECT_DIR / "output" / "us_local",
        "list_file":  PATHS.stock_lists.us_combined,
        "label":      "SP500 + NASDAQ",
        "benchmark":  "SPY",
    },
    "nse": {
        "data_dir":   PATHS.stock_data.nse_local,
        "output_dir": PROJECT_DIR / "output" / "nse_local",
        "list_file":  PATHS.stock_lists.nse_local,
        "label":      "NSE",
        "benchmark":  "^NSEI",
    },
}

EVAL_DIR = PROJECT_DIR / "output" / "evaluation"

RANK_BUCKETS = [
    ("Rank 1-5",   1,   5),
    ("Rank 6-15",  6,  15),
    ("Rank 16-30", 16, 30),
    ("Rank 31+",   31, 9999),
]


# ── Existing helpers ───────────────────────────────────────────────────────────

def _normalise_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise CSV columns to standard Date/Open/High/Low/Close/Volume."""
    rename = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "date":      rename[col] = "Date"
        elif cl == "open":    rename[col] = "Open"
        elif cl == "high":    rename[col] = "High"
        elif cl == "low":     rename[col] = "Low"
        elif cl == "close":   rename[col] = "Close"
        elif cl == "volume":  rename[col] = "Volume"
    df = df.rename(columns=rename)
    if "Date" not in df.columns and "ts" in df.columns:
        df["Date"] = pd.to_datetime(df["ts"], unit="s").dt.date.astype(str)
    if "Open" not in df.columns and "o" in df.columns:
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                 "c": "Close", "v": "Volume"})
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
    """Fetch OHLC for a single ticker/date from yfinance (fallback for stragglers).

    Uses yf.Ticker().history() — more reliable than yf.download() for single
    tickers in yfinance >= 0.2.50 where multi_level_index behaviour changed.
    """
    try:
        start = target - datetime.timedelta(days=5)
        end   = target + datetime.timedelta(days=5)
        df = yf.Ticker(ticker).history(
            start=str(start), end=str(end),
            auto_adjust=True, raise_errors=False,
        )
        if df.empty:
            return None
        # Strip timezone so .date comparisons work
        dates = pd.to_datetime(df.index).tz_localize(None).normalize().date
        candidates = [d for d in dates if d <= target]
        if not candidates:
            return None
        nearest = max(candidates)
        row = df.iloc[list(dates).index(nearest)]
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


def _read_local_ohlc(
    ticker: str,
    target: datetime.date,
    data_dir: Path,
    max_lag_days: int = 7,
) -> dict | None:
    """Read OHLC for ticker/date from local stock_data CSV (no network call).

    Returns None if:
      - the CSV file doesn't exist, or
      - the nearest available date is more than max_lag_days before target
        (meaning the CSV data ends before the forward date — caller falls
        back to yfinance rather than silently returning a stale base price).

    max_lag_days=7 allows for weekends + public holidays while still catching
    the common case where the CSV ends at base_date and is asked for a
    forward_date six months later — which would otherwise return the same
    close price for both dates, giving 0% change.
    """
    file_ticker = ticker.replace(".NS", "").replace(".BO", "")
    path = data_dir / f"{file_ticker}-1d.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        df = df.reset_index().rename(
            columns={"index": "Date", df.index.name or "index": "Date"})
        if "Date" not in df.columns:
            df.columns.values[0] = "Date"
        result = get_ohlc_on_date(df, target)
        if result is None:
            return None
        # Guard: if the nearest date is too far before target (CSV ends too
        # early), return None so the caller uses yfinance for the real price.
        nearest = datetime.date.fromisoformat(result["date"])
        if (target - nearest).days > max_lag_days:
            return None
        return result
    except Exception:
        return None


def fetch_batch_ohlc(
    tickers: list,
    target: datetime.date,
    data_dir: Path | None = None,
    batch_size: int = 100,
) -> dict:
    """Fetch OHLC for many tickers on one date — local CSV first, then batched yfinance.

    Strategy
    --------
    1. Check local stock_data CSV (instant, no network).
    2. Anything not found locally fetched in a SINGLE batched yf.download() call.

    Returns {ticker: ohlc_dict_or_None}
    """
    results: dict = {}
    need_yf: list = []

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

    start = str(target - datetime.timedelta(days=5))
    end   = str(target + datetime.timedelta(days=5))

    for i in range(0, len(need_yf), batch_size):
        batch = need_yf[i: i + batch_size]
        if not batch:
            continue

        if len(batch) == 1:
            results[batch[0]] = fetch_forward_ohlc(batch[0], target)
            continue

        try:
            raw = yf.download(
                batch, start=start, end=end,
                auto_adjust=True, progress=False,
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
                                   if not pd.isna(row.get(("Volume", ticker),
                                                           float("nan")))
                                   else None),
                    }
                except Exception:
                    results[ticker] = None

        except Exception:
            for ticker in batch:
                if results.get(ticker) is None:
                    results[ticker] = fetch_forward_ohlc(ticker, target)

    return results


def load_watchlist_scores(
    output_dir: Path,
    base_date: datetime.date,
    cap_tier: str | None = None,
) -> pd.DataFrame:
    """Load model scores from watchlist CSVs for the base date.

    Handles two file layouts:
      Old: {output_dir}/watchlist_{model}_{side}_{date}.csv
      New: {output_dir}/{date}/watchlist_{model}_{model_type}_{side}_{date}.csv

    cap_tier (None | "large" | "mid" | "small"):
      None  -> loads main bull/bear files only  (no _large/_mid/_small suffix)
      "large" -> loads watchlist_*_bull_large_{date}.csv etc.

    Normalises the 'side' column to lowercase.
    """
    date_str = str(base_date)
    search_dirs = [output_dir, output_dir / date_str]
    rows = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for side in ("bull", "bear"):
            if cap_tier:
                # e.g. watchlist_momentum_composite_bull_large_2024-08-12.csv
                pattern = f"watchlist_*_{side}_{cap_tier}_{date_str}.csv"
            else:
                # main files only: ends in _{side}_{date}.csv
                pattern = f"watchlist_*_{side}_{date_str}.csv"

            for f in sorted(search_dir.glob(pattern)):
                try:
                    df = pd.read_csv(f)
                    stem_parts = f.stem.split("_")
                    df["model"]      = stem_parts[1] if len(stem_parts) > 1 else ""
                    df["model_type"] = stem_parts[2] if len(stem_parts) > 3 else ""
                    rows.append(df)
                except Exception:
                    pass
        if rows:
            break

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    for col in ["ticker", "Ticker", "symbol", "Symbol"]:
        if col in combined.columns and col != "ticker":
            combined = combined.rename(columns={col: "ticker"})
            break

    if "side" in combined.columns:
        combined["side"] = combined["side"].str.lower()

    return combined


# ── Enrichment helpers ─────────────────────────────────────────────────────────

def _add_months(d: datetime.date, months: int) -> datetime.date:
    """Add N calendar months to a date, clamping day to month end."""
    month = d.month + months
    year  = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    days_in_month = [
        31, 28 + int(not (year % 4) and (year % 100 or not year % 400)),
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ][month - 1]
    return d.replace(year=year, month=month, day=min(d.day, days_in_month))


def load_watchlist_hf_json(hf_data_dir: Path, base_date: datetime.date) -> pd.DataFrame:
    """
    Load watchlist data from HF JSON files.

    Provides: rank, score, model_type, mode, cap_tier (inferred from tier
    sub-files), weight_pct, sdz_htf_score (when present).

    HF JSON format (bull.json / bear.json):
      [{"rank": 1, "side": "BULL", "ticker": "NVDA", "weight_pct": 3.24,
        "score": 1.0, "date": "2024-01-12", "mode": "momentum",
        "model_type": "Composite + PureML", ...}, ...]

    cap_tier is inferred: ticker in bull_large.json -> "large", etc.
    """
    date_str = str(base_date)
    date_dir = hf_data_dir / date_str
    if not date_dir.exists():
        candidates = sorted(
            hf_data_dir.glob("*/bull.json"),
            key=lambda f: f.parent.name, reverse=True
        )
        if candidates:
            date_dir = candidates[0].parent
            print(f"  HF JSON: {date_str} not found, using closest: {date_dir.name}")
        else:
            print(f"  HF JSON: no data found in {hf_data_dir}")
            return pd.DataFrame()

    main_rows = []
    for side in ("bull", "bear"):
        f = date_dir / f"{side}.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    row = dict(item)
                    row["hf_side"] = side
                    main_rows.append(row)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if not main_rows:
        print(f"  HF JSON: no bull.json / bear.json in {date_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(main_rows)

    for col in ["ticker", "Ticker", "symbol", "Symbol"]:
        if col in df.columns and col != "ticker":
            df = df.rename(columns={col: "ticker"})
            break

    # Build ticker -> cap_tier from tier sub-files
    tier_map: dict[str, str] = {}
    for side in ("bull", "bear"):
        for tier in ("large", "mid", "small", "micro"):
            tf = date_dir / f"{side}_{tier}.json"
            if not tf.exists():
                continue
            try:
                tdata = json.loads(tf.read_text(encoding="utf-8"))
                for item in tdata:
                    t = (item.get("ticker") or item.get("Ticker")
                         or item.get("symbol") or "")
                    if t and t not in tier_map:
                        tier_map[t] = tier
            except Exception:
                pass

    if tier_map:
        if "cap_tier" not in df.columns:
            df["cap_tier"] = df["ticker"].map(tier_map)
        else:
            mask = df["cap_tier"].isna()
            df.loc[mask, "cap_tier"] = df.loc[mask, "ticker"].map(tier_map)
    elif "cap_tier" not in df.columns:
        df["cap_tier"] = None

    # Rename to hf_ prefix to avoid column name clashes
    rename_hf = {
        "rank":       "hf_rank",
        "score":      "hf_score",
        "mode":       "hf_mode",
        "model_type": "hf_model_type",
        "cap_tier":   "hf_cap_tier",
        "weight_pct": "hf_weight_pct",
    }
    df = df.rename(columns={k: v for k, v in rename_hf.items()
                             if k in df.columns and k != "ticker"})

    keep = ["ticker", "hf_side", "hf_rank", "hf_score", "hf_mode",
            "hf_model_type", "hf_cap_tier", "hf_weight_pct"]
    for extra in ["sdz_htf_score", "ssz_htf_score", "adx_14"]:
        if extra in df.columns:
            keep.append(extra)
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def load_scores_detail(output_dir: Path, base_date: datetime.date) -> pd.DataFrame:
    """
    Load scores_detail JSON (produced by the scoring pipeline) for sector
    and cap_tier enrichment.

    File pattern: {output_dir}/scores_detail_momentum_{date_str}.json
    """
    date_str = str(base_date)
    rows = []
    for model in ("momentum", "reversal"):
        f = output_dir / f"scores_detail_{model}_{date_str}.json"
        if not f.exists():
            candidates = sorted(
                output_dir.glob(f"scores_detail_{model}_*.json"),
                key=lambda x: x.name, reverse=True
            )
            f = candidates[0] if candidates else None
        if f is None or not f.exists():
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                records = json.load(fh)
            df = pd.DataFrame(records)
            df["_detail_model"] = model
            rows.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f.name}: {e}")

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    if "ticker" in combined.columns:
        combined = combined.drop_duplicates(subset="ticker", keep="first")
    return combined


def fetch_benchmark_prices(
    benchmark: str,
    dates: list[datetime.date],
) -> dict[datetime.date, float | None]:
    """
    Fetch closing prices for a benchmark on multiple dates via yfinance.
    Returns {date: close_price}.
    """
    if not dates:
        return {}
    min_date = min(dates) - datetime.timedelta(days=5)
    max_date = max(dates) + datetime.timedelta(days=5)
    try:
        df = yf.Ticker(benchmark).history(
            start=str(min_date), end=str(max_date), auto_adjust=True
        )
        if df.empty:
            return {d: None for d in dates}
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index = df.index.date
    except Exception as e:
        print(f"  Warning: could not fetch benchmark {benchmark}: {e}")
        return {d: None for d in dates}

    result: dict = {}
    for target in dates:
        candidates = [d for d in df.index if d <= target]
        if not candidates:
            result[target] = None
        else:
            best = max(candidates)
            result[target] = float(df.loc[best, "Close"])
    return result


def fetch_daily_closes(
    tickers: list[str],
    start: datetime.date,
    end: datetime.date,
) -> pd.DataFrame:
    """
    Fetch daily Close prices for a list of tickers over [start, end].
    Returns a DataFrame with date as index and tickers as columns.
    Used for peak-timing analysis.
    """
    if not tickers:
        return pd.DataFrame()
    s = str(start - datetime.timedelta(days=3))
    e = str(end   + datetime.timedelta(days=3))
    try:
        if len(tickers) == 1:
            df = yf.Ticker(tickers[0]).history(start=s, end=e, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize().date
            return pd.DataFrame({tickers[0]: df["Close"]})

        raw = yf.download(tickers, start=s, end=e, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"]
        else:
            close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        close_df.index = pd.to_datetime(close_df.index).date
        return close_df
    except Exception as ex:
        print(f"  Warning: daily close fetch failed: {ex}")
        return pd.DataFrame()


def compute_peak_timing(
    close_df: pd.DataFrame,
    base_date: datetime.date,
    forward_date: datetime.date,
) -> pd.DataFrame:
    """
    For each ticker, find the date of peak Close within [base_date, forward_date].

    Returns: ticker, peak_date, peak_return_pct, peak_month (1-6), peak_period,
             peak_day_num.
    """
    rows = []
    for ticker in close_df.columns:
        series = close_df[ticker].dropna()
        in_window = [base_date <= d <= forward_date for d in series.index]
        series = series[in_window]
        if len(series) < 2:
            continue
        base_candidates = [(d, v) for d, v in zip(series.index, series.values)
                           if d >= base_date]
        if not base_candidates:
            continue
        base_val = float(base_candidates[0][1])
        if base_val == 0 or pd.isna(base_val):
            continue

        peak_idx = series.idxmax()
        peak_val = float(series[peak_idx])
        peak_ret = (peak_val - base_val) / base_val * 100
        peak_day_num = (peak_idx - base_date).days
        peak_month   = min(int(peak_day_num / 30) + 1, 6)

        if peak_month <= 2:
            peak_period = "Early (Month 1-2)"
        elif peak_month <= 4:
            peak_period = "Mid (Month 3-4)"
        else:
            peak_period = "Late (Month 5-6)"

        rows.append({
            "ticker":          ticker,
            "peak_date":       str(peak_idx),
            "peak_return_pct": round(peak_ret, 2),
            "peak_month":      peak_month,
            "peak_period":     peak_period,
            "peak_day_num":    peak_day_num,
        })
    return pd.DataFrame(rows)


def compute_peak_trough_12m(
    close_df: pd.DataFrame,
    base_date: datetime.date,
    end_date: datetime.date,
) -> pd.DataFrame:
    """
    For each ticker, scan the FULL window [base_date, end_date] (default 12 months)
    using daily closes and find:
      - the best moment  (peak_close / peak_date / peak_return_pct)
      - the worst moment (trough_close / trough_date / trough_return_pct)
      - the static end-of-window return (final_return_pct) for comparison
      - gap_pct = peak_return_pct - final_return_pct
                  ("how much was left on the table" by holding to a fixed date)

    This answers: "what was the max the price reached, and when" — rather than
    relying on a single static snapshot date, which can dramatically understate
    (or overstate) what the model's pick was actually capable of.

    Returns a DataFrame with one row per ticker.
    """
    rows = []
    for ticker in close_df.columns:
        series = close_df[ticker].dropna()
        in_window = [base_date <= d <= end_date for d in series.index]
        series = series[in_window]
        if len(series) < 2:
            continue
        base_candidates = [(d, v) for d, v in zip(series.index, series.values)
                           if d >= base_date]
        if not base_candidates:
            continue
        base_val = float(base_candidates[0][1])
        if base_val == 0 or pd.isna(base_val):
            continue

        peak_idx   = series.idxmax()
        peak_val   = float(series[peak_idx])
        trough_idx = series.idxmin()
        trough_val = float(series[trough_idx])
        final_idx  = series.index[-1]
        final_val  = float(series.iloc[-1])

        peak_ret   = (peak_val   - base_val) / base_val * 100
        trough_ret = (trough_val - base_val) / base_val * 100
        final_ret  = (final_val  - base_val) / base_val * 100

        rows.append({
            "ticker":              ticker,
            "peak_close":          round(peak_val, 2),
            "peak_date":           str(peak_idx),
            "days_to_peak":        (peak_idx - base_date).days,
            "peak_return_pct":     round(peak_ret, 2),
            "trough_close":        round(trough_val, 2),
            "trough_date":         str(trough_idx),
            "days_to_trough":      (trough_idx - base_date).days,
            "trough_return_pct":   round(trough_ret, 2),
            "final_close":         round(final_val, 2),
            "final_date":          str(final_idx),
            "final_return_pct":    round(final_ret, 2),
            "gap_pct":             round(peak_ret - final_ret, 2),
        })
    return pd.DataFrame(rows)


# ── Analysis functions ─────────────────────────────────────────────────────────

def rank_bucket_analysis(df_wl: pd.DataFrame) -> pd.DataFrame:
    """Rank bucket table: Rank 1-5, 6-15, 16-30, 31+, per side."""
    rank_col = "rank" if "rank" in df_wl.columns else (
        "hf_rank" if "hf_rank" in df_wl.columns else None
    )
    if rank_col is None or "excess_return_6m" not in df_wl.columns:
        return pd.DataFrame()

    df_wl = df_wl.copy()
    df_wl["_rank"] = pd.to_numeric(df_wl[rank_col], errors="coerce")
    df_wl = df_wl.dropna(subset=["_rank", "excess_return_6m"])

    side_col = "side" if "side" in df_wl.columns else (
        "hf_side" if "hf_side" in df_wl.columns else None
    )
    sides = df_wl[side_col].unique().tolist() if side_col else ["all"]

    rows = []
    for side in sides:
        sub_side = df_wl[df_wl[side_col] == side] if side_col else df_wl
        sign = -1 if str(side).lower() in ("bear", "short") else 1
        for label, r_min, r_max in RANK_BUCKETS:
            mask = (sub_side["_rank"] >= r_min) & (sub_side["_rank"] <= r_max)
            g = sub_side[mask]
            if g.empty:
                continue
            exc  = g["excess_return_6m"] * sign
            abs_ = g["close_pct_change"].dropna() if "close_pct_change" in g else pd.Series(dtype=float)
            rows.append({
                "Side":           str(side).upper(),
                "Rank Bucket":    label,
                "N":              len(g),
                "Avg Excess Ret": round(float(exc.mean()),   2),
                "Median Excess":  round(float(exc.median()), 2),
                "Hit Rate (>0%)": round(float((exc > 0).mean() * 100), 1),
                "Avg Abs Ret":    round(float(abs_.mean()),  2) if len(abs_) else None,
                "Best Excess":    round(float(exc.max()),    2),
                "Worst Excess":   round(float(exc.min()),    2),
            })
    return pd.DataFrame(rows)


def _side_group_analysis(
    df_wl: pd.DataFrame,
    group_col: str,
    group_label: str,
) -> pd.DataFrame:
    """Generic per-side group analysis for a categorical column."""
    if group_col not in df_wl.columns or "excess_return_6m" not in df_wl.columns:
        return pd.DataFrame()

    side_col = "side" if "side" in df_wl.columns else (
        "hf_side" if "hf_side" in df_wl.columns else None
    )
    sides = df_wl[side_col].unique().tolist() if side_col else ["all"]

    rows = []
    for side in sides:
        sub_side = df_wl[df_wl[side_col] == side] if side_col else df_wl
        sign = -1 if str(side).lower() in ("bear", "short") else 1
        for grp_val, g in sub_side.groupby(group_col, dropna=False):
            if g.empty:
                continue
            exc  = g["excess_return_6m"].dropna() * sign
            abs_ = g["close_pct_change"].dropna() if "close_pct_change" in g else pd.Series(dtype=float)
            rows.append({
                "Side":           str(side).upper(),
                group_label:      str(grp_val),
                "N":              len(g),
                "Avg Excess Ret": round(float(exc.mean()),   2) if len(exc) else None,
                "Hit Rate (>0%)": round(float((exc > 0).mean() * 100), 1) if len(exc) else None,
                "Avg Abs Ret":    round(float(abs_.mean()),  2) if len(abs_) else None,
                "Best":           round(float(exc.max()),    2) if len(exc) else None,
                "Worst":          round(float(exc.min()),    2) if len(exc) else None,
            })
    return pd.DataFrame(rows)


def mode_analysis(df_wl: pd.DataFrame) -> pd.DataFrame:
    """Momentum vs Reversal comparison."""
    col = next((c for c in ("hf_mode", "mode", "model")
                if c in df_wl.columns), None)
    if col is None:
        return pd.DataFrame()
    df_wl = df_wl.copy()
    df_wl["_mode"] = df_wl[col]
    return _side_group_analysis(df_wl, "_mode", "Mode")


def model_type_analysis(df_wl: pd.DataFrame) -> pd.DataFrame:
    """PureML vs Composite breakdown."""
    col = next((c for c in ("hf_model_type", "model_type")
                if c in df_wl.columns), None)
    if col is None:
        return pd.DataFrame()
    df_wl = df_wl.copy()
    df_wl["_mt"] = df_wl[col]
    return _side_group_analysis(df_wl, "_mt", "Model Type")


def cap_tier_analysis(df_wl: pd.DataFrame) -> pd.DataFrame:
    """Large / mid / small breakdown."""
    col = next((c for c in ("hf_cap_tier", "cap_tier")
                if c in df_wl.columns), None)
    if col is None:
        return pd.DataFrame()
    df_wl = df_wl.copy()
    df_wl["_ct"] = df_wl[col].fillna("unclassified")
    return _side_group_analysis(df_wl, "_ct", "Cap Tier")


def sdz_analysis(df_wl: pd.DataFrame) -> pd.DataFrame:
    """SDZ zone quality analysis (bull picks only)."""
    if "sdz_htf_score" not in df_wl.columns or "excess_return_6m" not in df_wl.columns:
        return pd.DataFrame()

    side_col = "side" if "side" in df_wl.columns else (
        "hf_side" if "hf_side" in df_wl.columns else None
    )
    bull = df_wl[df_wl[side_col].str.lower() == "bull"].copy() if side_col else df_wl.copy()
    if bull.empty:
        bull = df_wl.copy()

    bull["sdz_htf_score"] = pd.to_numeric(bull["sdz_htf_score"], errors="coerce")

    tiers = [
        ("Strong (>= 0.75)", bull["sdz_htf_score"] >= 0.75),
        ("Medium (0.25-0.75)", (bull["sdz_htf_score"] >= 0.25) & (bull["sdz_htf_score"] < 0.75)),
        ("Weak (< 0.25)",   bull["sdz_htf_score"] < 0.25),
        ("No SDZ (NaN)",    bull["sdz_htf_score"].isna()),
    ]
    rows = []
    for label, mask in tiers:
        g = bull[mask]
        if g.empty:
            continue
        exc = g["excess_return_6m"].dropna()
        abs_ = g["close_pct_change"].dropna() if "close_pct_change" in g else pd.Series(dtype=float)
        rows.append({
            "SDZ Tier":       label,
            "N":              len(g),
            "Avg Excess Ret": round(float(exc.mean()),   2) if len(exc) else None,
            "Hit Rate (>0%)": round(float((exc > 0).mean() * 100), 1) if len(exc) else None,
            "Avg Abs Ret":    round(float(abs_.mean()),  2) if len(abs_) else None,
        })
    return pd.DataFrame(rows)


def timing_analysis(peak_df: pd.DataFrame) -> pd.DataFrame:
    """Peak return period distribution summary."""
    if peak_df.empty or "peak_period" not in peak_df.columns:
        return pd.DataFrame()
    summary = (
        peak_df.groupby("peak_period")
        .agg(
            N=("ticker", "count"),
            Avg_Peak_Return=("peak_return_pct", "mean"),
            Median_Peak_Return=("peak_return_pct", "median"),
            Avg_Peak_Day=("peak_day_num", "mean"),
        )
        .reset_index()
    )
    summary.columns = ["Peak Period", "N", "Avg Peak Return %",
                        "Median Peak Return %", "Avg Peak Day"]
    summary["Avg Peak Return %"]    = summary["Avg Peak Return %"].round(2)
    summary["Median Peak Return %"] = summary["Median Peak Return %"].round(2)
    summary["Avg Peak Day"]         = summary["Avg Peak Day"].round(1)
    total = summary["N"].sum()
    summary["% of Picks"] = (summary["N"] / total * 100).round(1)
    return summary


def random_baseline(
    df_all: pd.DataFrame,
    bm_return: float,
    wl_tickers: set,
    n_samples: int = 5,
    sample_size: int = 30,
) -> pd.DataFrame:
    """
    Random-pick baseline: 5 samples of 30 stocks from the universe
    (excluding watchlist picks).
    """
    pool = df_all[
        df_all["close_pct_change"].notna() &
        ~df_all["ticker"].isin(wl_tickers)
    ].copy()
    pool["excess"] = pool["close_pct_change"] - bm_return
    rows = []
    np.random.seed(42)
    for i in range(n_samples):
        n = min(sample_size, len(pool))
        sample = pool.sample(n, random_state=i)
        exc = sample["excess"]
        rows.append({
            "Sample":         f"Sample {i + 1}",
            "N":              len(sample),
            "Avg Excess Ret": round(float(exc.mean()),   2),
            "Hit Rate (>0%)": round(float((exc > 0).mean() * 100), 1),
            "Avg Abs Ret":    round(float(sample["close_pct_change"].mean()), 2),
        })
    avg_exc = float(np.mean([r["Avg Excess Ret"] for r in rows]))
    avg_hit = float(np.mean([r["Hit Rate (>0%)"] for r in rows]))
    rows.append({
        "Sample":         "-- Avg of Samples --",
        "N":              sample_size,
        "Avg Excess Ret": round(avg_exc, 2),
        "Hit Rate (>0%)": round(avg_hit, 1),
        "Avg Abs Ret":    None,
    })
    return pd.DataFrame(rows)


def failed_picks(df_wl: pd.DataFrame, bm_label: str = "benchmark") -> pd.DataFrame:
    """Bull stocks that underperformed the benchmark, with flags."""
    if "excess_return_6m" not in df_wl.columns:
        return pd.DataFrame()

    side_col = "side" if "side" in df_wl.columns else (
        "hf_side" if "hf_side" in df_wl.columns else None
    )
    bull = df_wl[df_wl[side_col].str.lower() == "bull"].copy() if side_col else df_wl.copy()
    failed = bull[bull["excess_return_6m"] < 0].sort_values("excess_return_6m")
    if failed.empty:
        return pd.DataFrame()

    score_col = next((c for c in ("hf_score", "score") if c in failed.columns), None)
    sdz_col   = "sdz_htf_score" if "sdz_htf_score" in failed.columns else None

    rows = []
    for _, r in failed.iterrows():
        flags = []
        if score_col and pd.notna(r.get(score_col)) and r[score_col] < 0.70:
            flags.append("LOW-SCORE")
        if sdz_col and pd.notna(r.get(sdz_col)) and r[sdz_col] < 0.10:
            flags.append("NO-SDZ")
        rows.append({
            "ticker":           r.get("ticker", ""),
            "rank":             r.get("rank", r.get("hf_rank")),
            "score":            r.get(score_col) if score_col else None,
            "mode":             r.get("hf_mode", r.get("mode", r.get("model"))),
            "model_type":       r.get("hf_model_type", r.get("model_type")),
            "cap_tier":         r.get("hf_cap_tier", r.get("cap_tier")),
            "abs_return_6m":    r.get("close_pct_change"),
            "excess_return_6m": r.get("excess_return_6m"),
            "sdz_htf_score":    r.get(sdz_col) if sdz_col else None,
            "flags":            " | ".join(flags) if flags else "",
        })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    market: str,
    base_date: datetime.date,
    forward_date: datetime.date,
    custom_tickers: list[str] | None = None,
    hf_data_dir: Path | None = None,
    cap_tier: str | None = None,
    list_file: Path | None = None,
    simple: bool = False,
):
    cfg        = MARKET_CONFIG[market]
    data_dir   = cfg["data_dir"]
    output_dir = cfg["output_dir"]
    benchmark  = cfg["benchmark"]

    # Cap-tier overrides: swap universe list + label
    if cap_tier:
        tier_list = list_file or CAP_TIER_LISTS.get(cap_tier)
        if tier_list and Path(tier_list).exists():
            cfg = dict(cfg)
            cfg["list_file"] = Path(tier_list)
            cfg["label"]     = f"{cfg['label']} [{cap_tier.upper()}]"
    elif list_file:
        cfg = dict(cfg)
        cfg["list_file"] = Path(list_file)

    print("=" * 64)
    print(f"  Forward Performance Evaluation — {cfg['label']}")
    print(f"  Base date    : {base_date}")
    print(f"  Forward date : {forward_date}")
    print(f"  Window       : {(forward_date - base_date).days} days")
    print(f"  Benchmark    : {benchmark}")
    if cap_tier:
        print(f"  Cap tier     : {cap_tier}")
    print("=" * 64)

    # Intermediate dates
    date_4w  = base_date + datetime.timedelta(weeks=4)
    date_3m  = _add_months(base_date, 3)
    date_12m = _add_months(base_date, 12)
    print(f"  Intermediate : 4-week = {date_4w}  |  3-month = {date_3m}  |  12-month = {date_12m}")

    # ── Load ticker list ──────────────────────────────────────────────────
    if custom_tickers:
        tickers = custom_tickers
        print(f"\nUsing {len(tickers)} tickers from --tickers argument")
    else:
        list_file = cfg["list_file"]
        if not list_file.exists():
            print(f"ERROR: {list_file} not found")
            return
        df_list = pd.read_csv(list_file)
        sym_col = next(
            (c for c in df_list.columns if c.lower() in ("symbol", "ticker")), None
        )
        tickers = df_list[sym_col].dropna().str.strip().tolist()
        tickers = [t for t in tickers if t and not t.startswith("^")]
        print(f"\nLoaded {len(tickers)} tickers from {list_file.name}")

    # ── Load watchlist scores (CSV) ───────────────────────────────────────
    scores_df = load_watchlist_scores(output_dir, base_date, cap_tier=cap_tier)
    if scores_df.empty:
        print(f"  No watchlist CSVs found for {base_date} — scores will be blank")
    else:
        print(f"  Loaded {len(scores_df)} watchlist entries from CSV")

    # ── Load HF JSON enrichment (optional) ───────────────────────────────
    hf_df = pd.DataFrame()
    if hf_data_dir and Path(hf_data_dir).exists():
        print(f"\nLoading HF JSON enrichment ...")
        hf_df = load_watchlist_hf_json(Path(hf_data_dir), base_date)
        if not hf_df.empty:
            side_counts = {}
            if "hf_side" in hf_df.columns:
                side_counts = hf_df["hf_side"].value_counts().to_dict()
            print(f"  HF JSON: {len(hf_df)} entries  {side_counts}")
        else:
            print("  HF JSON: no data loaded")

    # ── Load scores_detail enrichment (sector, cap_tier) ─────────────────
    detail_df = load_scores_detail(output_dir, base_date)
    if not detail_df.empty:
        detail_cols = [c for c in ("ticker", "sector", "cap_tier")
                       if c in detail_df.columns]
        detail_df = detail_df[detail_cols]
        has_sector   = "sector"   in detail_df.columns
        has_cap_tier = "cap_tier" in detail_df.columns
        print(f"  scores_detail: {len(detail_df)} rows  "
              f"(sector={has_sector}, cap_tier={has_cap_tier})")

    # ── Fetch benchmark prices ────────────────────────────────────────────
    bench_dates = [base_date, date_4w, date_3m, forward_date]
    print(f"\nFetching {benchmark} benchmark prices ...")
    bench_prices = fetch_benchmark_prices(benchmark, bench_dates)
    b_base = bench_prices.get(base_date)
    b_4w   = bench_prices.get(date_4w)
    b_3m   = bench_prices.get(date_3m)
    b_fwd  = bench_prices.get(forward_date)

    def _bm_ret(p_start, p_end):
        if p_start and p_end and p_start != 0:
            return round((p_end - p_start) / p_start * 100, 2)
        return None

    bm_ret_6m = _bm_ret(b_base, b_fwd)
    bm_ret_4w = _bm_ret(b_base, b_4w)
    bm_ret_3m = _bm_ret(b_base, b_3m)
    print(f"  {benchmark} 4-week  : "
          f"{f'{bm_ret_4w:+.2f}%' if bm_ret_4w is not None else 'N/A'}")
    print(f"  {benchmark} 3-month : "
          f"{f'{bm_ret_3m:+.2f}%' if bm_ret_3m is not None else 'N/A'}")
    print(f"  {benchmark} 6-month : "
          f"{f'{bm_ret_6m:+.2f}%' if bm_ret_6m is not None else 'N/A'}")

    # ── Identify watchlist tickers ────────────────────────────────────────
    wl_tickers: list[str] = []
    if not scores_df.empty and "ticker" in scores_df.columns:
        wl_tickers = scores_df["ticker"].dropna().unique().tolist()
    elif not hf_df.empty and "ticker" in hf_df.columns:
        wl_tickers = hf_df["ticker"].dropna().unique().tolist()

    # ── Bulk price fetches ────────────────────────────────────────────────
    total = len(tickers)
    print(f"\nFetching base-date prices  ({base_date})  for {total} tickers ...")
    base_ohlc_map = fetch_batch_ohlc(tickers, base_date, data_dir)
    base_hit = sum(1 for v in base_ohlc_map.values() if v)
    print(f"  -> {base_hit}/{total} prices found")

    print(f"Fetching forward-date prices ({forward_date}) for {total} tickers ...")
    fwd_ohlc_map = fetch_batch_ohlc(tickers, forward_date, data_dir)
    fwd_hit = sum(1 for v in fwd_ohlc_map.values() if v)
    print(f"  -> {fwd_hit}/{total} prices found")

    # Intermediate prices — all tickers in simple mode, watchlist-only otherwise
    ohlc_4w_map:  dict = {}
    ohlc_3m_map:  dict = {}
    ohlc_12m_map: dict = {}
    inter_tickers = tickers if simple else wl_tickers
    if inter_tickers:
        lbl = "all" if simple else "watchlist"
        print(f"\nFetching 4-week prices  ({date_4w})  for "
              f"{len(inter_tickers)} {lbl} tickers ...")
        ohlc_4w_map = fetch_batch_ohlc(inter_tickers, date_4w, data_dir)
        h4w = sum(1 for v in ohlc_4w_map.values() if v)
        print(f"  -> {h4w}/{len(inter_tickers)} prices found")

        print(f"Fetching 3-month prices ({date_3m})  for "
              f"{len(inter_tickers)} {lbl} tickers ...")
        ohlc_3m_map = fetch_batch_ohlc(inter_tickers, date_3m, data_dir)
        h3m = sum(1 for v in ohlc_3m_map.values() if v)
        print(f"  -> {h3m}/{len(inter_tickers)} prices found")

        if date_12m <= datetime.date.today():
            print(f"Fetching 12-month prices ({date_12m}) for "
                  f"{len(inter_tickers)} {lbl} tickers ...")
            ohlc_12m_map = fetch_batch_ohlc(inter_tickers, date_12m, data_dir)
            h12m = sum(1 for v in ohlc_12m_map.values() if v)
            print(f"  -> {h12m}/{len(inter_tickers)} prices found")
        else:
            print(f"  12-month date {date_12m} is in the future — skipping")

    # ── Build results DataFrame ───────────────────────────────────────────
    print(f"\nBuilding results ...")
    results = []
    for i, ticker in enumerate(tickers, 1):
        base_ohlc = base_ohlc_map.get(ticker)
        fwd_ohlc  = fwd_ohlc_map.get(ticker)

        def _pct(p1, p2):
            if p1 and p2 and p1 != 0:
                return round((p2 - p1) / p1 * 100, 2)
            return None

        base_close = base_ohlc["close"] if base_ohlc else None
        fwd_close  = fwd_ohlc["close"]  if fwd_ohlc  else None
        pct_6m     = _pct(base_close, fwd_close)

        row: dict = {
            "ticker":           ticker,
            "base_date":        base_ohlc["date"]   if base_ohlc else None,
            "base_open":        base_ohlc["open"]   if base_ohlc else None,
            "base_high":        base_ohlc["high"]   if base_ohlc else None,
            "base_low":         base_ohlc["low"]    if base_ohlc else None,
            "base_close":       base_ohlc["close"]  if base_ohlc else None,
            "base_volume":      base_ohlc["volume"] if base_ohlc else None,
            "fwd_date":         fwd_ohlc["date"]    if fwd_ohlc  else None,
            "fwd_open":         fwd_ohlc["open"]    if fwd_ohlc  else None,
            "fwd_high":         fwd_ohlc["high"]    if fwd_ohlc  else None,
            "fwd_low":          fwd_ohlc["low"]     if fwd_ohlc  else None,
            "fwd_close":        fwd_ohlc["close"]   if fwd_ohlc  else None,
            "fwd_volume":       fwd_ohlc["volume"]  if fwd_ohlc  else None,
            "close_pct_change": pct_6m,
        }

        # Intermediate returns (watchlist tickers only, or all in simple mode)
        if ticker in ohlc_4w_map:
            ohlc_4w = ohlc_4w_map.get(ticker)
            row["close_4w"]      = ohlc_4w["close"] if ohlc_4w else None
            row["pct_change_4w"] = _pct(base_close, row["close_4w"])
        if ticker in ohlc_3m_map:
            ohlc_3m = ohlc_3m_map.get(ticker)
            row["close_3m"]      = ohlc_3m["close"] if ohlc_3m else None
            row["pct_change_3m"] = _pct(base_close, row["close_3m"])
        if ticker in ohlc_12m_map:
            ohlc_12m = ohlc_12m_map.get(ticker)
            row["close_12m"]      = ohlc_12m["close"] if ohlc_12m else None
            row["pct_change_12m"] = _pct(base_close, row["close_12m"])

        results.append(row)
        status = f"{pct_6m:+.1f}%" if pct_6m is not None else "no data"
        print(f"  [{i:>4}/{total}] {ticker:<20} {status}")

    if not results:
        print("No results generated.")
        return

    df_result = pd.DataFrame(results)

    # ── Merge scores & enrichment ─────────────────────────────────────────
    if not scores_df.empty:
        score_cols = [c for c in ["ticker", "rank", "score", "model", "side"]
                      if c in scores_df.columns]
        df_result = df_result.merge(scores_df[score_cols], on="ticker", how="left")

    if not hf_df.empty:
        df_result = df_result.merge(hf_df, on="ticker", how="left")

    if not detail_df.empty:
        merge_cols = ["ticker"] + [
            c for c in detail_df.columns
            if c != "ticker" and c not in df_result.columns
        ]
        if len(merge_cols) > 1:
            df_result = df_result.merge(detail_df[merge_cols], on="ticker", how="left")

    # ── Compute excess returns ────────────────────────────────────────────
    if bm_ret_6m is not None:
        df_result["excess_return_6m"] = (
            df_result["close_pct_change"] - bm_ret_6m
        ).round(2)
    if bm_ret_4w is not None and "pct_change_4w" in df_result.columns:
        df_result["excess_return_4w"] = (
            df_result["pct_change_4w"] - bm_ret_4w
        ).round(2)
    if bm_ret_3m is not None and "pct_change_3m" in df_result.columns:
        df_result["excess_return_3m"] = (
            df_result["pct_change_3m"] - bm_ret_3m
        ).round(2)

    # Sort by 6M return descending
    if "close_pct_change" in df_result.columns:
        df_result = df_result.sort_values("close_pct_change", ascending=False)

    # ── Watchlist subset for analysis ─────────────────────────────────────
    wl_mask = df_result["ticker"].isin(wl_tickers) if wl_tickers else pd.Series(
        False, index=df_result.index)
    df_wl = df_result[wl_mask].copy()

    # Ensure analysis functions have consistent side/rank columns
    if "side" not in df_wl.columns and "hf_side" in df_wl.columns:
        df_wl["side"] = df_wl["hf_side"]
    if "rank" not in df_wl.columns and "hf_rank" in df_wl.columns:
        df_wl["rank"] = df_wl["hf_rank"]

    # ── Peak / trough analysis over the FULL 12-month window ──────────────
    # A static snapshot at a fixed forward date can badly understate (or
    # overstate) what a pick was actually capable of — e.g. a stock that
    # ran +40% and gave it back by the 6M mark looks like a loser on a
    # snapshot, but the model's signal was correct; only the exit timing
    # was wrong. This scans daily closes for the whole 12M window and
    # records the best/worst moments and when they happened.
    if simple and wl_tickers and date_12m <= datetime.date.today():
        print(f"\nFetching daily price history for {len(wl_tickers)} watchlist "
              f"tickers ({base_date} -> {date_12m}) for peak/trough analysis ...")
        close_df_12m = fetch_daily_closes(wl_tickers, base_date, date_12m)
        df_peak = compute_peak_trough_12m(close_df_12m, base_date, date_12m)
        if not df_peak.empty:
            print(f"  -> peak/trough computed for {len(df_peak)}/{len(wl_tickers)} tickers")
            df_wl = df_wl.merge(df_peak, on="ticker", how="left")
        else:
            print("  -> no peak/trough data available")

    # ── Run analysis ──────────────────────────────────────────────────────
    df_rank     = rank_bucket_analysis(df_wl)
    df_mode     = mode_analysis(df_wl)
    df_mt       = model_type_analysis(df_wl)
    df_ct       = cap_tier_analysis(df_wl)
    df_sdz      = sdz_analysis(df_wl)
    df_baseline = (
        random_baseline(df_result, bm_ret_6m or 0.0, set(wl_tickers))
        if "close_pct_change" in df_result.columns else pd.DataFrame()
    )
    df_failed = failed_picks(df_wl, bm_label=benchmark)

    # ── Console summary ───────────────────────────────────────────────────
    valid = df_result["close_pct_change"].dropna()
    has_exc = "excess_return_6m" in df_result.columns
    exc_all = df_result["excess_return_6m"].dropna() if has_exc else pd.Series(dtype=float)
    exc_wl  = df_wl["excess_return_6m"].dropna() if (not df_wl.empty and has_exc) else pd.Series(dtype=float)

    print(f"\n{'='*64}")
    print(f"  SUMMARY")
    print(f"{'='*64}")
    print(f"  Tickers evaluated  : {len(df_result)}")
    print(f"  Forward fetched    : {df_result['fwd_close'].notna().sum()}")
    print(f"  Avg return (all)   : {valid.mean():.2f}%  "
          f"| Median: {valid.median():.2f}%")
    if bm_ret_6m is not None:
        print(f"  {benchmark} 6M ret   : {bm_ret_6m:+.2f}%")
    if len(exc_all):
        print(f"  Avg excess (all)   : {exc_all.mean():+.2f}%  "
              f"hit rate: {(exc_all > 0).mean()*100:.1f}%")
    if len(exc_wl):
        print(f"  Watchlist avg exc  : {exc_wl.mean():+.2f}%  "
              f"hit rate: {(exc_wl > 0).mean()*100:.1f}%  "
              f"({len(df_wl)} picks)")
    print()

    if not df_rank.empty:
        print("  Rank Bucket Summary (BULL):")
        bull_ranks = df_rank[df_rank["Side"] == "BULL"] if "Side" in df_rank.columns else df_rank
        for _, row in bull_ranks.iterrows():
            print(f"    {str(row.get('Rank Bucket','')):<15}  "
                  f"N={row.get('N',0):>3}  "
                  f"avg_exc={str(row.get('Avg Excess Ret','N/A')):>7}%  "
                  f"hit={str(row.get('Hit Rate (>0%)','N/A')):>5}%")
    print(f"{'='*64}")

    # ── Save to Excel ─────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    suffix   = f"_{custom_tickers[0]}" if custom_tickers and len(custom_tickers) <= 5 else ""
    out_file = EVAL_DIR / f"forward_eval_{market}_{base_date}_{forward_date}{suffix}.xlsx"
    if out_file.exists():
        import time as _time
        out_file = EVAL_DIR / (
            f"forward_eval_{market}_{base_date}_{forward_date}"
            f"{suffix}_{int(_time.time())}.xlsx"
        )

    # ── Metadata: read name/sector/indices from the universe list CSV ─────
    meta_df = pd.DataFrame()
    meta_list = cfg.get("list_file") or list_file
    if meta_list and Path(meta_list).exists():
        _ml = pd.read_csv(Path(meta_list))
        sym_col = next((c for c in _ml.columns if c.lower() in ("symbol", "ticker")), None)
        if sym_col:
            _ml = _ml.rename(columns={sym_col: "ticker"})
            meta_cols = ["ticker"] + [
                c for c in ("Name", "Sector", "Indices", "ETF")
                if c in _ml.columns
            ]
            meta_df = _ml[meta_cols].drop_duplicates("ticker")

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:

        if simple:
            # ── SIMPLE MODE: 2 sheets only ────────────────────────────────

            # Sheet 1 — Watchlist (bull + bear, sorted by side then rank)
            if not df_wl.empty:
                wl_out = df_wl.copy()
                # Merge name/sector/indices
                if not meta_df.empty:
                    wl_out = wl_out.merge(meta_df, on="ticker", how="left")
                rank_sort = "rank" if "rank" in wl_out.columns else None
                if rank_sort:
                    wl_out = wl_out.sort_values(
                        ["side", rank_sort], ascending=[True, True], na_position="last"
                    )
                # Select clean columns in order
                keep = ["ticker"]
                for c in ("side", "rank", "score", "model", "model_type",
                          "Name", "Sector", "Indices",
                          "base_close", "close_4w", "pct_change_4w",
                          "close_3m", "pct_change_3m",
                          "fwd_close", "close_pct_change",
                          "close_12m", "pct_change_12m",
                          # Peak / trough over the full 12M window — the
                          # "right view" of what the pick was capable of,
                          # vs. a single static snapshot date.
                          "peak_close", "peak_date", "days_to_peak", "peak_return_pct",
                          "trough_close", "trough_date", "days_to_trough", "trough_return_pct",
                          "gap_pct"):
                    if c in wl_out.columns:
                        keep.append(c)
                wl_out = wl_out[keep].rename(columns={"close_pct_change": "pct_change_6m"})
                wl_out.to_excel(writer, sheet_name="Watchlist", index=False)
            else:
                pd.DataFrame(columns=["ticker","side","rank","pct_change_4w",
                                       "pct_change_3m","pct_change_6m"]).to_excel(
                    writer, sheet_name="Watchlist", index=False)

            # Sheet 2 — All Tickers
            price_cols = [c for c in ("base_close", "close_4w", "close_3m", "fwd_close", "close_12m")
                          if c in df_result.columns]
            pct_cols_all = [c for c in ("pct_change_4w", "pct_change_3m", "close_pct_change", "pct_change_12m")
                            if c in df_result.columns]
            all_out = df_result[["ticker"] + price_cols + pct_cols_all].copy()
            all_out = all_out.rename(columns={"close_pct_change": "pct_change_6m"})
            if not meta_df.empty:
                all_out = all_out.merge(meta_df, on="ticker", how="left")
                front = ["ticker"] + [c for c in ("Name","Sector","Indices") if c in all_out.columns]
                rest  = [c for c in all_out.columns if c not in front]
                all_out = all_out[front + rest]
            all_out = all_out.sort_values("pct_change_6m", ascending=False, na_position="last")
            all_out.to_excel(writer, sheet_name="All Tickers", index=False)

        else:
            # ── FULL MODE: all 12 sheets ──────────────────────────────────

            # Sheet 1: All Tickers
            df_result.to_excel(writer, sheet_name="All Tickers", index=False)

            # Sheet 2: Summary
            exc_wl_mean = round(float(exc_wl.mean()),   2) if len(exc_wl) else None
            exc_wl_hit  = round(float((exc_wl > 0).mean() * 100), 1) if len(exc_wl) else None
            exc_all_mean = round(float(exc_all.mean()),   2) if len(exc_all) else None
            exc_all_hit  = round(float((exc_all > 0).mean() * 100), 1) if len(exc_all) else None
            stats = {
                "Metric": [
                    "Total tickers evaluated",
                    "Forward price fetched",
                    "Avg % change (all tickers)",
                    "Median % change (all tickers)",
                    "% positive (gainers)",
                    "% negative (losers)",
                    f"{benchmark} 4-week return (%)",
                    f"{benchmark} 3-month return (%)",
                    f"{benchmark} 6-month return (%)",
                    "Avg excess return — all tickers",
                    "Hit rate (>0 excess) — all tickers",
                    "Watchlist tickers (total picks)",
                    "Watchlist avg excess return (%)",
                    "Watchlist hit rate (>0 excess) (%)",
                    "Best performer",
                    "Best % change",
                    "Worst performer",
                    "Worst % change",
                ],
                "Value": [
                    len(df_result),
                    int(df_result["fwd_close"].notna().sum()),
                    round(float(valid.mean()),   2) if len(valid) else None,
                    round(float(valid.median()), 2) if len(valid) else None,
                    round(float((valid > 0).mean() * 100), 1) if len(valid) else None,
                    round(float((valid < 0).mean() * 100), 1) if len(valid) else None,
                    bm_ret_4w,
                    bm_ret_3m,
                    bm_ret_6m,
                    exc_all_mean,
                    exc_all_hit,
                    len(df_wl),
                    exc_wl_mean,
                    exc_wl_hit,
                    df_result.loc[valid.idxmax(), "ticker"] if len(valid) else "N/A",
                    round(float(valid.max()), 2) if len(valid) else None,
                    df_result.loc[valid.idxmin(), "ticker"] if len(valid) else "N/A",
                    round(float(valid.min()), 2) if len(valid) else None,
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name="Summary", index=False)

            # Sheets 3-4: Top/Bottom performers
            df_result.nlargest(50, "close_pct_change").to_excel(
                writer, sheet_name="Top 50 Gainers", index=False)
            df_result.nsmallest(50, "close_pct_change").to_excel(
                writer, sheet_name="Top 50 Losers", index=False)

            # Sheet 5: Watchlist Detail (enriched, sorted by rank)
            if not df_wl.empty:
                rank_sort_col = "rank" if "rank" in df_wl.columns else (
                    "hf_rank" if "hf_rank" in df_wl.columns else None
                )
                wl_sorted = (
                    df_wl.sort_values(rank_sort_col, ascending=True, na_position="last")
                    if rank_sort_col else df_wl
                )
                wl_sorted.to_excel(writer, sheet_name="Watchlist Detail", index=False)

            # Sheet 6: Rank Analysis
            if not df_rank.empty:
                df_rank.to_excel(writer, sheet_name="Rank Analysis", index=False)

            # Sheet 7: Mode Analysis
            if not df_mode.empty:
                df_mode.to_excel(writer, sheet_name="Mode Analysis", index=False)

            # Sheet 8: Model Type
            if not df_mt.empty:
                df_mt.to_excel(writer, sheet_name="Model Type", index=False)

            # Sheet 9: Cap Tier
            if not df_ct.empty:
                df_ct.to_excel(writer, sheet_name="Cap Tier", index=False)

            # Sheet 10: SDZ Analysis
            if not df_sdz.empty:
                df_sdz.to_excel(writer, sheet_name="SDZ Analysis", index=False)

            # Sheet 11: Random Baseline
            if not df_baseline.empty:
                df_baseline.to_excel(writer, sheet_name="Random Baseline", index=False)

            # Sheet 12: Failed Picks
            if not df_failed.empty:
                df_failed.to_excel(writer, sheet_name="Failed Picks", index=False)

    print(f"\nSaved: {out_file}")
    return out_file


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Forward performance evaluation with enrichment analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--market",       required=True, choices=["sp500", "nse"])
    p.add_argument("--base_date",    default=None,
                   help="Model scoring date YYYY-MM-DD "
                        "(default: auto-detect from latest watchlist)")
    p.add_argument("--forward_date", default=None,
                   help="Forward evaluation date (overrides --months)")
    p.add_argument("--months",       type=int, default=6,
                   help="Months ahead for forward date (default: 6)")
    p.add_argument("--tickers",      default=None,
                   help="Comma-separated ticker list, e.g. NVDA,AAPL "
                        "(overrides full market list)")
    p.add_argument("--watchlist_only", action="store_true",
                   help="Only evaluate tickers in the watchlist for this date "
                        "(much faster — skips full universe fetch).")
    p.add_argument("--output_dir",   default=None,
                   help="Override watchlist output directory "
                        "(e.g. /mnt/data/artefacts/us_local/output on Hetzner).")
    p.add_argument("--hf_data_dir",  default=None,
                   help="Path to HF JSON root, e.g. "
                        "C:/Victor/Projects/ml-stock-dashboard/public/data/us_local  "
                        "(provides model_type, mode, cap_tier enrichment).")
    p.add_argument("--cap_tier",     default=None, choices=["large", "mid", "small"],
                   help="Evaluate a single cap tier: loads watchlist_*_{side}_{cap_tier}_DATE.csv "
                        "and automatically swaps the universe to SPX / SP400 / SP600.")
    p.add_argument("--list_file",    default=None,
                   help="Override the universe CSV file "
                        "(e.g. C:/Victor/Learning_charts/stock_lists/constituents_spx.csv).")
    p.add_argument("--simple",       action="store_true",
                   help="Simple 2-sheet output: Watchlist (with sector/name) + All Tickers. "
                        "Fetches 4w/3m/6m for all tickers. No analysis sheets.")
    return p.parse_args()


def detect_latest_base_date(output_dir: Path) -> datetime.date | None:
    """Auto-detect base date from latest watchlist file."""
    files = sorted(output_dir.glob("watchlist_momentum_bull_*.csv"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None
    date_str = files[0].stem.replace("watchlist_momentum_bull_", "")
    try:
        return datetime.date.fromisoformat(date_str)
    except Exception:
        return None


if __name__ == "__main__":
    args = parse_args()
    cfg  = MARKET_CONFIG[args.market]

    if args.output_dir:
        cfg = dict(cfg)
        cfg["output_dir"] = Path(args.output_dir)

    # Resolve base date
    if args.base_date:
        base_date = datetime.date.fromisoformat(args.base_date)
    else:
        base_date = detect_latest_base_date(cfg["output_dir"])
        if base_date is None:
            print("ERROR: Could not auto-detect base date. Use --base_date YYYY-MM-DD")
            sys.exit(1)
        print(f"Auto-detected base date: {base_date}")

    # Resolve forward date
    if args.forward_date:
        forward_date = datetime.date.fromisoformat(args.forward_date)
    else:
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
        if scores_df.empty and args.hf_data_dir:
            # Try HF JSON as fallback
            hf = load_watchlist_hf_json(Path(args.hf_data_dir), base_date)
            if not hf.empty and "ticker" in hf.columns:
                custom_tickers = hf["ticker"].dropna().unique().tolist()
                print(f"--watchlist_only via HF JSON: {len(custom_tickers)} tickers")
            else:
                print("ERROR: No watchlist data found. Cannot use --watchlist_only.")
                sys.exit(1)
        elif not scores_df.empty:
            ticker_col = next(
                (c for c in scores_df.columns if c.lower() in ("ticker", "symbol")), None
            )
            if ticker_col:
                custom_tickers = scores_df[ticker_col].dropna().unique().tolist()
                print(f"--watchlist_only via CSV: {len(custom_tickers)} tickers")
            else:
                print("ERROR: No ticker column in watchlist. Cannot use --watchlist_only.")
                sys.exit(1)
        else:
            print("ERROR: No watchlist files found. Cannot use --watchlist_only.")
            sys.exit(1)

    run(
        market         = args.market,
        base_date      = base_date,
        forward_date   = forward_date,
        custom_tickers = custom_tickers,
        hf_data_dir    = Path(args.hf_data_dir) if args.hf_data_dir else None,
        cap_tier       = args.cap_tier,
        list_file      = Path(args.list_file) if args.list_file else None,
        simple         = args.simple,
    )
