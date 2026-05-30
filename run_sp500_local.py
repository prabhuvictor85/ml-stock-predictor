"""
run_sp500_local.py — End-to-end S&P 500 + NASDAQ 100 pipeline using LOCAL CSV stock data.

Universe  : S&P 500 + NASDAQ 100 combined (~540-560 unique tickers after dedup)
Benchmark : Blended SPX (40%) + NDX (60%) — reflects a tech-heavy portfolio

Reads:
  - Stock list  : C:/Victor/Learning_charts/stock_lists/constituents_us_combined.csv
                  Columns: Symbol, Name, Sector, Indices
  - Daily data  : C:/Victor/Learning_charts/stock_data/us_stocks/{TICKER}-1d.csv
                  Columns: Date, Close, High, Low, Open, Volume
  - Benchmarks  : ^GSPC-1d.csv and ^NDX-1d.csv in us_data/ (or fetched via yfinance)

Steps:
  1.  Load all local CSVs → master panel
  2.  Feature engineering (Wilder ATR/ADX, ICT, zones, multi-TF, regime)
  3.  Target building (cs_rank_20d, top_quintile, hit_target …)
  4.  Purged walk-forward CV (12-14 folds)
  5.  LGBMRanker training (momentum + reversal modes)
  6.  Isotonic calibration
  7.  Ensemble scoring on latest cross-section
  8.  Portfolio construction (top-30 stocks, equal weight)
  9.  SHAP global importance
 10.  Output: watchlist CSV + explanations JSON + HTML report

Usage:
    python run_sp500_local.py
    python run_sp500_local.py --top_n 15 --weighting inverse_vol
    python run_sp500_local.py --skip_train  (load existing artefacts and re-score)

First-time setup:
    python download_us_data.py   (downloads all stock CSVs and benchmark files)
"""
from __future__ import annotations

import sys, io
# Force UTF-8 stdout so Unicode characters don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
from pipeline.config.paths import PATHS
STOCK_LIST_CSV  = PATHS.stock_lists.us_combined
STOCK_DATA_DIR  = PATHS.stock_data.us
ARTEFACTS_DIR   = PATHS.artefacts_root / "us_local"
OUTPUT_DIR      = PATHS.artefacts_root / "us_local" / "output"
REPORTS_DIR     = PATHS.artefacts_root / "us_local" / "reports"

# ── Dual benchmark: SPX + NDX blended (tech-heavy portfolio) ───────────────
SPX_TICKER   = "^GSPC"
NDX_TICKER   = "^NDX"
SPX_WEIGHT   = 0.40   # 40% S&P 500
NDX_WEIGHT   = 0.60   # 60% NASDAQ 100 — adjust to match your portfolio mix
SPX_FILE     = STOCK_DATA_DIR / "^GSPC-1d.csv"
NDX_FILE     = STOCK_DATA_DIR / "^NDX-1d.csv"

# ── Two-ranker mode directories ─────────────────────────────────────────────
MOMENTUM_ARTEFACTS_DIR = ARTEFACTS_DIR / "momentum"
REVERSAL_ARTEFACTS_DIR = ARTEFACTS_DIR / "reversal"

# Universe filter thresholds for each mode
# momentum:  within 40% of 52w high — catches early-stage breakouts before
#            they're already near the high. Stocks at 60-85% of 52w high
#            sitting in SDZ/ICT zones are the highest-quality setups.
# reversal:  40%+ below 52w high — deep value / demand zone bounces
# No gap — every stock belongs to exactly one universe
MOMENTUM_DIST_THRESHOLD = -0.40
REVERSAL_DIST_THRESHOLD = -0.40
MIN_TRAIN_ROWS          = 1_000   # guard: refuse to train on a near-empty filtered universe

# ── Performance Timer ────────────────────────────────────────────────────────

class PerfTimer:
    """
    Lightweight stage-level performance monitor.

    Usage:
        perf = PerfTimer()
        with perf.stage("Feature engineering"):
            ...
        perf.report()          # prints summary table to console
        perf.save(log_path)    # appends timing table to the run log file
    """
    import time as _time

    def __init__(self):
        self._stages: list[dict] = []
        self._run_start = self._time.perf_counter()

    class _Stage:
        def __init__(self, timer, name):
            self._timer = timer
            self._name  = name
            self._t0    = None

        def __enter__(self):
            import time
            self._t0 = time.perf_counter()
            print(f"\n⏱  [{self._name}] starting ...", flush=True)
            return self

        def __exit__(self, *_):
            import time
            elapsed = time.perf_counter() - self._t0
            self._timer._stages.append({"stage": self._name, "seconds": elapsed})
            h, rem = divmod(int(elapsed), 3600)
            m, s   = divmod(rem, 60)
            label  = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
            print(f"⏱  [{self._name}] done — {label}", flush=True)

    def stage(self, name: str) -> "_Stage":
        return self._Stage(self, name)

    def report(self) -> None:
        import time
        total = time.perf_counter() - self._run_start
        print("\n" + "="*62)
        print("  PERFORMANCE SUMMARY")
        print("="*62)
        print(f"  {'Stage':<38} {'Time':>10}  {'%':>5}")
        print(f"  {'-'*38} {'-'*10}  {'-'*5}")
        for s in self._stages:
            secs = s["seconds"]
            pct  = secs / total * 100 if total > 0 else 0
            h, rem = divmod(int(secs), 3600)
            m, sc  = divmod(rem, 60)
            label  = f"{h}h {m:02d}m {sc:02d}s" if h else f"{m}m {sc:02d}s"
            bar    = "█" * int(pct / 5)
            print(f"  {s['stage']:<38} {label:>10}  {pct:>4.1f}%  {bar}")
        h, rem = divmod(int(total), 3600)
        m, sc  = divmod(rem, 60)
        total_label = f"{h}h {m:02d}m {sc:02d}s" if h else f"{m}m {sc:02d}s"
        print(f"  {'─'*38} {'─'*10}")
        print(f"  {'TOTAL':<38} {total_label:>10}")
        print("="*62 + "\n")

    def save(self, log_path) -> None:
        try:
            lines = ["\n" + "="*62, "  PERFORMANCE SUMMARY", "="*62]
            import time
            total = time.perf_counter() - self._run_start
            for s in self._stages:
                secs = s["seconds"]
                pct  = secs / total * 100 if total > 0 else 0
                h, rem = divmod(int(secs), 3600)
                m, sc  = divmod(rem, 60)
                label  = f"{h}h {m:02d}m {sc:02d}s" if h else f"{m}m {sc:02d}s"
                lines.append(f"  {s['stage']:<38} {label:>10}  {pct:>4.1f}%")
            h, rem = divmod(int(total), 3600)
            m, sc  = divmod(rem, 60)
            lines.append(f"  {'TOTAL':<38} {h}h {m:02d}m {sc:02d}s")
            lines.append("="*62)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass


# ── Helpers ─────────────────────────────────────────────────────────────────

def get_csv_max_date(data_dir: Path, min_ticker_pct: float = 0.80) -> Optional[pd.Timestamp]:
    """
    Scan all plain *-1d.csv files (excluding *-Drv.csv) in data_dir and
    return the latest date at which at least min_ticker_pct of tickers have data.

    Using raw MAX would return the date of whichever single ticker has the newest
    data, causing all other tickers to be missing from the scoring cross-section.
    """
    ticker_max_dates: List[pd.Timestamp] = []
    for csv_path in data_dir.glob("*-1d.csv"):
        if "Drv" in csv_path.name:
            continue
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in ("Date", "date", "datetime", "timestamp"))
            if df.empty:
                continue
            col = df.columns[0]
            dt = pd.to_datetime(df[col], errors="coerce").max()
            if pd.notna(dt):
                ticker_max_dates.append(dt)
        except Exception:
            continue

    if not ticker_max_dates:
        return None

    ticker_max_dates.sort()
    # Use the date at or below which min_ticker_pct of tickers have data.
    # e.g. with 80%: if 475/500 tickers end at 2023-12-08 and 1 ends at 2026-05-04,
    # the 80th-percentile cutoff is 2023-12-08, keeping the full cross-section.
    cutoff_idx = max(0, int(len(ticker_max_dates) * min_ticker_pct) - 1)
    majority_date = ticker_max_dates[cutoff_idx]
    abs_max = ticker_max_dates[-1]

    if abs_max != majority_date:
        outlier_count = sum(1 for d in ticker_max_dates if d > majority_date)
        print(
            f"  NOTE: {outlier_count} ticker(s) have data newer than the majority cutoff "
            f"({majority_date.date()}). Using majority date to ensure a full cross-section. "
            f"(Absolute max: {abs_max.date()})"
        )
    return majority_date


def set_seeds(seed: int = 42) -> None:
    import random, os
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US (S&P500 + NASDAQ100) Local Pipeline Runner")
    p.add_argument("--top_n",      type=int,   default=30)
    p.add_argument("--weighting",  choices=["equal", "inverse_vol"], default="equal")
    p.add_argument("--n_folds",    type=int,   default=8,
                   help="Walk-forward CV folds (8 = ~2yr test windows, faster; use 14 for exhaustive CV)")
    p.add_argument("--n_trials",   type=int,   default=25,
                   help="Optuna trials (set to 0 to skip HPO and use defaults; 25 = good balance of speed vs quality)")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training; load existing artefacts and re-score only")
    p.add_argument("--min_history_days", type=int, default=252,
                   help="Minimum trading days required per ticker")
    p.add_argument("--gpu", action="store_true",
                   help="Enable GPU acceleration for LightGBM (no effect — LightGBM GPU requires a custom CUDA build)")
    p.add_argument("--strict_data_check", action="store_true",
                   help="Treat data-staleness warnings as errors (block run on stale data). "
                        "Default: log warnings and continue.")
    p.add_argument("--max_data_lag_days", type=int, default=7,
                   help="Max calendar days since the latest bar before flagging staleness (default 7)")
    p.add_argument("--as_of", type=str, default=None,
                   help="Treat this date as 'today' for staleness checks and scoring "
                        "(format: YYYY-MM-DD). Use when running on historical data that "
                        "is intentionally capped at a past date. "
                        "Example: --as_of 2023-12-08")
    p.add_argument("--explain", type=str, default=None, metavar="TICKER",
                   help="Explain why TICKER was/wasn't selected. Uses the last run's scores. "
                        "E.g.: --explain AAPL")
    p.add_argument("--mode",
                   choices=["all", "momentum", "reversal", "legacy"],
                   default="all",
                   help=(
                       "Ranker mode(s) to run. "
                       "'all' trains and scores both momentum and reversal (default). "
                       "'momentum' = continuation plays, stocks within 40%% of 52w high. "
                       "'reversal' = demand zone bounce plays, stocks 40%%+ below 52w high. "
                       "'legacy' = original single-ranker (backward compat)."
                   ))
    p.add_argument("--train_start", type=str, default="2010-01-01",
                   help="Earliest date included in the training panel (default: 2010-01-01). "
                        "Rows before this date are dropped before walk-forward CV and HPO. "
                        "Scoring always uses the full panel. Example: --train_start 2010-01-01")
    p.add_argument("--n_jobs",     type=int,   default=1,
                   help="Parallel Optuna trials (default 1 = sequential). "
                        "Set to 4 on Hetzner CCX33 for ~3x HPO speedup. "
                        "Example: --n_jobs 4")
    return p.parse_args()


# ── 1. Load local CSV data ──────────────────────────────────────────────────

def load_local_ohlcv(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load {ticker}-1d.csv from data_dir. Returns DataFrame indexed by date."""
    path = data_dir / f"{ticker}-1d.csv"
    if not path.exists():
        return pd.DataFrame()   # legitimate skip — file simply not downloaded yet
    try:
        df = pd.read_csv(path)
    except MemoryError:
        raise MemoryError(f"Out of memory reading {path} — file may be corrupt or too large")
    except PermissionError:
        raise PermissionError(f"Permission denied reading {path} — check file locks")
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as _e:
        print(f"  WARNING [{ticker}]: CSV parse error — skipping ({_e})")
        return pd.DataFrame()   # corrupted file: skip but log visibly
    # Normalise column names to lower-case
    df.columns = [c.strip().lower() for c in df.columns]
    # Find date column
    date_col = next((c for c in df.columns if c in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        print(f"  WARNING [{ticker}]: no date column found in {path.name} — skipping")
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    # Ensure standard columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            print(f"  WARNING [{ticker}]: missing column '{col}' in {path.name} — skipping")
            return pd.DataFrame()
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(0, np.nan).dropna(subset=["close"])
    return df


# ── Zone timeframe config ──────────────────────────────────────────────────
# file suffix → weight (SDZ/SSZ get 2× base weight in engineer.py)
HTF_ZONE_FILES = {
    "1d":  1,   # daily
    "1wk": 2,   # weekly
    "1mo": 3,   # monthly
    "3mo": 4,   # quarterly
    "1y":  5,   # yearly
}


def load_htf_zones(ticker: str, data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load zone data from {ticker}-{tf}-Drv.csv files for each timeframe.
    Returns dict keyed by timeframe suffix e.g. {'1d': df, '1wk': df, ...}.

    Supports two Drv file formats:
      Format A (charting tool export): ZoneType, Zone, Proximal, Distal columns
      Format B (legacy):               zone_type, zone_valid, high/low columns
    Only rows where the validity column == 'Valid' are treated as active zones.
    """
    result: Dict[str, pd.DataFrame] = {}
    for tf in HTF_ZONE_FILES:
        path = data_dir / f"{ticker}-{tf}-Drv.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        # Lowercase columns for lookup but preserve originals
        col_lower = {c: c.strip().lower() for c in df.columns}
        df.columns = [c.strip().lower() for c in df.columns]

        date_col = next((c for c in df.columns if c in ("date", "datetime", "timestamp")), None)
        if date_col is None:
            continue

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        df.index.name = "date"

        # ── Detect format ──────────────────────────────────────────────────
        # Format A: charting tool — columns: zonetype, zone, proximal, distal
        # Format B: legacy       — columns: zone_type, zone_valid, high/low
        if "zonetype" in df.columns and "zone" in df.columns:
            # Format A
            df["zone_type"] = df["zonetype"].astype(str).str.strip().str.upper()
            df["zone_valid_flag"] = (
                df["zone"].astype(str).str.strip().str.lower() == "valid"
            ).astype(int)
            # Proximal = price closest to current (zone edge price hits first)
            # Distal   = far boundary; use as zone_high/zone_low
            if "proximal" in df.columns and "distal" in df.columns:
                prox = pd.to_numeric(df["proximal"], errors="coerce")
                dist = pd.to_numeric(df["distal"],   errors="coerce")
                df["zone_high"] = prox.combine(dist, max)
                df["zone_low"]  = prox.combine(dist, min)
        elif "zone_type" in df.columns and "zone_valid" in df.columns:
            # Format B
            df["zone_type"] = df["zone_type"].astype(str).str.strip().str.upper()
            df["zone_valid_flag"] = (
                df["zone_valid"].astype(str).str.strip().str.lower() == "valid"
            ).astype(int)
        else:
            continue

        # Invalid rows → clear zone type
        df.loc[df["zone_valid_flag"] == 0, "zone_type"] = ""
        keep = ["zone_type", "zone_valid_flag", "zone_high", "zone_low"]
        keep = [c for c in keep if c in df.columns]
        # Fallback price cols if zone_high/zone_low not derived
        for pc in ("high", "low", "open", "close"):
            if pc in df.columns and "zone_high" not in keep:
                keep.append(pc)
        result[tf] = df[keep].copy()
    return result


def merge_htf_zones_to_daily(
    daily_index: pd.DatetimeIndex,
    htf_zones: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Align each timeframe's zone_type to the daily index using merge_asof
    (carry the last known zone forward — no lookahead).
    Returns DataFrame with columns: zone_1d, zone_1wk, zone_1mo, zone_3mo, zone_1y.
    """
    daily_r = pd.DataFrame({"date": daily_index}).sort_values("date")
    out = pd.DataFrame(index=daily_index)
    out.index.name = "date"

    for tf, zone_df in htf_zones.items():
        col = f"zone_{tf}"
        if zone_df.empty:
            out[col] = ""
            continue
        z = zone_df.rename(columns={"zone_type": col}).reset_index().sort_values("date")
        if tf == "1d":
            merged = pd.merge(daily_r, z, on="date", how="left")
        else:
            merged = pd.merge_asof(daily_r, z, on="date", direction="backward")
        merged = merged.set_index("date")
        out[col] = merged[col].reindex(out.index).fillna("")

    return out


def load_benchmark(data_dir: Path) -> pd.Series:
    """Load blended SPX+NDX benchmark. Tech-heavy blend (SPX_WEIGHT/NDX_WEIGHT).

    Blending approach:
      1. Load SPX (^GSPC) and NDX (^NDX) close prices — local file first, yfinance fallback
      2. Compute daily returns for each index
      3. Blend: blend_return = SPX_WEIGHT * spx_ret + NDX_WEIGHT * ndx_ret
      4. Reconstruct price series from blended returns (base = 100)

    This gives a single benchmark that reflects a tech-heavy US portfolio rather
    than just SPX (which underweights tech) or just NDX (which ignores large-caps).
    """
    import yfinance as yf

    def _load_one(ticker: str, local_file: Path) -> pd.Series:
        if local_file.exists():
            raw_ticker = ticker.lstrip("^")
            df = load_local_ohlcv(ticker, data_dir)
            if df.empty:
                # try without caret in filename
                df = load_local_ohlcv(raw_ticker, data_dir)
            if not df.empty:
                return df["close"]
        try:
            bm = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)
            if isinstance(bm.columns, pd.MultiIndex):
                bm.columns = bm.columns.get_level_values(0)
            return bm["Close"]
        except Exception as e:
            print(f"  WARNING: Could not fetch {ticker}: {e}")
            return pd.Series(dtype=float)

    spx = _load_one(SPX_TICKER, SPX_FILE)
    ndx = _load_one(NDX_TICKER, NDX_FILE)

    if spx.empty and ndx.empty:
        print("WARNING: Both SPX and NDX unavailable — benchmark will be empty")
        return pd.Series(dtype=float)
    if spx.empty:
        print(f"  WARNING: SPX unavailable — using NDX only as benchmark")
        return ndx.rename("benchmark_close")
    if ndx.empty:
        print(f"  WARNING: NDX unavailable — using SPX only as benchmark")
        return spx.rename("benchmark_close")

    # Align dates and blend daily returns
    spx_r = spx.pct_change().dropna()
    ndx_r = ndx.pct_change().dropna()
    both  = pd.concat([spx_r.rename("spx"), ndx_r.rename("ndx")], axis=1).dropna()
    blend_r = SPX_WEIGHT * both["spx"] + NDX_WEIGHT * both["ndx"]

    # Reconstruct price index from blended returns (arbitrary base = 100)
    blend_close = (1 + blend_r).cumprod() * 100
    blend_close.name = "benchmark_close"
    print(
        f"  Blended benchmark: {SPX_WEIGHT:.0%} SPX + {NDX_WEIGHT:.0%} NDX  "
        f"({len(blend_close)} days, "
        f"{blend_close.index.min().date()} → {blend_close.index.max().date()})"
    )
    return blend_close


def build_panel_from_local(
    tickers: List[str],
    data_dir: Path,
    min_history_days: int = 252,
    sector_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load all local CSVs, compute adv_20d_usd (INR proxy — volume×close),
    set in_universe=True for tickers with enough history, assign group_date.
    Uses ThreadPoolExecutor for parallel CSV I/O.
    """
    from pipeline.config.nse import NSE_CONFIG
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cfg = NSE_CONFIG
    _sector_map = sector_map or {}

    print(f"Loading {len(tickers)} local CSV files (parallel I/O) ...")

    def _load_one(ticker: str):
        df = load_local_ohlcv(ticker, data_dir)
        if df.empty or len(df) < min_history_days:
            return None
        df = df.copy()
        htf_zones = load_htf_zones(ticker, data_dir)
        if htf_zones:
            zone_cols = merge_htf_zones_to_daily(df.index, htf_zones)
            df = df.join(zone_cols, how="left")
            for zcol in zone_cols.columns:
                df[zcol] = df[zcol].fillna("")
        df["ticker"] = ticker
        dollar_vol = df["volume"] * df["close"]
        df["adv_20d_usd"] = dollar_vol.rolling(20, min_periods=10).mean()
        df["market_cap_usd"] = np.nan
        df["sector"] = _sector_map.get(ticker, "US")
        df["in_universe"] = True
        return df

    frames: List[pd.DataFrame] = []
    skipped_history = 0   # < min_history_days or file not found
    skipped_errors:  List[str] = []   # files that existed but failed to parse
    n_workers = min(16, len(tickers))   # cap at 16 threads (I/O bound)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_one, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            ticker = futures[fut]
            try:
                result = fut.result()
            except Exception as _e:
                # MemoryError / PermissionError propagate here — re-raise so the run aborts
                raise RuntimeError(f"Fatal error loading {ticker}: {_e}") from _e
            if result is None:
                skipped_history += 1
            else:
                frames.append(result)
            if done % 100 == 0:
                print(f"  Loaded {len(frames)} tickers so far ({done}/{len(tickers)})...")

    if not frames:
        raise RuntimeError("No valid ticker data loaded.")

    print(f"  Loaded {len(frames)} tickers, skipped {skipped_history} (no file / < {min_history_days} days)")

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()

    # group_date = first day of each calendar month (monthly gives ~60 stocks/group
    # vs ~3 for weekly in early folds — Fix 1)
    dates = panel.index.get_level_values("date").to_series().reset_index(drop=True)
    group_dates = dates.dt.to_period("M").dt.to_timestamp()
    panel["group_date"] = group_dates.values

    print(f"  Panel shape: {panel.shape}  |  date range: "
          f"{panel.index.get_level_values('date').min().date()} -> "
          f"{panel.index.get_level_values('date').max().date()}")
    return panel


# ── 2. Train pipeline ────────────────────────────────────────────────────────

def train(panel: pd.DataFrame, benchmark_close: pd.Series,
          cfg, n_folds: int, n_trials: int, top_n: int,
          use_gpu: bool = False,
          mode: str = "legacy",
          mode_artefacts_dir: Optional[Path] = None,
          n_jobs: int = 1,
          as_of: Optional[str] = None) -> dict:
    """Full train: features → targets → CV → Optuna → models → calibration.

    mode: "legacy" | "momentum" | "reversal"
      legacy   — original single-ranker, no universe filter
      momentum — trains only on stocks within 40% of their 52w high
      reversal — trains only on stocks 40%+ below their 52w high
    mode_artefacts_dir: where to save mode-specific artefacts (ensemble, ranker, etc.)
      Defaults to ARTEFACTS_DIR for legacy mode.
    """
    import sys; sys.path.insert(0, ".")
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
    from pipeline.targets.builder import TargetBuilder
    from pipeline.validation.cv import PurgedWalkForwardCV
    from pipeline.validation.leakage_tests import LeakageTestSuite
    from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
    from pipeline.models.ensemble import EnsembleRanker
    from pipeline.selection.selector import FeatureSelector
    from pipeline.validation.metrics import compute_fold_metrics, ndcg_at_k
    from pipeline.monitoring.drift_monitor import FeatureDriftMonitor

    # Mode-specific artefacts dir (shared checkpoints always stay in ARTEFACTS_DIR)
    _art_dir = mode_artefacts_dir if mode_artefacts_dir is not None else ARTEFACTS_DIR
    _art_dir.mkdir(parents=True, exist_ok=True)

    ARTEFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT = ARTEFACTS_DIR / "checkpoints"   # shared — never mode-specific
    CKPT.mkdir(parents=True, exist_ok=True)

    # FeatureEngineer is always instantiated — needed for per-fold zone recompute
    fe = FeatureEngineer(cfg, benchmark_close)
    _train_perf = PerfTimer()

    # ── Checkpoint 1: Feature-engineered panel ────────────────────────────
    panel_ckpt = CKPT / "panel_features.pkl"
    feat_cols_ckpt = CKPT / "feat_cols.txt"

    with _train_perf.stage("[1/6] Feature engineering"):
        if panel_ckpt.exists() and feat_cols_ckpt.exists():
            print("\n[1/6] Feature engineering ... RESUMING from checkpoint")
            with open(panel_ckpt, "rb") as f:
                panel = pickle.load(f)
            feat_cols = feat_cols_ckpt.read_text().strip().split("\n")
            print(f"      Loaded checkpoint: {len(feat_cols)} features, panel shape {panel.shape}")
        else:
            print("\n[1/6] Feature engineering ...")
            panel = fe.build(panel)
            feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
            print(f"      {len(feat_cols)} feature columns — saving checkpoint ...")
            with open(panel_ckpt, "wb") as f:
                pickle.dump(panel, f)
            feat_cols_ckpt.write_text("\n".join(feat_cols))
            print(f"      Checkpoint saved: {panel_ckpt}")

    # ── Checkpoint 2: Targets ─────────────────────────────────────────────
    targets_ckpt = CKPT / "panel_targets.pkl"

    with _train_perf.stage("[2/6] Target building"):
        if targets_ckpt.exists():
            print("[2/6] Building targets ... RESUMING from checkpoint")
            with open(targets_ckpt, "rb") as f:
                panel = pickle.load(f)
            print(f"      cs_rank_20d non-null: {panel['cs_rank_20d'].notna().sum()}")
        else:
            print("[2/6] Building targets ...")
            tb = TargetBuilder(cfg)
            panel = tb.build(panel, benchmark_close)
            print(f"      cs_rank_20d non-null: {panel['cs_rank_20d'].notna().sum()} — saving checkpoint ...")
            with open(targets_ckpt, "wb") as f:
                pickle.dump(panel, f)
            print(f"      Checkpoint saved: {targets_ckpt}")

    # Leakage tests — run AFTER targets are built so cs_rank_20d is present
    suite = LeakageTestSuite(panel, feat_cols)
    suite.run_all()

    # ── Universe filter: restrict training rows based on mode ─────────────
    # Feature engineering runs on the full panel (shared checkpoint).
    # The mode filter narrows which stocks the ranker learns from.
    # full `panel` is kept unfiltered — returned for scoring all stocks.
    _dist_col = f"{FEATURE_PREFIX}high_52w_dist"
    if mode == "momentum" and _dist_col in panel.columns:
        _mask = panel[_dist_col] > MOMENTUM_DIST_THRESHOLD
        train_panel = panel[_mask].copy()
        n_tickers = train_panel.index.get_level_values("ticker").nunique()
        print(f"      [mode=momentum] within 40% of 52w high ({_dist_col} > {MOMENTUM_DIST_THRESHOLD}): "
              f"{len(train_panel):,} rows / {n_tickers} tickers kept of {len(panel):,}")
    elif mode == "reversal" and _dist_col in panel.columns:
        _mask = panel[_dist_col] <= REVERSAL_DIST_THRESHOLD
        train_panel = panel[_mask].copy()
        n_tickers = train_panel.index.get_level_values("ticker").nunique()
        print(f"      [mode=reversal] 40%+ below 52w high ({_dist_col} <= {REVERSAL_DIST_THRESHOLD}): "
              f"{len(train_panel):,} rows / {n_tickers} tickers kept of {len(panel):,}")
    else:
        train_panel = panel  # legacy: no filter

    if len(train_panel) < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"[mode={mode}] filtered training set has only {len(train_panel)} rows "
            f"(minimum: {MIN_TRAIN_ROWS}). Universe filter may be too restrictive "
            f"for the current data period."
        )

    with _train_perf.stage("[3/6] Walk-forward CV setup"):
        print(f"[3/6] Walk-forward CV ({n_folds} folds) ...")
        cv = PurgedWalkForwardCV(n_folds=n_folds, min_train_window=378)
        fold_specs = cv.get_fold_specs(train_panel)
    print(f"      {len(fold_specs)} folds generated")

    # ── Compute num_threads per LGBMRanker ──────────────────────────────────
    import os as _os
    _cpu_count = _os.cpu_count() or 8
    _lgbm_threads = max(1, _cpu_count // n_jobs)
    if n_jobs > 1:
        print(f"      Thread allocation: {n_jobs} parallel trials × "
              f"{_lgbm_threads} threads each = {n_jobs * _lgbm_threads}/{_cpu_count} cores")

    # ── Optuna HPO ──────────────────────────────────────────────────────────
    best_params = {
        "num_leaves": 63, "learning_rate": 0.05,
        "n_estimators": 500, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    }
    best_k = 30

    _hpo_stage = _train_perf.stage("[4/6] Optuna HPO")
    _hpo_stage.__enter__()
    if n_trials > 0:
        print(f"[4/6] Optuna HPO ({n_trials} trials) ...")
        import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ── SQLite storage: HPO trials persist across crashes / restarts ──
        _hpo_ckpt = _art_dir / "checkpoints"
        _hpo_ckpt.mkdir(parents=True, exist_ok=True)
        hpo_db    = _hpo_ckpt / "optuna_hpo.db"
        storage   = f"sqlite:///{hpo_db.as_posix()}"
        study_name = f"us_local_hpo_{mode}"
        print(f"      HPO progress saved to: {hpo_db}")
        print(f"      (delete {hpo_db.name} to restart HPO from scratch)")

        # ── Fix 4: Pre-select features ONCE before HPO — not per trial ───
        full_grp_hpo, _ = cv.build_group_array(train_panel, min_group_size=5)
        avail_hpo = [f for f in feat_cols if f in full_grp_hpo.columns]
        _cls_col_hpo = "bot_quintile" if mode == "reversal" else "top_quintile"
        cls_mask_hpo = full_grp_hpo[_cls_col_hpo].notna()
        X_hpo_cls = full_grp_hpo.loc[cls_mask_hpo, avail_hpo]
        y_hpo_cls = full_grp_hpo.loc[cls_mask_hpo, _cls_col_hpo].astype(int)
        _pre_sel = FeatureSelector(seed=cfg.random_seed)
        pre_selected = _pre_sel.select(X_hpo_cls, y_hpo_cls, top_k=50)  # generous pool
        print(f"      Pre-selected {len(pre_selected)} features for HPO")
        del full_grp_hpo, X_hpo_cls, y_hpo_cls  # free RAM

        MIN_VALID_FOLDS = 3   # Fix 3: need at least this many real folds per trial

        import time as _time

        # ── Pre-build fold cache: zone recompute runs ONCE per fold, not per trial ──
        # Zones depend only on price history up to the cutoff date — they are
        # identical for every Optuna trial on the same fold. Building them once
        # and reusing across all 50 trials gives a ~50x speedup on zone I/O.
        print("      Pre-computing fold zones (runs once, reused across all trials) ...")
        t_cache = _time.time()
        fold_cache = {}   # fold_id -> {"tr_grp", "tr_groups", "te_univ"}

        # ── Disk-backed fold cache ────────────────────────────────────────────
        FOLD_CACHE_DIR = _art_dir / "fold_cache"
        FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _as_of_str = as_of or pd.Timestamp.now().strftime("%Y-%m-%d")

        def _fold_cache_path(fold_id: int) -> "Path":
            return FOLD_CACHE_DIR / f"fold_{fold_id}.pkl"

        def _load_fold_cache(fold_id: int, spec) -> dict | None:
            p = _fold_cache_path(fold_id)
            if not p.exists():
                return None
            try:
                with open(p, "rb") as f:
                    cached = pickle.load(f)
                if (cached.get("train_end") == spec.train_end and
                        cached.get("test_end")  == spec.test_end and
                        cached.get("as_of")     == _as_of_str):
                    return cached["data"]
            except Exception:
                pass
            return None

        def _save_fold_cache(fold_id: int, spec, data: dict) -> None:
            try:
                with open(_fold_cache_path(fold_id), "wb") as f:
                    pickle.dump({
                        "train_end": spec.train_end,
                        "test_end":  spec.test_end,
                        "as_of":     _as_of_str,
                        "data":      data,
                    }, f, protocol=4)
            except Exception as e:
                print(f"        [warn] Could not save fold {fold_id} cache: {e}", flush=True)

        import gc as _gc
        for spec, tr_idx, te_idx in cv.split(train_panel):
            # ── Try loading from disk first ───────────────────────────────
            disk_hit = _load_fold_cache(spec.fold_id, spec)
            if disk_hit is not None:
                fold_cache[spec.fold_id] = True   # marker only — data stays on disk
                print(
                    f"        Fold {spec.fold_id}: loaded from disk  "
                    f"train={len(disk_hit['tr_grp']):,}rows/{len(disk_hit['tr_groups'])}groups "
                    f"test={len(disk_hit['te_univ']):,}",
                    flush=True,
                )
                del disk_hit
                _gc.collect()
                continue

            # ── Compute and save ──────────────────────────────────────────
            tr = train_panel.iloc[tr_idx]
            if len(tr) == 0:
                fold_cache[spec.fold_id] = None
                continue

            fold_cutoff = train_panel.index.get_level_values("date")[tr_idx].max()
            tr = fe.recompute_fold_features(tr, cutoff_date=fold_cutoff)
            tr_grp, tr_groups = cv.build_group_array(tr, min_group_size=5)
            del tr
            _gc.collect()

            if len(tr_grp) == 0:
                fold_cache[spec.fold_id] = None
                continue

            avg_group_size = len(tr_grp) / max(len(tr_groups), 1)
            if avg_group_size < 10:
                fold_cache[spec.fold_id] = None
                print(f"        Fold {spec.fold_id}: avg_group={avg_group_size:.1f} < 10 — skipping", flush=True)
                continue

            te_panel = train_panel.iloc[te_idx]
            te_panel = fe.recompute_fold_features(te_panel, cutoff_date=spec.test_end)
            te_univ  = te_panel[te_panel["in_universe"] == True]
            del te_panel
            _gc.collect()

            if len(te_univ) < 5:
                fold_cache[spec.fold_id] = None
                print(f"        Fold {spec.fold_id}: test universe too small ({len(te_univ)}) — skipping", flush=True)
                continue

            fold_data = {
                "tr_grp":    tr_grp,
                "tr_groups": tr_groups,
                "te_univ":   te_univ,
            }
            _save_fold_cache(spec.fold_id, spec, fold_data)
            fold_cache[spec.fold_id] = True   # marker only — data saved to disk, free RAM
            del fold_data, tr_grp, tr_groups, te_univ
            _gc.collect()
            print(
                f"        Fold {spec.fold_id}: cached to disk",
                flush=True,
            )

        valid_fold_ids = [fid for fid, v in fold_cache.items() if v is True]
        print(
            f"      Fold cache built: {len(valid_fold_ids)}/{len(fold_cache)} valid folds "
            f"in {_time.time()-t_cache:.0f}s",
            flush=True,
        )

        # Free in-memory fold data — trials will load from disk on demand.
        import gc
        fold_cache.clear()
        gc.collect()
        print("      Fold data cleared from RAM — trials will load per-fold from disk.")

        def objective(trial):
            top_k    = trial.suggest_categorical("feature_top_K", [20, 30, 40, 50])
            sf_trial = pre_selected[:top_k]

            params = {
                "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
                "learning_rate":     trial.suggest_float("lr", 0.005, 0.2, log=True),
                "n_estimators":      trial.suggest_int("n_estimators", 100, 250),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            print(
                f"\n  --- Trial {trial.number} ---\n"
                f"     features={top_k} | leaves={params['num_leaves']} | "
                f"lr={params['learning_rate']:.4f} | n_est={params['n_estimators']} | "
                f"min_child={params['min_child_samples']} | "
                f"subsample={params['subsample']:.2f} | lambda={params['reg_lambda']:.3f}",
                flush=True,
            )
            t_trial_start = _time.time()
            ndcg_vals, top_dec_vals = [], []

            for fold_id in valid_fold_ids:
                cached = _load_fold_cache(fold_id, cv.fold_specs[fold_id - 1])
                if cached is None:
                    continue

                tr_grp    = cached["tr_grp"]
                tr_groups = cached["tr_groups"]
                te_univ   = cached["te_univ"]

                sf_te = [f for f in sf_trial if f in tr_grp.columns]
                if not sf_te:
                    continue
                sf_te2 = [f for f in sf_te if f in te_univ.columns]
                if not sf_te2:
                    continue

                _rank = tr_grp["cs_rank_composite"].fillna(0)
                y_tr_r = 1.0 - _rank if mode == "reversal" else _rank

                t_fold = _time.time()

                # ── Early-stopping val split (last 20% of training dates) ─────
                # Keeps early stopping honest — never touches the held-out test fold.
                _tr_date_lvl  = tr_grp.index.get_level_values("date")
                _tr_uniq      = sorted(_tr_date_lvl.unique())
                _n_val_d      = max(5, len(_tr_uniq) // 5)
                _val_date_set = set(_tr_uniq[-_n_val_d:])
                _es_tr_mask   = ~_tr_date_lvl.isin(_val_date_set)
                _es_vl_mask   = _tr_date_lvl.isin(_val_date_set)
                _use_es       = (_es_vl_mask.sum() >= 20) and (_es_tr_mask.sum() >= 20)

                ranker = LGBMRanker(params, seed=cfg.random_seed, num_threads=_lgbm_threads)
                if _use_es:
                    X_es_tr  = tr_grp.loc[_es_tr_mask, sf_te].fillna(0)
                    X_es_val = tr_grp.loc[_es_vl_mask, sf_te].fillna(0)
                    y_es_tr  = y_tr_r[_es_tr_mask]
                    y_es_val = y_tr_r[_es_vl_mask]
                    g_es_tr  = X_es_tr.groupby(level="date").size().values
                    g_es_val = X_es_val.groupby(level="date").size().values
                    ranker.fit(X_es_tr, y_es_tr, g_es_tr,
                               X_val=X_es_val, y_val=y_es_val, group_val=g_es_val)
                else:
                    ranker.fit(tr_grp[sf_te].fillna(0), y_tr_r, tr_groups)

                n_trees = ranker.model_.num_trees()
                if n_trees == 0:
                    print(f"     Fold {fold_id}: ranker stalled (0 trees) — skipping", flush=True)
                    continue

                scores = pd.Series(ranker.predict(te_univ[sf_te2].fillna(0)), index=te_univ.index)
                m = compute_fold_metrics(te_univ, scores, sf_te2,
                                         benchmark_close.pct_change().fillna(0),
                                         cfg.commission_bps, cfg.get_slippage_bps(cfg.min_adv_usd),
                                         invert_relevance=(mode == "reversal"))
                fold_ndcg = m["mean_ndcg_at_10"]
                fold_exc  = m["top_decile_excess_return"]
                ndcg_vals.append(fold_ndcg)
                top_dec_vals.append(fold_exc)

                print(
                    f"     Fold {fold_id}: train={len(tr_grp):,}rows/{len(tr_groups)}groups "
                    f"| test={len(te_univ):,} | trees={n_trees} "
                    f"| NDCG@10={fold_ndcg:.4f} | TopDec_exc={fold_exc:+.4f} "
                    f"| fold_time={_time.time()-t_fold:.0f}s",
                    flush=True,
                )

                # Free fold data immediately after use
                del cached, tr_grp, tr_groups, te_univ
                gc.collect()

                trial.report(fold_ndcg, step=fold_id)
                if trial.should_prune():
                    print(f"     Trial {trial.number}: PRUNED after fold {fold_id}", flush=True)
                    raise optuna.exceptions.TrialPruned()

            if len(ndcg_vals) < MIN_VALID_FOLDS:
                print(f"     Trial {trial.number}: only {len(ndcg_vals)} valid folds (<{MIN_VALID_FOLDS}) — pruned", flush=True)
                raise optuna.exceptions.TrialPruned()
            if np.mean(top_dec_vals) <= 0:
                print(f"     Trial {trial.number}: mean top-decile excess={np.mean(top_dec_vals):.4f} <= 0 — pruned", flush=True)
                raise optuna.exceptions.TrialPruned()

            score = float(np.mean(ndcg_vals)) - 0.5 * float(np.std(ndcg_vals))
            print(
                f"  --- Trial {trial.number} COMPLETE | "
                f"mean_NDCG={np.mean(ndcg_vals):.4f} +/- {np.std(ndcg_vals):.4f} | "
                f"objective={score:.4f} | total_time={_time.time()-t_trial_start:.0f}s",
                flush=True,
            )
            return score

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,        # ← resumes from where it left off
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        )
        completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
        remaining = max(0, n_trials - completed)
        if completed > 0:
            print(f"      Resuming HPO: {completed} trials already done, {remaining} remaining")
        if n_jobs > 1:
            print(f"      Running {n_jobs} parallel trials (n_jobs={n_jobs})")
        study.optimize(objective, n_trials=remaining, n_jobs=n_jobs,
                       show_progress_bar=True)
        best_params = study.best_params.copy()
        best_k = best_params.pop("feature_top_K", 30)
        best_params["learning_rate"] = best_params.pop("lr", best_params.get("learning_rate", 0.05))
        print(f"      Best NDCG objective: {study.best_value:.4f}")
    else:
        print("[4/6] HPO skipped — using default params")
    _hpo_stage.__exit__(None, None, None)

    # Free fold cache memory before final training — it holds ~2-3 GB on large panels
    import gc
    if 'fold_cache' in dir():
        del fold_cache
    gc.collect()
    print(f"      RAM freed after HPO. Starting final training ...")

    with _train_perf.stage("[5/6] Final model training"):
        print(f"[5/6] Training final LightGBM ranker [{mode}] ...")
    full_grp, full_groups = cv.build_group_array(train_panel, min_group_size=5)
    avail    = [f for f in feat_cols if f in full_grp.columns]
    # Downcast to float32 to halve memory usage on large panels
    X_full   = full_grp[avail].astype(np.float32)
    _rank_raw = full_grp["cs_rank_composite"].fillna(0)
    # Reversal mode trains a bear ranker: invert rank so stocks expected to
    # decline most get the highest label, mirroring how the bull ranker works.
    y_full_r  = 1.0 - _rank_raw if mode == "reversal" else _rank_raw

    # FeatureSelector uses top_quintile (bull) or bot_quintile (reversal) as a
    # binary proxy. Reversal mode inverts so the selector learns features that
    # predict declines, not advances.
    _cls_col  = "bot_quintile" if mode == "reversal" else "top_quintile"
    cls_mask     = full_grp[_cls_col].notna()
    X_full_cls   = full_grp.loc[cls_mask, avail]
    y_full_c_cls = full_grp.loc[cls_mask, _cls_col].astype(int)
    if int(y_full_c_cls.sum()) == 0:
        raise RuntimeError(
            f"{_cls_col} has zero positive labels. Check targets were built correctly "
            "and the panel covers at least 20+ trading days before the most recent date."
        )

    sel = FeatureSelector(seed=cfg.random_seed)
    final_features = sel.select(X_full_cls, y_full_c_cls, top_k=best_k)
    print(f"      Selected {len(final_features)} features")

    X_fin = X_full[final_features].fillna(0)
    final_ranker = LGBMRanker(best_params, seed=cfg.random_seed, num_threads=-1)  # final model uses all cores
    final_ranker.fit(X_fin, y_full_r, full_groups)

    ensemble = EnsembleRanker(final_ranker)

    # ── Ensemble walk-forward CV validation ────────────────────────────────
    # HPO tuned LGBM hyperparameters; this validates the FULL ensemble (LGBM + inv-vol)
    # on each held-out fold so we have honest out-of-sample combined metrics.
    print("\n      [Ensemble CV] Validating ensemble on each held-out fold ...")
    bm_rets     = benchmark_close.pct_change().fillna(0)
    slip        = cfg.get_slippage_bps(cfg.min_adv_usd)
    dates_level = train_panel.index.get_level_values("date")
    ens_ndcg_vals, ens_top_dec_vals = [], []
    _hv_feat = f"{FEATURE_PREFIX}hist_vol_20d"

    for _spec in fold_specs:
        _te_mask = (dates_level >= _spec.test_start) & (dates_level <= _spec.test_end)
        _te_p    = train_panel.iloc[np.where(_te_mask)[0]]
        _te_u    = _te_p[_te_p["in_universe"] == True]
        _sf      = [f for f in final_features if f in _te_u.columns]
        if len(_te_u) < 5 or not _sf:
            continue
        _X_te  = _te_u[_sf].fillna(0)
        _vol_s = _te_u[_hv_feat] if _hv_feat in _te_u.columns else None
        _ens_s = pd.Series(ensemble.score(_X_te, _vol_s), index=_te_u.index)
        _m     = compute_fold_metrics(_te_u, _ens_s, _sf, bm_rets, cfg.commission_bps, slip,
                                      invert_relevance=(mode == "reversal"))
        ens_ndcg_vals.append(_m["mean_ndcg_at_10"])
        ens_top_dec_vals.append(_m["top_decile_excess_return"])

    if ens_ndcg_vals:
        print(f"      Ensemble CV  ({len(ens_ndcg_vals)} folds): "
              f"NDCG@10={np.mean(ens_ndcg_vals):.4f} ± {np.std(ens_ndcg_vals):.4f}  |  "
              f"top-decile excess={np.mean(ens_top_dec_vals)*100:.2f}%")

    # Drift monitor baseline
    drift_monitor = FeatureDriftMonitor(cfg, final_features)
    drift_monitor.fit_baseline(full_grp[final_features])

    with _train_perf.stage("[6/6] Save artefacts"):
        print("[6/6] Saving artefacts ...")
        _art_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [
            ("ensemble.pkl",      ensemble),
            ("lgbm_ranker.pkl",   final_ranker),
            ("drift_monitor.pkl", drift_monitor),
            ("panel.pkl",         panel),   # full unfiltered panel for scoring
        ]:
            with open(_art_dir / name, "wb") as f:
                pickle.dump(obj, f)
        (_art_dir / "selected_features.txt").write_text("\n".join(final_features))
        print(f"      Artefacts saved to {_art_dir}")

    _train_perf.report()

    return {
        "panel":             panel,       # full unfiltered — for scoring all stocks
        "ensemble":          ensemble,
        "final_features":    final_features,
        "drift_monitor":     drift_monitor,
        "benchmark_close":   benchmark_close,
        "feat_cols":         feat_cols,
        "mode":              mode,
        "mode_artefacts_dir": _art_dir,
    }


# ── 3. Score & rank ──────────────────────────────────────────────────────────

def score_and_rank(panel: pd.DataFrame, ensemble, final_features: List[str],
                   benchmark_close: pd.Series, cfg, top_n: int, weighting: str,
                   as_of_date: Optional[pd.Timestamp] = None,
                   mode: str = "legacy",
                   variant: str = "composite") -> dict:
    """Score the latest cross-section and build bull + bear portfolios.

    mode controls which stocks are eligible for the watchlist:
      legacy   — all stocks (original behaviour)
      momentum — only stocks within 40% of their 52w high
      reversal — only stocks 40%+ below their 52w high
    All stocks are scored regardless — the filter only affects watchlist selection.
    """
    from pipeline.portfolio.constructor import PortfolioConstructor
    from pipeline.explainability.shap_explainer import SHAPExplainer
    from pipeline.features.engineer import FEATURE_PREFIX

    # Pin to the max date present in the CSVs (as_of_date), falling back to
    # the max date in the panel if not provided.
    panel_max = panel.index.get_level_values("date").max()
    if as_of_date is not None:
        # Find the latest panel date that is <= as_of_date
        avail_dates = panel.index.get_level_values("date").unique().sort_values()
        candidates = avail_dates[avail_dates <= as_of_date]
        latest_date = candidates[-1] if len(candidates) > 0 else panel_max
    else:
        latest_date = panel_max
    print(f"\nScoring latest cross-section: {latest_date.date()}"
          + (f"  (CSV max date: {as_of_date.date()})" if as_of_date is not None else ""))

    cross = panel.xs(latest_date, level="date").copy()
    cross = cross[cross["in_universe"] == True]
    if cross.empty:
        dates = sorted(panel.index.get_level_values("date").unique())
        for d in reversed(dates[-6:]):
            cross = panel.xs(d, level="date").copy()
            cross = cross[cross["in_universe"] == True]
            if not cross.empty:
                latest_date = d
                print(f"  Fell back to date: {latest_date.date()}")
                break

    cross.index = pd.MultiIndex.from_arrays(
        [[latest_date] * len(cross), cross.index], names=["date", "ticker"]
    )

    avail   = [f for f in final_features if f in cross.columns]
    X_inf   = cross[avail].fillna(0)
    # Use HISTORICAL realized vol for inverse-vol weighting — future_vol_20d is a
    # forward-looking target (always NaN at inference time on the latest date).
    _hist_vol_feat = f"{FEATURE_PREFIX}hist_vol_20d"
    vol_col = _hist_vol_feat if _hist_vol_feat in cross.columns else None
    vol_series = cross[vol_col] if vol_col else None

    # ── Raw model score (0→1, higher = more bullish) ──────────────────────
    model_scores = ensemble.score(X_inf, vol_series)
    model_series = pd.Series(model_scores, index=cross.index)

    FP = FEATURE_PREFIX   # shorthand

    # ── Load signal weights from YAML (editable without code changes) ─────
    _weights_path = Path(__file__).parent / "signal_weights.yaml"
    import yaml
    try:
        with open(_weights_path, encoding="utf-8") as _f:
            _wcfg = yaml.safe_load(_f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"signal_weights.yaml not found at {_weights_path}. "
            f"This file is required — restore it from version control."
        )
    except yaml.YAMLError as _e:
        raise ValueError(
            f"signal_weights.yaml has a syntax error — fix it before running.\n{_e}"
        )

    # Mode-specific scoring weights fall back to base "scoring" section
    if mode == "momentum":
        _scoring = _wcfg.get("momentum_scoring", _wcfg.get("scoring", {}))
    elif mode == "reversal":
        _scoring = _wcfg.get("reversal_scoring", _wcfg.get("scoring", {}))
    else:
        _scoring = _wcfg.get("scoring", {})
    model_w      = float(_scoring.get("model_weight",     0.6))
    composite_w  = float(_scoring.get("composite_weight", 0.4))
    bull_signals = {k: float(v) for k, v in _wcfg.get("bull", {}).items()}
    bear_signals = {k: float(v) for k, v in _wcfg.get("bear", {}).items()}
    if variant == "pureml":
        model_w, composite_w = 1.0, 0.0
    print(f"  Signal weights loaded from: {_weights_path.name}  "
          f"(variant={variant}, model={model_w:.0%}, composite={composite_w:.0%}, "
          f"bull_signals={len(bull_signals)}, bear_signals={len(bear_signals)})")

    # ── Signal resolution map — how each key maps to actual feature data ──
    # Keys ending in known suffixes are auto-resolved as inverted features
    # Signals in signal_weights.yaml whose feature value must be inverted.
    # Format: yaml_key -> (feature_col_name, invert_flag)
    # Composite is SDZ/SSZ only — all signals map directly, no inversions needed.
    _INVERT_MAP: dict = {}

    def _get(col):
        full = f"{FP}{col}" if not col.startswith(FP) else col
        if full in cross.columns:
            return cross[full].fillna(0).values.astype(float)
        if col in cross.columns:
            return cross[col].fillna(0).values.astype(float)
        return np.zeros(len(cross))

    def _norm(arr):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx - mn < 1e-9:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - mn) / (mx - mn)

    def _composite(signals: dict, store: Dict[str, np.ndarray]) -> np.ndarray:
        total_weight = 0.0
        weighted_sum = np.zeros(len(cross))
        for sig, w in signals.items():
            if sig in _INVERT_MAP:
                base_col, invert = _INVERT_MAP[sig]
                raw = _norm(-_get(base_col) if invert else _get(base_col))
            else:
                raw = _norm(_get(sig))
            store[sig] = raw
            weighted_sum += w * raw
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum

    bull_sig_arrays: Dict[str, np.ndarray] = {}
    bull_composite = _composite(bull_signals, bull_sig_arrays)
    bear_sig_arrays: Dict[str, np.ndarray] = {}
    bear_composite = _composite(bear_signals, bear_sig_arrays)

    # ── Final scores: model% + composite% ────────────────────────────────
    bull_final = _norm(model_w * _norm(model_series.values) + composite_w * bull_composite)
    bear_final = _norm(model_w * _norm(1.0 - model_series.values) + composite_w * bear_composite)

    bull_score_series = pd.Series(bull_final, index=cross.index)
    bear_score_series = pd.Series(bear_final, index=cross.index)

    # ── Mode universe filter for watchlist selection ──────────────────────
    # All stocks are scored above. The filter narrows which stocks can appear
    # on the watchlist. Scores outside the filter universe are still available
    # for --explain and scores_detail output.
    _dist_col = f"{FEATURE_PREFIX}high_52w_dist"
    if mode == "momentum" and _dist_col in cross.columns:
        _wl_mask = cross[_dist_col] > MOMENTUM_DIST_THRESHOLD
        cross_wl = cross[_wl_mask].copy()
        print(f"  [{mode}] watchlist universe: {_wl_mask.sum()} of {len(cross)} stocks "
              f"(high_52w_dist > {MOMENTUM_DIST_THRESHOLD})")
    elif mode == "reversal" and _dist_col in cross.columns:
        _wl_mask = cross[_dist_col] <= REVERSAL_DIST_THRESHOLD
        cross_wl = cross[_wl_mask].copy()
        print(f"  [{mode}] watchlist universe: {_wl_mask.sum()} of {len(cross)} stocks "
              f"(high_52w_dist <= {REVERSAL_DIST_THRESHOLD})")
    else:
        _wl_mask = pd.Series(True, index=cross.index)
        cross_wl = cross

    bull_score_wl = bull_score_series[_wl_mask.values]
    bear_score_wl = bear_score_series[_wl_mask.values]

    # ── Composite presence filter ─────────────────────────────────────────
    # Require a minimum composite score so purely model-driven picks without
    # any SDZ/SSZ zone signal cannot appear on the watchlist.
    # Threshold: composite > 0 means at least one zone signal is active.
    _MIN_COMPOSITE = 0.0   # strict: must have at least some zone presence
    _bull_comp_wl = pd.Series(bull_composite, index=cross.index)[_wl_mask.values]
    _bear_comp_wl = pd.Series(bear_composite, index=cross.index)[_wl_mask.values]

    _bull_zone_mask = _bull_comp_wl > _MIN_COMPOSITE
    _bear_zone_mask = _bear_comp_wl > _MIN_COMPOSITE

    cross_wl_bull = cross_wl[_bull_zone_mask.values]
    cross_wl_bear = cross_wl[_bear_zone_mask.values]
    bull_score_wl  = bull_score_wl[_bull_zone_mask.values]
    bear_score_wl  = bear_score_wl[_bear_zone_mask.values]

    print(f"  [{mode}] after zone filter: {len(cross_wl_bull)} bull candidates  "
          f"| {len(cross_wl_bear)} bear candidates")

    # ── BULL portfolio: top_n highest bull scores ─────────────────────────
    cross_wl_bull["group_date"] = latest_date
    pc_bull = PortfolioConstructor(cfg, top_n=top_n, weighting=weighting)
    bull_ticker_scores, bull_weights = pc_bull.construct(cross_wl_bull, bull_score_wl)

    # ── BEAR portfolio: top_n highest bear scores ─────────────────────────
    cross_wl_bear["group_date"] = latest_date
    pc_bear = PortfolioConstructor(cfg, top_n=top_n, weighting=weighting)
    bear_ticker_scores, bear_weights = pc_bear.construct(cross_wl_bear, bear_score_wl)

    # Store original bull composite score for display
    bull_scores_orig = {}
    for t in bull_weights:
        try:
            val = bull_score_series.xs(t, level="ticker")
            bull_scores_orig[t] = round(float(val.iloc[0] if hasattr(val, "iloc") else val), 4)
        except Exception:
            bull_scores_orig[t] = 0.0

    # Store original bear composite score for display
    bear_scores_orig = {}
    for t in bear_weights:
        try:
            val = bear_score_series.xs(t, level="ticker")
            bear_scores_orig[t] = round(float(val.iloc[0] if hasattr(val, "iloc") else val), 4)
        except Exception:
            bear_scores_orig[t] = 0.0

    print(f"  Bull watchlist: {len(bull_weights)} stocks")
    print(f"  Bear watchlist: {len(bear_weights)} stocks")

    # ── SHAP explanations ─────────────────────────────────────────────────
    explanations_bull: List[dict] = []
    explanations_bear: List[dict] = []
    shap_img_path = REPORTS_DIR / "shap_global_us_local.png"

    try:
        shap_exp      = SHAPExplainer(ensemble.lgbm)
        recent_dates  = sorted(panel.index.get_level_values("date").unique())[-60:]
        recent        = panel[panel.index.get_level_values("date").isin(recent_dates)]
        recent_univ   = recent[recent["in_universe"] == True]
        X_shap        = recent_univ[[f for f in final_features
                                     if f in recent_univ.columns]].fillna(0)
        X_shap        = X_shap.sample(min(2000, len(X_shap)), random_state=42)
        shap_exp.compute(X_shap)
        shap_importance = shap_exp.global_importance(top_k=20)
        print(f"\n  Top 10 SHAP features:")
        print(shap_importance.head(10).to_string(index=False))
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        shap_exp.plot_global(X_shap, output_path=shap_img_path)

        rb  = f"{FEATURE_PREFIX}regime_bull"
        rbe = f"{FEATURE_PREFIX}regime_bear"
        regime = ("bull"   if rb  in cross.columns and cross[rb].mean()  > 0.5 else
                  "bear"   if rbe in cross.columns and cross[rbe].mean() > 0.5 else
                  "choppy")

        shap_inf   = shap_exp.compute(X_inf)
        feat_names = list(X_inf.columns)
        ticker_list = list(X_inf.index.get_level_values("ticker"))

        def _make_explanations(ranked_weights, side):
            expl = []
            for rank_pos, (ticker, weight) in enumerate(
                sorted(ranked_weights.items(), key=lambda x: -x[1]), 1
            ):
                if ticker not in ticker_list:
                    continue
                ti       = ticker_list.index(ticker)
                shap_row = shap_inf[ti]
                feat_shap = sorted(zip(feat_names, shap_row),
                                   key=lambda x: -abs(x[1]))[:5]
                # For bear: negative SHAP = bearish driver
                if side == "bear":
                    pos_f = [(f, round(v, 4)) for f, v in feat_shap if v < 0][:3]
                    neg_f = [(f, round(v, 4)) for f, v in feat_shap if v > 0][:3]
                else:
                    pos_f = [(f, round(v, 4)) for f, v in feat_shap if v > 0][:3]
                    neg_f = [(f, round(v, 4)) for f, v in feat_shap if v < 0][:3]
                expl.append({
                    "ticker":                ticker,
                    "side":                  side,
                    "rank":                  rank_pos,
                    "rank_score":            round(float(weight), 4),
                    "top_positive_features": pos_f,
                    "top_negative_features": neg_f,
                    "regime":                regime,
                })
            return expl

        explanations_bull = _make_explanations(bull_weights, "bull")
        explanations_bear = _make_explanations(bear_weights, "bear")

    except Exception as e:
        print(f"  SHAP warning: {e}")
        shap_img_path = None

    # ── Per-ticker signal details (for enhanced cards + --explain) ────────────
    _ticker_list_all = list(cross.index.get_level_values("ticker"))
    _n = len(_ticker_list_all)
    _idx_map = {t: i for i, t in enumerate(_ticker_list_all)}

    all_bull_final = pd.Series(bull_final, index=cross.index.get_level_values("ticker"))
    all_model_scores = pd.Series(
        model_series.values, index=cross.index.get_level_values("ticker")
    )
    all_bull_composites = pd.Series(bull_composite, index=cross.index.get_level_values("ticker"))
    all_bear_composites = pd.Series(bear_composite, index=cross.index.get_level_values("ticker"))
    all_bear_final = pd.Series(bear_final, index=cross.index.get_level_values("ticker"))

    signal_details_bull = {
        t: {sig: float(arr[_idx_map[t]]) for sig, arr in bull_sig_arrays.items()}
        for t in _ticker_list_all
    }
    signal_details_bear = {
        t: {sig: float(arr[_idx_map[t]]) for sig, arr in bear_sig_arrays.items()}
        for t in _ticker_list_all
    }

    return {
        # Bull
        "weights":              bull_weights,
        "ticker_scores":        bull_ticker_scores,
        "explanations":         explanations_bull,
        # Bear
        "bear_weights":         bear_weights,
        "bear_ticker_scores":   bear_ticker_scores,
        "bull_scores_orig":     bull_scores_orig,
        "bear_scores_orig":     bear_scores_orig,
        "explanations_bear":    explanations_bear,
        # Shared
        "latest_date":          latest_date,
        "shap_img":             shap_img_path,
        "cross":                cross,
        # Signal detail (for cards + --explain)
        "all_bull_final":       all_bull_final,
        "all_bear_final":       all_bear_final,
        "all_model_scores":     all_model_scores,
        "all_bull_composites":  all_bull_composites,
        "all_bear_composites":  all_bear_composites,
        "signal_details_bull":  signal_details_bull,
        "signal_details_bear":  signal_details_bear,
        "bull_signals_cfg":     bull_signals,
        "bear_signals_cfg":     bear_signals,
        "universe_size":        _n,
        "model_w":              model_w,
        "composite_w":          composite_w,
    }


# ── 4. Output ────────────────────────────────────────────────────────────────

# Confluence group definitions
_CONFLUENCE_GROUPS = {
    "YQM": ["1y", "3mo", "1mo"],   # Yearly + Quarterly + Monthly
    "QMW": ["3mo", "1mo", "1wk"],  # Quarterly + Monthly + Weekly
    "MWD": ["1mo", "1wk", "1d"],   # Monthly + Weekly + Daily
}
_BULL_ZONE_TYPES = {"SDZ", "DZ"}
_BEAR_ZONE_TYPES = {"SSZ", "SZ"}


def _get_ticker_zone_levels(ticker: str) -> Dict[str, dict]:
    """
    Load the LATEST valid zone per timeframe for a ticker directly from -Drv files.
    Returns dict: {'1d': {'zone_type': 'SDZ', 'high': 123.4, 'low': 120.1}, ...}
    Supports both Drv formats (ZoneType/Zone/Proximal/Distal and zone_type/zone_valid).
    """
    zones: Dict[str, dict] = {}
    for tf in HTF_ZONE_FILES:
        path = STOCK_DATA_DIR / f"{ticker}-{tf}-Drv.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]

            # ── Determine format ──────────────────────────────────────────
            if "zonetype" in df.columns and "zone" in df.columns:
                # Format A: charting tool export
                zt_col    = "zonetype"
                valid_col = "zone"
                valid_val = "valid"
                if "proximal" in df.columns and "distal" in df.columns:
                    df["proximal"] = pd.to_numeric(df["proximal"], errors="coerce")
                    df["distal"]   = pd.to_numeric(df["distal"],   errors="coerce")
                    df["_zh"] = df[["proximal","distal"]].max(axis=1)
                    df["_zl"] = df[["proximal","distal"]].min(axis=1)
                    high_col, low_col = "_zh", "_zl"
                else:
                    high_col, low_col = None, None
            elif "zone_type" in df.columns and "zone_valid" in df.columns:
                # Format B: legacy
                zt_col    = "zone_type"
                valid_col = "zone_valid"
                valid_val = "valid"
                high_col  = next((c for c in ("zone_high","zone_top","high") if c in df.columns), None)
                low_col   = next((c for c in ("zone_low","zone_bottom","low") if c in df.columns), None)
            else:
                continue

            valid_mask = df[valid_col].astype(str).str.strip().str.lower() == valid_val
            # Only keep actual zone rows (non-neutral ZoneType)
            zone_mask  = ~df[zt_col].astype(str).str.strip().str.upper().isin(["Z","NONE","0",""])
            df_valid = df[valid_mask & zone_mask].copy()
            if df_valid.empty:
                continue

            row = df_valid.iloc[-1]
            zt = str(row[zt_col]).strip().upper()
            if not zt:
                continue
            z_high = float(row[high_col]) if high_col and pd.notna(row.get(high_col)) else None
            z_low  = float(row[low_col])  if low_col  and pd.notna(row.get(low_col))  else None
            zones[tf] = {"zone_type": zt, "high": z_high, "low": z_low}
        except Exception:
            continue
    return zones


def _build_confluence(zones: Dict[str, dict], side: str) -> List[str]:
    """
    Return list of confluence strings where ALL timeframes in the group
    have a valid bull (DZ/SDZ) or bear (SZ/SSZ) zone.
    Each entry includes the price range: e.g. "YQM ✅ (₹2,820–₹2,865)"
    The price range shown is the OVERLAP (tightest common range) of all
    zone levels in the group — that is the highest-confidence entry area.
    """
    valid_types = _BULL_ZONE_TYPES if side == "bull" else _BEAR_ZONE_TYPES
    hits = []
    for grp_name, tfs in _CONFLUENCE_GROUPS.items():
        # Check all TFs in group have a valid zone
        group_zones = [zones.get(tf) for tf in tfs]
        if not all(z and z.get("zone_type", "") in valid_types for z in group_zones):
            continue

        # Compute the overlap of all zone ranges in the group
        # For bull: entry is the overlap low→high of DZ/SDZ bands
        # For bear: same logic (price should retrace INTO this area)
        highs = [z["high"] for z in group_zones if z.get("high") is not None]
        lows  = [z["low"]  for z in group_zones if z.get("low")  is not None]

        if highs and lows:
            # Overlap = max of lows → min of highs  (tightest common area)
            overlap_low  = max(lows)
            overlap_high = min(highs)
            if overlap_low <= overlap_high:
                # Clean overlap — price bands share a common area
                price_str = f"₹{overlap_low:,.0f} – ₹{overlap_high:,.0f}"
            else:
                # No strict overlap — show the average midpoints instead
                mid = (sum(lows) / len(lows) + sum(highs) / len(highs)) / 2
                price_str = f"~₹{mid:,.0f} (avg)"
            zone_labels = "+".join(z["zone_type"] for z in group_zones)
            hits.append(f"{grp_name} ✅  {zone_labels}  [{price_str}]")
        else:
            hits.append(f"{grp_name} ✅")
    return hits


def _best_entry(zones: Dict[str, dict], side: str,
                current_price: float) -> Optional[dict]:
    """
    Find the best entry zone — highest-timeframe valid bull/bear zone
    whose price range is near the current price (within 10%).
    Falls back to the nearest zone regardless of distance.
    """
    valid_types = _BULL_ZONE_TYPES if side == "bull" else _BEAR_ZONE_TYPES
    tf_priority = ["1y", "3mo", "1mo", "1wk", "1d"]
    candidates = []
    for tf in tf_priority:
        z = zones.get(tf)
        if not z or z.get("zone_type") not in valid_types:
            continue
        z_h = z.get("high")
        z_l = z.get("low")
        if z_h is None or z_l is None:
            continue
        # Distance from current price to zone midpoint
        mid   = (z_h + z_l) / 2
        dist  = abs(current_price - mid) / max(current_price, 1e-6)
        candidates.append({"tf": tf, "zone_type": z["zone_type"],
                            "high": z_h, "low": z_l, "dist_pct": dist * 100})
    if not candidates:
        return None
    # Prefer closest zone (price already in or nearest)
    return sorted(candidates, key=lambda x: x["dist_pct"])[0]


def _ict_signals(cross_row: pd.Series, side: str) -> List[str]:
    """Extract active ICT signals (OB, FVG) for bull or bear side."""
    FP = "features_"
    sigs = []
    if side == "bull":
        if cross_row.get(f"{FP}ict_bob_active",    0) > 0.5: sigs.append("Bull OB (Daily)")
        if cross_row.get(f"{FP}ict_bullfvg_active", 0) > 0.5: sigs.append("Bull FVG (Daily)")
    else:
        if cross_row.get(f"{FP}ict_sob_active", 0) > 0.5: sigs.append("Bear OB (Daily)")
        if cross_row.get(f"{FP}ict_bearfvg_active", 0) > 0.5: sigs.append("Bear FVG (Daily)")
    return sigs


def _trend_arrows(cross_row: pd.Series) -> str:
    """Return a compact trend string across all timeframes."""
    mapping = [("D", "features_regime_bull"),
               ("W", "weekly_trend"),
               ("M", "monthly_trend"),
               ("Q", "quarterly_trend"),
               ("Y", "yearly_trend")]
    parts = []
    for label, col in mapping:
        val = cross_row.get(col, np.nan)
        if pd.isna(val):
            parts.append(f"{label}?")
        elif float(val) >= 0.5:
            parts.append(f"{label}↑")
        else:
            parts.append(f"{label}↓")
    return "  ".join(parts)


def _sma_status(cross_row: pd.Series) -> str:
    """Human readable SMA position."""
    FP = "features_"
    parts = []
    for period in [20, 50, 200]:
        val = cross_row.get(f"{FP}price_vs_sma{period}", np.nan)
        if pd.notna(val):
            sym = "▲" if float(val) > 0 else "▼"
            parts.append(f"SMA{period}{sym}")
    adx = cross_row.get(f"{FP}adx_14", np.nan)
    if pd.notna(adx):
        adx_v = float(adx)
        strength = "Strong" if adx_v > 25 else "Weak"
        parts.append(f"ADX {adx_v:.0f} ({strength})")
    return "  ".join(parts)


def _fmt_price(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "N/A"
    return f"₹{p:,.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None or not np.isfinite(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def _stock_card(rank: int, ticker: str, side: str, score: float,
                cross_row: pd.Series, zones: Dict[str, dict],
                projected: Dict[int, float], panel: pd.DataFrame,
                scoring_detail: Optional[dict] = None) -> str:
    """Build a human-readable card for one stock."""
    emoji = "🟢" if side == "bull" else "🔴"
    title = f"{emoji}  #{rank}  {ticker:<20}  Score: {score*100:.0f}%"
    sep   = "─" * 64

    # ── Current price ─────────────────────────────────────────────────────
    cur_price = float(cross_row.get("close", 0)) if "close" in cross_row.index else 0.0

    # ── Best entry zone (nearest valid zone to current price) ─────────────
    entry = _best_entry(zones, side, cur_price)
    if entry:
        entry_str = (f"{_fmt_price(entry['low'])} – {_fmt_price(entry['high'])}"
                     f"  [{entry['zone_type']} {entry['tf'].upper()}]"
                     f"  ({entry['dist_pct']:.1f}% from price)")
    else:
        entry_str = "No active zone near current price"

    # ── Active zones across all TFs ───────────────────────────────────────
    valid_types = _BULL_ZONE_TYPES if side == "bull" else _BEAR_ZONE_TYPES
    zone_summary = []
    for tf in ["1y", "3mo", "1mo", "1wk", "1d"]:
        z = zones.get(tf, {})
        zt = z.get("zone_type", "")
        if zt in valid_types:
            zh = z.get("high"); zl = z.get("low")
            price_hint = (f" ₹{zl:,.0f}–₹{zh:,.0f}" if zh and zl else "")
            zone_summary.append(f"{zt}({tf}){price_hint}")
    zone_str = "  |  ".join(zone_summary) if zone_summary else "None"

    # ── Confluence groups with price ranges ───────────────────────────────
    confluences = _build_confluence(zones, side)
    ict_sigs    = _ict_signals(cross_row, side)

    # ── Projections ───────────────────────────────────────────────────────
    proj_str = "  |  ".join(
        f"{_fmt_pct(projected.get(h))} ({h//20}mo)" for h in [20, 40, 60]
    )

    # ── Trend + SMA ───────────────────────────────────────────────────────
    trend_str = _trend_arrows(cross_row)
    sma_str   = _sma_status(cross_row)

    # ── Assemble card ─────────────────────────────────────────────────────
    lines = [
        sep,
        title,
        sep,
        f"  Current Price : {_fmt_price(cur_price)}",
        f"  Best Entry    : {entry_str}",
        f"  Active Zones  : {zone_str}",
    ]

    # Confluence — one line per group, each with its price range
    if confluences or ict_sigs:
        lines.append(f"  Confluence    :")
        for cf in confluences:
            lines.append(f"    ▶  {cf}")
        for ic in ict_sigs:
            lines.append(f"    ▶  {ic}")
    else:
        lines.append(f"  Confluence    : None detected")

    lines += [
        f"  Projected     : {proj_str}",
        f"  Trend         : {trend_str}",
        f"  SMA / ADX     : {sma_str}",
    ]

    # HTF zone score + bias
    htf_col = "features_sdz_htf_score" if side == "bull" else "features_ssz_htf_score"
    htf_val = cross_row.get(htf_col, np.nan)
    if pd.notna(htf_val):
        lines.append(f"  HTF Zone Score: {float(htf_val)*100:.0f}%")
    conf_val = cross_row.get("features_zone_htf_confluence", np.nan)
    if pd.notna(conf_val):
        bias = "Bullish" if float(conf_val) > 0 else "Bearish"
        lines.append(f"  Zone Bias     : {bias}  ({float(conf_val)*100:+.0f}%)")

    # ── Scoring breakdown ─────────────────────────────────────────────────
    if scoring_detail:
        sd = scoring_detail
        u  = sd.get("universe_size", "?")
        lines.append(f"  {'─'*60}")
        lines.append(
            f"  Scoring  : Rank #{sd['rank_in_universe']}/{u}"
            f"  |  Model {sd['model_score']*100:.0f}%"
            f"  |  Composite {sd['composite_score']*100:.0f}%"
        )
        sig_weights = sd.get("signal_weights", {})
        sig_values  = sd.get("signal_values", {})
        if sig_weights:
            # Sort signals by weight × value (driving first)
            sigs_sorted = sorted(sig_weights.keys(),
                                 key=lambda s: -(sig_weights[s] * sig_values.get(s, 0.0)))
            # Top drivers (value >= 0.6)
            drivers = [(s, sig_weights[s], sig_values.get(s, 0.0))
                       for s in sigs_sorted if sig_values.get(s, 0.0) >= 0.6]
            # Weak/missing signals (value < 0.35)
            weak = [(s, sig_weights[s], sig_values.get(s, 0.0))
                    for s in sigs_sorted if sig_values.get(s, 0.0) < 0.35]
            if drivers:
                driver_str = "  |  ".join(
                    f"{s.replace('_score','').replace('_htf','').upper()} {v*100:.0f}% (w={w:.1f})"
                    for s, w, v in drivers[:4]
                )
                lines.append(f"  Driving  : {driver_str}")
            if weak:
                weak_str = "  |  ".join(
                    f"{s.replace('_score','').replace('_htf','').upper()} {v*100:.0f}%"
                    for s, w, v in weak[:4]
                )
                lines.append(f"  Weak/Low : {weak_str}")

    return "\n".join(lines)


def explain_ticker(ticker: str, date_str: Optional[str] = None) -> None:
    """
    Print a detailed scoring breakdown for any ticker (in or out of watchlist).
    Loads from the scores_detail_{date}.json saved by save_outputs().
    """
    import json, glob as _glob

    # Find the most recent (or date-specific) scores_detail JSON
    pattern = str(OUTPUT_DIR / f"scores_detail_{date_str or '*'}.json")
    matches = sorted(_glob.glob(pattern))
    if not matches:
        print(f"  No scores_detail file found at {OUTPUT_DIR}.")
        print(f"  Run without --explain first to generate scores.")
        return
    detail_path = Path(matches[-1])
    with open(detail_path) as f:
        scores_detail: dict = json.load(f)

    sep  = "═" * 66
    sep2 = "─" * 66

    # Normalise ticker for lookup (US tickers: AAPL, MSFT etc — no suffix)
    tkey = ticker.upper().strip()
    if tkey not in scores_detail:
        # Try case-insensitive match
        tkey_lower = tkey.lower()
        for k in scores_detail:
            if k.lower() == tkey_lower:
                tkey = k
                break

    if tkey not in scores_detail:
        # Fuzzy: find closest
        close = [t for t in scores_detail if ticker.upper() in t.upper()]
        if close:
            print(f"  '{ticker}' not found exactly. Did you mean: {close[:5]} ?")
        else:
            all_t = sorted(scores_detail.keys())
            print(f"  '{ticker}' not found in scores for {detail_path.name}.")
            print(f"  Universe has {len(all_t)} tickers. Sample: {all_t[:8]}")
        return

    entry       = scores_detail[tkey]
    bull_d      = entry["bull"]
    bear_d      = entry["bear"]
    bull_rank   = entry["bull_rank"]
    bear_rank   = entry["bear_rank"]
    in_bull     = entry["in_bull_watchlist"]
    in_bear     = entry["in_bear_watchlist"]
    u_size      = bull_d["universe_size"]

    # Watchlist status
    status = []
    if in_bull:
        status.append("IN BULL WATCHLIST")
    if in_bear:
        status.append("IN BEAR WATCHLIST")
    if not status:
        status.append("NOT in watchlist")

    print(f"\n{sep}")
    print(f"  EXPLAIN: {tkey}   [{' | '.join(status)}]")
    print(f"  Source  : {detail_path.name}")
    print(sep)

    # ── Bull breakdown ─────────────────────────────────────────────────────
    m_score = bull_d["model_score"]
    c_score = bull_d["composite_score"]
    m_w     = bull_d["model_weight"]
    c_w     = bull_d["composite_weight"]
    print(f"\n  BULL SIDE   rank #{bull_rank} / {u_size}")
    print(f"    Model score     : {m_score*100:.1f}%  (weight {m_w:.0%})")
    print(f"    Composite score : {c_score*100:.1f}%  (weight {c_w:.0%})")
    print(f"    (Cutoff to enter top-12: roughly top {12/u_size*100:.1f}% ≈ rank #{12})")

    sig_weights = bull_d.get("signal_weights", {})
    sig_values  = bull_d.get("signal_values", {})
    if sig_weights:
        print(f"\n    Signal breakdown (normalized 0-100%):")
        print(f"    {'Signal':<35} {'Weight':>6}  {'Value':>7}  {'Contribution':>12}")
        print(f"    {sep2[4:]}")
        sigs_sorted = sorted(sig_weights.keys(),
                             key=lambda s: -(sig_weights[s] * sig_values.get(s, 0.0)))
        for s in sigs_sorted:
            w  = sig_weights[s]
            v  = sig_values.get(s, 0.0)
            contrib = w * v
            bar = "#" * int(v * 15)
            flag = "  <-- HIGH" if v >= 0.65 else ("  <-- LOW" if v < 0.30 else "")
            print(f"    {s:<35} {w:>6.1f}  {v*100:>6.1f}%  {contrib:>12.2f}{flag}")

    # ── Bear breakdown ─────────────────────────────────────────────────────
    m_score_b = bear_d["model_score"]   # already (1 - raw) — inverted in _scoring_detail
    c_score_b = bear_d["composite_score"]
    print(f"\n  BEAR SIDE   rank #{bear_rank} / {u_size}")
    print(f"    Model (bearish) : {m_score_b*100:.1f}%  (inverted model)")
    print(f"    Composite score : {c_score_b*100:.1f}%  (weight {c_w:.0%})")
    sig_weights_b = bear_d.get("signal_weights", {})
    sig_values_b  = bear_d.get("signal_values", {})
    if sig_weights_b:
        print(f"\n    Signal breakdown (bear, normalized 0-100%):")
        print(f"    {'Signal':<35} {'Weight':>6}  {'Value':>7}  {'Contribution':>12}")
        print(f"    {sep2[4:]}")
        sigs_sorted_b = sorted(sig_weights_b.keys(),
                               key=lambda s: -(sig_weights_b[s] * sig_values_b.get(s, 0.0)))
        for s in sigs_sorted_b:
            w  = sig_weights_b[s]
            v  = sig_values_b.get(s, 0.0)
            contrib = w * v
            flag = "  <-- HIGH" if v >= 0.65 else ("  <-- LOW" if v < 0.30 else "")
            print(f"    {s:<35} {w:>6.1f}  {v*100:>6.1f}%  {contrib:>12.2f}{flag}")

    # ── What would help ────────────────────────────────────────────────────
    if sig_weights:
        low_sigs = [(s, sig_weights[s], sig_values.get(s, 0.0))
                    for s in sig_weights if sig_values.get(s, 0.0) < 0.35]
        if low_sigs:
            print(f"\n  WHAT WOULD HELP (bull — signals currently LOW):")
            for s, w, v in sorted(low_sigs, key=lambda x: -x[1]):
                gain = w * (0.80 - v)   # simulated gain if signal reaches 80th pct
                print(f"    {s:<35}  current {v*100:.0f}%  →  if 80%: +{gain:.2f} composite pts")

    print(f"\n{sep}\n")


def save_outputs(result: dict, panel: pd.DataFrame,
                 benchmark_close: pd.Series, cfg,
                 mode: str = "legacy",
                 cap_tier_map: dict | None = None,
                 variant: str = "composite") -> None:
    from pipeline.features.engineer import FEATURE_PREFIX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    bull_weights        = result["weights"]
    bear_weights        = result["bear_weights"]
    bull_scores_orig    = result["bull_scores_orig"]
    bear_scores         = result["bear_scores_orig"]
    latest_date         = result["latest_date"]
    cross               = result["cross"]
    date_str            = str(latest_date.date())
    all_bull_final      = result.get("all_bull_final", pd.Series(dtype=float))
    all_bear_final      = result.get("all_bear_final", pd.Series(dtype=float))
    all_model_scores    = result.get("all_model_scores", pd.Series(dtype=float))
    all_bull_composites = result.get("all_bull_composites", pd.Series(dtype=float))
    all_bear_composites = result.get("all_bear_composites", pd.Series(dtype=float))
    sig_details_bull    = result.get("signal_details_bull", {})
    sig_details_bear    = result.get("signal_details_bear", {})
    bull_signals_cfg    = result.get("bull_signals_cfg", {})
    bear_signals_cfg    = result.get("bear_signals_cfg", {})
    universe_size       = result.get("universe_size", len(cross))
    model_w             = result.get("model_w", 0.6)
    composite_w         = result.get("composite_w", 0.4)

    # ── Build bull ranking for universe rank lookup ────────────────────────
    _bull_rank = {t: r for r, t in enumerate(
        all_bull_final.sort_values(ascending=False).index, 1
    )} if not all_bull_final.empty else {}
    _bear_rank = {t: r for r, t in enumerate(
        all_bear_final.sort_values(ascending=False).index, 1
    )} if not all_bear_final.empty else {}

    def _safe_scalar(series: pd.Series, ticker: str, default: float = 0.0) -> float:
        """Extract a scalar from a possibly non-unique-indexed Series."""
        val = series.get(ticker, default)
        if isinstance(val, pd.Series):
            return float(val.iloc[-1]) if not val.empty else default
        return float(val) if val is not None else default

    def _scoring_detail(ticker: str, side: str) -> dict:
        """Build per-ticker scoring breakdown dict."""
        sig_cfg  = bull_signals_cfg if side == "bull" else bear_signals_cfg
        sig_vals = sig_details_bull.get(ticker, {}) if side == "bull" else sig_details_bear.get(ticker, {})
        rank_map = _bull_rank if side == "bull" else _bear_rank
        _raw_m   = _safe_scalar(all_model_scores, ticker)
        m_score  = (1.0 - _raw_m) if side == "bear" else _raw_m
        c_score  = _safe_scalar(all_bull_composites if side == "bull"
                                else all_bear_composites, ticker)
        return {
            "ticker":           ticker,
            "side":             side,
            "rank_in_universe": rank_map.get(ticker, 0),
            "universe_size":    universe_size,
            "model_score":      m_score,
            "composite_score":  c_score,
            "model_weight":     model_w,
            "composite_weight": composite_w,
            "signal_weights":   sig_cfg,
            "signal_values":    sig_vals,
        }

    # ── Full universe scores detail JSON ──────────────────────────────────
    all_tickers = list(cross.index.get_level_values("ticker"))
    scores_detail: Dict[str, dict] = {}
    for t in all_tickers:
        scores_detail[t] = {
            "bull": _scoring_detail(t, "bull"),
            "bear": _scoring_detail(t, "bear"),
            "bull_rank":    _bull_rank.get(t, 0),
            "bear_rank":    _bear_rank.get(t, 0),
            "in_bull_watchlist": t in bull_weights,
            "in_bear_watchlist": t in bear_weights,
        }
    import json
    _sfx = f"_{mode}" if mode != "legacy" else ""
    detail_path = OUTPUT_DIR / f"scores_detail{_sfx}_{date_str}.json"
    with open(detail_path, "w") as f:
        json.dump(scores_detail, f, indent=2)
    print(f"    Scores detail   : {detail_path}")

    # Pre-compute historical return quantile series for each horizon (done once).
    # Used to derive per-stock projected returns based on score percentile.
    _hist_ret_series: Dict[int, pd.Series] = {}
    for _h in [20, 40, 60]:
        _rc = f"future_{_h}d_return"
        if _rc in panel.columns:
            _hist_ret_series[_h] = panel[_rc].dropna()

    # Pre-compute score percentile ranks for bull and bear (over full universe).
    _bull_pct = all_bull_final.rank(pct=True) if not all_bull_final.empty else pd.Series(dtype=float)
    _bear_pct = all_bear_final.rank(pct=True) if not all_bear_final.empty else pd.Series(dtype=float)

    def _build_rows(weights_dict, side, score_override=None):
        rows = []
        score_pct_series = _bull_pct if side == "BULL" else _bear_pct
        for rank_pos, (ticker, weight) in enumerate(
            sorted(weights_dict.items(), key=lambda x: -x[1]), 1
        ):
            display_score = score_override[ticker] if score_override else weight
            row = {
                "rank":       rank_pos,
                "side":       side,
                "ticker":     ticker,
                "weight_pct": round(weight * 100, 2),
                "score":      round(float(display_score), 4),
                "date":       date_str,
            }
            if ticker in cross.index.get_level_values("ticker"):
                t_row = cross.xs(ticker, level="ticker", drop_level=False).iloc[0]
                for feat in [f"{FEATURE_PREFIX}adx_14",
                             f"{FEATURE_PREFIX}return_20d",
                             f"{FEATURE_PREFIX}vol_contraction",
                             f"{FEATURE_PREFIX}sector_rs_20d",
                             f"{FEATURE_PREFIX}rolling_beta_60d",
                             f"{FEATURE_PREFIX}regime_bull",
                             f"{FEATURE_PREFIX}regime_bear",
                             f"{FEATURE_PREFIX}sdz_htf_score",
                             f"{FEATURE_PREFIX}ssz_htf_score",
                             f"{FEATURE_PREFIX}zone_htf_confluence"]:
                    if feat in t_row.index:
                        row[feat.replace("features_", "")] = round(float(t_row[feat]), 4)

                # Multi-horizon projected returns — per-stock, based on score percentile.
                # Bull: high score → high return quantile (top of distribution).
                # Bear: high bear score → expected to decline → invert to bottom quantile.
                for h in [20, 40, 60]:
                    rank_col = f"cs_rank_{h}d"
                    if h in _hist_ret_series and ticker in score_pct_series.index:
                        pct = float(score_pct_series.get(ticker, 0.5))
                        lookup_pct = (1.0 - pct) if side == "BEAR" else pct
                        projected = _hist_ret_series[h].quantile(lookup_pct)
                        if np.isfinite(projected):
                            row[f"projected_{h}d_pct"] = round(projected * 100, 2)
                    if rank_col in t_row.index and pd.notna(t_row[rank_col]):
                        row[f"cs_rank_{h}d"] = round(float(t_row[rank_col]), 3)
            rows.append(row)
        return rows

    bull_rows = _build_rows(bull_weights, "BULL", score_override=bull_scores_orig)
    bear_rows = _build_rows(bear_weights, "BEAR", score_override=bear_scores)

    bull_df = pd.DataFrame(bull_rows)
    bear_df = pd.DataFrame(bear_rows)

    # ── Save separate CSVs ────────────────────────────────────────────────
    _sfx = f"_{mode}" if mode != "legacy" else ""
    _vsuffix = "_pureml" if variant == "pureml" else "_composite"
    bull_path = OUTPUT_DIR / f"watchlist{_sfx}{_vsuffix}_bull_{date_str}.csv"
    bear_path = OUTPUT_DIR / f"watchlist{_sfx}{_vsuffix}_bear_{date_str}.csv"
    bull_df.to_csv(bull_path, index=False)
    bear_df.to_csv(bear_path, index=False)

    # ── Save combined CSV ─────────────────────────────────────────────────
    combined_df = pd.concat([bull_df, bear_df], ignore_index=True)
    combined_path = OUTPUT_DIR / f"watchlist{_sfx}{_vsuffix}_combined_{date_str}.csv"
    combined_df.to_csv(combined_path, index=False)

    # ── Per-cap-tier top-10 watchlists (Large / Mid / Small) ─────────────
    if cap_tier_map:
        _TIER_DEFS = [("large", "Large Cap"), ("mid", "Mid Cap"), ("small", "Small Cap")]
        for side_key, side_label, all_final in [
            ("bull", "BULL", all_bull_final),
            ("bear", "BEAR", all_bear_final),
        ]:
            if all_final.empty:
                continue
            # Deduplicated sort (guards against non-unique index)
            sorted_tickers = list(dict.fromkeys(
                all_final.sort_values(ascending=False).index.tolist()
            ))
            for tier_key, tier_label in _TIER_DEFS:
                tier_tickers = [t for t in sorted_tickers
                                if cap_tier_map.get(t) == tier_key][:10]
                if not tier_tickers:
                    continue
                tier_weights = {t: _safe_scalar(all_final, t) for t in tier_tickers}
                tier_rows = _build_rows(tier_weights, side_label)
                for i, row in enumerate(tier_rows, 1):
                    row["rank"] = i
                    row["cap_tier"] = tier_label
                tier_df_out = pd.DataFrame(tier_rows)
                tier_path = OUTPUT_DIR / f"watchlist{_sfx}{_vsuffix}_{side_key}_{tier_key}_{date_str}.csv"
                tier_df_out.to_csv(tier_path, index=False)
                print(f"    {tier_label} {side_label} top-10: {tier_path}")

    # ── SHAP explanations JSON ────────────────────────────────────────────
    all_expl = result["explanations"] + result["explanations_bear"]
    expl_path = OUTPUT_DIR / f"explanations_{date_str}.json"
    with open(expl_path, "w") as f:
        json.dump(all_expl, f, indent=2)

    # ── Console output — human readable cards ─────────────────────────────
    def _render_side(weights_dict, side, score_override=None):
        label = "🟢  BULL" if side == "bull" else "🔴  BEAR"
        print(f"\n{'═'*66}")
        print(f"  {label} WATCHLIST — {len(weights_dict)} stocks — {date_str}")
        print(f"{'═'*66}")
        score_pct_series = _bull_pct if side == "bull" else _bear_pct
        for rank_pos, (ticker, weight) in enumerate(
            sorted(weights_dict.items(), key=lambda x: -x[1]), 1
        ):
            display_score = score_override[ticker] if score_override else weight
            row = {}
            if ticker in cross.index.get_level_values("ticker"):
                row = cross.xs(ticker, level="ticker", drop_level=False).iloc[0]

            # Projected returns — same percentile-lookup logic as _build_rows.
            # Bear inverts the percentile so high bear score → negative projection.
            proj: Dict[int, float] = {}
            for h in [20, 40, 60]:
                if h in _hist_ret_series and ticker in score_pct_series.index:
                    pct = float(score_pct_series.get(ticker, 0.5))
                    lookup_pct = (1.0 - pct) if side == "bear" else pct
                    projected = _hist_ret_series[h].quantile(lookup_pct)
                    if np.isfinite(projected):
                        proj[h] = projected * 100

            # Zone levels from raw -Drv files
            zones = _get_ticker_zone_levels(ticker)

            sd = _scoring_detail(ticker, side)
            card = _stock_card(rank_pos, ticker, side, float(display_score),
                               row, zones, proj, panel, scoring_detail=sd)
            print(card)
        print(f"\n{'═'*66}")

    _render_side(bull_weights, "bull", score_override=bull_scores_orig)
    _render_side(bear_weights, "bear", score_override=bear_scores)

    print(f"\n  Projection legend:")
    print(f"    1mo = 20 trading days  |  2mo = 40 trading days  |  3mo = 60 trading days")
    print(f"\n  Files saved:")
    print(f"    Bull watchlist  : {bull_path}")
    print(f"    Bear watchlist  : {bear_path}")
    print(f"    Combined        : {combined_path}")
    print(f"    Explanations    : {expl_path}")
    print(f"    Scores detail   : {detail_path}")

    # ── HTML report ───────────────────────────────────────────────────────
    try:
        from pipeline.backtest.reporter import PerformanceReporter
        from pipeline.reports.generator import ReportGenerator

        univ = panel[panel["in_universe"] == True]
        weekly_port_rets = (
            univ.groupby("group_date")["future_20d_return"]
            .apply(lambda x: x.nlargest(max(1, len(x) // 5)).mean())
            .dropna()
        )
        bm_weekly = benchmark_close.resample("W-FRI").last().pct_change().dropna()
        bm_weekly = bm_weekly.reindex(weekly_port_rets.index, method="ffill").fillna(0)
        net_rets  = weekly_port_rets - cfg.commission_bps / 10000

        rpt_obj = PerformanceReporter(
            weekly_gross_returns=weekly_port_rets,
            weekly_net_returns=net_rets,
            weekly_bm_returns=bm_weekly,
        )
        perf = rpt_obj.report()
        rpt_obj.print_summary(perf)

        all_expl = result.get("explanations", []) + result.get("explanations_bear", [])
        rg = ReportGenerator(
            report=perf,
            market_id="US_LOCAL",
            shap_img=result.get("shap_img"),
            stock_explanations=all_expl,
        )
        html_path = rg.generate(REPORTS_DIR / "report_us_local.html")
        print(f"\n  HTML report: {html_path}")
    except Exception as e:
        print(f"  Report warning: {e}")



# ── MAIN ─────────────────────────────────────────────────────────────────────

def prevent_sleep() -> None:
    """Prevent Windows from sleeping/hibernating while training runs."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        print("Sleep prevention ENABLED — computer will stay awake during training.")
    except Exception:
        pass   # non-Windows or failed — not critical


def allow_sleep() -> None:
    """Restore normal Windows sleep behaviour."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass


def main() -> None:
    import sys; sys.path.insert(0, ".")
    args = parse_args()

    # ── Fast path: --explain only (no pipeline run needed) ───────────────
    if args.explain:
        # Check if a scores_detail file already exists — if so, skip pipeline
        import glob as _glob
        existing = sorted(_glob.glob(str(OUTPUT_DIR / "scores_detail_*.json")))
        if existing:
            explain_ticker(args.explain)
            return
        # No existing detail file → must run pipeline first
        print(f"  No scores_detail file found. Running --skip_train pipeline first...")

    set_seeds(42)
    prevent_sleep()
    perf = PerfTimer()

    # ── Initialise per-run log file ───────────────────────────────────────
    from datetime import datetime
    from pathlib import Path
    _log_dir = Path("artefacts/us_local/logs")
    _log_dir.mkdir(parents=True, exist_ok=True)
    _run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = _log_dir / f"run_{_run_ts}.log"
    with open(_log_path, "w", encoding="utf-8") as _lf:
        _lf.write(
            f"{'='*70}\n"
            f"RUN START: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"  |  args: n_folds={args.n_folds} n_trials={args.n_trials}\n"
            f"{'='*70}\n"
        )
    print(f"Log file: {_log_path.resolve()}", flush=True)
    from pipeline.utils.logging import configure_log_file
    configure_log_file(_log_path)

    try:
        # ── GPU detection ─────────────────────────────────────────────────
        use_gpu = False
        if args.gpu:
            try:
                import subprocess
                subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
                use_gpu = True
                print("GPU flag set — note: LightGBM GPU requires a custom CUDA build; running on CPU")
            except Exception:
                print("WARNING: --gpu specified but no CUDA GPU detected — running on CPU")

        from pipeline.config.nse import NSE_CONFIG as cfg

        # ── Load ticker list ──────────────────────────────────────────────
        with perf.stage("Load ticker list"):
            ticker_df = pd.read_csv(STOCK_LIST_CSV)
            # Filter out benchmark index symbols (^GSPC, ^NDX etc.) — not tradeable stocks
            ticker_df = ticker_df[~ticker_df["Symbol"].str.startswith("^", na=True)]
            tickers = ticker_df["Symbol"].str.strip().dropna().tolist()
            print(f"Ticker list: {len(tickers)} symbols from {STOCK_LIST_CSV.name}")

        # Build sector map from CSV if a sector/industry column is present.
        # Without real sectors, sector_rs and portfolio sector-cap are meaningless.
        _sec_col = next(
            (c for c in ticker_df.columns
             if c.strip().lower() in ("sector", "industry", "industryname", "sectorname", "gics_sector", "gics sector")),
            None,
        )
        sector_map: Dict[str, str] = {}
        if _sec_col:
            sector_map = {
                str(sym).strip(): str(sec).strip()
                for sym, sec in zip(
                    ticker_df["Symbol"].str.strip(),
                    ticker_df[_sec_col].str.strip(),
                )
                if pd.notna(sec) and str(sec).strip()
            }
            print(f"  Sector map: {len(sector_map)} tickers mapped from column '{_sec_col}'")
        else:
            print(
                "  WARNING: No sector column found in ticker CSV — all stocks default to 'US'.\n"
                "           Run download_us_data.py to get a constituent CSV with sector data."
            )

        # Build cap-tier map: SPX/NDX → large, MID → mid, SML → small
        cap_tier_map: Dict[str, str] = {}
        if "Indices" in ticker_df.columns:
            for _sym, _idx in zip(ticker_df["Symbol"].str.strip(),
                                   ticker_df["Indices"].fillna("")):
                _idx_s = str(_idx).strip()
                if "SPX" in _idx_s or "NDX" in _idx_s:
                    cap_tier_map[str(_sym).strip()] = "large"
                elif _idx_s == "MID":
                    cap_tier_map[str(_sym).strip()] = "mid"
                elif _idx_s == "SML":
                    cap_tier_map[str(_sym).strip()] = "small"
            _lc = sum(1 for v in cap_tier_map.values() if v == "large")
            _mc = sum(1 for v in cap_tier_map.values() if v == "mid")
            _sc = sum(1 for v in cap_tier_map.values() if v == "small")
            print(f"  Cap tier map: {_lc} large-cap, {_mc} mid-cap, {_sc} small-cap")

        # ── Benchmark ─────────────────────────────────────────────────────
        with perf.stage("Load benchmark"):
            print(f"Loading blended benchmark ({SPX_WEIGHT:.0%} SPX + {NDX_WEIGHT:.0%} NDX)...")
            benchmark_close = load_benchmark(STOCK_DATA_DIR)
            if benchmark_close.empty:
                print("  WARNING: Benchmark unavailable — using equal-weight index proxy")

        # ── Determine which modes to run ──────────────────────────────────
        MODES_TO_RUN = {
            "all":      ["momentum", "reversal"],
            "momentum": ["momentum"],
            "reversal": ["reversal"],
            "legacy":   ["legacy"],
        }[args.mode]

        MODE_DIRS = {
            "momentum": MOMENTUM_ARTEFACTS_DIR,
            "reversal": REVERSAL_ARTEFACTS_DIR,
            "legacy":   ARTEFACTS_DIR,
        }

        # Build fresh panel once — shared across all modes
        with perf.stage("Load CSV panel (parallel I/O)"):
            panel = build_panel_from_local(tickers, STOCK_DATA_DIR,
                                           min_history_days=args.min_history_days,
                                           sector_map=sector_map)
            if benchmark_close.empty:
                benchmark_close = panel.groupby(level="date")["close"].mean().rename("benchmark_close")

        # ── Resolve as_of date ─────────────────────────────────────────────
        as_of_dt: Optional[datetime] = None
        if args.as_of:
            as_of_dt = datetime.strptime(args.as_of, "%Y-%m-%d")
            print(f"\n  [as_of={args.as_of}] Using {args.as_of} as reference date "
                  f"for staleness checks and scoring.")

        # ── Data freshness checks (StaleDataGuard) ─────────────────────────
        from pipeline.monitoring.stale_data_guard import StaleDataGuard, StaleDataError
        guard = StaleDataGuard(
            max_lag_days=args.max_data_lag_days,
            min_coverage_pct=0.80,
        )
        try:
            if args.strict_data_check:
                guard.assert_fresh(panel, benchmark_close, as_of=as_of_dt)
            else:
                issues = guard.check(panel, benchmark_close, as_of=as_of_dt)
                if any(i.severity == "error" for i in issues):
                    print("  ⚠ Data freshness issues detected (use --strict_data_check to block):")
                    for i in issues:
                        if i.severity == "error":
                            print(f"    {i}")
        except StaleDataError as _e:
            print(f"\n✗ Stale data check failed:\n{_e}\n")
            print("  Rerun without --strict_data_check to proceed anyway, "
                  "or refresh source data.")
            raise

        results_by_mode: Dict[str, dict] = {}

        if args.skip_train:
            print("\nLoading existing model artefacts (--skip_train) ...")

            def _load_mode(art_dir: Path, m: str):
                def _load(name):
                    p = art_dir / name
                    if not p.exists():
                        raise FileNotFoundError(
                            f"Artefact not found: {p}. "
                            f"Run without --skip_train --mode {m} first."
                        )
                    with open(p, "rb") as f:
                        return pickle.load(f)
                return {
                    "ensemble":       _load("ensemble.pkl"),
                    "drift_monitor":  _load("drift_monitor.pkl"),
                    "final_features": (art_dir / "selected_features.txt")
                                      .read_text().strip().split("\n"),
                }

            for m in MODES_TO_RUN:
                results_by_mode[m] = _load_mode(MODE_DIRS[m], m)

            with perf.stage("Feature engineering (skip_train)"):
                print("Re-running feature engineering on fresh panel ...")
                from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
                fe = FeatureEngineer(cfg, benchmark_close)
                panel = fe.build(panel)
                print(f"  Features ready. Panel date range: "
                      f"{panel.index.get_level_values('date').min().date()} → "
                      f"{panel.index.get_level_values('date').max().date()}")
        else:
            # Train each mode sequentially.
            # Checkpoints 1+2 (feature engineering + targets) are shared and built once —
            # the second mode call loads them instantly from disk.

            # ── Apply train_start cutoff ──────────────────────────────────────
            full_panel = panel
            train_start_ts = pd.Timestamp(args.train_start)
            panel_dates = panel.index.get_level_values("date")
            if panel_dates.min() < train_start_ts:
                panel_for_train = panel[panel_dates >= train_start_ts].copy()
                n_dropped = len(panel) - len(panel_for_train)
                print(f"\n  [train_start={args.train_start}] Trimmed training panel: "
                      f"{len(panel_for_train):,} rows kept "
                      f"({n_dropped:,} pre-{args.train_start} rows excluded from CV/HPO)")
            else:
                panel_for_train = panel
                print(f"\n  [train_start={args.train_start}] Panel already starts at "
                      f"{panel_dates.min().date()} — no rows dropped")

            # ── Auto-compute n_folds from date range if not explicitly set ────
            # With 1-year test windows, n_folds = data_years - warmup_years.
            # Warmup: ~2 years needed for sufficient history before first fold.
            # This ensures the most recent years are always covered in CV.
            if args.n_folds == 8:   # default — user didn't explicitly set it
                _as_of_year      = pd.Timestamp(args.as_of).year if args.as_of else pd.Timestamp.now().year
                _start_year      = pd.Timestamp(args.train_start).year
                _warmup_years    = 2
                _auto_folds      = max(6, _as_of_year - _start_year - _warmup_years + 1)
                if _auto_folds != args.n_folds:
                    print(f"\n  [n_folds] Auto-computed {_auto_folds} folds "
                          f"({_as_of_year} - {_start_year} - {_warmup_years} warmup + 1) "
                          f"— overrides default of {args.n_folds}")
                    args.n_folds = _auto_folds

            for m in MODES_TO_RUN:
                print(f"\n{'='*62}")
                print(f"  TRAINING  MODE: {m.upper()}")
                print(f"{'='*62}")
                with perf.stage(f"Full training — {m}"):
                    artefacts = train(
                        panel=panel_for_train,
                        benchmark_close=benchmark_close,
                        cfg=cfg,
                        n_folds=args.n_folds,
                        n_trials=args.n_trials,
                        top_n=args.top_n,
                        use_gpu=use_gpu,
                        mode=m,
                        mode_artefacts_dir=MODE_DIRS[m],
                        n_jobs=args.n_jobs,
                        as_of=args.as_of,
                    )
                panel = artefacts["panel"]   # feature-engineered panel — required for score_and_rank
                results_by_mode[m] = {
                    "ensemble":      artefacts["ensemble"],
                    "final_features": artefacts["final_features"],
                    "drift_monitor": artefacts["drift_monitor"],
                }

        # ── Drift check (per mode) ─────────────────────────────────────────
        with perf.stage("Drift monitoring"):
            for m, art in results_by_mode.items():
                try:
                    latest_date = panel.index.get_level_values("date").max()
                    art["drift_monitor"].compute_weekly_drift(panel, latest_date)
                    art["drift_monitor"].save(Path(f"monitoring/{m}"))
                except Exception as e:
                    print(f"Drift monitor warning [{m}]: {e}")

        # ── Determine scoring date ─────────────────────────────────────────
        # Priority: --as_of flag > CSV max date > panel max date
        if as_of_dt is not None:
            csv_max_date = pd.Timestamp(as_of_dt)
            print(f"\nScoring as-of date: {csv_max_date.date()} (from --as_of)")
        else:
            csv_max_date = get_csv_max_date(STOCK_DATA_DIR)
            if csv_max_date is not None:
                print(f"\nCSV max date detected: {csv_max_date.date()} — using as scoring as-of date")
            else:
                print("\nCould not determine CSV max date — falling back to panel max date")

        # ── Score & rank each mode, save outputs ───────────────────────────
        for m, art in results_by_mode.items():
            for variant in ["pureml", "composite"]:
                print(f"\n{'='*62}")
                print(f"  SCORING   MODE: {m.upper()}  VARIANT: {variant.upper()}")
                print(f"{'='*62}")
                with perf.stage(f"Score & rank — {m} ({variant})"):
                    result = score_and_rank(
                        panel=panel,
                        ensemble=art["ensemble"],
                        final_features=art["final_features"],
                        benchmark_close=benchmark_close,
                        cfg=cfg,
                        top_n=args.top_n,
                        weighting=args.weighting,
                        as_of_date=csv_max_date,
                        mode=m,
                        variant=variant,
                    )
                with perf.stage(f"Save outputs — {m} ({variant})"):
                    save_outputs(result, panel, benchmark_close, cfg, mode=m,
                                 cap_tier_map=cap_tier_map if cap_tier_map else None,
                                 variant=variant)

        # ── --explain (post-run, if requested) ───────────────────────────
        if args.explain:
            explain_ticker(args.explain)

    finally:
        allow_sleep()
        perf.report()
        perf.save(_log_path)


if __name__ == "__main__":
    main()

