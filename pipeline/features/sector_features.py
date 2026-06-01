"""
sector_features.py
------------------
Computes sector-level zone scores from ETF OHLC data and attaches them
to the stock cross-section as features.

Pipeline:
  1. Fetch OHLC for 12 ETFs (11 SPDR sector + SOXX) from yfinance.
  2. Run compute_zone_features on each ETF (same logic as individual stocks).
  3. Extract zone_type values at as_of_date → compute sdz/ssz HTF scores.
  4. Map each stock to its sector ETF via the constituent CSV 'ETF' column.
  5. Attach features_sector_etf_bull_score / features_sector_etf_bear_score
     to the cross-section DataFrame so signal_weights.yaml can reference them.

Results are disk-cached per as_of_date so re-runs within the same day are fast.

Usage (called automatically from score_and_rank):
    from pipeline.features.sector_features import attach_sector_etf_scores
    attach_sector_etf_scores(cross, as_of_date, constituent_csv_path, cache_dir)
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

# US market — 11 SPDR sector ETFs + SOXX for semiconductors
ETF_UNIVERSE = [
    "XLC",   # Communication Services
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLK",   # Information Technology
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "SOXX",  # Semiconductors (iShares)
]

# NSE market — Nifty sector indices (all validated on yfinance)
NSE_INDEX_UNIVERSE = [
    "^NSEBANK",    # Banking + Financial Services  (Nifty Bank)
    "^CNXPSUBANK", # PSU Banks
    "^CNXIT",      # Technology / IT               (Nifty IT)
    "^CNXPHARMA",  # Healthcare / Pharma            (Nifty Pharma)
    "^CNXFMCG",    # FMCG / Consumer Defensive      (Nifty FMCG)
    "^CNXMETAL",   # Metal / Basic Materials        (Nifty Metal)
    "^CNXENERGY",  # Energy / Oil & Gas / Utilities (Nifty Energy)
    "^CNXREALTY",  # Real Estate / Realty           (Nifty Realty)
    "^CNXMEDIA",   # Media / Communication Services (Nifty Media)
    "^CNXINFRA",   # Industrials / Infrastructure   (Nifty Infra)
    "^CNXAUTO",    # Auto / Consumer Cyclical       (Nifty Auto)
    "^NSEI",       # Nifty 50 broad (Others / catch-all)
]

# Mirrors the HTF weights in engineer.py  (1d:1, 1wk:2, 1mo:3, 3mo:4, 1y:5)
_HTF_W: Dict[str, int] = {
    "zone_type_1d":  1,
    "zone_type_1wk": 2,
    "zone_type_1mo": 3,
    "zone_type_3mo": 4,
    "zone_type_1y":  5,
}
_MAX_SCORE = sum(_HTF_W.values()) * 2   # 30  (SDZ weight = 2× DZ weight)

FEATURE_PREFIX = "features_"
_BULL_COL = f"{FEATURE_PREFIX}sector_etf_bull_score"
_BEAR_COL = f"{FEATURE_PREFIX}sector_etf_bear_score"

# Lookback for ETF data: 6 years covers annual zones comfortably
_LOOKBACK_YEARS = 6


# ── Core: compute zone score for one ETF ──────────────────────────────────────

def _fetch_etf_ohlcv(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download ETF OHLC from yfinance. Returns lowercase-column DataFrame or None."""
    try:
        import yfinance as yf
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        # Ensure required columns exist
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = np.nan
        return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    except Exception:
        return None


def _zone_score_for_etf(
    ticker: str,
    as_of_date: pd.Timestamp,
) -> Dict[str, float]:
    """
    Fetch ETF OHLC, run zone features, return {'bull': float, 'bear': float}.
    Returns zeros on any failure so a bad ETF never blocks the pipeline.
    """
    from pipeline.features.zone_features import compute_zone_features

    start = (as_of_date - pd.DateOffset(years=_LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    end   = (as_of_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    ohlcv = _fetch_etf_ohlcv(ticker, start, end)
    if ohlcv is None or len(ohlcv) < 60:
        return {"bull": 0.0, "bear": 0.0}

    try:
        cutoff = as_of_date.normalize()   # midnight — same convention as pipeline
        zone_df = compute_zone_features(ohlcv, cutoff_date=cutoff)
    except Exception:
        return {"bull": 0.0, "bear": 0.0}

    # Find the row at (or nearest before) as_of_date
    zone_df.index = pd.to_datetime(zone_df.index)
    candidates = zone_df[zone_df.index <= as_of_date]
    if candidates.empty:
        return {"bull": 0.0, "bear": 0.0}

    row = candidates.iloc[-1]

    # Compute HTF-weighted SDZ / SSZ scores (mirrors engineer.py logic)
    sdz_score = 0.0
    ssz_score = 0.0
    for col, weight in _HTF_W.items():
        zt = str(row.get(col, "")).strip().upper()
        if zt == "SDZ":
            sdz_score += weight * 2
        elif zt == "DZ":
            sdz_score += weight * 1
        elif zt == "SSZ":
            ssz_score += weight * 2
        elif zt == "SZ":
            ssz_score += weight * 1

    bull = float(np.clip(sdz_score / _MAX_SCORE, 0.0, 1.0))
    bear = float(np.clip(ssz_score / _MAX_SCORE, 0.0, 1.0))
    return {"bull": bull, "bear": bear}


# ── Cached batch scorer ────────────────────────────────────────────────────────

def compute_etf_zone_scores(
    as_of_date: pd.Timestamp,
    universe: Optional[list] = None,
    cache_dir: Optional[Path] = None,
    cache_key: str = "us",
    force_refresh: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bull/bear zone scores for all sector ETFs/indices on as_of_date.

    Parameters
    ----------
    as_of_date    : Scoring date.
    universe      : List of ETF/index tickers. Defaults to US ETF_UNIVERSE.
    cache_dir     : Directory for JSON cache files.
    cache_key     : Market identifier used in the cache filename (e.g. "us", "nse").
    force_refresh : Ignore existing cache and recompute.

    Returns
    -------
    {etf_ticker: {"bull": float [0-1], "bear": float [0-1]}}
    """
    if universe is None:
        universe = ETF_UNIVERSE

    date_str = as_of_date.strftime("%Y-%m-%d")

    # ── Try disk cache ─────────────────────────────────────────────────────
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"sector_etf_scores_{cache_key}_{date_str}.json"
        if cache_file.exists() and not force_refresh:
            try:
                with open(cache_file, encoding="utf-8") as f:
                    cached = json.load(f)
                print(f"  [sector_etf] Loaded from cache: {cache_file.name}")
                return cached
            except Exception:
                pass   # cache corrupt — recompute

    # ── Compute fresh ──────────────────────────────────────────────────────
    print(f"  [sector_etf] Computing zone scores for {len(universe)} indices "
          f"({cache_key}) as-of {date_str} ...")
    results: Dict[str, Dict[str, float]] = {}
    for etf in universe:
        scores = _zone_score_for_etf(etf, as_of_date)
        results[etf] = scores
        bull_pct = f"{scores['bull']*100:.0f}%"
        bear_pct = f"{scores['bear']*100:.0f}%"
        print(f"    {etf:<15}  bull={bull_pct:>4}  bear={bear_pct:>4}")

    # ── Write cache ────────────────────────────────────────────────────────
    if cache_file is not None:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass   # cache write failure is non-fatal

    return results


# ── Attachment helper (called from score_and_rank) ─────────────────────────────

def attach_sector_etf_scores(
    cross: pd.DataFrame,
    as_of_date: pd.Timestamp,
    constituent_csv: Path,
    cache_dir: Optional[Path] = None,
    universe: Optional[list] = None,
    cache_key: str = "us",
) -> None:
    """
    Add sector_etf_bull_score and sector_etf_bear_score columns to `cross`
    (in-place).  Missing tickers and unmapped ETFs default to 0.

    Parameters
    ----------
    cross           : Cross-section DataFrame indexed by (date, ticker).
    as_of_date      : The scoring date.
    constituent_csv : Path to constituent CSV with ETF column
                      (constituents_us_combined.csv for US, constituentsi.csv for NSE).
    cache_dir       : Optional directory for caching ETF/index zone scores.
    universe        : ETF/index ticker list. Defaults to US ETF_UNIVERSE.
    cache_key       : Market label for cache filename ("us" or "nse").
    """
    # Load ETF scores
    etf_scores = compute_etf_zone_scores(
        as_of_date, universe=universe, cache_dir=cache_dir, cache_key=cache_key
    )

    # Load ticker → ETF mapping
    # Build both bare (RELIANCE) and Yahoo-suffixed (RELIANCE.NS) keys so the
    # lookup works regardless of whether the pipeline uses suffix or not.
    ticker_to_etf: Dict[str, str] = {}
    if Path(constituent_csv).exists():
        try:
            df_const = pd.read_csv(constituent_csv, usecols=["Symbol", "ETF"])
            df_const = df_const.dropna(subset=["ETF"])
            for _, row in df_const.iterrows():
                sym = str(row["Symbol"]).strip()
                etf = str(row["ETF"]).strip()
                ticker_to_etf[sym] = etf                      # bare:  RELIANCE
                if not sym.endswith(".NS"):
                    ticker_to_etf[sym + ".NS"] = etf          # NS:    RELIANCE.NS
                if not sym.endswith(".BO"):
                    ticker_to_etf[sym + ".BO"] = etf          # BSE:   RELIANCE.BO
        except Exception as e:
            print(f"  [sector_etf] WARNING: could not load constituent CSV ({e}) — "
                  f"sector scores will be 0")
    else:
        print(f"  [sector_etf] WARNING: constituent CSV not found at {constituent_csv} — "
              f"sector scores will be 0")

    # Map each stock in cross to its ETF score
    tickers = cross.index.get_level_values("ticker")

    bull_vals = np.zeros(len(cross), dtype=np.float32)
    bear_vals = np.zeros(len(cross), dtype=np.float32)

    for i, ticker in enumerate(tickers):
        etf = ticker_to_etf.get(ticker)
        if etf and etf in etf_scores:
            bull_vals[i] = etf_scores[etf]["bull"]
            bear_vals[i] = etf_scores[etf]["bear"]

    cross[_BULL_COL] = bull_vals
    cross[_BEAR_COL] = bear_vals

    mapped = int((bull_vals > 0).sum() + (bear_vals > 0).sum()) // 2
    print(f"  [sector_etf] Attached scores: {mapped}/{len(cross)} tickers mapped to an ETF")
