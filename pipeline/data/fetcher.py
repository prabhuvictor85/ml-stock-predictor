"""
DataFetcher — fetches OHLCV + market cap data from primary / fallback sources.
All market-specific parameters come from cfg: MarketConfig.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ── FX helper ──────────────────────────────────────────────────────────────

def _fetch_fx_to_usd(currency: str, start: str, end: str) -> pd.Series:
    """Return daily USD/currency FX series. If already USD return 1.0 series."""
    if currency == "USD":
        idx = pd.date_range(start, end, freq="B")
        return pd.Series(1.0, index=idx, name="fx_usd")
    # For INR use yfinance USDINR=X
    try:
        import yfinance as yf
        ticker_sym = "USDINR=X"
        df = yf.download(ticker_sym, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("Empty FX data")
        rate = df["Close"].rename("fx_usd")
        # rate is INR per 1 USD; invert to get USD per 1 INR
        return (1.0 / rate).ffill().bfill()
    except Exception as e:
        log.warning(f"FX fetch failed for {currency}: {e}. Using 1/75 as fallback.")
        idx = pd.date_range(start, end, freq="B")
        return pd.Series(1.0 / 75.0, index=idx, name="fx_usd")


# ── yfinance adapter ────────────────────────────────────────────────────────

def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted OHLCV from Yahoo Finance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"yfinance missing columns {missing} for {ticker}")
    return df[required].copy()


# ── polygon adapter ─────────────────────────────────────────────────────────

def _fetch_polygon(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch adjusted OHLCV from Polygon.io REST API."""
    import requests

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day"
        f"/{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("resultsCount", 0) == 0:
        return pd.DataFrame()
    rows = data["results"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return df


# ── tiingo adapter ───────────────────────────────────────────────────────────

def _fetch_tiingo(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """Fetch adjusted OHLCV from Tiingo."""
    import requests

    url = (
        f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        f"?startDate={start}&endDate={end}&resampleFreq=daily&token={api_key}"
    )
    resp = requests.get(url, timeout=30, headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.rename(columns={"adjOpen": "open", "adjHigh": "high", "adjLow": "low",
                             "adjClose": "close", "adjVolume": "volume"})
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    return df


class DataFetcher:
    """
    Fetches backward-adjusted OHLCV data for a list of tickers.
    Reads all source/calendar/currency config from cfg.

    Parameters
    ----------
    cfg : MarketConfig
    polygon_api_key : str, optional
    tiingo_api_key : str, optional
    """

    def __init__(
        self,
        cfg: MarketConfig,
        polygon_api_key: str = "",
        tiingo_api_key: str = "",
    ) -> None:
        self.cfg = cfg
        self.polygon_api_key = polygon_api_key
        self.tiingo_api_key = tiingo_api_key

    def fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch OHLCV for one ticker. Tries primary source then fallback.
        Returns DataFrame indexed by date with columns: open,high,low,close,volume.
        """
        df = pd.DataFrame()
        primary = self.cfg.data_source_primary

        try:
            if primary in ("nsepy", "kite"):
                # NSE via yfinance (nsepy is end-of-life; yfinance .NS suffix works)
                df = _fetch_yfinance(ticker, start, end)
            elif primary == "polygon":
                if not self.polygon_api_key:
                    raise ValueError("polygon_api_key not set")
                df = _fetch_polygon(ticker, start, end, self.polygon_api_key)
        except Exception as e:
            log.warning(f"Primary source '{primary}' failed for {ticker}: {e}. Trying fallback.")

        if df.empty:
            fallback = self.cfg.data_source_fallback
            try:
                if fallback == "yfinance":
                    df = _fetch_yfinance(ticker, start, end)
                elif fallback == "tiingo":
                    if not self.tiingo_api_key:
                        raise ValueError("tiingo_api_key not set")
                    df = _fetch_tiingo(ticker, start, end, self.tiingo_api_key)
            except Exception as e2:
                log.warning(f"Fallback source '{fallback}' also failed for {ticker}: {e2}.")

        if df.empty:
            log.warning(f"No data retrieved for {ticker}. Returning empty DataFrame.")
        return df

    def fetch_benchmark(self, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV for cfg.benchmark_ticker."""
        log.info(f"Fetching benchmark {self.cfg.benchmark_ticker} [{start} → {end}]")
        return self.fetch_single(self.cfg.benchmark_ticker, start, end)

    def fetch_many(
        self,
        tickers: List[str],
        start: str,
        end: str,
        sleep_between: float = 0.1,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple tickers. Returns dict ticker → DataFrame."""
        results: Dict[str, pd.DataFrame] = {}
        for i, ticker in enumerate(tickers):
            log.info(f"Fetching {ticker} ({i+1}/{len(tickers)})")
            results[ticker] = self.fetch_single(ticker, start, end)
            if sleep_between > 0:
                time.sleep(sleep_between)
        return results

    def fetch_fx(self, start: str, end: str) -> pd.Series:
        """Return daily USD rate for cfg.currency."""
        return _fetch_fx_to_usd(self.cfg.currency, start, end)

