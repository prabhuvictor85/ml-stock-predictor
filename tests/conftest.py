"""
conftest.py — pytest configuration for the ml-stock-predictor test suite.

Ensures the project root is in sys.path so tests can `from pipeline.X import Y`
regardless of where pytest is invoked from.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `from pipeline.X import Y` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pytest
import numpy as np
import pandas as pd


# ── Skip legacy ad-hoc scripts (kept for ref but not real tests) ─────────────
# These files predate the test suite; they're inspection scripts that make
# live HTTP calls or print to stdout, not pytest tests.
collect_ignore = [
    "test.py",
    "test_sources.py",
    "test_sources2.py",
]


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def project_root() -> Path:
    """Project root directory."""
    return _PROJECT_ROOT


@pytest.fixture
def tiny_ohlcv() -> pd.DataFrame:
    """
    Synthetic daily OHLCV for a single ticker — 500 bars of mild upward drift
    with realistic volatility. Used by unit tests that need a small, fast,
    deterministic dataset.
    """
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2022-01-01", periods=n)
    log_rets = rng.normal(0.0005, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(log_rets))
    # Realistic OHLC: open near prev close, high/low around close
    intraday_vol = np.abs(rng.normal(0, 0.01, size=n))
    high = close * (1 + intraday_vol)
    low  = close * (1 - intraday_vol)
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.002, size=n - 1))])
    volume = rng.integers(100_000, 1_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.Index(dates, name="date"),
    )


@pytest.fixture
def tiny_panel(tiny_ohlcv) -> pd.DataFrame:
    """
    Synthetic multi-ticker panel — 5 tickers × 500 bars with MultiIndex
    (date, ticker). Each ticker has independent price paths.
    """
    rng = np.random.default_rng(43)
    parts = []
    for ticker in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        df = tiny_ohlcv.copy()
        # Add ticker-specific noise so they're not all identical
        bump = rng.normal(1.0, 0.05)
        df["close"] = df["close"] * bump
        df["high"]  = df["high"]  * bump
        df["low"]   = df["low"]   * bump
        df["open"]  = df["open"]  * bump
        df["ticker"] = ticker
        df["in_universe"] = True
        df["group_date"]  = df.index
        df = df.set_index("ticker", append=True)
        parts.append(df)
    panel = pd.concat(parts).sort_index()
    return panel
