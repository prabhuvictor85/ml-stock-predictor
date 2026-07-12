"""
Tests for StaleDataGuard — verifies the freshness checks actually fire.

This catches the bug we observed in production where the TradingView panel
was 5+ trading days stale but no warning was raised.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pipeline.monitoring.stale_data_guard import (
    StaleDataGuard,
    StaleDataError,
)


def _make_panel(last_date: pd.Timestamp, n_tickers: int = 5, n_bars: int = 100) -> pd.DataFrame:
    """Build a small panel ending at last_date."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range(end=last_date, periods=n_bars)
    n = len(dates)  # use actual length (bdate_range may round to nearest business day)
    parts = []
    for i in range(n_tickers):
        close = 100.0 + np.cumsum(rng.normal(0, 1, size=n))
        df = pd.DataFrame(
            {
                "open":   close,
                "high":   close * 1.01,
                "low":    close * 0.99,
                "close":  close,
                "volume": rng.integers(100_000, 500_000, size=n).astype(float),
            },
            index=pd.Index(dates, name="date"),
        )
        df["ticker"] = f"T{i}"
        df = df.set_index("ticker", append=True)
        parts.append(df)
    return pd.concat(parts).sort_index()


def test_stale_data_guard_fresh_panel_passes():
    """Recent data should not raise."""
    today = pd.Timestamp.now().normalize()
    panel = _make_panel(today)
    bm    = panel.groupby(level="date")["close"].mean()
    guard = StaleDataGuard(max_lag_days=7)
    # Should not raise
    guard.assert_fresh(panel, bm, as_of=today.to_pydatetime())


def test_stale_data_guard_old_panel_raises():
    """Panel that's 30 days old should raise in strict mode."""
    old_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    panel = _make_panel(old_date)
    bm    = panel.groupby(level="date")["close"].mean()
    guard = StaleDataGuard(max_lag_days=7)
    with pytest.raises(StaleDataError):
        guard.assert_fresh(panel, bm)


def test_stale_data_guard_check_returns_issues():
    """check() (non-strict) should return issues without raising."""
    old_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    panel = _make_panel(old_date)
    bm    = panel.groupby(level="date")["close"].mean()
    guard = StaleDataGuard(max_lag_days=7)
    issues = guard.check(panel, bm)
    assert len(issues) > 0
    assert any(i.severity == "error" for i in issues)
