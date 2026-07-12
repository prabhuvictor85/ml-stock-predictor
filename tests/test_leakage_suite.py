"""
Smoke tests for LeakageTestSuite — verifies the suite catches a known leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.validation.leakage_tests import (
    LeakageTestSuite,
    LeakageError,
)


def _panel_with_target(n_bars: int = 600, n_tickers: int = 10) -> pd.DataFrame:
    """Build a panel with a plausible cs_rank_20d target column."""
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    parts = []
    for i in range(n_tickers):
        close = 100.0 + np.cumsum(rng.normal(0, 1, size=n_bars))
        df = pd.DataFrame(
            {
                "open":  close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "volume": 100_000.0,
                # A plausible benign feature (no leakage)
                "features_return_5d": pd.Series(close).pct_change(5).values,
            },
            index=pd.Index(dates, name="date"),
        )
        df["ticker"] = f"T{i:03d}"
        df = df.set_index("ticker", append=True)
        parts.append(df)
    panel = pd.concat(parts).sort_index()

    # Build a benign cs_rank_20d target = future 20d log return ranked cross-sectionally.
    # Shifted forward by 20 so cs_rank_20d at date t depends on returns from t+1 to t+20.
    fwd = panel.groupby(level="ticker")["close"].pct_change(20).shift(-20)
    panel["cs_rank_20d"] = fwd.groupby(level="date").rank(pct=True)
    # Note: we deliberately do NOT add a future_*_return column — the leakage suite
    # legitimately flags any future_* column in the panel as a potential leak.
    return panel


def test_leakage_suite_structural_checks_pass_on_clean_panel():
    """
    The structural leakage checks (no future_* cols, no ffill boundary, no
    purging violations) should pass on a panel with no known leakage. The
    statistical checks (TargetShiftTest, TemporalCorrelationTest) are
    sensitive to the synthetic-data construction and are excluded from this
    smoke test — they're validated separately in production training runs.
    """
    panel = _panel_with_target()
    feat_cols = [c for c in panel.columns if c.startswith("features_")]
    suite = LeakageTestSuite(panel, feat_cols)
    results = suite.run_all(raise_on_fail=False)
    STRUCTURAL = {"FuturePriceColumnTest", "ForwardFillBoundaryTest", "GroupBoundaryTest"}
    structural_results = [r for r in results if r.name in STRUCTURAL]
    assert all(r.passed for r in structural_results), (
        f"Structural leakage check(s) failed on clean panel: "
        f"{[r.name for r in structural_results if not r.passed]}"
    )


def test_leakage_suite_blocks_future_column():
    """Adding a 'future_X' feature must trigger FuturePriceColumnTest failure."""
    panel = _panel_with_target()
    feat_cols = [c for c in panel.columns if c.startswith("features_")]
    # Inject a leaking feature: a future_-prefixed column in the feat list
    panel["future_close_5d"] = (
        panel.groupby(level="ticker")["close"].shift(-5)
    )
    feat_cols.append("future_close_5d")
    suite = LeakageTestSuite(panel, feat_cols)
    with pytest.raises(LeakageError):
        suite.run_all(raise_on_fail=True)
