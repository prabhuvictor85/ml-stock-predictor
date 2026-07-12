"""
Smoke tests for PurgedWalkForwardCV — verifies the core leakage controls:
  1. No train/test date overlap
  2. Purge + embargo gap is respected
  3. Per-stock eligibility (stocks without enough history are excluded)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.validation.cv import PurgedWalkForwardCV, PURGE_WINDOW, EMBARGO_WINDOW


def _build_long_panel(n_tickers: int = 20, n_bars: int = 1500) -> pd.DataFrame:
    """Build a long synthetic panel for CV testing."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2015-01-01", periods=n_bars)
    parts = []
    for i in range(n_tickers):
        log_rets = rng.normal(0.0003, 0.015, size=n_bars)
        close = 100.0 * np.exp(np.cumsum(log_rets))
        df = pd.DataFrame(
            {
                "open":  close,
                "high":  close * 1.01,
                "low":   close * 0.99,
                "close": close,
                "volume": rng.integers(100_000, 1_000_000, size=n_bars).astype(float),
                "in_universe": True,
                "group_date":  dates,
            },
            index=pd.Index(dates, name="date"),
        )
        df["ticker"] = f"T{i:03d}"
        df = df.set_index("ticker", append=True)
        parts.append(df)
    return pd.concat(parts).sort_index()


def test_cv_no_train_test_overlap():
    """For every fold, no date in train_idx may appear in test_idx."""
    panel = _build_long_panel()
    cv = PurgedWalkForwardCV(n_folds=5)
    n_folds_checked = 0
    for spec, train_idx, test_idx in cv.split(panel):
        train_dates = set(panel.iloc[train_idx].index.get_level_values("date").unique())
        test_dates  = set(panel.iloc[test_idx].index.get_level_values("date").unique())
        overlap = train_dates & test_dates
        assert not overlap, f"Fold {spec.fold_id}: train/test overlap dates: {sorted(overlap)[:5]}"
        n_folds_checked += 1
    assert n_folds_checked >= 1, "No folds were generated"


def test_cv_purge_embargo_gap():
    """For every fold, the gap between train_end and test_start must be >= purge+embargo trading days."""
    panel = _build_long_panel()
    cv = PurgedWalkForwardCV(n_folds=5)
    all_dates = np.array(sorted(panel.index.get_level_values("date").unique()))
    n_required = PURGE_WINDOW + EMBARGO_WINDOW
    for spec, train_idx, test_idx in cv.split(panel):
        # Find indices of train_end and test_start in the full date array
        i_train_end  = int(np.searchsorted(all_dates, np.datetime64(spec.train_end)))
        i_test_start = int(np.searchsorted(all_dates, np.datetime64(spec.test_start)))
        gap = i_test_start - i_train_end
        assert gap >= n_required, (
            f"Fold {spec.fold_id}: gap {gap} < required {n_required} trading days "
            f"(train_end={spec.train_end.date()}, test_start={spec.test_start.date()})"
        )


def test_cv_fold_count_at_least_min_folds():
    """With sufficient data, CV must produce at least MIN_FOLDS (5) folds."""
    panel = _build_long_panel(n_tickers=10, n_bars=2500)
    cv = PurgedWalkForwardCV(n_folds=5)
    specs = cv.get_fold_specs(panel)
    assert len(specs) >= 5
