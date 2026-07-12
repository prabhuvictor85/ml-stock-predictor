"""
Phase-4 feature family tests (Exp-401..404: GK vol, skew/kurt, VWAP/CMF/OBV,
residual momentum, choppiness, variance ratio, cross-sectional z-scores,
A/D thrust).

1. Gate — PHASE4_FEATURES=0 must produce a panel with ZERO Phase-4 columns,
   so a with/without A/B is a single env var, not a git revert.
2. Default ON — a build without the env var set carries every Phase-4 column.
3. NaN-native z-scores — undefined cross-sectional std (singleton sector,
   zero-dispersion date) yields NaN, never a fabricated "exactly average" 0.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features.engineer import (
    FEATURE_PREFIX,
    PHASE4_FEATURE_COLS,
    FeatureEngineer,
)

PHASE4_PREFIXED = [f"{FEATURE_PREFIX}{c}" for c in PHASE4_FEATURE_COLS]
_CSZ_BASES = ["return_20d", "return_60d", "vol_ratio_20d",
              "residual_mom_20d", "residual_mom_60d"]
CSZ_PREFIXED = (
    [f"{FEATURE_PREFIX}{b}_csz" for b in _CSZ_BASES]
    + [f"{FEATURE_PREFIX}{b}_sec_csz" for b in _CSZ_BASES]
)


def _make_panel(n_tickers: int = 5, n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    dates   = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng     = np.random.default_rng(seed)
    n       = len(idx)
    close   = np.maximum(10.0, 100.0 + rng.normal(0, 1, n).cumsum())
    # Two sectors → each has >=2 members, so sector-neutral z-scores are defined.
    sector_map = {f"T{i}": ["IT", "Finance"][i % 2]
                  for i in range(n_tickers)}
    return pd.DataFrame({
        "open":        close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":        close * (1 + rng.uniform(0.001, 0.015, n)),
        "low":         close * (1 - rng.uniform(0.001, 0.015, n)),
        "close":       close,
        "volume":      rng.integers(50_000, 500_000, n).astype(float),
        "in_universe": True,
        "sector":      [sector_map[t] for _, t in idx],
    }, index=idx)


def _make_benchmark(n_days: int = 300) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rng   = np.random.default_rng(99)
    return pd.Series(np.maximum(1.0, 1000.0 + rng.normal(0, 5, n_days).cumsum()),
                     index=dates, name="benchmark_close")


def _build() -> pd.DataFrame:
    from pipeline.config import get_config
    fe = FeatureEngineer(get_config("nse"), _make_benchmark())
    return fe.build(_make_panel())


def test_gate_off_by_default_adds_no_phase4_columns(monkeypatch):
    # Default is OFF: the baseline panel must be Phase-4-free with no env set
    # (same convention as PIVOT_FEATURES).
    monkeypatch.delenv("PHASE4_FEATURES", raising=False)
    out = _build()
    present = [c for c in PHASE4_PREFIXED + CSZ_PREFIXED if c in out.columns]
    assert not present, f"default (gate-off) build still carries: {present}"
    stray = [c for c in out.columns if c.endswith("_csz") or c.endswith("_sec_csz")]
    assert not stray, f"unexpected z-score columns with gate off: {stray}"


def test_gate_on_adds_all_phase4_columns(monkeypatch):
    monkeypatch.setenv("PHASE4_FEATURES", "1")
    out = _build()
    missing = [c for c in PHASE4_PREFIXED + CSZ_PREFIXED if c not in out.columns]
    assert not missing, f"PHASE4_FEATURES=1 build is missing: {missing}"


def test_csz_nan_when_std_undefined():
    """Singleton sector and zero-dispersion dates must yield NaN, not 0.0."""
    from pipeline.config import get_config
    fe    = FeatureEngineer(get_config("nse"), _make_benchmark(10))
    dates = pd.bdate_range("2020-01-01", periods=2)
    idx   = pd.MultiIndex.from_product([dates, ["A", "B", "C"]],
                                       names=["date", "ticker"])
    col   = f"{FEATURE_PREFIX}return_20d"
    panel = pd.DataFrame({
        # date 0: dispersed values; date 1: all equal (zero cross-sectional std)
        col:      [1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
        # sector Y has exactly one member (ticker C) → sector std undefined
        "sector": ["X", "X", "Y", "X", "X", "Y"],
    }, index=idx)

    out = fe._add_cross_sectional_zscores(panel.copy())

    sec = out[f"{col}_sec_csz"].xs("C", level="ticker")
    assert sec.isna().all(), "singleton-sector z-score must be NaN, not 0.0"

    d1 = out.loc[(dates[1], slice(None)), f"{col}_csz"]
    assert d1.isna().all(), "zero-dispersion date z-score must be NaN, not 0.0"

    d0 = out.loc[(dates[0], slice(None)), f"{col}_csz"]
    assert np.isfinite(d0).all()
    assert abs(d0.mean()) < 1e-9, "well-defined z-scores must be centered per date"
