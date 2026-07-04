"""
Pivot feature family tests (pipeline/features/pivots.py + engineer.py wiring).

Two layers:
  - Engine-level (fast): PivotFeatureEngine.compute() directly — golden formula
    values, TC/BC normalization, neutral trend band, truncation invariance
    (the property that licenses the fold-recompute no-op), MTF no-lookahead,
    warmup NaNs, per-ticker statelessness.
  - Build-level (slower, shared module fixtures): the features_pivot_* vocabulary
    in the assembled panel, default-OFF guarantee, fold-recompute passthrough,
    winsorization survival, per-ticker isolation on a non-winsorized column.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
from pipeline.features.ict_features import _wilder_atr
from pipeline.features.pivots import (
    PivotFeatureEngine, PIVOT_FEATURE_COLS, PIVOT_WINSORIZE_EXCLUDE,
    pivot_features_enabled,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_atr(g: pd.DataFrame) -> np.ndarray:
    a = _wilder_atr(g["high"].values, g["low"].values, g["close"].values, 14)
    floor = np.abs(g["close"].values) * 5e-4
    return np.where(np.isnan(a) | (a <= 0), np.nan, np.where(a > floor, a, floor))


def _synth_ticker(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = np.maximum(10.0, 100.0 + rng.normal(0, 1, n).cumsum())
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.006, 0.006, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def _make_panel(n_tickers: int, n_days: int = 320, seed: int = 42) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = np.maximum(10.0, 100.0 + rng.normal(0, 1, n).cumsum())
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.integers(50_000, 500_000, n).astype(float)
    sectors = [["IT", "Finance", "Energy"][i % 3] for _, t in idx for i in [int(t[1:])]]
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
        "in_universe": True, "sector": sectors,
    }, index=idx)


def _benchmark(n_days: int = 320) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rng = np.random.default_rng(99)
    return pd.Series(np.maximum(1.0, 1000.0 + rng.normal(0, 5, n_days).cumsum()), index=dates)


def _build_with_env(panel, pivot_on: bool):
    """Build a panel with PIVOT_FEATURES forced on/off, restoring env after."""
    from pipeline.config import get_config
    prev = os.environ.get("PIVOT_FEATURES")
    if pivot_on:
        os.environ["PIVOT_FEATURES"] = "1"
    else:
        os.environ.pop("PIVOT_FEATURES", None)
    try:
        return FeatureEngineer(get_config("nse"), _benchmark(len(panel.index.levels[0]))).build(panel.copy())
    finally:
        if prev is None:
            os.environ.pop("PIVOT_FEATURES", None)
        else:
            os.environ["PIVOT_FEATURES"] = prev


@pytest.fixture(scope="module")
def raw2():
    """Shared 2-ticker raw panel so built_on and built_solo see IDENTICAL T0 data
    (a fresh _make_panel(1) would draw different interleaved RNG values for T0)."""
    return _make_panel(2)


@pytest.fixture(scope="module")
def built_on(raw2):
    return _build_with_env(raw2, pivot_on=True)


@pytest.fixture(scope="module")
def built_solo(raw2):
    solo = raw2[raw2.index.get_level_values("ticker") == "T0"].copy()
    return _build_with_env(solo, pivot_on=True)


@pytest.fixture(scope="module")
def built_off(raw2):
    solo = raw2[raw2.index.get_level_values("ticker") == "T0"].copy()
    return _build_with_env(solo, pivot_on=False)


# ── Engine-level (fast) ────────────────────────────────────────────────────────

def test_golden_values():
    """Hand-computed levels for prev bar H=110, L=90, C=102; evaluated on the next
    bar with ATR=2 so dist = (close - level)/2."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    grp = pd.DataFrame({
        "open": [100.0, 100.0, 100.0],
        "high": [110.0, 108.0, 108.0],
        "low":  [90.0, 96.0, 96.0],
        "close": [102.0, 100.0, 100.0],
    }, index=idx)
    atr = np.array([np.nan, 2.0, 2.0])
    f = PivotFeatureEngine().compute(grp, atr)
    row = f.iloc[1]
    close = 100.0
    PP = (110 + 90 + 102) / 3
    exp = {
        "pivot_dist_pp_atr": (close - PP) / 2,
        "pivot_dist_r1_atr": (close - (2 * PP - 90)) / 2,
        "pivot_dist_s1_atr": (close - (2 * PP - 110)) / 2,
        "pivot_dist_bc_atr": (close - 100.0) / 2,             # BC = (110+90)/2 = 100
        "pivot_dist_tc_atr": (close - ((PP - 100.0) + PP)) / 2,
        "pivot_dist_h3_atr": (close - (102 + 20 * 1.1 / 4)) / 2,
        "pivot_dist_h4_atr": (close - (102 + 20 * 1.1 / 2)) / 2,
        "pivot_dist_l3_atr": (close - (102 - 20 * 1.1 / 4)) / 2,
        "pivot_dist_l4_atr": (close - (102 - 20 * 1.1 / 2)) / 2,
        "pivot_dist_h5_atr": (close - (110 / 90) * 102) / 2,
        "pivot_dist_l5_atr": (close - (102 - ((110 / 90) * 102 - 102))) / 2,
    }
    for col, want in exp.items():
        assert abs(float(row[col]) - want) < 1e-3, f"{col}: got {row[col]} want {want}"


def test_tc_bc_normalized():
    """When the raw TC formula lands below the raw BC (prev H=110,L=90,C=95), the
    engine must swap so emitted TC >= BC and width >= 0."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    grp = pd.DataFrame({
        "open": [100.0, 100.0, 100.0],
        "high": [110.0, 108.0, 108.0],
        "low":  [90.0, 96.0, 96.0],
        "close": [95.0, 100.0, 100.0],   # PP=98.33, BC=100 > raw TC=96.67 → swap
    }, index=idx)
    atr = np.array([np.nan, 2.0, 2.0])
    f = PivotFeatureEngine().compute(grp, atr)
    row = f.iloc[1]
    # dist = (close - level)/atr → a HIGHER level gives a LOWER (more negative) dist.
    # TC must be the higher of the two, so dist_tc <= dist_bc.
    assert float(row["pivot_dist_tc_atr"]) <= float(row["pivot_dist_bc_atr"]) + 1e-9
    # width is non-negative everywhere it is defined
    w = f["pivot_cpr_width_atr"].dropna()
    assert (w >= -1e-9).all()


def test_trend_side_neutral_inside_band():
    """A close strictly inside [BC, TC] must set neither trend dummy."""
    # Build a flat series so the band brackets the close.
    idx = pd.bdate_range("2020-01-01", periods=40)
    base = 100.0
    grp = pd.DataFrame({
        "open": base, "high": base + 3.0, "low": base - 3.0, "close": base,
    }, index=idx).astype(float)
    atr = _safe_atr(grp)
    f = PivotFeatureEngine().compute(grp, atr)
    valid = f["pivot_cpr_trend_bull"].notna()
    # close == PP == mid of band → inside band → both flags 0
    assert (f.loc[valid, "pivot_cpr_trend_bull"] == 0).all()
    assert (f.loc[valid, "pivot_cpr_trend_bear"] == 0).all()


def test_truncation_invariance():
    """Load-bearing: pivot features are pure trailing functions of OHLC, so
    truncating future rows must not change any past row's value. This licenses the
    fold-recompute no-op in engineer.recompute_fold_features()."""
    grp = _synth_ticker(300, seed=7)
    eng = PivotFeatureEngine()
    full = eng.compute(grp, _safe_atr(grp))
    cut = grp.iloc[:200]
    part = eng.compute(cut, _safe_atr(cut))
    a = full.iloc[:200].to_numpy(dtype=float)
    b = part.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    assert (np.isclose(a, b, equal_nan=False) | both_nan).all(), "pivot features are not truncation-invariant"


def test_mtf_no_lookahead():
    """Monthly pivot on any day of month M+1 must derive from month M — and the
    LAST day of month M must still see month M-1 (allow_exact_matches=False)."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(3)
    close = 100 + rng.normal(0, 1, len(idx)).cumsum()
    grp = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    eng = PivotFeatureEngine()
    mtf = eng._mtf_pivots(grp["high"], grp["low"], grp["close"], grp.index)
    monthly_pp = mtf["monthly_pp"]
    # Independently compute April's monthly pivot from April OHLC.
    apr = grp[(grp.index >= "2020-04-01") & (grp.index <= "2020-04-30")]
    apr_pp = (apr["high"].max() + apr["low"].min() + apr["close"].iloc[-1]) / 3.0
    # A mid-May row should carry April's pivot.
    mid_may = pd.Timestamp("2020-05-15")
    while mid_may not in monthly_pp.index:
        mid_may += pd.Timedelta(days=1)
    assert abs(float(monthly_pp.loc[mid_may]) - apr_pp) < 1e-6
    # The last trading day of May must NOT yet see May's own pivot.
    may_rows = monthly_pp[(monthly_pp.index >= "2020-05-01") & (monthly_pp.index <= "2020-05-31")]
    may = grp[(grp.index >= "2020-05-01") & (grp.index <= "2020-05-31")]
    may_pp = (may["high"].max() + may["low"].min() + may["close"].iloc[-1]) / 3.0
    assert abs(float(may_rows.iloc[-1]) - may_pp) > 1e-9, "month-end row leaked its own (incomplete) month"


def test_warmup_nans():
    grp = _synth_ticker(300, seed=1)
    f = PivotFeatureEngine().compute(grp, _safe_atr(grp))
    # First row: no prior session → level distances NaN.
    assert np.isnan(f["pivot_dist_pp_atr"].iloc[0])
    # Width percentile needs min_periods=10 of width history.
    assert f["pivot_cpr_width_pctile"].iloc[:10].isna().all()
    # Yearly MTF pivot needs a completed prior year → NaN through the first year.
    first_year = f[f.index < pd.Timestamp("2020-01-01")]["pivot_dist_yearly_pp_atr"]
    assert first_year.isna().all()


def test_engine_is_stateless():
    """Same input → same output regardless of prior calls (no hidden state that
    could leak between tickers in the build loop)."""
    eng = PivotFeatureEngine()
    g0 = _synth_ticker(200, seed=11)
    g1 = _synth_ticker(200, seed=22)
    first = eng.compute(g0, _safe_atr(g0))
    _ = eng.compute(g1, _safe_atr(g1))          # different ticker in between
    again = eng.compute(g0, _safe_atr(g0))
    pd.testing.assert_frame_equal(first, again)


def test_vocabulary_frozen():
    assert len(PIVOT_FEATURE_COLS) == 69
    assert all(c.startswith("pivot_") for c in PIVOT_FEATURE_COLS)
    g = _synth_ticker(120, seed=5)
    f = PivotFeatureEngine().compute(g, _safe_atr(g))
    assert list(f.columns) == PIVOT_FEATURE_COLS
    assert all(str(dt) == "float32" for dt in f.dtypes)


# ── Build-level (shared module fixtures) ───────────────────────────────────────

def test_prefix_and_vocabulary_in_panel(built_on):
    piv = sorted(c for c in built_on.columns if c.startswith(f"{FEATURE_PREFIX}pivot_"))
    expected = sorted(f"{FEATURE_PREFIX}{c}" for c in PIVOT_FEATURE_COLS)
    assert piv == expected, "features_pivot_* vocabulary mismatch in built panel"
    # No raw (un-prefixed) pivot_* columns leaked.
    raw = [c for c in built_on.columns if c.startswith("pivot_")]
    assert not raw, f"raw pivot_* columns leaked into panel: {raw}"


def test_default_off_no_pivot_columns(built_off):
    piv = [c for c in built_off.columns if c.startswith(f"{FEATURE_PREFIX}pivot_")]
    assert not piv, "pivot columns present despite PIVOT_FEATURES unset (default must be OFF)"
    assert not pivot_features_enabled()


def test_fold_recompute_passthrough(built_on):
    """recompute_fold_features must leave features_pivot_* untouched (it only
    rebuilds ICT/zone state)."""
    from pipeline.config import get_config
    fe = FeatureEngineer(get_config("nse"), _benchmark(320))
    cutoff = built_on.index.get_level_values("date").min() + pd.Timedelta(days=200)
    before = built_on[[c for c in built_on.columns if c.startswith(f"{FEATURE_PREFIX}pivot_")]].copy()
    recomputed = fe.recompute_fold_features(built_on, cutoff)
    after = recomputed[[c for c in recomputed.columns if c.startswith(f"{FEATURE_PREFIX}pivot_")]]
    # align (recompute reorders index levels) and compare
    after = after.reorder_levels(["date", "ticker"]).sort_index()
    before = before.sort_index()
    common = before.columns
    a = before[common].to_numpy(dtype=float)
    b = after.loc[before.index, common].to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    assert (np.isclose(a, b, equal_nan=False) | both_nan).all(), "pivot columns changed during fold recompute"


def test_winsorize_excludes_streaks(built_on):
    """The excluded columns (streaks/counts/ages/percentile) must be present and
    keep their discrete/bounded ranges — not clipped into fractional values."""
    for name in PIVOT_WINSORIZE_EXCLUDE:
        assert name in built_on.columns, f"{name} missing from panel"
    pctile = built_on[f"{FEATURE_PREFIX}pivot_cpr_width_pctile"].dropna()
    assert (pctile >= -1e-6).all() and (pctile <= 1 + 1e-6).all()
    # Streaks/counts remain whole numbers (winsorization would introduce fractions).
    streak = built_on[f"{FEATURE_PREFIX}pivot_cpr_trend_streak"].dropna()
    assert np.allclose(streak, np.round(streak)), "trend streak was winsorized into non-integers"


def test_per_ticker_isolation(built_on, built_solo):
    """A non-winsorized pivot column for T0 must be identical whether T0 is built
    alone or alongside T1 (per-ticker computation, no cross-ticker leakage).
    Uses an excluded column so cross-sectional winsorization can't confound it."""
    col = f"{FEATURE_PREFIX}pivot_virgin_cpr_age"
    t0_multi = built_on.xs("T0", level="ticker")[col]
    t0_solo = built_solo.xs("T0", level="ticker")[col]
    joined = pd.concat([t0_multi.rename("multi"), t0_solo.rename("solo")], axis=1).dropna()
    assert len(joined) > 50
    assert np.allclose(joined["multi"], joined["solo"]), "pivot values leaked across tickers"
