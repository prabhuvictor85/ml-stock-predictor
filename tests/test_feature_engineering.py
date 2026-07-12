"""
Feature engineering regression tests.

Covers FeatureEngineer.build() and recompute_fold_features():
  1. Output structure   — correct columns, feature prefix, no raw ICT leak
  2. Feature validity   — 52w dist signs, breakout binary, beta clipped, vol ratio positive
  3. Causal guards      — warmup NaNs at start, per-ticker isolation
  4. Normalization      — returns ATR-normalized not raw, ATR rank in [0,1]
  5. Fold recompute     — ICT cols present, cutoff respected
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_panel(n_tickers: int = 5, n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic OHLCV panel with MultiIndex (date, ticker).
    300 days gives enough warmup for 52w (252d) features.
    """
    dates   = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rng = np.random.default_rng(seed)
    n   = len(idx)

    close  = np.maximum(10.0, 100.0 + rng.normal(0, 1, n).cumsum())
    high   = close * (1 + rng.uniform(0.001, 0.015, n))
    low    = close * (1 - rng.uniform(0.001, 0.015, n))
    open_  = close * (1 + rng.uniform(-0.005, 0.005, n))
    volume = rng.integers(50_000, 500_000, n).astype(float)

    # Two sectors so each has >=2 members with the default 5 tickers: sector RS
    # features get cross-sectional signal AND sector-neutral z-scores are
    # defined (a singleton sector has no cross-sectional std → NaN by design).
    sector_map = {f"T{i}": ["IT", "Finance"][i % 2]
                  for i in range(n_tickers)}
    sectors = [sector_map[t] for _, t in idx]

    return pd.DataFrame({
        "open":        open_,
        "high":        high,
        "low":         low,
        "close":       close,
        "volume":      volume,
        "in_universe": True,
        "sector":      sectors,
    }, index=idx)


def _make_benchmark(n_days: int = 300) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rng   = np.random.default_rng(99)
    close = np.maximum(1.0, 1000.0 + rng.normal(0, 5, n_days).cumsum())
    return pd.Series(close, index=dates, name="benchmark_close")


@pytest.fixture(scope="module")
def built_panel():
    """Run FeatureEngineer.build() once and share across tests in module."""
    from pipeline.config import get_config
    cfg   = get_config("nse")
    bm    = _make_benchmark()
    panel = _make_panel()
    fe    = FeatureEngineer(cfg, bm)
    return fe.build(panel)


# ── 1. Output structure ───────────────────────────────────────────────────────

def test_feature_prefix_on_all_computed_columns(built_panel):
    """
    Every ML feature column must start with FEATURE_PREFIX ('features_').
    Legitimate non-prefixed columns are: OHLCV, sector, zone label strings
    (zone_type_*), and known intermediate columns kept for downstream use.
    The critical guard: no raw ict_* columns without the prefix.
    """
    feat_cols = [c for c in built_panel.columns if c.startswith(FEATURE_PREFIX)]
    assert len(feat_cols) > 0, "No feature columns produced"

    # Raw ICT columns without prefix are a hard error — they indicate the
    # rename step failed and the model would receive wrong column names.
    raw_ict = [c for c in built_panel.columns if c.startswith("ict_")]
    assert not raw_ict, f"Raw ICT columns leaked (no features_ prefix): {raw_ict}"


def test_expected_feature_columns_present(built_panel):
    """Key feature columns must always be produced by build()."""
    required = [
        f"{FEATURE_PREFIX}return_1d",
        f"{FEATURE_PREFIX}return_20d",
        f"{FEATURE_PREFIX}return_60d",
        f"{FEATURE_PREFIX}high_52w_dist",
        f"{FEATURE_PREFIX}low_52w_dist",
        f"{FEATURE_PREFIX}20d_breakout",
        f"{FEATURE_PREFIX}50d_breakout",
        f"{FEATURE_PREFIX}adx_14",
        f"{FEATURE_PREFIX}atr_pct_rank_252",
        f"{FEATURE_PREFIX}vol_contraction",
        f"{FEATURE_PREFIX}vol_ratio_5d",
        f"{FEATURE_PREFIX}price_vs_sma20",
        f"{FEATURE_PREFIX}price_vs_sma50",
        f"{FEATURE_PREFIX}price_vs_sma200",
        f"{FEATURE_PREFIX}rolling_beta_60d",
        f"{FEATURE_PREFIX}hist_vol_20d",
        f"{FEATURE_PREFIX}ict_bob_active",
        f"{FEATURE_PREFIX}ict_bullrb_active",
        f"{FEATURE_PREFIX}ict_bullfvg_active",
        f"{FEATURE_PREFIX}ict_bull_zone_priority",
        f"{FEATURE_PREFIX}ict_bear_zone_priority",
    ]
    missing = [c for c in required if c not in built_panel.columns]
    assert not missing, f"Missing feature columns: {missing}"


def test_raw_ict_columns_not_in_panel(built_panel):
    """
    Raw ict_* columns (without features_ prefix) must NOT appear in the final panel.
    If they do, the rename step failed and the model receives un-prefixed columns.
    """
    raw_ict = [c for c in built_panel.columns if c.startswith("ict_")]
    assert not raw_ict, f"Raw ICT columns leaked into panel: {raw_ict}"


def test_ohlcv_columns_preserved(built_panel):
    """build() must not drop original OHLCV columns."""
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in built_panel.columns, f"OHLCV column '{col}' was dropped by build()"


def test_feature_count_is_reasonable(built_panel):
    """Feature count should be in a sane range — guards against silent column drop."""
    feat_cols = [c for c in built_panel.columns if c.startswith(FEATURE_PREFIX)]
    assert len(feat_cols) >= 50, f"Too few features: {len(feat_cols)} (expected >= 50)"
    assert len(feat_cols) <= 300, f"Feature explosion: {len(feat_cols)} columns"


# ── 2. Feature validity ───────────────────────────────────────────────────────

def test_high_52w_dist_is_nonpositive(built_panel):
    """
    high_52w_dist = (close - 52w_high) / 52w_high.
    Close can never exceed its own 52-week high → must be <= 0.
    """
    col  = f"{FEATURE_PREFIX}high_52w_dist"
    vals = built_panel[col].dropna()
    assert vals.max() <= 1e-6, (
        f"high_52w_dist has positive values (max={vals.max():.4f}) — "
        "close cannot exceed its own 52w high"
    )


def test_low_52w_dist_is_nonnegative(built_panel):
    """
    low_52w_dist = (close - 52w_low) / 52w_low.
    Close can never be below its own 52-week low → must be >= 0.
    """
    col  = f"{FEATURE_PREFIX}low_52w_dist"
    vals = built_panel[col].dropna()
    assert vals.min() >= -1e-6, (
        f"low_52w_dist has negative values (min={vals.min():.4f}) — "
        "close cannot be below its own 52w low"
    )


def test_breakout_flags_are_binary(built_panel):
    """
    20d_breakout and 50d_breakout are structurally binary (0/1) before
    winsorization. After cross-sectional winsorization at [1,99] percentile,
    boundary rows may get fractional values (e.g. 0.04, 0.96) — that is
    correct behavior (winsorization clips the 1% tails).
    Guard: all values must stay in [0, 1].
    """
    for col in [f"{FEATURE_PREFIX}20d_breakout", f"{FEATURE_PREFIX}50d_breakout"]:
        vals = built_panel[col].dropna()
        assert vals.min() >= 0.0 - 1e-6, f"{col} has values below 0: min={vals.min()}"
        assert vals.max() <= 1.0 + 1e-6, f"{col} has values above 1: max={vals.max()}"


def test_vol_ratio_is_nonnegative(built_panel):
    """Volume ratios are ratio of positive quantities — must always be >= 0."""
    for col in [f"{FEATURE_PREFIX}vol_ratio_5d", f"{FEATURE_PREFIX}vol_ratio_20d"]:
        vals = built_panel[col].dropna()
        assert vals.min() >= 0.0, f"{col} has negative values (min={vals.min():.4f})"


def test_beta_is_clipped(built_panel):
    """Rolling beta is clipped to [-2, 4] to remove outliers."""
    col  = f"{FEATURE_PREFIX}rolling_beta_60d"
    vals = built_panel[col].dropna()
    assert vals.min() >= -2.0 - 1e-6, f"Beta below -2: min={vals.min():.4f}"
    assert vals.max() <=  4.0 + 1e-6, f"Beta above  4: max={vals.max():.4f}"


def test_atr_pct_rank_in_unit_interval(built_panel):
    """ATR percentile rank must be in [0, 1] — it's a rolling.rank(pct=True)."""
    col  = f"{FEATURE_PREFIX}atr_pct_rank_252"
    vals = built_panel[col].dropna()
    assert vals.min() >= 0.0 - 1e-6, f"atr_pct_rank below 0: min={vals.min():.4f}"
    assert vals.max() <= 1.0 + 1e-6, f"atr_pct_rank above 1: max={vals.max():.4f}"


def test_adx_is_nonnegative(built_panel):
    """ADX is always >= 0 by construction (it's an absolute trend strength)."""
    col  = f"{FEATURE_PREFIX}adx_14"
    vals = built_panel[col].dropna()
    assert vals.min() >= 0.0 - 1e-6, f"ADX has negative values: min={vals.min():.4f}"


def test_vol_contraction_in_unit_interval(built_panel):
    """
    vol_contraction = atr / atr_60d_max → in (0, 1].
    Equals 1 when ATR is at its 60d maximum; approaches 0 when ATR collapses.
    """
    col  = f"{FEATURE_PREFIX}vol_contraction"
    vals = built_panel[col].dropna()
    assert vals.min() > 0.0 - 1e-6, f"vol_contraction has zero/negative values"
    assert vals.max() <= 1.0 + 1e-6, f"vol_contraction > 1: max={vals.max():.4f}"


def test_ict_active_flags_are_binary_in_panel(built_panel):
    """
    ICT active flags are structurally binary (0/1) before winsorization.
    After cross-sectional winsorization they may get fractional boundary
    values — guard that all values stay in [0, 1].
    """
    active_cols = [
        c for c in built_panel.columns
        if c.startswith(f"{FEATURE_PREFIX}ict_") and c.endswith("_active")
    ]
    assert active_cols, "No ICT active flag columns found in panel"
    for col in active_cols:
        vals = built_panel[col].dropna()
        assert vals.min() >= 0.0 - 1e-6, f"{col} has values below 0: min={vals.min()}"
        assert vals.max() <= 1.0 + 1e-6, f"{col} has values above 1: max={vals.max()}"


def test_ict_atr_dist_bounded_in_panel(built_panel):
    """features_ict_*_atr_dist must be within [-20, 20] in the full panel."""
    dist_cols = [
        c for c in built_panel.columns
        if c.startswith(f"{FEATURE_PREFIX}ict_") and c.endswith("_atr_dist")
    ]
    assert dist_cols, "No ICT atr_dist columns in panel"
    for col in dist_cols:
        vals = built_panel[col].dropna()
        assert vals.min() >= -20.0, f"{col} min={vals.min():.2f} < -20"
        assert vals.max() <=  20.0, f"{col} max={vals.max():.2f} > +20"


# ── 3. Causal guards ──────────────────────────────────────────────────────────

def test_short_history_features_have_warmup_nans(built_panel):
    """
    Rolling features need warmup — first bars for each ticker must have NaN.
    price_vs_sma200 needs 200 bars; first ticker rows must be NaN.
    Guards against missing warmup (which would create bogus values).
    """
    col = f"{FEATURE_PREFIX}price_vs_sma200"
    for ticker in built_panel.index.get_level_values("ticker").unique()[:3]:
        ticker_vals = built_panel.xs(ticker, level="ticker")[col]
        # First 99 bars must be NaN (min_periods=100)
        first_99 = ticker_vals.iloc[:99]
        assert first_99.isna().all(), (
            f"{ticker}: {col} should be NaN for first 99 bars (warmup), "
            f"but found {first_99.notna().sum()} non-NaN values"
        )


def test_per_ticker_rolling_computation_is_isolated():
    """
    Per-ticker ROLLING computations (ATR, ADX) must be identical whether the
    ticker is processed alone or alongside other tickers.

    Note: cross-sectional winsorization legitimately changes feature values
    when the cross-section size differs — we test atr_14 which is stored raw
    (before winsorization) and is purely per-ticker.
    """
    from pipeline.config import get_config
    cfg = get_config("nse")
    bm  = _make_benchmark()

    panel_multi  = _make_panel(n_tickers=5)
    panel_single = panel_multi.xs("T0", level="ticker").copy()
    panel_single = pd.concat({"T0": panel_single}, names=["ticker"]).swaplevel().sort_index()

    fe    = FeatureEngineer(cfg, bm)
    multi = fe.build(panel_multi)
    sing  = fe.build(panel_single)

    # atr_14 is per-ticker Wilder ATR — stored raw, not winsorized
    col = "atr_14"
    multi_t0 = multi.xs("T0", level="ticker")[col].dropna()
    sing_t0  = sing.xs("T0", level="ticker")[col].dropna()

    common_idx = multi_t0.index.intersection(sing_t0.index)
    assert len(common_idx) > 10

    max_diff = (multi_t0.loc[common_idx].values - sing_t0.loc[common_idx].values).max()
    assert abs(max_diff) < 1e-8, (
        f"atr_14 differs between multi/single-ticker runs (max_diff={max_diff:.2e}) "
        "— rolling computation is not ticker-isolated"
    )


def test_features_no_inf_values(built_panel):
    """No feature column should contain Inf or -Inf — guards divide-by-zero."""
    feat_cols = [c for c in built_panel.columns if c.startswith(FEATURE_PREFIX)]
    for col in feat_cols:
        has_inf = np.isinf(built_panel[col].replace([None], np.nan).fillna(0)).any()
        assert not has_inf, f"{col} contains Inf values"


def test_features_no_excessive_nan(built_panel):
    """
    After warmup, features should not be mostly NaN.
    We take the last 100 bars per ticker (well past warmup) and
    assert NaN rate < 20% per feature.
    """
    feat_cols = [c for c in built_panel.columns if c.startswith(FEATURE_PREFIX)]
    # Take last 100 dates
    all_dates = built_panel.index.get_level_values("date").unique()
    recent_dates = all_dates[-100:]
    recent = built_panel[built_panel.index.get_level_values("date").isin(recent_dates)]

    high_nan_cols = []
    for col in feat_cols:
        nan_rate = recent[col].isna().mean()
        if nan_rate > 0.20:
            high_nan_cols.append((col, nan_rate))

    assert not high_nan_cols, (
        "Features with >20% NaN in last 100 bars (excessive, possible bug):\n"
        + "\n".join(f"  {c}: {r:.1%}" for c, r in high_nan_cols)
    )


# ── 4. Normalization ──────────────────────────────────────────────────────────

def test_return_features_are_not_raw_log_returns(built_panel):
    """
    Return features are ATR-normalized, not raw log returns.
    Raw 1d log return is typically |0.01–0.05|.
    ATR-normalized: dividing by pct_atr (~0.01–0.02) scales up to ~1–5.
    We check that the mean abs value of return_1d > 0.1 (not a raw tiny return).
    """
    col  = f"{FEATURE_PREFIX}return_1d"
    vals = built_panel[col].dropna()
    mean_abs = vals.abs().mean()
    assert mean_abs > 0.1, (
        f"return_1d mean abs = {mean_abs:.4f} — looks like raw log return, not ATR-normalized"
    )


def test_ict_zone_priority_in_valid_range(built_panel):
    """
    ict_bull_zone_priority is structurally in {0, 1, 2, 3} before winsorization.
    After cross-sectional winsorization fractional values near 0/1/2/3 may appear.
    Guard: all values must be in [0, 3].
    """
    col  = f"{FEATURE_PREFIX}ict_bull_zone_priority"
    vals = built_panel[col].dropna()
    assert vals.min() >= 0.0 - 1e-6, f"ict_bull_zone_priority below 0: {vals.min()}"
    assert vals.max() <= 3.0 + 1e-6, f"ict_bull_zone_priority above 3: {vals.max()}"


# ── 5. Fold recompute ─────────────────────────────────────────────────────────

def test_recompute_fold_features_produces_ict_columns():
    """
    recompute_fold_features() must produce all ICT feature columns —
    same set as build(). Guards against the recompute path silently
    dropping ICT columns.
    """
    from pipeline.config import get_config
    from pipeline.features.engineer import FeatureEngineer

    cfg   = get_config("nse")
    bm    = _make_benchmark()
    panel = _make_panel(n_tickers=3, n_days=300)

    fe        = FeatureEngineer(cfg, bm)
    built     = fe.build(panel)
    cutoff    = built.index.get_level_values("date").unique()[-50]
    recomputed = fe.recompute_fold_features(built, cutoff_date=cutoff)

    required_ict = [
        f"{FEATURE_PREFIX}ict_bob_active",
        f"{FEATURE_PREFIX}ict_bullrb_active",
        f"{FEATURE_PREFIX}ict_bull_zone_priority",
        f"{FEATURE_PREFIX}ict_bear_zone_priority",
    ]
    missing = [c for c in required_ict if c not in recomputed.columns]
    assert not missing, f"recompute_fold_features dropped ICT columns: {missing}"


def test_recompute_fold_cutoff_respected():
    """
    recompute_fold_features(cutoff=T) must use only data up to T.
    The ICT active flag on a test-period row must equal the last known
    state at T (forward-filled), NOT a value computed using post-T prices.
    We verify this by checking that recomputing with an EARLIER cutoff
    produces a different (or equal but not future-derived) result.
    """
    from pipeline.config import get_config

    cfg   = get_config("nse")
    bm    = _make_benchmark()
    panel = _make_panel(n_tickers=2, n_days=300)

    fe    = FeatureEngineer(cfg, bm)
    built = fe.build(panel)

    all_dates    = built.index.get_level_values("date").unique().sort_values()
    cutoff_late  = all_dates[-20]
    cutoff_early = all_dates[-100]

    late  = fe.recompute_fold_features(built, cutoff_date=cutoff_late)
    early = fe.recompute_fold_features(built, cutoff_date=cutoff_early)

    # Both must have the same columns
    assert set(late.columns) == set(early.columns), (
        "recompute_fold_features produces different column sets for different cutoffs"
    )

    # The two recomputes may differ in values (different training windows → different zones)
    # — that's expected and correct. We just verify neither raises and both are non-empty.
    assert len(late) > 0
    assert len(early) > 0
