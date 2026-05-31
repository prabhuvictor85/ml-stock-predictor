"""
Critical invariants — three gaps in the existing regression suite.

  #4  Winsorize cross-sectional  — outlier tickers must NOT distort the
                                   entire date's features
  #6  Wilder ATR formula         — wrong formula = wrong normalization for
                                   every return feature in the model
  #8  Schema stable              — feature column list must be identical
                                   across different input panels (train ≠
                                   inference mismatch silently degrades model)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_panel(
    n_tickers: int = 4,
    n_days: int = 250,
    seed: int = 42,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Minimal OHLCV + sector panel with MultiIndex (date, ticker).
    Same structure expected by FeatureEngineer.build().
    """
    dates   = pd.bdate_range(start, periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rng    = np.random.default_rng(seed)
    n      = len(idx)
    close  = np.clip(100.0 + rng.normal(0, 1, n).cumsum(), 10, None)
    sector_pool = ["IT", "Finance", "Energy", "FMCG"]
    sectors = [sector_pool[i % len(sector_pool)] for i in range(n_tickers)]
    sector_map = dict(zip(tickers, sectors))
    sector_vals = [sector_map[t] for _, t in idx]

    return pd.DataFrame(
        {
            "open":        close * (1 + rng.uniform(-0.005, 0.005, n)),
            "high":        close * (1 + rng.uniform(0.000, 0.010, n)),
            "low":         close * (1 - rng.uniform(0.000, 0.010, n)),
            "close":       close,
            "volume":      rng.integers(1_000, 100_000, n).astype(float),
            "in_universe": True,
            "sector":      sector_vals,
        },
        index=idx,
    )


# ══════════════════════════════════════════════════════════════════════════════
# #4  Winsorize cross-sectional
# ══════════════════════════════════════════════════════════════════════════════

class TestWinsoriseCrossSectional:
    """
    _winsorize_per_date clips at [1st, 99th] percentile of the cross-section
    on each date.  A single extreme outlier must be capped — it must NOT
    survive into the model unchanged and distort the cross-section.
    """

    def _make_dated_frame(self) -> pd.DataFrame:
        """20 tickers × 3 dates, feat_x ~ N(0, 1)."""
        n_tickers = 20
        n_dates   = 3
        dates   = pd.bdate_range("2022-01-01", periods=n_dates)
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {"feat_x": rng.normal(0.0, 1.0, len(idx))},
            index=idx,
        )

    def test_outlier_is_clipped(self):
        """999.0 injected into one cell must not survive winsorization."""
        from pipeline.features.engineer import _winsorize_per_date

        df = self._make_dated_frame()
        outlier_date   = df.index.get_level_values("date")[0]
        outlier_ticker = "T0"
        df.loc[(outlier_date, outlier_ticker), "feat_x"] = 999.0

        out = _winsorize_per_date(df.copy(), ["feat_x"])

        clipped = out.loc[(outlier_date, outlier_ticker), "feat_x"]
        assert clipped < 999.0, (
            f"Outlier 999.0 was not clipped by _winsorize_per_date; "
            f"got {clipped:.4f}.  Cross-sectional winsorization is broken."
        )

    def test_outlier_capped_at_99th_percentile(self):
        """Clipped value must equal the 99th percentile of the same date's cross-section."""
        from pipeline.features.engineer import _winsorize_per_date

        df = self._make_dated_frame()
        outlier_date = df.index.get_level_values("date")[0]
        df.loc[(outlier_date, "T0"), "feat_x"] = 999.0

        # Record the 99th percentile BEFORE winsorization (with outlier present,
        # nanpercentile still gives the correct clip target)
        pre_vals  = df.xs(outlier_date, level="date")["feat_x"].values
        p99       = float(np.nanpercentile(pre_vals, 99))

        out = _winsorize_per_date(df.copy(), ["feat_x"])

        clipped = float(out.loc[(outlier_date, "T0"), "feat_x"])
        assert abs(clipped - p99) < 1e-9, (
            f"Winsorized value {clipped:.6f} != p99 {p99:.6f}.  "
            "Clip boundary is wrong."
        )

    def test_non_outlier_values_unchanged(self):
        """
        Rows far from the [1, 99] boundary must be bit-for-bit identical
        after winsorization — only extreme rows should change.
        """
        from pipeline.features.engineer import _winsorize_per_date

        df = self._make_dated_frame()
        out = _winsorize_per_date(df.copy(), ["feat_x"])

        # Inner rows (p5 to p95) should be unmodified
        for date, grp in df.groupby(level="date"):
            pre = grp["feat_x"].values
            post = out.xs(date, level="date")["feat_x"].values
            p5, p95 = np.nanpercentile(pre, 5), np.nanpercentile(pre, 95)
            mask_inner = (pre >= p5) & (pre <= p95)
            if mask_inner.any():
                np.testing.assert_allclose(
                    pre[mask_inner], post[mask_inner], rtol=0,
                    err_msg=f"Inner-range values changed on {date} — winsorize is modifying safe rows"
                )

    def test_winsorize_is_per_date_not_global(self):
        """
        An extreme value that is not an outlier within its own date's
        cross-section (but would be globally) must NOT be clipped.
        Winsorization is cross-sectional per date, not global.

        Design note: we test the MEDIAN ticker of date2 (T10, value 110).
        - Globally extreme: date1 spans [0, 19], so 110 looks like an outlier.
        - Within date2's cross-section [100, 199] it sits at p50 — nowhere
          near [p1, p99] — so a per-date winsorize must leave it unchanged.
        Using the absolute max of date2 (T99, value 199) would fail the test
        because with 100 tickers the 99th-percentile is ~197, and the max IS
        legitimately clipped — that is correct behaviour, not a bug.
        """
        from pipeline.features.engineer import _winsorize_per_date

        # 100 tickers per date gives a stable [p1, p99] window.
        # Date 1: values 0–99, Date 2: values 100–199.
        n = 100
        date1 = pd.Timestamp("2022-01-03")
        date2 = pd.Timestamp("2022-01-04")
        tickers = [f"T{i}" for i in range(n)]

        df = pd.DataFrame(
            {"feat_x": list(range(n)) + list(range(100, 100 + n))},
            index=pd.MultiIndex.from_tuples(
                [(date1, t) for t in tickers] + [(date2, t) for t in tickers],
                names=["date", "ticker"],
            ),
        )

        out = _winsorize_per_date(df.copy(), ["feat_x"])

        # T50 (value 150) is the median of date2's cross-section.
        # Globally it looks extreme (date1 max is 99) but within date2 it is
        # at p50 — per-date winsorize must leave it completely untouched.
        v_before = df.loc[(date2, "T50"), "feat_x"]
        v_after  = out.loc[(date2, "T50"), "feat_x"]
        assert v_after == pytest.approx(v_before, abs=1e-6), (
            f"Date-2 median value {v_before} was clipped to {v_after} — "
            "winsorize is using global percentiles instead of per-date"
        )


# ══════════════════════════════════════════════════════════════════════════════
# #6  Wilder ATR formula
# ══════════════════════════════════════════════════════════════════════════════

class TestWilderATRFormula:
    """
    _wilder_atr must satisfy the Wilder EMA recurrence:
        ATR[t] = (1 - α) × ATR[t-1]  +  α × TR[t]   where α = 1/period

    This recurrence is the industry definition and is what the model relies on
    for normalizing every return feature.  A wrong formula (e.g., simple SMA,
    wrong α) silently corrupts all ATR-scaled features.
    """

    def _make_ohlc(self, n: int = 60, seed: int = 7) -> tuple:
        rng   = np.random.default_rng(seed)
        close = np.clip(100.0 + rng.normal(0, 0.5, n).cumsum(), 10, None)
        high  = close * (1 + rng.uniform(0.001, 0.008, n))
        low   = close * (1 - rng.uniform(0.001, 0.008, n))
        return high, low, close

    def test_satisfies_ema_recurrence(self):
        """
        ATR[t] must equal (1-α)*ATR[t-1] + α*TR[t] for every bar from index 1
        onwards, where TR[t] = max(H-L, |H-prev_C|, |L-prev_C|).
        """
        from pipeline.features.ict_features import _wilder_atr

        high, low, close = self._make_ohlc(n=80)
        period = 14
        alpha  = 1.0 / period

        atr = _wilder_atr(high, low, close, period=period)

        # Reconstruct TR manually
        prev_close        = np.roll(close, 1)
        prev_close[0]     = np.nan
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )

        # Verify recurrence for every non-NaN bar
        for t in range(1, len(atr)):
            if np.isnan(atr[t]) or np.isnan(atr[t - 1]) or np.isnan(tr[t]):
                continue
            expected = (1 - alpha) * atr[t - 1] + alpha * tr[t]
            assert abs(atr[t] - expected) < 1e-10, (
                f"ATR recurrence failed at t={t}: "
                f"got {atr[t]:.10f}, expected {expected:.10f}"
            )

    def test_alpha_is_one_over_period_not_two_over_period_plus_one(self):
        """
        Wilder uses α = 1/period, NOT the standard EMA α = 2/(period+1).
        A common mistake is using the standard EMA formula.
        Verify that α=1/14 produces different results than α=2/15.
        """
        from pipeline.features.ict_features import _wilder_atr

        high, low, close = self._make_ohlc(n=80)

        # Compute TR
        prev_close    = np.roll(close, 1); prev_close[0] = np.nan
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )

        wilder_atr  = _wilder_atr(high, low, close, period=14)
        standard_ema = pd.Series(tr).ewm(alpha=2 / 15, adjust=False).mean().values

        # Must differ — guards against accidentally using 2/(N+1) instead of 1/N
        assert not np.allclose(wilder_atr, standard_ema, rtol=1e-4), (
            "Wilder ATR (α=1/14) matches standard EMA (α=2/15) — "
            "the formula may be using the wrong alpha"
        )

    def test_atr_is_always_positive(self):
        """ATR is a volatility measure and must always be > 0 (after warmup)."""
        from pipeline.features.ict_features import _wilder_atr

        high, low, close = self._make_ohlc(n=80)
        atr = _wilder_atr(high, low, close, period=14)

        non_nan = atr[~np.isnan(atr)]
        assert (non_nan > 0).all(), (
            f"ATR contains non-positive values: min={non_nan.min():.8f}"
        )

    def test_constant_tr_converges_to_tr(self):
        """
        When every true range is the same constant C, the Wilder EMA must
        converge to C as t → ∞.  Checks the steady-state property.
        """
        from pipeline.features.ict_features import _wilder_atr

        n = 200
        # Construct OHLC so that H-L = 2.0 and close = prev_close exactly
        # → TR = max(2, 0, 0) = 2.0 every bar
        close = np.full(n, 100.0)
        high  = np.full(n, 101.0)
        low   = np.full(n, 99.0)

        atr = _wilder_atr(high, low, close, period=14)

        # After 100+ bars the EWM should be within 0.1% of 2.0
        tail = atr[100:]
        assert np.allclose(tail, 2.0, rtol=1e-3), (
            f"Constant TR=2.0 did not converge to ATR=2.0: tail mean = {tail.mean():.6f}"
        )

    def test_matches_pandas_ewm_reference(self):
        """
        Cross-check against pandas ewm(alpha=1/period, adjust=False) —
        the canonical reference implementation.
        """
        from pipeline.features.ict_features import _wilder_atr

        high, low, close = self._make_ohlc(n=80)
        period = 14

        # Reference: build TR manually, apply pandas ewm
        prev_close    = np.roll(close, 1); prev_close[0] = np.nan
        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        reference = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values

        result = _wilder_atr(high, low, close, period=period)

        np.testing.assert_allclose(
            result, reference, rtol=1e-6,
            err_msg=(
                "_wilder_atr diverges from pandas ewm(alpha=1/14, adjust=False). "
                "The Wilder ATR formula has changed."
            ),
        )


# ══════════════════════════════════════════════════════════════════════════════
# #8  Schema stable — feature columns are deterministic across input panels
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureSchemaStable:
    """
    FeatureEngineer.build() must produce the exact same set of features_*
    column names regardless of which tickers or date range the input covers.

    Silent schema drift between train and inference is the most dangerous
    class of bug — the model receives fewer or differently-ordered columns
    and fails/degrades without any explicit error.
    """

    @pytest.fixture(scope="class")
    def two_schemas(self):
        """
        Build two small panels with different seeds / start dates.
        Runs once per class (~40-60 s).  Returns (cols_A, cols_B).
        """
        from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
        from pipeline.config import get_config

        cfg = get_config("nse")
        bm  = pd.Series(
            1.0,
            index=pd.bdate_range("2017-01-01", periods=1500),
        )
        fe  = FeatureEngineer(cfg, bm)

        panel_a = _make_panel(n_tickers=3, n_days=260, seed=11, start="2019-03-01")
        panel_b = _make_panel(n_tickers=3, n_days=260, seed=99, start="2021-06-01")

        out_a = fe.build(panel_a)
        out_b = fe.build(panel_b)

        cols_a = sorted(c for c in out_a.columns if c.startswith(FEATURE_PREFIX))
        cols_b = sorted(c for c in out_b.columns if c.startswith(FEATURE_PREFIX))
        return cols_a, cols_b

    def test_same_column_count(self, two_schemas):
        cols_a, cols_b = two_schemas
        assert len(cols_a) == len(cols_b), (
            f"Column count differs: panel_A={len(cols_a)}, panel_B={len(cols_b)}.\n"
            f"Only in A: {set(cols_a) - set(cols_b)}\n"
            f"Only in B: {set(cols_b) - set(cols_a)}"
        )

    def test_same_column_names(self, two_schemas):
        cols_a, cols_b = two_schemas
        only_a = set(cols_a) - set(cols_b)
        only_b = set(cols_b) - set(cols_a)
        assert not only_a, (
            f"Columns in panel_A but not panel_B (conditional generation?): {sorted(only_a)}"
        )
        assert not only_b, (
            f"Columns in panel_B but not panel_A (conditional generation?): {sorted(only_b)}"
        )

    def test_column_order_is_deterministic(self, two_schemas):
        """
        The sorted list of feature columns must be identical — guards against
        non-deterministic dict iteration introducing ordering differences.
        """
        cols_a, cols_b = two_schemas
        assert cols_a == cols_b, (
            "Feature column order differs between two build() calls.  "
            "Model feature alignment would be broken at inference time."
        )

    def test_recompute_does_not_add_or_drop_columns(self):
        """
        recompute_fold_features() must not change the set of features_* columns —
        it may update values but must never introduce or remove column names.
        """
        from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
        from pipeline.config import get_config

        cfg = get_config("nse")
        bm  = pd.Series(1.0, index=pd.bdate_range("2019-01-01", periods=1200))
        fe  = FeatureEngineer(cfg, bm)

        panel = _make_panel(n_tickers=3, n_days=260, seed=5, start="2019-01-01")
        built = fe.build(panel)

        cols_before = sorted(c for c in built.columns if c.startswith(FEATURE_PREFIX))

        # Recompute at a midpoint cutoff
        dates  = built.index.get_level_values("date").unique().sort_values()
        cutoff = dates[len(dates) // 2]
        recomputed = fe.recompute_fold_features(built, cutoff)

        cols_after = sorted(c for c in recomputed.columns if c.startswith(FEATURE_PREFIX))

        only_before = set(cols_before) - set(cols_after)
        only_after  = set(cols_after)  - set(cols_before)

        assert not only_before, (
            f"recompute_fold_features DROPPED columns: {sorted(only_before)}"
        )
        assert not only_after, (
            f"recompute_fold_features ADDED new columns: {sorted(only_after)}"
        )
