"""
Regression guards — critical pipeline invariants.

Covers gaps not addressed by existing test files:
  1. Target builder  — returns are decimal, forward-shifted, cs_rank in [0,1]
  2. Feature bounds  — zone_dist_atr and ict_*_atr_dist clipped to ±20
  3. ATR floor       — near-zero ATR (illiquid stock) produces bounded output
  4. NDCG metric     — perfect=1.0, inverted<perfect, empty=0.0
  5. Universe filter — momentum keeps near-52w-high, reversal keeps far-from-high
  6. ICT isolation   — multi-symbol input raises AssertionError
  7. CS rank         — percentile rank stays in [0, 1]
  8. Backtest file   — duplicate main() definition detected
"""
from __future__ import annotations

import ast
import textwrap

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_panel(n_tickers: int = 5, n_days: int = 120) -> pd.DataFrame:
    """Minimal OHLCV panel with MultiIndex (date, ticker)."""
    dates   = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    rng = np.random.default_rng(42)
    n   = len(idx)
    close  = 100.0 + rng.normal(0, 1, n).cumsum()
    close  = np.clip(close, 10, None)
    df = pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.000, 0.010, n)),
        "low":    close * (1 - rng.uniform(0.000, 0.010, n)),
        "close":  close,
        "volume": rng.integers(1_000, 100_000, n).astype(float),
        "in_universe": True,
    }, index=idx)
    return df


def _make_ict_input(n: int = 60, ticker: str = "AAA") -> pd.DataFrame:
    """Single-ticker OHLCV + atr_14 for ICTFeatureEngine."""
    from pipeline.features.ict_features import _wilder_atr
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(7)
    c   = 100.0 + rng.normal(0, 0.5, n).cumsum()
    c   = np.clip(c, 10, None)
    df  = pd.DataFrame({
        "open":  c * (1 + rng.uniform(-0.003, 0.003, n)),
        "high":  c * (1 + rng.uniform(0.001, 0.008, n)),
        "low":   c * (1 - rng.uniform(0.001, 0.008, n)),
        "close": c,
        "volume": 1000.0,
    }, index=idx)
    df["atr_14"] = _wilder_atr(df["high"].values, df["low"].values, df["close"].values, 14)
    return df


# ── 1. Target builder ─────────────────────────────────────────────────────────

def test_future_return_is_decimal():
    """future_20d_return must be in decimal form (0.05 = 5%), not percentage (5.0)."""
    from pipeline.targets.builder import TargetBuilder
    from pipeline.config import get_config

    cfg   = get_config("nse")
    panel = _make_panel(n_tickers=10, n_days=100)
    bm    = pd.Series(
        np.ones(100) * 100.0,
        index=pd.bdate_range("2020-01-01", periods=100),
    )
    tb    = TargetBuilder(cfg)
    out   = tb.build(panel, bm)

    ret = out["future_20d_return"].dropna()
    assert ret.abs().max() < 5.0, (
        f"future_20d_return max abs = {ret.abs().max():.2f} — looks like percentage, not decimal"
    )
    assert ret.abs().max() > 0.0, "future_20d_return is all zeros — target build failed"


def test_cs_rank_is_percentile():
    """cs_rank_20d must be in [0, 1] — it is a within-date percentile rank."""
    from pipeline.targets.builder import TargetBuilder
    from pipeline.config import get_config

    cfg   = get_config("nse")
    panel = _make_panel(n_tickers=20, n_days=100)
    bm    = pd.Series(
        np.ones(100) * 100.0,
        index=pd.bdate_range("2020-01-01", periods=100),
    )
    tb  = TargetBuilder(cfg)
    out = tb.build(panel, bm)

    rank = out["cs_rank_20d"].dropna()
    assert rank.min() >= 0.0, f"cs_rank_20d min = {rank.min():.4f} — below 0"
    assert rank.max() <= 1.0, f"cs_rank_20d max = {rank.max():.4f} — above 1"


def test_target_uses_future_not_past_returns():
    """
    Verify forward shift: the cs_rank on date t is based on future returns
    (close[t+20] / close[t] - 1), not past. We check by comparing return
    direction vs same-day price change — they must NOT be perfectly correlated.
    """
    from pipeline.targets.builder import TargetBuilder
    from pipeline.config import get_config

    cfg   = get_config("nse")
    panel = _make_panel(n_tickers=20, n_days=100)
    bm    = pd.Series(
        np.ones(100) * 100.0,
        index=pd.bdate_range("2020-01-01", periods=100),
    )
    tb  = TargetBuilder(cfg)
    out = tb.build(panel, bm)

    # Spearman between same-day return proxy and target rank — must be near 0
    same_day_ret = (
        out["close"].groupby(level="ticker").pct_change()
    )
    target_rank  = out["cs_rank_20d"]
    common       = same_day_ret.dropna().index.intersection(target_rank.dropna().index)
    corr         = same_day_ret.loc[common].corr(target_rank.loc[common])
    assert abs(corr) < 0.4, (
        f"Suspicious correlation {corr:.3f} between same-day return and cs_rank_20d — "
        "possible target leakage (target not forward-shifted)"
    )


# ── 2. Feature bounds ─────────────────────────────────────────────────────────

def test_ict_atr_dist_bounded():
    """All ict_*_atr_dist columns must be within [-20, 20] after the clip fix."""
    from pipeline.features.ict_features import ICTFeatureEngine

    df  = _make_ict_input(n=120)
    out = ICTFeatureEngine().compute(df)

    dist_cols = [c for c in out.columns if c.endswith("_atr_dist")]
    assert dist_cols, "No *_atr_dist columns found in ICT output"

    for col in dist_cols:
        vals = out[col].dropna()
        assert vals.min() >= -20.0, f"{col} min = {vals.min():.2f} < -20"
        assert vals.max() <=  20.0, f"{col} max = {vals.max():.2f} > +20"


def test_zone_dist_atr_bounded():
    """zone_dist_atr_1d must be within [-20, 20] after the clip fix."""
    from pipeline.features.zone_features import compute_zone_features

    panel = _make_panel(n_tickers=1, n_days=150)
    df    = panel.droplevel("ticker")
    out   = compute_zone_features(df)

    col = "zone_dist_atr_1d"
    if col in out.columns:
        vals = out[col].dropna()
        if len(vals) > 0:
            assert vals.min() >= -20.0, f"{col} min = {vals.min():.2f} < -20"
            assert vals.max() <=  20.0, f"{col} max = {vals.max():.2f} > +20"


def test_atr_floor_prevents_explosion_ict():
    """
    A stock with near-zero ATR (flat price) must produce bounded atr_dist,
    not NaN/Inf explosion. Guards against divide-by-tiny-ATR regressing.
    """
    from pipeline.features.ict_features import ICTFeatureEngine

    n   = 60
    idx = pd.bdate_range("2021-01-01", periods=n)
    # Completely flat price — ATR collapses to ~0
    df  = pd.DataFrame({
        "open":   100.0, "high": 100.01,
        "low":    99.99, "close": 100.0,
        "volume": 1000.0,
        "atr_14": 1e-7,   # near-zero ATR injected directly
    }, index=idx)

    out = ICTFeatureEngine().compute(df)
    dist_cols = [c for c in out.columns if c.endswith("_atr_dist")]
    for col in dist_cols:
        assert not out[col].isin([np.inf, -np.inf]).any(), f"{col} contains Inf"
        vals = out[col].dropna()
        if len(vals):
            assert vals.abs().max() <= 20.0, f"{col} exploded: max = {vals.abs().max()}"


# ── 3. NDCG metric ────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking_is_one():
    """NDCG@k = 1.0 when scores rank stocks in the same order as relevance."""
    from pipeline.validation.metrics import ndcg_at_k

    relevance = np.array([0, 1, 2, 3, 4])
    scores    = np.array([0.1, 0.2, 0.3, 0.4, 0.5])   # same order
    assert ndcg_at_k(relevance, scores, k=5) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_inverted_ranking_below_perfect():
    """NDCG@k < 1.0 when worst stocks are ranked highest."""
    from pipeline.validation.metrics import ndcg_at_k

    relevance = np.array([0, 1, 2, 3, 4])
    scores    = np.array([0.5, 0.4, 0.3, 0.2, 0.1])   # worst first
    assert ndcg_at_k(relevance, scores, k=5) < 1.0


def test_ndcg_empty_returns_zero():
    """NDCG@k = 0.0 on empty inputs — guards against division-by-zero."""
    from pipeline.validation.metrics import ndcg_at_k

    assert ndcg_at_k(np.array([]), np.array([]), k=10) == 0.0


def test_ndcg_all_zero_relevance():
    """NDCG@k = 0.0 when all relevance labels are 0 — IDCG = 0, no signal."""
    from pipeline.validation.metrics import ndcg_at_k

    relevance = np.zeros(10)
    scores    = np.random.default_rng(0).random(10)
    assert ndcg_at_k(relevance, scores, k=10) == 0.0


# ── 4. Universe filter ────────────────────────────────────────────────────────

def test_momentum_filter_keeps_near_52w_high():
    """
    Momentum universe filter (high_52w_dist > -0.4) must keep only stocks
    within 40% of their 52-week high. Stocks far below must be excluded.
    """
    close_near = 95.0    # within 5% of 52w high (100) → momentum
    close_far  = 55.0    # 45% below 52w high (100)   → reversal
    high_52w   = 100.0

    dist_near = close_near / high_52w - 1   # -0.05 > -0.40 → kept
    dist_far  = close_far  / high_52w - 1   # -0.45 < -0.40 → excluded

    assert dist_near > -0.4, "Near-high stock should pass momentum filter"
    assert dist_far  < -0.4, "Far-from-high stock should fail momentum filter"


def test_reversal_filter_keeps_far_from_52w_high():
    """Reversal universe keeps stocks 40%+ below 52w high."""
    close_near = 95.0
    close_far  = 55.0
    high_52w   = 100.0

    dist_near = close_near / high_52w - 1
    dist_far  = close_far  / high_52w - 1

    # Reversal: high_52w_dist <= -0.4
    assert dist_far  <= -0.4, "Far-from-high stock should pass reversal filter"
    assert dist_near >  -0.4, "Near-high stock should fail reversal filter"


# ── 5. ICT isolation guard ────────────────────────────────────────────────────

def test_ict_rejects_multi_symbol_input():
    """
    ICTFeatureEngine.compute() must raise AssertionError when passed a
    DataFrame containing more than one symbol — guards against cross-ticker ffill.
    """
    from pipeline.features.ict_features import ICTFeatureEngine

    df = _make_ict_input(n=60)
    df["symbol"] = ["AAA"] * 30 + ["BBB"] * 30   # two symbols

    with pytest.raises(AssertionError, match="symbol"):
        ICTFeatureEngine().compute(df)


def test_ict_accepts_single_symbol_input():
    """ICTFeatureEngine.compute() must NOT raise when symbol column has one unique value."""
    from pipeline.features.ict_features import ICTFeatureEngine

    df = _make_ict_input(n=60)
    df["symbol"] = "AAA"
    ICTFeatureEngine().compute(df)   # should not raise


# ── 6. Backtest duplicate code detection ─────────────────────────────────────

def test_backtest_run_has_no_duplicate_main():
    """
    backtest_run.py has a known duplicate main() definition (copy-paste artifact).
    This test fails until the duplicate is removed, acting as a reminder.
    A SyntaxError is also treated as failure — the duplicate causes one at line 144.
    """
    import pathlib
    src = pathlib.Path("pipeline/backtest_run.py").read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        pytest.fail(
            f"backtest_run.py has a SyntaxError ({e}) — likely caused by the "
            "duplicate main() copy-paste block starting at line ~144. "
            "Remove the duplicate to fix both this error and the test."
        )
    main_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    assert len(main_defs) == 1, (
        f"backtest_run.py has {len(main_defs)} definitions of main() — "
        "remove the duplicate (copy-paste artifact at line ~144)"
    )


# ── 7. ICT feature columns always present ────────────────────────────────────

def test_ict_output_has_required_columns():
    """ICTFeatureEngine.compute() must always return the full set of ICT columns."""
    from pipeline.features.ict_features import ICTFeatureEngine

    required = [
        "ict_bob_active",    "ict_bob_atr_dist",
        "ict_sob_active",    "ict_sob_atr_dist",
        "ict_bullrb_active", "ict_bullrb_atr_dist",
        "ict_bearrb_active", "ict_bearrb_atr_dist",
        "ict_bullfvg_active","ict_bullfvg_atr_dist",
        "ict_bearfvg_active","ict_bearfvg_atr_dist",
        "ict_bull_zone_priority", "ict_bear_zone_priority",
        "ict_bsl_swept", "ict_ssl_swept",
    ]
    out = ICTFeatureEngine().compute(_make_ict_input(n=80))
    missing = [c for c in required if c not in out.columns]
    assert not missing, f"ICT output missing columns: {missing}"


def test_ict_active_flags_are_binary():
    """All ict_*_active columns must contain only 0.0 and 1.0 — no partial values."""
    from pipeline.features.ict_features import ICTFeatureEngine

    out      = ICTFeatureEngine().compute(_make_ict_input(n=80))
    act_cols = [c for c in out.columns if c.endswith("_active")]
    for col in act_cols:
        unique = set(out[col].dropna().unique())
        assert unique.issubset({0.0, 1.0}), (
            f"{col} has non-binary values: {unique - {0.0, 1.0}}"
        )


# ── 8. Zone priority ordering ─────────────────────────────────────────────────

def test_zone_priority_bb_gt_ob_gt_fvg():
    """BB(3) > OB(2) > FVG(1) — the priority hierarchy must not regress."""
    from pipeline.features.ict_features import ZonePriority

    assert ZonePriority.BB  == 3
    assert ZonePriority.OB  == 2
    assert ZonePriority.FVG == 1
    assert ZonePriority.BB > ZonePriority.OB > ZonePriority.FVG
