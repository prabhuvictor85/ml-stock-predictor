"""
Tests for the shared momentum-bull quality gate (pipeline/gating.py).

The gate was extracted verbatim from run_sp500_local.py and is now shared by
all three run scripts — these tests pin its veto semantics so a future edit
can't silently change watchlist filtering across markets.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.features.engineer import FEATURE_PREFIX
from pipeline.gating import momentum_bull_quality_gate


def _make_cross_wl(n: int = 1, **overrides) -> pd.DataFrame:
    """One healthy momentum candidate per row; override columns to break it."""
    base = {
        "ssz_htf_score":      0.0,   # no overhead supply
        "ict_bear_htf_score": 0.0,   # no bear ICT structure
        "price_vs_sma50":     0.05,  # above rising SMA50
        "sma50_slope_5":      0.01,
        "sma200_slope_10":    0.0,   # flat SMA200 is acceptable (>= 0)
        "plus_di":            25.0,  # bulls own ADX
        "minus_di":           10.0,
    }
    base.update(overrides)
    data = {f"{FEATURE_PREFIX}{k}": np.full(n, v, dtype=float) for k, v in base.items()}
    return pd.DataFrame(data, index=[f"TICK{i}" for i in range(n)])


def test_healthy_candidate_kept():
    gate = momentum_bull_quality_gate(_make_cross_wl(), "momentum", FEATURE_PREFIX)
    assert gate.all()


def test_non_momentum_mode_is_noop():
    df = _make_cross_wl(ssz_htf_score=0.9)  # would be vetoed in momentum mode
    for mode in ("reversal", "legacy"):
        gate = momentum_bull_quality_gate(df, mode, FEATURE_PREFIX)
        assert gate.all(), f"mode={mode} must not filter"


def test_ssz_supply_veto():
    gate = momentum_bull_quality_gate(
        _make_cross_wl(ssz_htf_score=0.61), "momentum", FEATURE_PREFIX)
    assert not gate.any()
    # boundary: exactly 0.6 is NOT vetoed (strict >)
    gate = momentum_bull_quality_gate(
        _make_cross_wl(ssz_htf_score=0.60), "momentum", FEATURE_PREFIX)
    assert gate.all()


def test_ict_bear_structure_veto():
    gate = momentum_bull_quality_gate(
        _make_cross_wl(ict_bear_htf_score=0.41), "momentum", FEATURE_PREFIX)
    assert not gate.any()
    gate = momentum_bull_quality_gate(
        _make_cross_wl(ict_bear_htf_score=0.40), "momentum", FEATURE_PREFIX)
    assert gate.all()


def test_broken_trend_stack_veto():
    # price below SMA50
    gate = momentum_bull_quality_gate(
        _make_cross_wl(price_vs_sma50=-0.02), "momentum", FEATURE_PREFIX)
    assert not gate.any()
    # SMA50 falling
    gate = momentum_bull_quality_gate(
        _make_cross_wl(sma50_slope_5=-0.01), "momentum", FEATURE_PREFIX)
    assert not gate.any()
    # SMA200 falling
    gate = momentum_bull_quality_gate(
        _make_cross_wl(sma200_slope_10=-0.01), "momentum", FEATURE_PREFIX)
    assert not gate.any()


def test_bearish_adx_veto():
    gate = momentum_bull_quality_gate(
        _make_cross_wl(plus_di=10.0, minus_di=25.0), "momentum", FEATURE_PREFIX)
    assert not gate.any()


def test_missing_columns_disable_gate():
    # No gate features at all → inactive, everything passes
    df = pd.DataFrame({"close": [100.0, 200.0]}, index=["AAA", "BBB"])
    gate = momentum_bull_quality_gate(df, "momentum", FEATURE_PREFIX)
    assert gate.all()


def test_nan_treated_as_zero():
    # NaN in a veto column must not veto (fillna(0) semantics) — but NaN in
    # the trend stack means pvs50=0.0 which fails the strict > 0 check.
    df = _make_cross_wl(ssz_htf_score=np.nan)
    gate = momentum_bull_quality_gate(df, "momentum", FEATURE_PREFIX)
    assert gate.all()
    df = _make_cross_wl(price_vs_sma50=np.nan)
    gate = momentum_bull_quality_gate(df, "momentum", FEATURE_PREFIX)
    assert not gate.any()


def test_mixed_universe_counts():
    df = pd.concat([
        _make_cross_wl(3),                        # 3 healthy
        _make_cross_wl(2, ssz_htf_score=0.9),     # 2 under supply
    ])
    df.index = [f"T{i}" for i in range(5)]
    gate = momentum_bull_quality_gate(df, "momentum", FEATURE_PREFIX)
    assert int(gate.sum()) == 3 and len(gate) == 5
