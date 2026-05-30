"""
Regression guards for the ICT feature engine.

Locks down three fixes that the generic leakage suite did not catch:
  1. HTF resample look-ahead leak  (MS/QS/YS left-label -> ME/QE/YE right-label)
  2. Zone-violation mitigation latch (a violated zone must NOT resurrect)
  3. Persistent zone priority        (derived from active flags, not trigger-instant)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features.ict_features import _wilder_atr, ICTFeatureEngine
from pipeline.features.engineer import _ICT_HTF_RESAMPLE, _ICT_DISP_MULT


def _ob_then_violation_path() -> pd.DataFrame:
    """
    Deterministic OHLC path that forms a high-priority bull zone, holds it,
    then violates it and recovers — used to test the mitigation latch.
    """
    o: list[float] = []
    h: list[float] = []
    l: list[float] = []
    c: list[float] = []

    def bar(op: float, cl: float, hi: float | None = None, lo: float | None = None):
        h.append(max(op, cl) if hi is None else hi)
        l.append(min(op, cl) if lo is None else lo)
        o.append(op)
        c.append(cl)

    px = 100.0
    for i in range(20):                       # warmup: tiny ranges -> low ATR
        bar(px, px + (0.2 if i % 2 else -0.2))
        px = c[-1]
    bar(100.0, 96.0)                          # bar20: bearish down candle, zone [96,100]
    bar(97.0, 112.0)                          # bar21: bull displacement (>3xATR), trigger
    for _ in range(8):                        # bars22-29: HOLD above zone low
        bar(c[-1], c[-1] + 0.8)
    bar(c[-1], 90.0, lo=89.0)                 # bar30: VIOLATION (close below zone low)
    for _ in range(9):                        # bars31-39: RECOVER above zone low
        bar(c[-1], c[-1] + 1.5)

    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c},
        index=pd.date_range("2020-01-01", periods=len(o), freq="B"),
    )
    df["atr_14"] = _wilder_atr(
        df["high"].values, df["low"].values, df["close"].values, 14
    )
    return df


def test_htf_resample_rules_are_right_labelled():
    """
    Monthly/quarterly/yearly resample rules must be period-END anchored. A
    period-START rule (MS/QS/YS) labels the bar on the left edge, so merge_asof
    backward attaches a still-incomplete (future) period to early daily bars —
    a look-ahead leak.
    """
    assert _ICT_HTF_RESAMPLE["1mo"] == "ME"
    assert _ICT_HTF_RESAMPLE["3mo"] == "QE"
    assert _ICT_HTF_RESAMPLE["1y"] == "YE"
    # Weekly is right-anchored on Friday already — safe.
    assert _ICT_HTF_RESAMPLE["1wk"] == "W-FRI"


def test_no_lookahead_in_monthly_merge_asof():
    """
    End-to-end: with ME labelling, a daily bar mid-month must see the PRIOR
    completed month, never the current (incomplete) month's aggregate.
    """
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    df = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)
    monthly = df.resample(_ICT_HTF_RESAMPLE["1mo"]).agg({"close": "last"})
    m = monthly.reset_index()
    m.columns = ["date", "m_close"]
    probe = pd.DataFrame({"date": pd.to_datetime(["2024-05-15"])})
    merged = pd.merge_asof(probe, m, on="date", direction="backward")
    april_close = float(df.loc["2024-04-30", "close"])
    may_close = float(df.loc["2024-05-31", "close"])
    seen = float(merged["m_close"].iloc[0])
    assert seen == april_close, "mid-May must see April's completed month"
    assert seen != may_close, "mid-May must NOT see May's future close (leak)"


def test_zone_violation_latch_is_sticky():
    """
    Once price violates a live zone, the zone must stay inactive even if price
    later re-crosses the (forward-filled) boundary. Without the latch the active
    flag would flicker back on, contradicting ICT mitigation semantics.
    """
    out = ICTFeatureEngine().compute(_ob_then_violation_path(), disp_mult=3.0)
    # The path forms a bull Breaker Block (outranks the OB) at bar 21.
    a = out["ict_bullbb_active"].values
    assert np.all(a[21:30] == 1), "zone should be live while price holds above it"
    assert a[30] == 0, "zone must deactivate on violation"
    assert np.all(a[31:40] == 0), "violated zone must NOT resurrect on recovery"


def test_zone_priority_is_persistent_not_trigger_instant():
    """
    Exported zone priority must persist for the life of the active zone (it is
    derived from forward-filled active flags), not spike on a single trigger
    candle. A trigger-instant export collapses the MTF composite to ~0.
    """
    out = ICTFeatureEngine().compute(_ob_then_violation_path(), disp_mult=3.0)
    prio = out["ict_bull_zone_priority"].values
    # Live window 21..29 should all carry the BB priority (3), not a lone spike.
    assert np.all(prio[21:30] == 3.0)
    assert prio[30] == 0.0
    # Distance features must be zeroed once the zone is dead.
    dist = out["ict_bullbb_atr_dist"].values
    assert np.all(dist[21:30] != 0)
    assert np.all(dist[31:40] == 0)


def test_disp_mult_param_threads_through():
    """A looser displacement gate must admit at least as many zone triggers."""
    df = _ob_then_violation_path()
    strict = ICTFeatureEngine().compute(df.copy(), disp_mult=3.0)
    loose = ICTFeatureEngine().compute(df.copy(), disp_mult=1.0)
    n_strict = int(strict[[c for c in strict.columns if c.endswith("_active")]].sum().sum())
    n_loose = int(loose[[c for c in loose.columns if c.endswith("_active")]].sum().sum())
    assert n_loose >= n_strict


def _modest_ob_path() -> pd.DataFrame:
    """
    A bullish Order Block whose rally body clears the 1.2x prior-body rule but is
    SMALLER than 3x ATR (high-wick warmup inflates ATR). The reference Pine
    indicator forms this OB; a 3x-ATR displacement gate would wrongly reject it.
    """
    o, h, l, c = [], [], [], []

    def bar(op, cl, hi, lo):
        o.append(op); c.append(cl); h.append(hi); l.append(lo)

    for _ in range(20):                       # warmup: tiny body, wide range -> high ATR
        bar(100.0, 100.2, 102.5, 97.5)
    bar(100.0, 98.0, 100.0, 97.5)             # bar20: bearish (prev red), body 2
    bar(99.0, 104.0, 104.0, 99.0)             # bar21: rally, body 5 (>1.2x2) but <3xATR
    for _ in range(8):                        # hold above the OB
        bar(c[-1], c[-1] + 0.3, c[-1] + 0.5, c[-1] - 0.2)

    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c},
        index=pd.date_range("2021-01-01", periods=len(o), freq="B"),
    )
    df["atr_14"] = _wilder_atr(df["high"].values, df["low"].values, df["close"].values, 14)
    return df


def test_order_block_fires_without_displacement_gate():
    """
    Fidelity to the Pine indicator: an OB qualified purely by structure + the
    1.2x relative-body rule must form under the DEFAULT engine (gate off), and
    that same OB must be (wrongly) rejected by a strict absolute ATR gate.
    Guards against re-introducing the 3x gate that annihilated all Order Blocks.
    """
    df = _modest_ob_path()
    # Default engine: displacement gate OFF -> OB forms.
    default = ICTFeatureEngine().compute(df.copy())
    assert default["ict_bob_active"].sum() > 0, "OB must form under the structural (Pine) rule"
    # Strict absolute ATR gate would reject this modest-body OB.
    gated = ICTFeatureEngine().compute(df.copy(), disp_mult=3.0)
    assert gated["ict_bob_active"].sum() == 0, "3x ATR gate wrongly kills the OB"


def test_default_disp_mult_is_gate_off():
    """The engine default and the engineer per-timeframe map must keep the gate
    disabled (0.0) so OB/FVG match the reference indicator out of the box."""
    import inspect
    sig = inspect.signature(ICTFeatureEngine.compute)
    assert sig.parameters["disp_mult"].default == 0.0
    assert all(v == 0.0 for v in _ICT_DISP_MULT.values())
