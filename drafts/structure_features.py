"""
structure_features.py — Causal BOS / CHoCH market-structure features (v2).

DRAFT — NOT INTEGRATED. This module lives in drafts/ and is intentionally
not imported by the pipeline (engineer.py). Review before wiring in.

v2 changes vs. the first draft
------------------------------
+ Liquidity sweep detection (BSL/SSL): a wick beyond the active level
  that CLOSES back inside is recorded separately from a genuine
  close-based break, and — critically — does NOT consume the level.
  Naming mirrors your existing ict_bsl_swept/ict_ssl_swept convention
  in ict_features.py for consistency.
+ compute_multiscale_structure_features(): runs the engine at two
  fractal scales (major/internal) and derives the cross-scale
  confluence features that give each scale's signals their ICT
  meaning — alignment and the Judas-swing setup.

Concept
-------
A "structural break" occurs when price CLOSES beyond the most recent
confirmed swing high/low. The SAME break event is labeled differently
depending on the prevailing trend:

  - Break agrees with the current trend   -> BOS   (continuation)
  - Break opposes the current trend       -> CHoCH (reversal; flips trend)

A WICK beyond the level that closes back inside is a different animal
entirely — a liquidity sweep / stop hunt / "Judas swing": smart money
raids resting orders without committing to a break, often immediately
BEFORE a reversal in the opposite direction. Collapsing sweeps and
breaks into one high/low-based test (as a naive structure detector
would) misreads hunts as reversals; this module keeps them distinct.

LEAKAGE DISCIPLINE (mirrors zone_features.py)
----------------------------------------------
A swing at bar i can only be "known" `swing_length` bars later. Every
swing is tagged with a causal `confirmed_at` index, and when
cutoff_date is given, only swings confirmed BY the cutoff feed the
labeler — though the labeler still runs across the full index, so
breaks/sweeps of pre-cutoff-KNOWN levels are correctly observed as
bars unfold in the test window (causal, not leakage: exactly what live
inference would see). Call with cutoff_date=<training end> per fold,
exactly like compute_zone_features.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.features.ict_features import _wilder_atr
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


class TrendState(IntEnum):
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


# ── Swing detection (unchanged from v1) ───────────────────────────────────────
def _detect_confirmed_swings(
    high: np.ndarray, low: np.ndarray, swing_length: int
) -> list[tuple[int, int, float, int]]:
    """
    Detect swing highs/lows via a centered rolling extremum window, tag
    each with its CAUSAL confirmation index = swing_index + swing_length.
    Returns (swing_index, swing_type [+1 high / -1 low], level, confirmed_at)
    sorted by confirmed_at, with consecutive same-type swings pruned to
    their most extreme point (re-derivation of smc.swing_highs_lows's dedup).
    """
    n = len(high)
    w = 2 * swing_length + 1
    if n < w:
        return []

    high_s, low_s = pd.Series(high), pd.Series(low)
    roll_max = high_s.rolling(w, center=True, min_periods=w).max().values
    roll_min = low_s.rolling(w, center=True, min_periods=w).min().values

    is_high = (high == roll_max) & np.isfinite(roll_max)
    is_low = (low == roll_min) & np.isfinite(roll_min)

    raw = [(i, 1, float(high[i])) for i in np.flatnonzero(is_high)]
    raw += [(i, -1, float(low[i])) for i in np.flatnonzero(is_low)]
    raw.sort(key=lambda x: x[0])

    pruned: list[tuple[int, int, float]] = []
    for s in raw:
        if pruned and pruned[-1][1] == s[1]:
            more_extreme = (s[2] > pruned[-1][2]) if s[1] == 1 else (s[2] < pruned[-1][2])
            if more_extreme:
                pruned[-1] = s
            continue
        pruned.append(s)

    swings = [(i, t, lvl, min(i + swing_length, n - 1)) for (i, t, lvl) in pruned]
    swings.sort(key=lambda x: (x[3], x[0]))
    return swings


# ── Event labeling: BOS / CHoCH / BSL / SSL state machine ─────────────────────
def _label_structure_events(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swings: list[tuple[int, int, float, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single forward pass. Maintains the active (most recently confirmed,
    not-yet-broken) swing-high/-low levels and the trend state.

    Two DISTINCT tests run against the active levels each bar:

      1. Sweep test (wick-based, level-PRESERVING):
         high > active_high but close <= active_high  -> BSL swept
         low  < active_low  but close >= active_low   -> SSL swept
         The level survives — smart money raided the liquidity resting
         beyond it without committing to a break. This is independent
         of, and can co-occur with, a break on the OPPOSITE side.

      2. Break test (close-based, level-CONSUMING):
         close > active_high -> structural break up   (BOS or CHoCH)
         close < active_low  -> structural break down (BOS or CHoCH)
         Labeled BOS if it agrees with the current trend, CHoCH (+ trend
         flip) if it opposes it. One-shot: consumes the level until the
         next confirmed swing of that type replaces it.

    Returns: bos, choch, bsl_swept, ssl_swept, level, trend
      bos/choch   : +1 bullish / -1 bearish / 0
      bsl_swept   : 1.0 if buy-side liquidity (above active_high) raided
      ssl_swept   : 1.0 if sell-side liquidity (below active_low) raided
      level       : the level a genuine break consumed (NaN otherwise)
      trend       : state AFTER processing bar i (persists between events)
    """
    n = len(close)
    bos = np.zeros(n, dtype=np.int8)
    choch = np.zeros(n, dtype=np.int8)
    bsl_swept = np.zeros(n, dtype=np.float32)
    ssl_swept = np.zeros(n, dtype=np.float32)
    level = np.full(n, np.nan, dtype=np.float64)
    trend = np.zeros(n, dtype=np.int8)

    swing_ptr = 0
    n_swings = len(swings)
    active_high = np.nan
    active_low = np.nan
    state = TrendState.NEUTRAL

    for i in range(n):
        while swing_ptr < n_swings and swings[swing_ptr][3] <= i:
            _, s_type, s_level, _ = swings[swing_ptr]
            swing_ptr += 1
            if s_type == 1:
                active_high = s_level
            else:
                active_low = s_level

        h_i, l_i, c_i = high[i], low[i], close[i]

        # ── 1. Sweep test — independent per side, level-preserving ────
        # (An outside bar that wicks both ways and closes mid-range can
        # legitimately set BOTH flags — that's correct, not a conflict:
        # both liquidity pools genuinely got raided on the same bar.)
        if np.isfinite(active_high) and h_i > active_high and c_i <= active_high:
            bsl_swept[i] = 1.0
        if np.isfinite(active_low) and l_i < active_low and c_i >= active_low:
            ssl_swept[i] = 1.0

        # ── 2. Break test — close-based, mutually exclusive, consuming ─
        broke_up = np.isfinite(active_high) and c_i > active_high
        broke_down = np.isfinite(active_low) and c_i < active_low

        if broke_up and broke_down:
            # Rare gap-day double break: keep the more extreme side as
            # the "real" structural event; document/revisit if SHAP
            # shows this tie-break path matters in practice.
            if abs(c_i - active_high) >= abs(c_i - active_low):
                broke_down = False
            else:
                broke_up = False

        if broke_up:
            choch[i] = 1 if state == TrendState.BEARISH else 0
            bos[i] = 0 if choch[i] else 1
            level[i] = active_high
            state = TrendState.BULLISH
            active_high = np.nan
        elif broke_down:
            choch[i] = -1 if state == TrendState.BULLISH else 0
            bos[i] = 0 if choch[i] else -1
            level[i] = active_low
            state = TrendState.BEARISH
            active_low = np.nan

        trend[i] = int(state)

    return bos, choch, bsl_swept, ssl_swept, level, trend


def _bars_since(flag: np.ndarray) -> np.ndarray:
    """Causal recency: bars since the last nonzero entry (inf before the first)."""
    n = len(flag)
    out = np.full(n, np.inf, dtype=np.float64)
    last = -1
    for i in range(n):
        if flag[i] != 0:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


# ── Single-scale entry point ──────────────────────────────────────────────────
def compute_structure_features(
    df: pd.DataFrame,
    swing_length: int = 10,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute causal BOS/CHoCH/BSL/SSL structure features for one ticker's
    daily OHLCV DataFrame. Call once per fold with cutoff_date=<training
    end>, exactly like zone_features.compute_zone_features.

    Adds columns
    ------------
    structure_trend_state        : -1 / 0 / +1, persists between events
    structure_bos_flag           : +1 bullish / -1 bearish / 0   (sparse)
    structure_choch_flag         : +1 bullish / -1 bearish / 0   (sparse)
    structure_bsl_swept          : 1.0 where buy-side liquidity raided
    structure_ssl_swept          : 1.0 where sell-side liquidity raided
    structure_bars_since_bos     : recency (inf before the first event)
    structure_bars_since_choch   : recency
    structure_bars_since_bsl_sweep / _ssl_sweep : recency
    structure_level_dist_atr     : ATR-normalised signed distance from
                                   close to the most recently broken
                                   level, sticky, clipped +/-20 ATR
                                   (same convention as ict_features.py)
    """
    result = df.copy()
    idx = pd.to_datetime(result.index)

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(c)

    atr = _wilder_atr(h, l, c, 14)
    atr_floor = np.abs(c) * 5e-4
    safe_atr = np.where(np.isfinite(atr) & (atr > atr_floor), atr, atr_floor)
    safe_atr = np.where(safe_atr > 0, safe_atr, np.nan)

    cols_f = [
        "structure_trend_state", "structure_bos_flag", "structure_choch_flag",
        "structure_bsl_swept", "structure_ssl_swept", "structure_level_dist_atr",
    ]
    cols_inf = [
        "structure_bars_since_bos", "structure_bars_since_choch",
        "structure_bars_since_bsl_sweep", "structure_bars_since_ssl_sweep",
    ]

    if n < 2 * swing_length + 1:
        for col in cols_f:
            result[col] = 0.0
        for col in cols_inf:
            result[col] = np.inf
        return result

    swings = _detect_confirmed_swings(h, l, swing_length)

    if cutoff_date is not None:
        cutoff_i = int(np.searchsorted(idx.values, np.datetime64(cutoff_date), side="right")) - 1
        swings = [s for s in swings if s[3] <= cutoff_i] if cutoff_i >= 0 else []

    bos, choch, bsl_swept, ssl_swept, level, trend = _label_structure_events(h, l, c, swings)

    result["structure_trend_state"] = trend.astype(np.float32)
    result["structure_bos_flag"] = bos.astype(np.float32)
    result["structure_choch_flag"] = choch.astype(np.float32)
    result["structure_bsl_swept"] = bsl_swept
    result["structure_ssl_swept"] = ssl_swept

    result["structure_bars_since_bos"] = _bars_since(bos)
    result["structure_bars_since_choch"] = _bars_since(choch)
    result["structure_bars_since_bsl_sweep"] = _bars_since(bsl_swept)
    result["structure_bars_since_ssl_sweep"] = _bars_since(ssl_swept)

    level_ff = pd.Series(level).ffill().values
    valid = np.isfinite(level_ff) & np.isfinite(safe_atr)
    dist = np.where(valid, np.clip((c - level_ff) / safe_atr, -20.0, 20.0), 0.0)
    result["structure_level_dist_atr"] = dist.astype(np.float32)

    return result


# ── Multi-scale (fractal) wrapper ─────────────────────────────────────────────
def compute_multiscale_structure_features(
    df: pd.DataFrame,
    major_swing_length: int = 25,
    minor_swing_length: int = 5,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Run the structure engine at two fractal scales — major (HTF-bias
    proxy) and internal (LTF-shift proxy) — and derive the cross-scale
    confluence features that give each scale's raw signals their actual
    ICT meaning. A single-scale engine cannot produce these; they are
    the entire point of running two windows rather than one.

    Adds (beyond the per-scale columns, prefixed major_/internal_,
    e.g. major_trend_state, internal_choch_flag, internal_ssl_swept):

    structure_alignment : sign(major_trend_state * internal_trend_state)
        +1 = internal shift agrees with major trend (continuation context
             — the ICT-preferred entry: "trade the internal shift in the
             direction of HTF bias")
        -1 = internal shift fights the major trend (either an early major
             reversal, or a lower-quality counter-trend setup)
         0 = either scale has no defined trend yet

    NOTE: structure_judas_setup was removed after validation — it fired on
    ~1 of 3508 bars (too sparse for a tree to learn from). The raw
    ingredients (internal_bsl_swept / internal_ssl_swept / internal_choch_flag
    + major_trend_state) remain exported, so the model can rediscover the
    interaction itself, and a denser hand-built version can be revisited later.
    """
    if minor_swing_length >= major_swing_length:
        raise ValueError(
            f"minor_swing_length ({minor_swing_length}) must be < "
            f"major_swing_length ({major_swing_length}) - they represent "
            "distinct fractal scales (HTF bias vs. LTF internal shifts)."
        )

    major = compute_structure_features(df, swing_length=major_swing_length, cutoff_date=cutoff_date)
    minor = compute_structure_features(df, swing_length=minor_swing_length, cutoff_date=cutoff_date)

    result = df.copy()
    prefix_len = len("structure_")
    for col in major.columns:
        if col.startswith("structure_"):
            result[f"major_{col[prefix_len:]}"] = major[col].values
    for col in minor.columns:
        if col.startswith("structure_"):
            result[f"internal_{col[prefix_len:]}"] = minor[col].values

    major_trend = result["major_trend_state"].values
    internal_trend = result["internal_trend_state"].values
    result["structure_alignment"] = np.sign(major_trend * internal_trend).astype(np.float32)

    return result
