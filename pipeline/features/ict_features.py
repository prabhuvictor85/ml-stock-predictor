import numpy as np
import pandas as pd
from enum import IntEnum

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR implementation."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Wilder's EMA is equivalent to ewm(alpha=1/period, adjust=False)
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

def _wilder_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ADX implementation."""
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    up_move = high - prev_high
    down_move = prev_low - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    atr = _wilder_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-8)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    return dx.ewm(alpha=1/period, adjust=False).mean().values

# ── Zone Priority Hierarchy (ICT: BB > OB > FVG) ──────────────────────────────
class ZonePriority(IntEnum):
    FVG = 1
    OB  = 2
    BB  = 3

# ── Session Masks (UTC hours) ──────────────────────────────────────────────────
SESSION_WINDOWS = {
    "asia":   (0,  8),
    "london": (7,  16),
    "ny":     (13, 22),
}

SESSION_WEIGHTS = {
    "asia":   0.5,
    "london": 1.0,
    "ny":     1.0,
    "overlap": 1.5,   # London/NY overlap 13:00–16:00 UTC
}


class ICTFeatureEngine:
    """
    Institution-Grade ICT Feature Engine.

    Fixes applied vs v3:
      1. Boundary-safe shift (num=0 edge case fixed).
      2. FVG zone boundary ordering corrected (zh > zl guaranteed).
      3. Bear distance sign-flipped to match bull convention (+ve = in zone, -ve = away).
      4. Liquidity / Stop-Hunt detection (equal highs/lows sweep).
      5. Zone deduplication with priority (BB > OB > FVG per candle).
      6. Session filtering with session weights baked into dist features.
      7. MTF bias input parameter (htf_bias: +1 bull, -1 bear, 0 neutral).
      8. Symbol isolation guard (asserts single symbol per call).
    """

    # ── Public API ─────────────────────────────────────────────────────────────
    def compute(
        self,
        grp: pd.DataFrame,
        pct_more: float = 20.0,
        htf_bias: int = 0,
        eq_thresh_atr: float = 0.1,
        session_filter: bool = True,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        grp            : OHLCV DataFrame for a SINGLE symbol, sorted by time ascending.
                         Must contain: open, high, low, close, atr_14.
                         Optional: timestamp (datetime-like) for session filtering.
        pct_more       : OB body size multiplier threshold (default 20%).
        htf_bias       : Higher-timeframe directional bias (+1=bull, -1=bear, 0=neutral).
                         Signals against bias are suppressed.
        eq_thresh_atr  : ATR multiples within which two swing points are "equal" for
                         liquidity detection.
        session_filter : If True and 'timestamp' column exists, weight signals by session.
        """
        self._assert_single_symbol(grp)
        original_index = grp.index
        grp = grp.copy().reset_index(drop=True)

        h, l, o, c = [grp[x].values.astype(float) for x in ["high", "low", "open", "close"]]
        atr = grp["atr_14"].values.astype(float)
        n   = len(c)

        # Mask invalid ATRs → NaN distances instead of explosions
        safe_atr = np.where((atr > 0) & ~np.isnan(atr), atr, np.nan)
        mult = 1.0 + pct_more / 100.0

        # ── 1. Boundary-Safe Shift (fixes num=0 edge case) ────────────────────
        def shift(arr: np.ndarray, num: int, fill: float = np.nan) -> np.ndarray:
            if num == 0:
                return arr.copy()
            res = np.full_like(arr, fill, dtype=float)
            if num > 0:
                res[num:] = arr[:-num]
            else:
                res[:num] = arr[-num:]
            return res

        h1, l1, o1, c1 = shift(h,1), shift(l,1), shift(o,1), shift(c,1)
        h2, l2          = shift(h,2), shift(l,2)
        h3, l3          = shift(h,3), shift(l,3)

        # ── 2. Session Weights ─────────────────────────────────────────────────
        sess_weight = np.ones(n, dtype=float)
        if session_filter and "timestamp" in grp.columns:
            ts   = pd.to_datetime(grp["timestamp"])
            hour = ts.dt.hour.values
            is_london  = (hour >= 7)  & (hour < 16)
            is_ny      = (hour >= 13) & (hour < 22)
            is_asia    = (hour >= 0)  & (hour < 8)
            is_overlap = is_london & is_ny

            sess_weight = np.where(is_overlap, SESSION_WEIGHTS["overlap"],
                         np.where(is_ny,       SESSION_WEIGHTS["ny"],
                         np.where(is_london,   SESSION_WEIGHTS["london"],
                         np.where(is_asia,     SESSION_WEIGHTS["asia"], 1.0))))

        # ── 3. Liquidity / Stop-Hunt Detection ────────────────────────────────
        # Equal highs/lows: two swing points within eq_thresh_atr of each other,
        # followed by a wick that sweeps beyond and closes back inside.
        def _liquidity_sweep(swing_vals: np.ndarray, direction: str) -> np.ndarray:
            """
            Returns boolean array; True where a stop-hunt sweep is confirmed.
            direction: 'high' (BSL sweep) or 'low' (SSL sweep).
            """
            swept = np.zeros(n, dtype=bool)
            for i in range(2, n):
                prev  = swing_vals[i - 1]
                prev2 = swing_vals[i - 2]
                if np.isnan(prev) or np.isnan(prev2) or np.isnan(safe_atr[i]):
                    continue
                eq = abs(prev - prev2) <= eq_thresh_atr * safe_atr[i]
                if not eq:
                    continue
                if direction == "high":
                    # Wick above equal highs, close back below
                    swept[i] = (h[i] > max(prev, prev2)) and (c[i] < max(prev, prev2))
                else:
                    swept[i] = (l[i] < min(prev, prev2)) and (c[i] > min(prev, prev2))
            return swept

        bsl_swept = _liquidity_sweep(h1, "high")   # Buy-side liquidity taken
        ssl_swept = _liquidity_sweep(l1, "low")    # Sell-side liquidity taken

        # ── 4. Signal Detection ───────────────────────────────────────────────
        d_body_max = np.maximum(o1, c1)
        d_body_min = np.minimum(o1, c1)
        r_blen     = np.abs(c  - o)
        d_blen     = np.abs(c1 - o1)

        # Order Blocks
        is_bob = ((c1 < o1) & (c > o) & (o > c1) &
                  (c > d_body_max) & (r_blen >= mult * d_blen))
        is_sob = ((c1 > o1) & (c < o) & (o < c1) &
                  (c < d_body_min) & (r_blen >= mult * d_blen))

        # Breaker Blocks (require confirmed swing + stop-hunt)
        is_swing_low  = (l1 < l2) & (l1 < l3) & (l1 < l)
        is_swing_high = (h1 > h2) & (h1 > h3) & (h1 > h)

        is_bull_bb = (c1 < o1) & is_swing_low  & ssl_swept & (c > d_body_max)
        is_bear_bb = (c1 > o1) & is_swing_high & bsl_swept & (c < d_body_min)

        # Fair Value Gaps — corrected boundary ordering (zh > zl guaranteed)
        is_bull_fvg = l > h2          # gap: h2 (bottom) to l (top)  → zh=l,  zl=h2  ✓
        is_bear_fvg = h < l2          # gap: h (top)    to l2 (bottom)→ zh=l2, zl=h   ✓

        # ── 5. HTF Bias Filter ────────────────────────────────────────────────
        if htf_bias == 1:             # Bull bias → suppress bear signals
            is_sob     = np.zeros(n, dtype=bool)
            is_bear_bb = np.zeros(n, dtype=bool)
            is_bear_fvg= np.zeros(n, dtype=bool)
        elif htf_bias == -1:          # Bear bias → suppress bull signals
            is_bob     = np.zeros(n, dtype=bool)
            is_bull_bb = np.zeros(n, dtype=bool)
            is_bull_fvg= np.zeros(n, dtype=bool)

        # ── 6. Zone Priority Deduplication ────────────────────────────────────
        # When multiple signal types fire on the same candle, keep highest priority only.
        # Bull side: BB(3) > OB(2) > FVG(1)
        bull_priority = (is_bull_bb.astype(int) * ZonePriority.BB  +
                         is_bob.astype(int)      * ZonePriority.OB  +
                         is_bull_fvg.astype(int) * ZonePriority.FVG)

        bear_priority = (is_bear_bb.astype(int)  * ZonePriority.BB  +
                         is_sob.astype(int)       * ZonePriority.OB  +
                         is_bear_fvg.astype(int)  * ZonePriority.FVG)

        # Suppress lower-priority signals on conflicting candles
        is_bob      = is_bob      & (bull_priority <= ZonePriority.OB  * is_bob)
        is_bull_fvg = is_bull_fvg & (bull_priority <= ZonePriority.FVG * is_bull_fvg)
        is_sob      = is_sob      & (bear_priority <= ZonePriority.OB  * is_sob)
        is_bear_fvg = is_bear_fvg & (bear_priority <= ZonePriority.FVG * is_bear_fvg)

        # ── 7. Unified Zone Forward-Fill ──────────────────────────────────────
        def _ffill_zone(
            trigger:    np.ndarray,
            zh:         np.ndarray,    # zone top (zh > zl always)
            zl:         np.ndarray,    # zone bottom
            price:      np.ndarray,
            is_bull:    bool,
            mid_cancel: bool = False,
            flip_sign:  bool = False,  # flip dist sign for bear zones → +ve = inside
            weight:     np.ndarray = None,
        ):
            ah = np.where(trigger, zh, np.nan)
            al = np.where(trigger, zl, np.nan)
            ff_h = pd.Series(ah).ffill().values
            ff_l = pd.Series(al).ffill().values
            mid  = (ff_h + ff_l) / 2.0

            if mid_cancel:
                still = (price >= mid) if is_bull else (price <= mid)
            else:
                still = (price >= ff_l) if is_bull else (price <= ff_h)

            active = (~np.isnan(ff_h) & still).astype(float)

            raw_dist = (price - mid) / safe_atr
            if flip_sign:
                raw_dist = -raw_dist          # bear: +ve when price is below mid (inside zone)

            dist = np.where(active == 1, raw_dist, 0.0)

            # Bake in session weight
            if weight is not None:
                dist = dist * weight

            return active, dist

        w = sess_weight   # shorthand

        # Bull zones
        grp["ict_bob_active"],     grp["ict_bob_dist"]     = _ffill_zone(is_bob,      d_body_max, d_body_min, c, True,  weight=w)
        grp["ict_bullbb_active"],  grp["ict_bullbb_dist"]  = _ffill_zone(is_bull_bb,  d_body_max, d_body_min, c, True,  weight=w)
        grp["ict_bullfvg_active"], grp["ict_bullfvg_dist"] = _ffill_zone(is_bull_fvg, l,          h2,         c, True,  mid_cancel=True, weight=w)

        # Bear zones (flip_sign=True → dist is +ve when price is inside zone)
        grp["ict_sob_active"],     grp["ict_sob_dist"]     = _ffill_zone(is_sob,      d_body_max, d_body_min, c, False, flip_sign=True, weight=w)
        grp["ict_bearbb_active"],  grp["ict_bearbb_dist"]  = _ffill_zone(is_bear_bb,  d_body_max, d_body_min, c, False, flip_sign=True, weight=w)
        grp["ict_bearfvg_active"], grp["ict_bearfvg_dist"] = _ffill_zone(is_bear_fvg, l2,         h,          c, False, mid_cancel=True, flip_sign=True, weight=w)

        # ── 8. Liquidity Sweep Flags ──────────────────────────────────────────
        grp["ict_bsl_swept"] = bsl_swept.astype(float)
        grp["ict_ssl_swept"] = ssl_swept.astype(float)

        # ── 9. Zone Priority Metadata ─────────────────────────────────────────
        grp["ict_bull_zone_priority"] = bull_priority.astype(float)
        grp["ict_bear_zone_priority"] = bear_priority.astype(float)

        # ── 10. Session Weight ─────────────────────────────────────────────────
        grp["ict_session_weight"] = sess_weight

        # Restore original index (reset_index(drop=True) at the start replaced it
        # with integers; we must put the date index back so engineer.py can re-attach
        # the ticker level correctly).
        grp.index = original_index

        return grp

    # ── Private Helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _assert_single_symbol(grp: pd.DataFrame):
        """Guard against cross-symbol ffill leakage."""
        if "symbol" in grp.columns:
            unique = grp["symbol"].nunique()
            assert unique == 1, (
                f"ICTFeatureEngine.compute() received {unique} symbols. "
                "Call once per symbol group to prevent ffill leakage across symbols."
            )