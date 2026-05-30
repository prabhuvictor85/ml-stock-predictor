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


class ICTFeatureEngine:
    """
    Institution-Grade ICT Feature Engine.

    Fixes applied vs v3:
      1. Boundary-safe shift (num=0 edge case fixed).
      2. FVG zone boundary ordering corrected (zh > zl guaranteed).
      3. Bear distance sign-flipped to match bull convention (+ve = in zone, -ve = away).
      4. Liquidity / Stop-Hunt detection (equal highs/lows sweep).
      5. Zone deduplication with priority (BB > OB > FVG per candle).
      6. Session filter removed — pure spatial distance features, no temporal contamination.
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
        zone_expiry_bars: int = 63,
        disp_mult: float = 3.0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        grp           : OHLCV DataFrame for a SINGLE symbol, sorted by time ascending.
                        Must contain: open, high, low, close, atr_14.
        pct_more      : OB body size multiplier threshold (default 20%).
        htf_bias      : Higher-timeframe directional bias (+1=bull, -1=bear, 0=neutral).
                        Signals against bias are suppressed.
        eq_thresh_atr : ATR multiples within which two swing points are "equal" for
                        liquidity detection.
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

        # ── 2. Liquidity / Stop-Hunt Detection ────────────────────────────────
        # Equal highs/lows: two swing points within eq_thresh_atr of each other,
        # followed by a wick that sweeps beyond and closes back inside.
        def _liquidity_sweep(swing_vals: np.ndarray, direction: str) -> np.ndarray:
            """
            Returns boolean array; True where a stop-hunt sweep is confirmed.
            direction: 'high' (BSL sweep) or 'low' (SSL sweep).

            Fully vectorised with NumPy shifts — O(n) array ops instead of an
            O(n) Python for-loop (typically 50–200× faster on real panels).
            """
            # Shift by 1 and 2 bars; mask the look-back boundary with NaN
            prev  = np.empty(n, dtype=float); prev[:]  = np.nan
            prev2 = np.empty(n, dtype=float); prev2[:] = np.nan
            prev[1:]  = swing_vals[:-1]
            prev2[2:] = swing_vals[:-2]

            # All three values must be non-NaN for the test to be valid
            valid = ~np.isnan(prev) & ~np.isnan(prev2) & ~np.isnan(safe_atr)

            # Equal-highs / equal-lows condition
            eq = valid & (np.abs(prev - prev2) <= eq_thresh_atr * safe_atr)

            if direction == "high":
                # Wick above equal highs, close back below the level
                level = np.maximum(prev, prev2)
                swept = eq & (h > level) & (c < level)
            else:
                # Wick below equal lows, close back above the level
                level = np.minimum(prev, prev2)
                swept = eq & (l < level) & (c > level)

            swept[:2] = False   # first two bars can never have two prior bars
            return swept

        bsl_swept = _liquidity_sweep(h1, "high")   # Buy-side liquidity taken
        ssl_swept = _liquidity_sweep(l1, "low")    # Sell-side liquidity taken

        # ── 4. Signal Detection ───────────────────────────────────────────────
        d_body_max = np.maximum(o1, c1)
        d_body_min = np.minimum(o1, c1)
        r_blen     = np.abs(c  - o)
        d_blen     = np.abs(c1 - o1)

        # ── ATR Displacement Gates ─────────────────────────────────────────────
        # Require genuine institutional displacement — body > disp_mult × ATR.
        # 1.5× was too loose: on NSE daily it fired on 50-60% of bars, making
        # ict_bob_active near-constant and SHAP importance near-zero. Daily uses
        # 3.0× (strict). Higher timeframes pass a smaller multiple because they
        # have very few bars (a yearly series is ~14 bars) and a fixed 3.0× yields
        # ~0 triggers there — see engineer._ICT_DISP_MULT.
        _DISP_MULT = disp_mult
        has_displacement  = r_blen > (_DISP_MULT * safe_atr)          # current bar
        has_displacement1 = d_blen > (_DISP_MULT * shift(safe_atr, 1))# previous bar (FVG)

        # Order Blocks — now gated by displacement on the signal candle
        is_bob = ((c1 < o1) & (c > o) & (o > c1) &
                  (c > d_body_max) & (r_blen >= mult * d_blen) &
                  has_displacement)
        is_sob = ((c1 > o1) & (c < o) & (o < c1) &
                  (c < d_body_min) & (r_blen >= mult * d_blen) &
                  has_displacement)

        # Breaker Blocks (require confirmed swing + stop-hunt)
        is_swing_low  = (l1 < l2) & (l1 < l3) & (l1 < l)
        is_swing_high = (h1 > h2) & (h1 > h3) & (h1 > h)

        is_bull_bb = (c1 < o1) & is_swing_low  & ssl_swept & (c > d_body_max)
        is_bear_bb = (c1 > o1) & is_swing_high & bsl_swept & (c < d_body_min)

        # Fair Value Gaps — gated by displacement on the candle that created the gap
        # (candle i-1 must be a strong displacement candle for the gap to be institutional)
        is_bull_fvg = (l > h2) & has_displacement1   # gap: h2→l, displacement on bar i-1
        is_bear_fvg = (h < l2) & has_displacement1   # gap: h→l2, displacement on bar i-1

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
        #
        # FIX: use np.maximum (element-wise peak) instead of addition (+).
        # Summation caused OB(2)+FVG(1)=3 to exceed both thresholds, wiping
        # out both signals instead of keeping the higher-priority one.
        bull_priority = np.maximum(
            np.maximum(is_bull_bb.astype(int) * ZonePriority.BB,
                       is_bob.astype(int)     * ZonePriority.OB),
            is_bull_fvg.astype(int) * ZonePriority.FVG,
        )

        bear_priority = np.maximum(
            np.maximum(is_bear_bb.astype(int) * ZonePriority.BB,
                       is_sob.astype(int)     * ZonePriority.OB),
            is_bear_fvg.astype(int) * ZonePriority.FVG,
        )

        # Suppress lower-priority signals on conflicting candles.
        # A zone is kept only when it IS the peak priority for that candle:
        #   OB kept  → only when no BB fired on the same candle (peak == OB)
        #   FVG kept → only when neither BB nor OB fired (peak == FVG)
        #   BB is always kept (highest priority — no suppression needed).
        is_bob      = is_bob      & (bull_priority == ZonePriority.OB)
        is_bull_fvg = is_bull_fvg & (bull_priority == ZonePriority.FVG)
        is_sob      = is_sob      & (bear_priority == ZonePriority.OB)
        is_bear_fvg = is_bear_fvg & (bear_priority == ZonePriority.FVG)

        # ── 7. Unified Zone Forward-Fill ──────────────────────────────────────
        # Zone expiry: a zone older than zone_expiry_bars without a price
        # reaction is considered stale and marked inactive.
        # Default 63 bars = ~3 months on daily. Caller should pass
        # timeframe-appropriate values for HTF resampled data:
        #   daily=63, weekly=26, monthly=12, quarterly=8, yearly=3
        ZONE_EXPIRY_BARS = zone_expiry_bars

        def _ffill_zone(
                trigger: np.ndarray,
                zh: np.ndarray,
                zl: np.ndarray,
                price: np.ndarray,
                is_bull: bool,
                mid_cancel: bool = False,
                flip_sign: bool = False,
        ):
            ah = np.where(trigger, zh, np.nan)
            al = np.where(trigger, zl, np.nan)
            ff_h = pd.Series(ah).ffill().values
            ff_l = pd.Series(al).ffill().values
            mid = (ff_h + ff_l) / 2.0

            # Track age of current zone (bars since last trigger)
            trigger_idx = np.where(trigger)[0]
            age = np.full(n, np.inf)
            if len(trigger_idx) > 0:
                # For each bar, find the most recent trigger and compute age
                last_trigger = np.searchsorted(trigger_idx, np.arange(n), side='right') - 1
                valid_trigger = last_trigger >= 0
                age[valid_trigger] = np.arange(n)[valid_trigger] - trigger_idx[last_trigger[valid_trigger]]

            not_expired = age <= ZONE_EXPIRY_BARS

            if mid_cancel:
                still = (price >= mid) if is_bull else (price <= mid)
            else:
                still = (price >= ff_l) if is_bull else (price <= ff_h)

            # One-way mitigation latch: once a zone is violated it stays dead
            # until a NEW trigger overwrites it. Without this, a zone resurrects
            # whenever price re-crosses the (still forward-filled) boundary —
            # contradicting ICT semantics (a mitigated zone is dead) and making
            # `active` flicker, which corrupts the distance features and SHAP.
            # cumsum(trigger) gives a per-zone segment id; cummax within each
            # segment makes the violation sticky until the next trigger resets it.
            seg      = np.cumsum(trigger)
            violated = (pd.Series((~still).astype(int))
                        .groupby(seg).cummax().values.astype(bool))

            active = (~np.isnan(ff_h) & ~violated & not_expired).astype(float)

            # ── Raw % distance from zone mid (SMC-faithful) ──────────────────────────
            pct_dist = np.where(mid != 0, (price - mid) / mid * 100, 0.0)
            if flip_sign:
                pct_dist = -pct_dist

            # ── ATR-normalized distance (ML cross-asset comparability) ───────────────
            atr_dist = (price - mid) / safe_atr
            if flip_sign:
                atr_dist = -atr_dist

            # Zero out when zone inactive
            pct_dist = np.where(active == 1, pct_dist, 0.0)
            atr_dist = np.where(active == 1, atr_dist, 0.0)

            return active, pct_dist, atr_dist

        # Bull zones
        (grp["ict_bob_active"],
         grp["ict_bob_pct_dist"],
         grp["ict_bob_atr_dist"]) = _ffill_zone(is_bob, d_body_max, d_body_min, c, True)

        (grp["ict_bullbb_active"],
         grp["ict_bullbb_pct_dist"],
         grp["ict_bullbb_atr_dist"]) = _ffill_zone(is_bull_bb, d_body_max, d_body_min, c, True)

        (grp["ict_bullfvg_active"],
         grp["ict_bullfvg_pct_dist"],
         grp["ict_bullfvg_atr_dist"]) = _ffill_zone(is_bull_fvg, l, h2, c, True, mid_cancel=True)

        # Bear zones
        (grp["ict_sob_active"],
         grp["ict_sob_pct_dist"],
         grp["ict_sob_atr_dist"]) = _ffill_zone(is_sob, d_body_max, d_body_min, c, False, flip_sign=True)

        (grp["ict_bearbb_active"],
         grp["ict_bearbb_pct_dist"],
         grp["ict_bearbb_atr_dist"]) = _ffill_zone(is_bear_bb, d_body_max, d_body_min, c, False, flip_sign=True)

        (grp["ict_bearfvg_active"],
         grp["ict_bearfvg_pct_dist"],
         grp["ict_bearfvg_atr_dist"]) = _ffill_zone(is_bear_fvg, l2, h, c, False, mid_cancel=True, flip_sign=True)

        # ── 8. Liquidity Sweep Flags ──────────────────────────────────────────
        grp["ict_bsl_swept"] = bsl_swept.astype(float)
        grp["ict_ssl_swept"] = ssl_swept.astype(float)

        # ── 9. Zone Priority Metadata (PERSISTENT — derived from active flags) ─
        # Export the priority of the highest-priority CURRENTLY-LIVE zone, taken
        # from the forward-filled `active` flags rather than the trigger-instant
        # `bull_priority`/`bear_priority` (which are kept above only for signal
        # dedup). A trigger-instant export is nonzero on ~1 candle per zone, so on
        # higher timeframes (few bars) it is ~0 everywhere and collapses the MTF
        # composite (which weights HTF most) to zero. Deriving from active flags
        # makes the feature persist for the life of the zone.
        grp["ict_bull_zone_priority"] = np.maximum.reduce([
            grp["ict_bullbb_active"].values  * int(ZonePriority.BB),
            grp["ict_bob_active"].values     * int(ZonePriority.OB),
            grp["ict_bullfvg_active"].values * int(ZonePriority.FVG),
        ]).astype(float)
        grp["ict_bear_zone_priority"] = np.maximum.reduce([
            grp["ict_bearbb_active"].values  * int(ZonePriority.BB),
            grp["ict_sob_active"].values     * int(ZonePriority.OB),
            grp["ict_bearfvg_active"].values * int(ZonePriority.FVG),
        ]).astype(float)

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