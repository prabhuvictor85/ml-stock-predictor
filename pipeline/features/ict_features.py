import numpy as np
import pandas as pd
from enum import IntEnum
import numba

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ATR implementation."""
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Wilder's EMA is equivalent to ewm(alpha=1/period, adjust=False)
    return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

def _wilder_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """Wilder's directional indicators. Returns (plus_di, minus_di) as np arrays.

    +DI quantifies upward directional pressure, -DI downward. Their relationship
    (+DI > -DI vs -DI > +DI) gives trend DIRECTION — the piece ADX discards.
    """
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

    plus_di = (100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-8)).values
    minus_di = (100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / (atr_s + 1e-8)).values
    return plus_di, minus_di


def _wilder_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ADX (directionless trend strength). Direction lives in +DI/-DI."""
    plus_di, minus_di = _wilder_di(high, low, close, period)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    return pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values

# ── Liquidity Pool Feature Column Names ───────────────────────────────────────
# Feature layout per side (offset 0–8 = BSL, 9–17 = SSL):
#   0  dist_atr     — distance from price to pool near-edge (ATR, always +ve)
#   1  sweep_dep    — max wick penetration depth through pool (ATR)
#   2  width        — pool price range in ATR units
#   3  density      — touches / log1p(pool age) — normalised touch frequency
#   4  sweep_decay  — exp decay since last sweep (halflife ≈ 14 bars); 0 if never swept
#   5  touches      — raw touch count
#   6  sweep_cnt    — number of times the pool has been swept
#   7  log_untouched — log1p(bars since last touch)
#   8  strength     — recency-weighted composite: touches × width × recency / (1 + sweeps)
_LIQ_BSL_COLS = [
    "ict_liq_bsl_dist_atr",
    "ict_liq_bsl_sweep_dep",
    "ict_liq_bsl_width",
    "ict_liq_bsl_density",
    "ict_liq_bsl_sweep_decay",
    "ict_liq_bsl_touches",
    "ict_liq_bsl_sweep_cnt",
    "ict_liq_bsl_untouched",
    "ict_liq_bsl_strength",
]
_LIQ_SSL_COLS = [c.replace("bsl", "ssl") for c in _LIQ_BSL_COLS]


@numba.njit
def _run_liquidity_engine(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    n_swing: int,
    expiry: int,
    max_pools: int,
    merge_prox: float = 0.25,
    sweep_halflife: float = 20.0,
) -> np.ndarray:
    """
    JIT-compiled liquidity pool engine.

    Detects BSL (buy-side) and SSL (sell-side) liquidity pools from confirmed
    swing highs/lows, tracks their lifecycle (touches, sweeps, expiry), and
    emits 20 features per bar (9 BSL + 9 SSL + 2 sweep flags) for the nearest
    live pool on each side.

    Fix log:
      F1 — slot overflow: recycles inactive slots; evicts weakest when full.
      F2 — strength uses recency weight (exp decay) not stale log(untouched).
      F3 — deferred deactivation via p_deact_next[]: a swept pool stays active
           through its own bar so sweep features are visible to the model, then
           dies at the start of the next bar. Without this, sweep_cnt / sweep_dep
           / sweep_decay are structurally zero everywhere (vanishing signal bug).
      F4 — displacement kill: close >= pool_top (BSL) or close <= pool_bot (SSL)
           means liquidity was absorbed into a displacement; pool dies next bar.
      F5 — timestamp anchor: p_first_t / p_last_t stored as i - n_swing (the
           actual extremum bar), not i (the confirmation bar). Fixes age/decay
           features being skewed by n_swing bars.
      F6 — ATR anchor: merge threshold uses atr[i - n_swing] (swing bar), not
           atr[i] (current bar). Prevents earnings-spike ATR from collapsing
           unrelated historical structure levels into the same pool.
      F7 — off+1 is sweep_dep (wick depth), not the redundant dist_bot.
    """
    n       = len(close)
    n_pools = 0

    # ── SoA pool storage ──────────────────────────────────────────────────────
    p_side        = np.zeros(max_pools, dtype=np.int32)
    p_centroid    = np.zeros(max_pools, dtype=np.float64)
    p_min         = np.zeros(max_pools, dtype=np.float64)
    p_max         = np.zeros(max_pools, dtype=np.float64)
    p_first_t     = np.zeros(max_pools, dtype=np.int32)
    p_last_t      = np.zeros(max_pools, dtype=np.int32)
    p_last_swp    = np.full(max_pools, -1, dtype=np.int32)
    p_touches     = np.zeros(max_pools, dtype=np.int32)
    p_sweep_cnt   = np.zeros(max_pools, dtype=np.int32)
    p_sweep_dep   = np.zeros(max_pools, dtype=np.float64)
    p_active      = np.zeros(max_pools, dtype=np.bool_)
    p_extr_atr    = np.zeros(max_pools, dtype=np.float64)
    # F3: deferred deactivation — True means "die at the start of next bar"
    p_deact_next  = np.zeros(max_pools, dtype=np.bool_)

    features = np.zeros((n, 20), dtype=np.float64)

    for i in range(2 * n_swing, n):
        safe_atr  = atr[i]           if atr[i]           > 1e-6 else 1e-6
        # F6: ATR at the swing bar (i - n_swing), not the current discovery bar
        swing_atr = atr[i - n_swing] if atr[i - n_swing] > 1e-6 else 1e-6

        # ── F3: apply deferred deactivations from the previous bar ───────────
        # This runs BEFORE everything else so that pools swept on bar i-1 are
        # dead before we process bar i, but were alive for bar i-1's features.
        for p in range(n_pools):
            if p_deact_next[p]:
                p_active[p]     = False
                p_deact_next[p] = False

        # ── Swing detection: N-bar confirmation on each side ──────────────────
        sh = high[i - n_swing]
        sl = low[i - n_swing]

        is_high = True
        for j in range(i - 2 * n_swing, i - n_swing):
            if high[j] >= sh:
                is_high = False
                break
        if is_high:
            for j in range(i - n_swing + 1, i + 1):
                if high[j] >= sh:
                    is_high = False
                    break

        is_low = True
        for j in range(i - 2 * n_swing, i - n_swing):
            if low[j] <= sl:
                is_low = False
                break
        if is_low:
            for j in range(i - n_swing + 1, i + 1):
                if low[j] <= sl:
                    is_low = False
                    break

        # ── 1. Pool lifecycle ─────────────────────────────────────────────────
        for p in range(n_pools):
            if not p_active[p]:
                continue

            if p_side[p] == 1:   # BSL pool — equal highs above current price
                if high[i] > p_max[p] and close[i] < p_max[p]:
                    # SWEEP: wick above the pool, close back below.
                    # Record the event, then defer deactivation (F3) so features
                    # on THIS bar reflect sweep_cnt=1 and sweep_dep>0.
                    p_last_swp[p]  = i
                    p_sweep_cnt[p] += 1
                    pen = (high[i] - p_max[p]) / safe_atr
                    if pen > p_sweep_dep[p]:
                        p_sweep_dep[p] = pen
                    p_deact_next[p] = True
                    features[i, 18] = 1.0
                elif close[i] >= p_max[p]:
                    # F4: DISPLACEMENT — clean close through the pool. Stops
                    # absorbed into the move; pool has no remaining liquidity.
                    p_deact_next[p] = True

            else:                # SSL pool — equal lows below current price
                if low[i] < p_min[p] and close[i] > p_min[p]:
                    # SWEEP: wick below the pool, close back above.
                    p_last_swp[p]  = i
                    p_sweep_cnt[p] += 1
                    pen = (p_min[p] - low[i]) / safe_atr
                    if pen > p_sweep_dep[p]:
                        p_sweep_dep[p] = pen
                    p_deact_next[p] = True
                    features[i, 19] = 1.0
                elif close[i] <= p_min[p]:
                    # F4: DISPLACEMENT — clean close below the pool.
                    p_deact_next[p] = True

            # Expiry: also deferred so age features are correct on the last bar
            max_age = float(expiry) * (1.0 + np.log1p(float(p_touches[p])))
            if float(i - p_last_t[p]) > max_age:
                p_deact_next[p] = True

        # ── 2. Pool add / merge ───────────────────────────────────────────────
        if is_high or is_low:
            price = sh if is_high else sl
            s     = 1  if is_high else -1

            found = False
            for p in range(n_pools):
                # Skip pools being deactivated — don't merge a new swing into
                # a pool that was just swept or displaced this bar
                if not p_active[p] or p_deact_next[p] or p_side[p] != s:
                    continue
                # Use the pool's originating ATR for merge tolerance, not current
                thresh = merge_prox * p_extr_atr[p]
                
                d = p_centroid[p] - price
                if d < 0.0:
                    d = -d
                if d < thresh:
                    if price < p_min[p]:
                        p_min[p] = price
                    if price > p_max[p]:
                        p_max[p] = price
                    p_centroid[p] = (p_min[p] + p_max[p]) * 0.5
                    # F5: anchor last-touch time to the actual swing bar
                    p_last_t[p]   = i - n_swing
                    p_touches[p] += 1
                    found = True
                    break

            if not found:
                # Allocate: inactive slot first, then extend, then evict oldest
                slot = -1
                for p in range(n_pools):
                    if not p_active[p]:
                        slot = p
                        break
                if slot == -1:
                    if n_pools < max_pools:
                        slot    = n_pools
                        n_pools += 1
                    else:
                        # Preserve HTF-like pools: old and lightly-touched levels
                        # are often the strongest liquidity magnets in SMC.
                        min_score = 1e18
                        slot   = 0
                        for p in range(max_pools):
                            width   = p_max[p] - p_min[p]
                            touches = float(p_touches[p])
                            age     = float(i - p_last_t[p])

                            # Base weakness: narrow + repeatedly interacted pools.
                            score = width * (1.0 + touches)

                            # Protection bonus: keep long-lived, mostly untouched pools.
                            if touches <= 1.0 and age >= float(expiry):
                                score += 1e6
                            elif age >= float(2 * expiry):
                                score += 1e3
                            
                            # Protect pools that are scheduled for natural deactivation next bar
                            if p_deact_next[p]:
                                score += 1e9

                            if score < min_score:
                                min_score = score
                                slot   = p

                p_side[slot]       = s
                p_centroid[slot]   = price
                p_min[slot]        = price
                p_max[slot]        = price
                p_extr_atr[slot]   = swing_atr
                # F5: anchor timestamps to actual extremum bar, not discovery bar
                p_first_t[slot]    = i - n_swing
                p_last_t[slot]     = i - n_swing
                p_last_swp[slot]   = -1
                p_touches[slot]    = 1
                p_sweep_cnt[slot]  = 0
                p_sweep_dep[slot]  = 0.0
                p_active[slot]     = True
                p_deact_next[slot] = False

        # ── 3. Feature extraction ─────────────────────────────────────────────
        # Pools with p_deact_next=True were swept or displaced THIS bar and are
        # still p_active=True — intentional. Their sweep_cnt / sweep_dep /
        # sweep_decay are non-zero here for the first (and only) time, giving
        # the model a signal exactly when the liquidity purge happens.
        best_bsl = -1
        best_ssl = -1
        min_bsl  = 1e18
        min_ssl  = 1e18

        for p in range(n_pools):
            if not p_active[p]:
                continue
            if p_side[p] == 1:
                dist = abs(p_max[p] - close[i]) / safe_atr
                if dist < min_bsl:
                    min_bsl  = dist
                    best_bsl = p
            elif p_side[p] == -1:
                dist = abs(close[i] - p_min[p]) / safe_atr
                if dist < min_ssl:
                    min_ssl  = dist
                    best_ssl = p

        for idx in range(2):
            p   = best_bsl if idx == 0 else best_ssl
            off = idx * 9
            if p == -1:
                continue

            width     = (p_max[p] - p_min[p]) / safe_atr
            untouched = float(i - p_last_t[p])
            recency   = np.exp(-untouched / float(expiry))
            strength  = (float(p_touches[p]) * (1.0 + width) * recency) / (1.0 + float(p_sweep_cnt[p]))

            if idx == 0:
                features[i, off + 0] = abs(p_max[p] - close[i]) / safe_atr
            else:
                features[i, off + 0] = abs(close[i] - p_min[p]) / safe_atr
            features[i, off + 1] = p_sweep_dep[p]
            features[i, off + 2] = width
            features[i, off + 3] = float(p_touches[p]) / np.log1p(float(i - p_first_t[p] + 1))
            if p_last_swp[p] != -1:
                features[i, off + 4] = np.exp(-float(i - p_last_swp[p]) / sweep_halflife)
            else:
                features[i, off + 4] = np.exp(-float(i - p_first_t[p]) / sweep_halflife)
            features[i, off + 5] = float(p_touches[p])
            features[i, off + 6] = float(p_sweep_cnt[p])
            features[i, off + 7] = np.log1p(untouched)
            features[i, off + 8] = strength

    return features


# ── Zone Priority Hierarchy (ICT: BK > RB > OB > FVG) ────────────────────────
class ZonePriority(IntEnum):
    FVG = 1
    OB  = 2
    BB  = 3   # Rejection Block (legacy name)
    BK  = 4   # True Breaker Block — mitigated OB revisited from opposite side


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
      7. Symbol isolation guard (asserts single symbol per call).
    """

    # ── Public API ─────────────────────────────────────────────────────────────
    def compute(
        self,
        grp: pd.DataFrame,
        implementation_mode: str = "legacy",
        pct_more: float = 20.0,
        eq_thresh_atr: float = 0.1,
        zone_expiry_bars: int = 63,
        disp_mult: float = 0.0,
        proximity_pct: float = 0.0,
        liq_n_swing: int = 5,
        liq_merge_prox: float = 0.25,
        liq_sweep_halflife: float = 20.0,
        fvg_sweep_lookback: int = 3,
        bos_lookback: int = 3,
        fvg_min_gap_atr: float = 0.1,
        ob_bos_hard_gate: bool = False,
        fvg_bos_hard_gate: bool = False,
        ob_pd_hard_gate: bool = False,
        fvg_pd_hard_gate: bool = False,
        fvg_sweep_hard_gate: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        grp           : OHLCV DataFrame for a SINGLE symbol, sorted by time ascending.
                        Must contain: open, high, low, close, atr_14.
        implementation_mode : "legacy" (default) keeps backward-compatible
                        behavior; "institutional" enables BOS hard gates only
                        (OB/FVG must form near a recent break of structure);
                        "strict" matches the full reference Pine "Strict ICT
                        Mode" — BOS gate + premium/discount alignment gate
                        (bull OB/FVG only valid in discount, bear only in
                        premium) + FVG opposite-side sweep confirmation gate.
                        "institutional" alone is a partial replication of the
                        reference strict logic; use "strict" for the full set.
        pct_more      : OB body size multiplier threshold (default 20%).
        eq_thresh_atr : ATR multiples within which two swing points are "equal" for
                        liquidity detection.
        proximity_pct : if > 0, a zone is INACTIVE while price has run more than
                        this fraction beyond it (bull: price > zone_top*(1+pct);
                        bear: price < zone_bottom*(1-pct)). Mirrors the
                        ZoneAnalyzer proximity gate so both zone engines agree
                        on when price has "left a zone behind". Transient, not
                        sticky: a retest back inside range reactivates the zone.
                        0.0 (default) = gate off — without it the engine calls
                        ~88%% of stock-days "in a zone" (measured), because
                        nothing ever dies from price running away.
        fvg_sweep_lookback : bars to look back for opposite-side liquidity sweep
                        confirmation on high-conviction FVG triggers.
        bos_lookback : bars to look back for recent BOS context used by
                        OB/FVG BOS-conditioned companion features.
        fvg_min_gap_atr : minimum FVG gap size in ATR units (default 0.1).
        ob_bos_hard_gate : if True, OB triggers require a recent BOS.
        fvg_bos_hard_gate : if True, FVG triggers require a recent BOS.
        ob_pd_hard_gate : if True, bull OB requires price in discount,
                        bear OB requires price in premium.
        fvg_pd_hard_gate : if True, bull FVG requires price in discount,
                        bear FVG requires price in premium.
        fvg_sweep_hard_gate : if True, bull FVG requires a recent SSL sweep,
                        bear FVG requires a recent BSL sweep.
        """
        self._assert_single_symbol(grp)
        mode = implementation_mode.lower().strip()
        if mode not in ("legacy", "institutional", "strict"):
            raise ValueError("implementation_mode must be 'legacy', 'institutional', or 'strict'")

        # Mode presets; explicit flags can still force stricter behavior.
        if mode == "institutional":
            ob_bos_hard_gate = True
            fvg_bos_hard_gate = True
        elif mode == "strict":
            ob_bos_hard_gate = True
            fvg_bos_hard_gate = True
            ob_pd_hard_gate = True
            fvg_pd_hard_gate = True
            fvg_sweep_hard_gate = True

        original_index = grp.index
        grp = grp.copy().reset_index(drop=True)

        h, l, o, c = [grp[x].values.astype(float) for x in ["high", "low", "open", "close"]]
        atr = grp["atr_14"].values.astype(float)
        n   = len(c)

        # Mask invalid ATRs and floor the denominator at 5 bps of price. On
        # illiquid/penny names ATR can collapse to ~1e-6, and dividing a zone
        # distance by it explodes the *_atr_dist features. A sub-5bps daily
        # range is noise, not signal; flooring keeps the feature finite.
        atr_floor = np.abs(c) * 5e-4
        safe_atr = np.where(np.isnan(atr) | (atr <= 0), np.nan,
                            np.where(atr > atr_floor, atr, atr_floor))
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

        # BOS events from most recent confirmed swing extremes.
        # Use liq_n_swing to match the liquidity engine (instead of a shallow 1-bar confirmation)
        s_n = int(liq_n_swing)
        s_h = pd.Series(h)
        s_l = pd.Series(l)

        def _is_swing(series: pd.Series, n_bars: int, is_high: bool) -> np.ndarray:
            """Strict n-bar swing: bar i-n_bars is a local extremum confirmed at bar i.
            Center must be strictly greater (or less) than every other bar in [i-2*n_bars, i]."""
            res = pd.Series(True, index=series.index)
            center = series.shift(n_bars)
            for j in range(2 * n_bars + 1):
                if j == n_bars:
                    continue
                val = series.shift(j)
                res = res & (center > val if is_high else center < val)
            return res.values

        # BOS swing: n_swing-bar confirmation — matches the liquidity engine pool anchors.
        is_swing_high = _is_swing(s_h, s_n, True)
        is_swing_low  = _is_swing(s_l, s_n, False)
        
        swing_high_lvl = np.where(is_swing_high, shift(h, s_n), np.nan)
        swing_low_lvl  = np.where(is_swing_low, shift(l, s_n), np.nan)
        last_swing_high = pd.Series(swing_high_lvl).ffill().values
        last_swing_low  = pd.Series(swing_low_lvl).ffill().values
        bull_bos_state = (~np.isnan(last_swing_high)) & (c > last_swing_high)
        # fill=0.0 so bar 0's prior state is False — NaN fill would convert to True
        # via float→bool, masking a genuine bar-0 event and confusing reviewers.
        bull_bos_evt = bull_bos_state & ~shift(bull_bos_state.astype(float), 1, fill=0.0).astype(bool)
        bear_bos_state = (~np.isnan(last_swing_low)) & (c < last_swing_low)
        bear_bos_evt = bear_bos_state & ~shift(bear_bos_state.astype(float), 1, fill=0.0).astype(bool)

        # ── Market Structure State & CHoCH ────────────────────────────────────
        # State: +1=bullish, -1=bearish, 0=neutral (last BOS direction wins).
        # CHoCH fires when a BOS goes AGAINST the current structure state —
        # a structural reversal signal rather than a continuation.
        _bos_dir  = np.where(bull_bos_evt, 1.0, np.where(bear_bos_evt, -1.0, 0.0))
        mss_arr   = pd.Series(np.where(_bos_dir != 0, _bos_dir, np.nan)).ffill().fillna(0.0).values
        _prev_mss = shift(mss_arr, 1, fill=0.0)
        bull_choch_evt = bull_bos_evt & (_prev_mss <= 0)   # bull break against neutral/bear
        bear_choch_evt = bear_bos_evt & (_prev_mss >= 0)   # bear break against neutral/bull

        # ── Premium / Discount Arrays ─────────────────────────────────────────
        # ICT dealing range = [last confirmed swing low, last confirmed swing high].
        # Equilibrium (EQ) = midpoint. Price above EQ = premium (favour sells),
        # price below EQ = discount (favour buys). Primary context filter for setups.
        _eq_level   = (last_swing_high + last_swing_low) / 2.0
        _deal_range = last_swing_high - last_swing_low
        _safe_dr    = np.where(_deal_range > 0, _deal_range, np.nan)
        _pdr        = (c - last_swing_low) / _safe_dr  # 0=at swing low, 0.5=EQ, 1=at swing high

        def _recent_true(mask: np.ndarray, lookback: int) -> np.ndarray:
            if lookback <= 1:
                return mask.copy()
            return pd.Series(mask).rolling(lookback, min_periods=1).max().fillna(0).astype(bool).values

        # Backward-compatible stop-hunt detector used by BB triggers and tests.
        # The structural engine sweeps are OR'ed in below.
        def _liquidity_sweep(swing_vals: np.ndarray, direction: str) -> np.ndarray:
            prev  = np.empty(n, dtype=float); prev[:]  = np.nan
            prev2 = np.empty(n, dtype=float); prev2[:] = np.nan
            prev[1:]  = swing_vals[:-1]
            prev2[2:] = swing_vals[:-2]

            valid = ~np.isnan(prev) & ~np.isnan(prev2) & ~np.isnan(safe_atr)
            eq = valid & (np.abs(prev - prev2) <= eq_thresh_atr * safe_atr)

            if direction == "high":
                level = np.maximum(prev, prev2)
                swept = eq & (h > level) & (c < level)
            else:
                level = np.minimum(prev, prev2)
                swept = eq & (l < level) & (c > level)

            swept[:2] = False
            return swept

        legacy_bsl_swept = _liquidity_sweep(h1, "high")
        legacy_ssl_swept = _liquidity_sweep(l1, "low")

        # ── 2. Unified Liquidity & Stop-Hunt Detection ────────────────────────
        # We run the deep structural Numba engine first so its pristine sweep events
        # can feed into the Breaker Block (Rejection Block) logic, ensuring those
        # signals trigger off true multi-timeframe liquidity pools, not noise.
        if len(grp) > 2 * liq_n_swing:
            _nan_atr = int(np.sum(np.isnan(safe_atr)))
            if _nan_atr > 0:
                import warnings
                warnings.warn(
                    f"ICTFeatureEngine: {_nan_atr} NaN ATR rows passed to liquidity engine; "
                    "those bars will use 1e-6 floor — ATR-normalised pool features may be distorted.",
                    UserWarning, stacklevel=3,
                )
            liq_feats = _run_liquidity_engine(
                h, l, c, safe_atr,
                liq_n_swing, zone_expiry_bars, 300,
                merge_prox=liq_merge_prox, sweep_halflife=liq_sweep_halflife
            )
            bsl_swept = legacy_bsl_swept | (liq_feats[:, 18] > 0.5)
            ssl_swept = legacy_ssl_swept | (liq_feats[:, 19] > 0.5)
        else:
            liq_feats = np.zeros((n, 20), dtype=np.float64)
            bsl_swept = legacy_bsl_swept
            ssl_swept = legacy_ssl_swept

        # ── 4. Signal Detection ───────────────────────────────────────────────
        d_body_max = np.maximum(o1, c1)
        d_body_min = np.minimum(o1, c1)
        r_blen     = np.abs(c  - o)
        d_blen     = np.abs(c1 - o1)

        # ── ATR Displacement Gate (OPTIONAL — OFF by default) ──────────────────
        # The reference Pine indicator (Smart-Money-Trading-Complete) qualifies an
        # OB purely structurally: prev red + rally + close>prev-body-top + rally
        # body >= (1+pctMore%) * prev body. It has NO absolute ATR displacement
        # gate. An earlier 3.0× ATR gate was bolted on here to tame active-flag
        # saturation, but it annihilated 100% of Order Blocks (6678→0) and 99% of
        # FVGs — breaking fidelity to the source indicator. We therefore drop it
        # from the creation rules. disp_mult is kept as an OPTIONAL extra filter
        # (default disabled: a value <= 0 means "no gate") for experimentation.
        _DISP_MULT = disp_mult
        if _DISP_MULT and _DISP_MULT > 0:
            has_displacement  = r_blen > (_DISP_MULT * safe_atr)
            has_displacement1 = d_blen > (_DISP_MULT * shift(safe_atr, 1))
        else:
            has_displacement  = np.ones(n, dtype=bool)
            has_displacement1 = np.ones(n, dtype=bool)

        # Order Blocks — SMC structural definition
        #   prevWasRed & isRally & close>dBodyMax & rBLen >= mult*dBLen
        # Fix: Removed the restrictive (o > c1) gap-down exclusion rule so 
        # massive overnight open-sweep origins are correctly embraced.
        is_bob = ((c1 < o1) & (c > o) & 
                  (c > d_body_max) & (r_blen >= mult * d_blen) &
                  has_displacement)
        is_sob = ((c1 > o1) & (c < o) & 
                  (c < d_body_min) & (r_blen >= mult * d_blen) &
                  has_displacement)

        # RB swing: n_bars=1 is the minimum causal window for the immediate-reversal
        # pattern — the reversal candle (bar i) confirms bar i-1 as the local extremum.
        # Using a larger n would require future bars for confirmation and introduce lag.
        rb_swing_low  = _is_swing(s_l, 1, False)   # l[i-1] strictly below l[i] and l[i-2]
        rb_swing_high = _is_swing(s_h, 1, True)    # h[i-1] strictly above h[i] and h[i-2]

        # Breaker taxonomy note:
        # Current production trigger is a liquidity-sweep rejection pattern
        # (often called a rejection block / turtle soup variant), exported via
        # legacy RB columns.
        is_bull_rb = (c1 < o1) & rb_swing_low  & ssl_swept & (c > d_body_max)
        is_bear_rb = (c1 > o1) & rb_swing_high & bsl_swept & (c < d_body_min)
        is_bull_bb = is_bull_rb
        is_bear_bb = is_bear_rb

        # Fair Value Gaps €” structural SMC definition
        # a 3-bar gap (low[i] > high[i+2] bull / high[i] < low[i+2] bear). 
        fvg_formation_atr = shift(safe_atr, 1)
        fvg_min_gap = float(fvg_min_gap_atr) * fvg_formation_atr
        is_bull_fvg = (l > h2) & has_displacement1 & ((l - h2) >= fvg_min_gap)   # gap: h2†’l
        is_bear_fvg = (h < l2) & has_displacement1 & ((l2 - h) >= fvg_min_gap)   # gap: h†’l2

        # Pre-compute recent BOS windows here; reused by both the hard gate and the
        # companion features section below — avoids computing _recent_true twice.
        _recent_bull_bos = _recent_true(bull_bos_evt, int(bos_lookback))
        _recent_bear_bos = _recent_true(bear_bos_evt, int(bos_lookback))

        # Hard gate: use a recent WINDOW (not exact-bar match) so the OB that CAUSED
        # the BOS — typically 1-3 bars before the crossing — is still admitted.
        # The old exact-bar gate (bull_bos_evt) almost never co-occurs with the OB
        # trigger on the same candle, making the institutional mode a near-total filter.
        if ob_bos_hard_gate:
            is_bob = is_bob & _recent_bull_bos
            is_sob = is_sob & _recent_bear_bos
        if fvg_bos_hard_gate:
            is_bull_fvg = is_bull_fvg & _recent_bull_bos
            is_bear_fvg = is_bear_fvg & _recent_bear_bos

        # Premium/discount alignment gate (matches reference Pine "Strict ICT
        # Mode" strictRequirePDForOBFVG): bull setups only count when price is
        # in discount (cheap side of the dealing range), bear only in premium.
        # NaN-safe: an undefined _pdr (no confirmed swing range yet) fails
        # both sides, which is correct — no range means no premium/discount
        # context to validate against.
        _in_discount = ~np.isnan(_pdr) & (_pdr < 0.5)
        _in_premium  = ~np.isnan(_pdr) & (_pdr > 0.5)
        if ob_pd_hard_gate:
            is_bob = is_bob & _in_discount
            is_sob = is_sob & _in_premium
        if fvg_pd_hard_gate:
            is_bull_fvg = is_bull_fvg & _in_discount
            is_bear_fvg = is_bear_fvg & _in_premium

        # FVG opposite-side sweep confirmation gate (matches Pine
        # strictRequireSweepForFVG): a bull FVG only counts if sell-side
        # liquidity was recently swept (stop-hunt before the reversal up),
        # bear FVG only if buy-side liquidity was recently swept. Computed
        # here (not just in the companion-feature section below) so the
        # gate can actually constrain is_bull_fvg/is_bear_fvg themselves.
        _recent_ssl_swept_gate = _recent_true(ssl_swept, int(fvg_sweep_lookback))
        _recent_bsl_swept_gate = _recent_true(bsl_swept, int(fvg_sweep_lookback))
        if fvg_sweep_hard_gate:
            is_bull_fvg = is_bull_fvg & _recent_ssl_swept_gate
            is_bear_fvg = is_bear_fvg & _recent_bsl_swept_gate

        # ── 5. Zone Priority Deduplication ────────────────────────────────────
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
            # Canonical ordering guard: if caller accidentally passes zh < zl (e.g. FVG
            # with inverted bounds from degenerate input), swap so ff_h is always the ceiling.
            # np.maximum/minimum propagate NaN correctly, so pre-trigger NaN rows are unaffected.
            ff_h, ff_l = np.maximum(ff_h, ff_l), np.minimum(ff_h, ff_l)
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

            # Proximity gate (transient, mirrors ZoneAnalyzer semantics): while
            # price has run more than proximity_pct beyond the zone, the zone is
            # not tradeable context — price left it behind. A retest back within
            # range reactivates it (unlike mitigation, which is sticky).
            if proximity_pct and proximity_pct > 0:
                with np.errstate(invalid="ignore"):
                    if is_bull:
                        too_far = price > ff_h * (1.0 + proximity_pct)
                    else:
                        too_far = price < ff_l * (1.0 - proximity_pct)
                too_far = np.where(np.isnan(ff_h), False, too_far)
            else:
                too_far = np.zeros(n, dtype=bool)

            active = (~np.isnan(ff_h) & ~violated & not_expired & ~too_far).astype(float)

            # ── Raw % distance from zone mid (SMC-faithful) ──────────────────────────
            pct_dist = np.where(mid != 0, (price - mid) / mid * 100, 0.0)
            if flip_sign:
                pct_dist = -pct_dist

            # ── ATR-normalized distance (ML cross-asset comparability) ───────────────
            # Clip to ±20 ATR: beyond that the distance is saturated/meaningless
            # and a defensive cap against any residual tiny-ATR explosion.
            atr_dist = np.clip((price - mid) / safe_atr, -20.0, 20.0)
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

        (grp["ict_bullrb_active"],
         grp["ict_bullrb_pct_dist"],
         grp["ict_bullrb_atr_dist"]) = _ffill_zone(is_bull_bb, d_body_max, d_body_min, c, True)

        (grp["ict_bullfvg_active"],
         grp["ict_bullfvg_pct_dist"],
         grp["ict_bullfvg_atr_dist"]) = _ffill_zone(is_bull_fvg, l, h2, c, True, mid_cancel=True)

        # Bear zones
        (grp["ict_sob_active"],
         grp["ict_sob_pct_dist"],
         grp["ict_sob_atr_dist"]) = _ffill_zone(is_sob, d_body_max, d_body_min, c, False, flip_sign=True)

        (grp["ict_bearrb_active"],
         grp["ict_bearrb_pct_dist"],
         grp["ict_bearrb_atr_dist"]) = _ffill_zone(is_bear_bb, d_body_max, d_body_min, c, False, flip_sign=True)

        (grp["ict_bearfvg_active"],
         grp["ict_bearfvg_pct_dist"],
         grp["ict_bearfvg_atr_dist"]) = _ffill_zone(is_bear_fvg, l2, h, c, False, mid_cancel=True, flip_sign=True)

        # ── 8. Derived Zone Attributes ────────────────────────────────────────
        # Forward-fill OB/SOB body bounds (post-dedup) used for fill-pct and Breaker Blocks.
        _ff_bob_h = pd.Series(np.where(is_bob, d_body_max, np.nan)).ffill().values
        _ff_bob_l = pd.Series(np.where(is_bob, d_body_min, np.nan)).ffill().values
        _ff_sob_h = pd.Series(np.where(is_sob, d_body_max, np.nan)).ffill().values
        _ff_sob_l = pd.Series(np.where(is_sob, d_body_min, np.nan)).ffill().values

        # OB fill pct: 0.0=price at zone top (untouched), 1.0=at zone bottom (mitigated).
        _bob_rng = np.maximum(_ff_bob_h - _ff_bob_l, 1e-6)
        _sob_rng = np.maximum(_ff_sob_h - _ff_sob_l, 1e-6)
        grp["ict_bob_fill_pct"] = np.where(
            grp["ict_bob_active"].values == 1,
            np.clip((_ff_bob_h - c) / _bob_rng, 0.0, 1.0), 0.0)
        grp["ict_sob_fill_pct"] = np.where(
            grp["ict_sob_active"].values == 1,
            np.clip((c - _ff_sob_l) / _sob_rng, 0.0, 1.0), 0.0)

        # FVG fill pct.
        _ff_bfvg_h = pd.Series(np.where(is_bull_fvg, l,  np.nan)).ffill().values
        _ff_bfvg_l = pd.Series(np.where(is_bull_fvg, h2, np.nan)).ffill().values
        _ff_rfvg_h = pd.Series(np.where(is_bear_fvg, l2, np.nan)).ffill().values
        _ff_rfvg_l = pd.Series(np.where(is_bear_fvg, h,  np.nan)).ffill().values
        _bfvg_rng  = np.maximum(_ff_bfvg_h - _ff_bfvg_l, 1e-6)
        _rfvg_rng  = np.maximum(_ff_rfvg_h - _ff_rfvg_l, 1e-6)
        grp["ict_bullfvg_fill_pct"] = np.where(
            grp["ict_bullfvg_active"].values == 1,
            np.clip((_ff_bfvg_h - c) / _bfvg_rng, 0.0, 1.0), 0.0)
        grp["ict_bearfvg_fill_pct"] = np.where(
            grp["ict_bearfvg_active"].values == 1,
            np.clip((c - _ff_rfvg_l) / _rfvg_rng, 0.0, 1.0), 0.0)

        # Displacement quality: body / total_range on the OB formation candle.
        # 1.0 = pure impulsive candle, 0.0 = doji / wick-dominated.
        _disp_q = r_blen / np.maximum(h - l, 1e-6)
        grp["ict_bob_disp_quality"] = np.where(
            grp["ict_bob_active"].values == 1,
            pd.Series(np.where(is_bob, _disp_q, np.nan)).ffill().fillna(0.0).values, 0.0)
        grp["ict_sob_disp_quality"] = np.where(
            grp["ict_sob_active"].values == 1,
            pd.Series(np.where(is_sob, _disp_q, np.nan)).ffill().fillna(0.0).values, 0.0)

        # ── 8b. Zone Entry & Rejection Signals ───────────────────────────────
        # Both BOB and SOB fill_pct share the same direction:
        #   0.0 = price at the zone entry edge (untouched)
        #   1.0 = price at the violation edge (deeply penetrated)
        # BOB: entry edge = zone top (bob_h), violation edge = zone bottom (bob_l)
        # SOB: entry edge = zone bottom (sob_l), violation edge = zone top (sob_h)
        # Rejection logic is therefore identical for both zones.
        #
        # Entry event tracks fill_pct going 0 → >0 (price physically enters the zone),
        # NOT the active flag going 0→1 (which is the OB formation event, when price
        # is far from the zone due to the displacement that just created it).
        _OB_ENTRY_LOOKBACK      = 5    # bars (~1 week): "recently entered" window
        _OB_REJECTION_LOOKBACK  = 10   # bars (~2 weeks): rejection check window
        _OB_REJECTION_MIN_FILL  = 0.2  # must penetrate ≥20% into zone to count as a test
        _OB_REJECTION_EXIT_FILL = 0.05 # must recover to within 5% of zone entry edge

        _bob_active_f = grp["ict_bob_active"].values
        _sob_active_f = grp["ict_sob_active"].values
        _bob_fill = grp["ict_bob_fill_pct"].values
        _sob_fill = grp["ict_sob_fill_pct"].values

        _bob_entry_evt = (_bob_active_f > 0) & (_bob_fill > 0) & (shift(_bob_fill, 1, fill=0.0) == 0.0)
        _sob_entry_evt = (_sob_active_f > 0) & (_sob_fill > 0) & (shift(_sob_fill, 1, fill=0.0) == 0.0)
        grp["ict_bob_entered_recent"] = _recent_true(_bob_entry_evt, _OB_ENTRY_LOOKBACK).astype(float)
        grp["ict_sob_entered_recent"] = _recent_true(_sob_entry_evt, _OB_ENTRY_LOOKBACK).astype(float)

        _bob_max_fill = (pd.Series(_bob_fill)
                         .rolling(_OB_REJECTION_LOOKBACK, min_periods=1).max().values)
        _sob_max_fill = (pd.Series(_sob_fill)
                         .rolling(_OB_REJECTION_LOOKBACK, min_periods=1).max().values)

        grp["ict_bob_rejection"] = (
            (_bob_active_f > 0) &
            (_bob_max_fill >= _OB_REJECTION_MIN_FILL) &
            (_bob_fill     <= _OB_REJECTION_EXIT_FILL)
        ).astype(float)
        grp["ict_sob_rejection"] = (
            (_sob_active_f > 0) &
            (_sob_max_fill >= _OB_REJECTION_MIN_FILL) &
            (_sob_fill     <= _OB_REJECTION_EXIT_FILL)
        ).astype(float)

        # ── 9. True Breaker Blocks ────────────────────────────────────────────
        # A mitigated Bull OB → Bear Breaker Block (prior support = new resistance).
        # A mitigated Bear OB → Bull Breaker Block (prior resistance = new support).
        # Only triggers on PRICE VIOLATION (close through body), not expiry.
        _bob_prev_act = shift(grp["ict_bob_active"].values.astype(float), 1, fill=0.0).astype(bool)
        _sob_prev_act = shift(grp["ict_sob_active"].values.astype(float), 1, fill=0.0).astype(bool)
        _bear_bk_trig = _bob_prev_act & (grp["ict_bob_active"].values == 0) & (c < _ff_bob_l)
        _bull_bk_trig = _sob_prev_act & (grp["ict_sob_active"].values == 0) & (c > _ff_sob_h)

        (grp["ict_bullbk_active"],
         grp["ict_bullbk_pct_dist"],
         grp["ict_bullbk_atr_dist"]) = _ffill_zone(
            _bull_bk_trig, _ff_sob_h, _ff_sob_l, c, True)
        (grp["ict_bearbk_active"],
         grp["ict_bearbk_pct_dist"],
         grp["ict_bearbk_atr_dist"]) = _ffill_zone(
            _bear_bk_trig, _ff_bob_h, _ff_bob_l, c, False, flip_sign=True)

        # ── 10. Prior Session Levels ──────────────────────────────────────────
        # PDH/PDL: previous day high/low (= h1/l1 on daily data).
        # PWH/PWL: prior week high/low — institutional weekly delivery range.
        try:
            _h_s = pd.Series(h, index=original_index)
            _l_s = pd.Series(l, index=original_index)
            _pwh = _h_s.resample("W-FRI").max().shift(1).reindex(original_index, method="ffill").values
            _pwl = _l_s.resample("W-FRI").min().shift(1).reindex(original_index, method="ffill").values
        except Exception:
            _pwh = np.full(n, np.nan)
            _pwl = np.full(n, np.nan)
        _pw_eq = (_pwh + _pwl) / 2.0

        grp["ict_pdh_dist_atr"]  = np.where(np.isfinite(h1 / safe_atr),
                                             np.clip((c - h1) / safe_atr, -20.0, 20.0), 0.0)
        grp["ict_pdl_dist_atr"]  = np.where(np.isfinite(l1 / safe_atr),
                                             np.clip((c - l1) / safe_atr, -20.0, 20.0), 0.0)
        grp["ict_pwh_dist_atr"]  = np.where(~np.isnan(_pwh),
                                             np.clip((c - _pwh) / safe_atr, -20.0, 20.0), 0.0)
        grp["ict_pwl_dist_atr"]  = np.where(~np.isnan(_pwl),
                                             np.clip((c - _pwl) / safe_atr, -20.0, 20.0), 0.0)
        grp["ict_pw_eq_atr_dist"] = np.where(~np.isnan(_pw_eq),
                                              np.clip((c - _pw_eq) / safe_atr, -20.0, 20.0), 0.0)

        # ── 11. Liquidity Sweep Flags ─────────────────────────────────────────
        grp["ict_bsl_swept"] = bsl_swept.astype(float)
        grp["ict_ssl_swept"] = ssl_swept.astype(float)

        # Discrete BOS event (sparse ~0.5% prevalence — useful as a gate, not a feature)
        grp["ict_bull_bos"] = bull_bos_evt.astype(float)
        grp["ict_bear_bos"] = bear_bos_evt.astype(float)
        # Persistent state: price is above/below the last confirmed swing — richer ML signal
        grp["ict_bull_bos_state"]  = bull_bos_state.astype(float)
        grp["ict_bear_bos_state"]  = bear_bos_state.astype(float)

        # CHoCH: BOS against current structure state → higher-quality reversal signal
        grp["ict_bull_choch"] = bull_choch_evt.astype(float)
        grp["ict_bear_choch"] = bear_choch_evt.astype(float)
        # Market Structure State: +1=bullish, -1=bearish, 0=neutral
        grp["ict_mss"] = mss_arr

        # Premium / Discount position within the current dealing range
        grp["ict_premium_disc_ratio"] = np.where(~np.isnan(_pdr), np.clip(_pdr, -0.5, 1.5), 0.5)
        grp["ict_in_premium"]         = np.where(~np.isnan(_pdr), (_pdr > 0.5).astype(float), 0.0)
        grp["ict_in_discount"]        = np.where(~np.isnan(_pdr), (_pdr < 0.5).astype(float), 0.0)
        grp["ict_eq_atr_dist"]        = np.where(
            ~np.isnan(_eq_level),
            np.clip((c - _eq_level) / safe_atr, -20.0, 20.0), 0.0)

        # High-conviction FVG trigger: require recent opposite-side sweep.
        # Reuses the gate-section computation (same fvg_sweep_lookback) —
        # see _recent_ssl_swept_gate/_recent_bsl_swept_gate above.
        grp["ict_bullfvg_sweep_conf"] = (is_bull_fvg & _recent_ssl_swept_gate).astype(float)
        grp["ict_bearfvg_sweep_conf"] = (is_bear_fvg & _recent_bsl_swept_gate).astype(float)

        # BOS-conditioned companion features (additive context, no hard gate).
        # _recent_bull/bear_bos was already computed above for the hard gate.
        recent_bull_bos = _recent_bull_bos
        recent_bear_bos = _recent_bear_bos
        grp["ict_bob_bos_conf"] = (is_bob & recent_bull_bos).astype(float)
        grp["ict_sob_bos_conf"] = (is_sob & recent_bear_bos).astype(float)
        grp["ict_bullfvg_bos_conf"] = (is_bull_fvg & recent_bull_bos).astype(float)
        grp["ict_bearfvg_bos_conf"] = (is_bear_fvg & recent_bear_bos).astype(float)
        # Recent-window BOS state: smoother than the event, more useful as a direct feature
        grp["ict_bull_bos_recent"] = recent_bull_bos.astype(float)
        grp["ict_bear_bos_recent"] = recent_bear_bos.astype(float)

        # Macro trend regime — consecutive BOS streak counters.
        # Mirrors the Pine macro filter (bearBosCount/bullBosCount) but as features,
        # not hard gates. High bear_streak = sustained downtrend context (long signals
        # historically weak). High bull_streak = sustained uptrend (short signals weak).
        # macro_regime is signed: positive = bullish trend, negative = bearish trend.
        _bear_streak = np.zeros(n, dtype=np.float32)
        _bull_streak = np.zeros(n, dtype=np.float32)
        _bc, _uc = 0, 0
        for _i in range(n):
            if bear_bos_evt[_i]:
                _bc += 1
                _uc  = 0
            elif bull_bos_evt[_i]:
                _uc += 1
                _bc  = 0
            _bear_streak[_i] = _bc
            _bull_streak[_i] = _uc
        grp["ict_bear_bos_streak"] = _bear_streak
        grp["ict_bull_bos_streak"] = _bull_streak
        grp["ict_macro_regime"]    = (_bull_streak - _bear_streak).astype(np.float32)

        # True confluence: both zones active AND both within 5 ATR of current price.
        # Without the proximity check, an OB at $100 and FVG at $150 with price at $120
        # both show active=1 and trigger confluence — the zones are structurally unrelated.
        _CONF_DIST_ATR = 5.0
        _ob_near   = (grp["ict_bob_active"].values > 0)      & (np.abs(grp["ict_bob_atr_dist"].values)      < _CONF_DIST_ATR)
        _fvg_near  = (grp["ict_bullfvg_active"].values > 0)  & (np.abs(grp["ict_bullfvg_atr_dist"].values)  < _CONF_DIST_ATR)
        _sob_near  = (grp["ict_sob_active"].values > 0)      & (np.abs(grp["ict_sob_atr_dist"].values)      < _CONF_DIST_ATR)
        _bfvg_near = (grp["ict_bearfvg_active"].values > 0)  & (np.abs(grp["ict_bearfvg_atr_dist"].values)  < _CONF_DIST_ATR)
        grp["ict_bull_ob_fvg_confluence"] = (_ob_near  & _fvg_near).astype(float)
        grp["ict_bear_ob_fvg_confluence"] = (_sob_near & _bfvg_near).astype(float)

        # ── 9. Zone Priority Metadata (PERSISTENT — derived from active flags) ─
        # Export the priority of the highest-priority CURRENTLY-LIVE zone, taken
        # from the forward-filled `active` flags rather than the trigger-instant
        # `bull_priority`/`bear_priority` (which are kept above only for signal
        # dedup). A trigger-instant export is nonzero on ~1 candle per zone, so on
        # higher timeframes (few bars) it is ~0 everywhere and collapses the MTF
        # composite (which weights HTF most) to zero. Deriving from active flags
        # makes the feature persist for the life of the zone.
        grp["ict_bull_zone_priority"] = np.maximum.reduce([
            grp["ict_bullbk_active"].values  * int(ZonePriority.BK),
            grp["ict_bullrb_active"].values  * int(ZonePriority.BB),
            grp["ict_bob_active"].values     * int(ZonePriority.OB),
            grp["ict_bullfvg_active"].values * int(ZonePriority.FVG),
        ]).astype(float)
        grp["ict_bear_zone_priority"] = np.maximum.reduce([
            grp["ict_bearbk_active"].values  * int(ZonePriority.BK),
            grp["ict_bearrb_active"].values  * int(ZonePriority.BB),
            grp["ict_sob_active"].values     * int(ZonePriority.OB),
            grp["ict_bearfvg_active"].values * int(ZonePriority.FVG),
        ]).astype(float)

        # ── 10. Liquidity Pool Feature Mapping ────────────────────────────────
        for j, col in enumerate(_LIQ_BSL_COLS):
            grp[col] = liq_feats[:, j]
        for j, col in enumerate(_LIQ_SSL_COLS):
            grp[col] = liq_feats[:, 9 + j]

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
