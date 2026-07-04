"""
PivotFeatureEngine — floor-pivot / Central-Pivot-Range / Camarilla feature family.

Source concepts: Franklin O. Ochoa, *Secrets of a Pivot Boss*. Formulas verified
against the text:
  - Floor pivots:  PP = (H+L+C)/3,  R1 = 2·PP−L,  S1 = 2·PP−H, ...
  - CPR:           BC = (H+L)/2,  TC = (PP−BC)+PP   (mirror of BC around PP)
  - Camarilla:     H3 = C + RANGE·1.1/4,  H4 = C + RANGE·1.1/2,  H5 = (H/L)·C  (Ch.7)

Design contract (mirrors ICTFeatureEngine):
  - `compute(grp, safe_atr)` takes ONE ticker's daily frame (lowercase
    open/high/low/close, DatetimeIndex, no ticker level) plus the pipeline's
    floored Wilder ATR array, and returns a DataFrame with EXACTLY the columns in
    PIVOT_FEATURE_COLS — the raw `pivot_*` names (engineer.py prefixes them to
    `features_pivot_*`). All float32, NaN-native (never fillna — the LGBM path
    reads NaN as "unknown").
  - Every level is built from the PRIOR session's H/L/C (shift(1)), so a level for
    day t is fixed before day t opens — leak-free against today's open/close.
  - Categorical states are emitted as FIXED-vocabulary 0/1 columns (never
    pd.get_dummies): every ticker yields the identical column set, so the panel
    concat aligns.

Two corrections to the source draft, applied consistently everywhere:
  1. TC/BC normalization. The book notes the TC/BC formulas can invert (TC landing
     below BC); its software assigns the higher value to TC, the lower to BC. We do
     the same at the source (tc = max, bc = min) so width ≥ 0 and every downstream
     comparison (two-day state, trend side, opening relation, virgin band) is
     well-defined.
  2. Trend side ordering. Bull = close > TC, Bear = close < BC, inside the band =
     Neutral. (A naive [close>BC, close<TC] select reads an inside-band close as
     Bullish, which is wrong.)

One deliberate enrichment over the draft: the two-day CPR relationship uses the
book's overlap definition (Ch.4/6) so all seven states are reachable — clean
Higher/Lower require a gap (bc > tc_prev / tc < bc_prev), Overlapping Higher/Lower
are the partial-overlap cases. The draft's if-order left the overlapping states
structurally unreachable. Logged as a deviation in PROTOCOL.md §3.1.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

try:
    import numba
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep alongside ict_features
    _HAVE_NUMBA = False

# Period-end resample aliases — picks the spelling the installed pandas accepts
# (M/Q/Y on <2.2, ME/QE/YE on >=2.2). Same guard the ICT/zone code uses; a
# hardcoded "ME" silently zeroes MTF pivots on pandas <2.2.
from pipeline.features.zone_features import _MONTH_END, _YEAR_END

# ── Frozen feature vocabulary (69 columns) ────────────────────────────────────
# Order is load-bearing: the engine builds a dict and reindexes to this list, so a
# missing/renamed key raises immediately instead of silently shifting the panel.
_LEVEL_DIST_COLS = [
    "dist_pp_atr", "dist_r1_atr", "dist_s1_atr", "dist_tc_atr", "dist_bc_atr",
    "dist_h3_atr", "dist_h4_atr", "dist_h5_atr", "dist_l3_atr", "dist_l4_atr", "dist_l5_atr",
]
_WIDTH_COLS = [
    "cpr_width_atr", "cpr_width_pctile", "cpr_width_chg_pct", "cpr_expanding",
    "cpr_extreme_narrow", "cpr_regime_narrow", "cpr_regime_wide", "cpr_compression_streak",
]
_VIRGIN_COLS = ["virgin_cpr", "virgin_cpr_age", "untouched_cpr_count"]
_CAM_BEHAVIOR_COLS = [
    "cam_h3_reversal", "cam_l3_reversal", "cam_h4_breakout", "cam_l4_breakout",
    "cam_return_from_h4", "cam_return_from_l4",
]
_OPEN_COLS = [
    "open_above_tc", "open_below_bc", "open_gap_above_r1", "open_gap_below_s1",
    "open_above_h4", "open_above_h3", "open_below_l3", "open_below_l4",
]
_TWODAY_COLS = [
    "twoday_higher_value", "twoday_lower_value", "twoday_inside_value",
    "twoday_outside_value", "twoday_overlap_higher", "twoday_overlap_lower",
    "twoday_unchanged", "twoday_streak",
    "bias_confirmed", "bias_rejected", "bias_confirmed_breakout",
]
_TREND_COLS = [
    "cpr_trend_bull", "cpr_trend_bear", "cpr_trend_streak",
    "pp_slope_3d_atr", "pp_slope_5d_atr",
]
_ACCEPT_COLS = ["pp_accept_streak", "pp_accept_count_5d", "pp_accept_count_10d"]
_NEAREST_COLS = [
    "dist_nearest_level_atr", "dist_nearest_res_atr", "dist_nearest_sup_atr",
    "above_all_levels", "below_all_levels",
]
_MTF_COLS = [
    "dist_weekly_pp_atr", "dist_weekly_r1_atr", "dist_weekly_s1_atr",
    "dist_monthly_pp_atr", "dist_monthly_r1_atr", "dist_monthly_s1_atr",
    "dist_yearly_pp_atr", "dist_yearly_r1_atr", "dist_yearly_s1_atr",
]

# Internal assembly order uses bare names (dist_pp_atr, ...); the engine emits
# them with a `pivot_` family tag so FeatureEngineer's `features_` prefix yields
# the canonical `features_pivot_*` (mirrors ICT: engine emits `ict_*`).
_ASSEMBLY_ORDER: list[str] = (
    _LEVEL_DIST_COLS + _WIDTH_COLS + _VIRGIN_COLS + _CAM_BEHAVIOR_COLS + _OPEN_COLS
    + _TWODAY_COLS + _TREND_COLS + _ACCEPT_COLS + _NEAREST_COLS + _MTF_COLS
)
PIVOT_FEATURE_COLS: list[str] = [f"pivot_{c}" for c in _ASSEMBLY_ORDER]
assert len(PIVOT_FEATURE_COLS) == 69, f"expected 69 pivot cols, got {len(PIVOT_FEATURE_COLS)}"

FEATURE_PREFIX = "features_"

# Non-winsorize list (fully prefixed names). The engineer's discrete auto-detector
# catches 0/1 flags, but streaks/counts/ages run 0..60 and a percentile is bounded
# [0,1] — none of those have per-date outliers to clip, and clipping a streak would
# distort it. ATR distances / slopes / width_atr / width_chg_pct DO winsorize.
PIVOT_WINSORIZE_EXCLUDE: frozenset[str] = frozenset(
    f"{FEATURE_PREFIX}pivot_{name}" for name in (
        "cpr_width_pctile", "cpr_compression_streak", "cpr_trend_streak",
        "twoday_streak", "pp_accept_streak", "pp_accept_count_5d",
        "pp_accept_count_10d", "virgin_cpr_age", "untouched_cpr_count",
    )
)

# Rolling / window knobs (book/draft defaults — prevalence-style, NOT
# outcome-validated; counted as one DOF family in PROTOCOL.md §3.1).
_WIDTH_LOOKBACK = 60
_WIDTH_MIN_PERIODS = 10
_NARROW_PCTILE = 0.25
_WIDE_PCTILE = 0.75
_EXTREME_PCTILE = 0.10
_VIRGIN_LOOKBACK = 60
_PP_SLOPE_WINDOWS = (3, 5)
_ACCEPT_WINDOWS = (5, 10)
_CAM_MULT = 1.1  # Camarilla range multiplier (book)


def pivot_features_enabled() -> bool:
    """Call-time env read (TWAP pattern), so tests can monkeypatch without reimport.
    Frozen default OFF — the production panel/recipe is unchanged unless enabled."""
    return os.environ.get("PIVOT_FEATURES", "0").strip().lower() in {"1", "true", "on", "yes"}


def _run_length(mask: pd.Series) -> pd.Series:
    """Consecutive-True run length ending at each row; 0 on False rows."""
    mask = mask.astype(bool)
    change = mask.ne(mask.shift())
    grp_id = change.cumsum()
    cc = mask.groupby(grp_id).cumcount() + 1
    return cc.where(mask, 0)


def _virgin_cpr_numpy(high, low, band_lo, band_hi, lookback):
    """Pure-python/numpy fallback (only used if numba import failed)."""
    n = len(high)
    virgin_today = np.full(n, np.nan)
    age = np.full(n, np.nan)
    count_untouched = np.full(n, np.nan)
    for t in range(n):
        if np.isnan(band_lo[t]):
            continue
        virgin_today[t] = 0.0 if (high[t] >= band_lo[t] and low[t] <= band_hi[t]) else 1.0
        start = max(0, t - lookback)
        if start >= t:
            age[t] = 0.0
            count_untouched[t] = 0.0
            continue
        running_max = -np.inf
        running_min = np.inf
        cnt = 0.0
        max_age = 0.0
        for j in range(t, start - 1, -1):
            if high[j] > running_max:
                running_max = high[j]
            if low[j] < running_min:
                running_min = low[j]
            if j == t:
                continue
            if np.isnan(band_lo[j]):
                continue
            touched = (running_max >= band_lo[j]) and (running_min <= band_hi[j])
            if not touched:
                cnt += 1.0
                a = t - j
                if a > max_age:
                    max_age = float(a)
        count_untouched[t] = cnt
        age[t] = max_age
    return virgin_today, age, count_untouched


if _HAVE_NUMBA:
    _virgin_cpr_engine = numba.njit(cache=True)(_virgin_cpr_numpy)
else:  # pragma: no cover
    _virgin_cpr_engine = _virgin_cpr_numpy


class PivotFeatureEngine:
    """Stateless per-ticker pivot feature computer. Instantiate once, call
    `compute(grp, safe_atr)` per ticker inside the FeatureEngineer loop."""

    def compute(self, grp: pd.DataFrame, safe_atr: np.ndarray) -> pd.DataFrame:
        idx = grp.index
        if isinstance(idx, pd.MultiIndex):
            raise ValueError("PivotFeatureEngine.compute expects a single-ticker frame "
                             "(DatetimeIndex, no ticker level)")

        o = grp["open"].astype(float)
        h = grp["high"].astype(float)
        l = grp["low"].astype(float)
        c = grp["close"].astype(float)
        atr = pd.Series(np.asarray(safe_atr, dtype=float), index=idx)

        ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
        prng = ph - pl
        valid = pc.notna()  # prior-day levels available

        # ── Floor pivots (from prior session) ─────────────────────────────
        pp = (ph + pl + pc) / 3.0
        r1 = 2 * pp - pl
        s1 = 2 * pp - ph
        r2 = pp + prng
        s2 = pp - prng
        r3 = ph + 2 * (pp - pl)
        s3 = pl - 2 * (ph - pp)

        # ── CPR (with TC/BC normalization) ────────────────────────────────
        bc_raw = (ph + pl) / 2.0
        tc_raw = (pp - bc_raw) + pp
        tc = np.maximum(tc_raw, bc_raw)
        bc = np.minimum(tc_raw, bc_raw)
        width = (tc - bc).abs()

        # ── Camarilla (from prior session) ────────────────────────────────
        h3 = pc + prng * _CAM_MULT / 4
        h4 = pc + prng * _CAM_MULT / 2
        l3 = pc - prng * _CAM_MULT / 4
        l4 = pc - prng * _CAM_MULT / 2
        h5 = (ph / pl.replace(0, np.nan)) * pc
        l5 = pc - (h5 - pc)

        out: dict[str, pd.Series] = {}

        def _dist(level):
            return (c - level) / atr

        # ── Level distances ───────────────────────────────────────────────
        out["dist_pp_atr"] = _dist(pp)
        out["dist_r1_atr"] = _dist(r1)
        out["dist_s1_atr"] = _dist(s1)
        out["dist_tc_atr"] = _dist(tc)
        out["dist_bc_atr"] = _dist(bc)
        out["dist_h3_atr"] = _dist(h3)
        out["dist_h4_atr"] = _dist(h4)
        out["dist_h5_atr"] = _dist(h5)
        out["dist_l3_atr"] = _dist(l3)
        out["dist_l4_atr"] = _dist(l4)
        out["dist_l5_atr"] = _dist(l5)

        # ── CPR width regime ──────────────────────────────────────────────
        width_pctile = width.rolling(_WIDTH_LOOKBACK, min_periods=_WIDTH_MIN_PERIODS).rank(pct=True)
        out["cpr_width_atr"] = width / atr
        out["cpr_width_pctile"] = width_pctile
        out["cpr_width_chg_pct"] = width.pct_change()
        out["cpr_expanding"] = (width.diff() > 0).astype(float).where(valid)
        out["cpr_extreme_narrow"] = (width_pctile <= _EXTREME_PCTILE).astype(float).where(width_pctile.notna())
        narrow = width_pctile <= _NARROW_PCTILE
        out["cpr_regime_narrow"] = narrow.astype(float).where(width_pctile.notna())
        out["cpr_regime_wide"] = (width_pctile >= _WIDE_PCTILE).astype(float).where(width_pctile.notna())
        out["cpr_compression_streak"] = _run_length(narrow & width_pctile.notna()).astype(float).where(width_pctile.notna())

        # ── Virgin CPR (numba) ────────────────────────────────────────────
        band_lo = pd.Series(bc, index=idx).to_numpy(dtype=float)
        band_hi = pd.Series(tc, index=idx).to_numpy(dtype=float)
        v_today, v_age, v_count = _virgin_cpr_engine(
            h.to_numpy(dtype=float), l.to_numpy(dtype=float), band_lo, band_hi, _VIRGIN_LOOKBACK
        )
        out["virgin_cpr"] = pd.Series(v_today, index=idx)
        out["virgin_cpr_age"] = pd.Series(v_age, index=idx)
        out["untouched_cpr_count"] = pd.Series(v_count, index=idx)

        # ── Camarilla behavior (today's OHLC vs prior-day levels) ─────────
        out["cam_h3_reversal"] = ((h >= h3) & (c < h3)).astype(float).where(valid)
        out["cam_l3_reversal"] = ((l <= l3) & (c > l3)).astype(float).where(valid)
        out["cam_h4_breakout"] = (c > h4).astype(float).where(valid)
        out["cam_l4_breakout"] = (c < l4).astype(float).where(valid)
        out["cam_return_from_h4"] = ((h >= h4) & (c <= h4)).astype(float).where(valid)
        out["cam_return_from_l4"] = ((l <= l4) & (c >= l4)).astype(float).where(valid)

        # ── Opening relationships (today's open vs prior-day levels) ──────
        tc_s = pd.Series(tc, index=idx)
        bc_s = pd.Series(bc, index=idx)
        out["open_above_tc"] = (o > tc_s).astype(float).where(valid)
        out["open_below_bc"] = (o < bc_s).astype(float).where(valid)
        out["open_gap_above_r1"] = (o > r1).astype(float).where(valid)
        out["open_gap_below_s1"] = (o < s1).astype(float).where(valid)
        out["open_above_h4"] = (o > h4).astype(float).where(valid)
        out["open_above_h3"] = ((o > h3) & (o <= h4)).astype(float).where(valid)
        out["open_below_l3"] = ((o < l3) & (o >= l4)).astype(float).where(valid)
        out["open_below_l4"] = (o < l4).astype(float).where(valid)

        # ── Two-day CPR relationship (book overlap definition) ────────────
        tcp, bcp = tc_s.shift(1), bc_s.shift(1)
        both_valid = tc_s.notna() & tcp.notna()
        higher_clean = bc_s > tcp
        lower_clean = tc_s < bcp
        overlap_higher = (tc_s > tcp) & (bc_s > bcp) & (bc_s <= tcp)
        overlap_lower = (tc_s < tcp) & (bc_s < bcp) & (tc_s >= bcp)
        inside = (tc_s <= tcp) & (bc_s >= bcp)
        outside = (tc_s >= tcp) & (bc_s <= bcp)
        conds = [higher_clean, lower_clean, overlap_higher, overlap_lower, inside, outside]
        # int code 0..6 (6 = unchanged/default), then explicit dummies
        code = np.select(conds, [0, 1, 2, 3, 4, 5], default=6)
        code = pd.Series(code, index=idx).where(both_valid)
        out["twoday_higher_value"] = (code == 0).astype(float).where(both_valid)
        out["twoday_lower_value"] = (code == 1).astype(float).where(both_valid)
        out["twoday_overlap_higher"] = (code == 2).astype(float).where(both_valid)
        out["twoday_overlap_lower"] = (code == 3).astype(float).where(both_valid)
        out["twoday_inside_value"] = (code == 4).astype(float).where(both_valid)
        out["twoday_outside_value"] = (code == 5).astype(float).where(both_valid)
        out["twoday_unchanged"] = (code == 6).astype(float).where(both_valid)
        same_code = code.eq(code.shift()) & both_valid
        out["twoday_streak"] = _run_length(same_code).astype(float).where(both_valid)

        # Bias confirm/reject (book Ch.6): bullish states confirm if open >= BC,
        # bearish if open <= TC; breakout states confirm on a gap past prior range.
        bull_state = (code == 0) | (code == 2)
        bear_state = (code == 1) | (code == 3)
        breakout_state = (code == 6) | (code == 4)
        beyond_prior = (o > ph) | (o < pl)
        confirmed = (bull_state & (o >= bc_s)) | (bear_state & (o <= tc_s))
        rejected = (bull_state & (o < bc_s)) | (bear_state & (o > tc_s))
        out["bias_confirmed"] = confirmed.astype(float).where(both_valid)
        out["bias_rejected"] = rejected.astype(float).where(both_valid)
        out["bias_confirmed_breakout"] = (breakout_state & beyond_prior).astype(float).where(both_valid)

        # ── Pivot trend (fixed ordering: bull>TC, bear<BC, else neutral) ──
        trend_bull = c > tc_s
        trend_bear = c < bc_s
        out["cpr_trend_bull"] = trend_bull.astype(float).where(valid)
        out["cpr_trend_bear"] = trend_bear.astype(float).where(valid)
        # side code for streak: +1 bull / -1 bear / 0 neutral
        side = pd.Series(np.select([trend_bull, trend_bear], [1, -1], default=0), index=idx).where(valid)
        same_side = side.eq(side.shift()) & valid
        out["cpr_trend_streak"] = _run_length(same_side).astype(float).where(valid)
        for w in _PP_SLOPE_WINDOWS:
            out[f"pp_slope_{w}d_atr"] = (pp - pp.shift(w)) / atr

        # ── PP acceptance ─────────────────────────────────────────────────
        # Mask the warmup row (NaN pivot) to NaN before counting so it doesn't
        # get folded into a "below" run / lower the rolling counts.
        above_pp = (c > pp).where(valid)
        change = above_pp.ne(above_pp.shift())
        run = above_pp.groupby(change.cumsum()).cumcount() + 1
        out["pp_accept_streak"] = run.where(above_pp == 1, -run).astype(float).where(valid)
        for w in _ACCEPT_WINDOWS:
            out[f"pp_accept_count_{w}d"] = above_pp.astype(float).rolling(w, min_periods=w).sum()

        # ── Nearest level (over the full internal level stack) ────────────
        stack = pd.concat(
            [pp, r1, r2, r3, s1, s2, s3, bc_s, tc_s, h3, h4, h5, l3, l4, l5,
             pd.Series(pc + prng * _CAM_MULT / 12, index=idx),   # H1
             pd.Series(pc + prng * _CAM_MULT / 6, index=idx),    # H2
             pd.Series(pc - prng * _CAM_MULT / 12, index=idx),   # L1
             pd.Series(pc - prng * _CAM_MULT / 6, index=idx)],   # L2
            axis=1,
        )
        levels = stack.to_numpy(dtype=float)                 # (n, 19)
        cvals = c.to_numpy(dtype=float)[:, None]
        atrv = atr.to_numpy(dtype=float)
        diffs = levels - cvals                               # level - close: >0 above
        absd = np.abs(diffs)
        any_level = ~np.all(np.isnan(diffs), axis=1)
        with warnings.catch_warnings():
            # first rows have an all-NaN level stack (shift(1) warmup) → nanmin
            # legitimately returns NaN there; suppress the noisy all-NaN warning.
            warnings.simplefilter("ignore", RuntimeWarning)
            nearest = np.where(any_level, np.nanmin(absd, axis=1), np.nan) / atrv
            res = np.where(diffs > 0, diffs, np.nan)
            sup = np.where(diffs < 0, -diffs, np.nan)
            all_nan_res = np.all(np.isnan(res), axis=1)
            all_nan_sup = np.all(np.isnan(sup), axis=1)
            nearest_res = np.where(all_nan_res, np.nan, np.nanmin(np.where(np.isnan(res), np.inf, res), axis=1)) / atrv
            nearest_sup = np.where(all_nan_sup, np.nan, np.nanmin(np.where(np.isnan(sup), np.inf, sup), axis=1)) / atrv
        out["dist_nearest_level_atr"] = pd.Series(nearest, index=idx)
        out["dist_nearest_res_atr"] = pd.Series(nearest_res, index=idx)
        out["dist_nearest_sup_atr"] = pd.Series(nearest_sup, index=idx)
        out["above_all_levels"] = pd.Series(np.where(any_level, all_nan_res.astype(float), np.nan), index=idx)
        out["below_all_levels"] = pd.Series(np.where(any_level, all_nan_sup.astype(float), np.nan), index=idx)

        # ── Multi-timeframe pivots (weekly/monthly/yearly) ────────────────
        mtf = self._mtf_pivots(h, l, c, idx)
        for tf in ("weekly", "monthly", "yearly"):
            for lvl in ("pp", "r1", "s1"):
                out[f"dist_{tf}_{lvl}_atr"] = (c - mtf[f"{tf}_{lvl}"]) / atr

        # ── Assemble in frozen order, tag family, float32 ─────────────────
        frame = pd.DataFrame({name: out[name] for name in _ASSEMBLY_ORDER}, index=idx)
        frame = frame.add_prefix("pivot_")   # engine emits pivot_* raw names
        return frame.astype(np.float32)

    @staticmethod
    def _mtf_pivots(h: pd.Series, l: pd.Series, c: pd.Series, idx: pd.Index) -> dict[str, pd.Series]:
        """Weekly/monthly/yearly floor pivots (PP/R1/S1), each computed from the
        LAST completed period's H/L/C and held for the following period.

        Uses period-END resample labels + merge_asof(direction='backward',
        allow_exact_matches=False): every day picks the pivot of the most recent
        period that ended STRICTLY before today, so the current (incomplete) period
        is invisible and no shift is needed. This is the leak-free equivalent of
        the draft's shift-then-ffill, and reuses the pandas-version alias guard.
        """
        ohlc = pd.DataFrame({"high": h, "low": l, "close": c}, index=idx)
        specs = [("weekly", "W-FRI"), ("monthly", _MONTH_END), ("yearly", _YEAR_END)]
        daily = pd.DataFrame({"date": idx}).sort_values("date")
        result: dict[str, pd.Series] = {}
        for tf, rule in specs:
            per = ohlc.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna(subset=["close"])
            pp = (per["high"] + per["low"] + per["close"]) / 3.0
            r1 = 2 * pp - per["low"]
            s1 = 2 * pp - per["high"]
            lvl = pd.DataFrame({"date": per.index, f"{tf}_pp": pp.values,
                                f"{tf}_r1": r1.values, f"{tf}_s1": s1.values}).sort_values("date")
            merged = pd.merge_asof(daily, lvl, on="date", direction="backward",
                                   allow_exact_matches=False).set_index("date").reindex(idx)
            for lname in ("pp", "r1", "s1"):
                result[f"{tf}_{lname}"] = merged[f"{tf}_{lname}"]
        return result
