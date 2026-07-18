"""
FeatureEngineer — computes all features from §5.2 of the spec.

RULE: All rolling statistics computed inside df.groupby('ticker') — NEVER on full panel.
RULE: ATR is Wilder's EMA (α = 1/N), not simple rolling mean.
RULE: Removed: raw ICT price levels, handcrafted aggregate scores.
RULE: All distance features: (close − level) / ATR_14.
RULE: All return features: log-return.
RULE: Winsorize every feature at [1, 99] percentile per date.
"""
from __future__ import annotations

import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from pipeline.config.base import MarketConfig
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr, _wilder_adx, _wilder_di
from pipeline.features.multitf_merger import MultiTFMerger
from pipeline.features.zone_features import (
    compute_zone_features, _MONTH_END, _QUARTER_END, _YEAR_END,
    _ZONE_PROXIMITY_PCT,
)
from pipeline.features.structure_features import structure_feature_frame
from pipeline.features.pivots import (
    PivotFeatureEngine, pivot_features_enabled, PIVOT_WINSORIZE_EXCLUDE,
)
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

FEATURE_PREFIX = "features_"

# ── Multi-timeframe ICT constants ─────────────────────────────────────────────
# Period-end aliases come from zone_features.py, which picks the spelling the
# installed pandas accepts (M/Q/Y on <2.2, ME/QE/YE on >=2.2). Hardcoded "ME"
# silently zeroed the 1mo/3mo/1y ICT contributions on pandas <2.2 — the
# per-TF try/except logged it at DEBUG and moved on.
_ICT_HTF_RESAMPLE = {"1wk": "W-FRI", "1mo": _MONTH_END, "3mo": _QUARTER_END, "1y": _YEAR_END}

# Zone expiry in bars per timeframe — stale zones deactivate after this many bars.
# daily=63 (~3mo), weekly=26 (~6mo), monthly=12 (1yr), quarterly=8 (2yr), yearly=3 (3yr)
_ICT_ZONE_EXPIRY = {"1d": 63, "1wk": 26, "1mo": 12, "3mo": 8, "1y": 3}
# Displacement-gate ATR multiple per timeframe: an OB/FVG only registers when
# the move that created it spans >= this many ATRs (institutional displacement,
# not drift). Measured dose-response on 25 US tickers, 750d, daily TF
# (proximity gate on):
#   disp 0.0 -> OB 39% / FVG 40% of days active  (saturated - no discrimination)
#   disp 1.0 -> OB 14% / FVG 25%                 (selective)
#   disp 1.5 -> OB  3%                            (near-annihilation; 3.0 killed 100%)
# 1.0 thins creation rate without breaking structural fidelity; Breaker Blocks
# are unaffected (already gated by swing + sweep + engulfing structure).
_ICT_DISP_MULT   = {"1d": 1.0, "1wk": 1.0, "1mo": 1.0, "3mo": 1.0, "1y": 1.0}
_ICT_IMPL_MODE   = "legacy"  # switch to "institutional" for stricter OB/FVG defaults
_ICT_HTF_W        = {"1d": 1, "1wk": 2, "1mo": 3, "3mo": 4, "1y": 5}
_ICT_SIGNAL_MAX   = float(sum(_ICT_HTF_W.values()))   # 15.0
_ICT_PRIORITY_MAX = 4.0   # max ZonePriority value (BK = 4)

# ── Phase-4 feature families (Exp-401..404) ────────────────────────────────
# GK vol, skew/kurt, VWAP/CMF/OBV, residual momentum, choppiness, variance
# ratio, cross-sectional z-scores, A/D thrust. Gated by env PHASE4_FEATURES
# (read at call time, same pattern as PIVOT_FEATURES/TARGET_TWAP_WINDOW).
# Default OFF — the conservative baseline; set PHASE4_FEATURES=1
# to build the Phase-4 panel for a with/without A/B.
PHASE4_FEATURE_COLS = [
    "gk_vol_20d", "ret_skew_20d", "ret_kurt_20d",
    "vwap_dist_20d", "cmf_20d", "obv_osc_20d",
    "residual_mom_20d", "residual_mom_60d",
    "chop_idx_14d", "var_ratio_5d", "ad_thrust_10d",
]  # cross-sectional *_csz / *_sec_csz columns are additionally gated with these


def phase4_features_enabled() -> bool:
    return os.environ.get("PHASE4_FEATURES", "0").strip().lower() in {"1", "true", "on", "yes"}


# Construction scaffolding: unprefixed intermediates consumed while building
# their features_ twins — the same values would otherwise sit on the panel
# twice, riding through every copy, checkpoint and fold slice (~0.5 GB at
# production scale). atr_14 is deliberately NOT here (raw, unwinsorized —
# pinned by the ticker-isolation regression test); zone_type_* labels are
# kept for watchlist display. Dropped with `del` (in-place, per-column) —
# a DataFrame.drop on the assembled panel would copy the consolidated block.
_SCAFFOLD_COLS = [
    "atr_pct", "return_20d", "return_60d",
    "zone_active_1d", "zone_dist_atr_1d", "zone_strength_1d",
    "weekly_trend", "weekly_vol", "monthly_trend", "monthly_vol",
    "quarterly_trend", "quarterly_vol", "yearly_trend", "yearly_vol",
]


def _drop_scaffold(df: pd.DataFrame) -> None:
    for _c in _SCAFFOLD_COLS:
        if _c in df.columns:
            del df[_c]


def feature_build_workers() -> int:
    """Env FEATURE_BUILD_WORKERS: process count for the per-ticker feature
    section. Default 1 = the serial path, bit-identical to before. Set to the
    machine's core count (e.g. 4 on the Hetzner box) to parallelise the
    ~2h/1,500-ticker build; cross-sectional steps always stay single-process."""
    try:
        return max(1, int(os.environ.get("FEATURE_BUILD_WORKERS", "1")))
    except ValueError:
        return 1


def _parallel_batch_worker(cfg, benchmark_close, skip_ict, sub_panel):
    """Runs in a worker process: per-ticker features for one batch of tickers.
    Module-level so it pickles under both fork (Linux) and spawn (Windows).
    A fresh engine per process; numba JIT warms once per worker (~5s)."""
    fe = FeatureEngineer(cfg, benchmark_close, skip_ict=skip_ict)
    return fe.build(sub_panel, _per_ticker_only=True)

# Columns carried from each HTF ICT run back to the daily index
_ICT_CARRY_COLS = [
    "ict_bob_active",   "ict_bullrb_active",   "ict_bullfvg_active",
    "ict_sob_active",   "ict_bearrb_active",   "ict_bearfvg_active",
    "ict_bull_bos", "ict_bear_bos",
    "ict_bob_bos_conf", "ict_sob_bos_conf",
    "ict_bullfvg_bos_conf", "ict_bearfvg_bos_conf",
    "ict_bullfvg_sweep_conf", "ict_bearfvg_sweep_conf",
    "ict_bull_ob_fvg_confluence", "ict_bear_ob_fvg_confluence",
    "ict_bsl_swept",    "ict_ssl_swept",
    "ict_bull_zone_priority", "ict_bear_zone_priority",
    "ict_bob_entered_recent", "ict_sob_entered_recent",
    "ict_bob_rejection",      "ict_sob_rejection",
]


def _rolling_beta(ticker_log_rets: pd.Series, bm_log_rets: pd.Series, window: int = 60) -> pd.Series:
    """60-day rolling OLS beta — vectorised via pandas rolling cov/var."""
    cov = ticker_log_rets.rolling(window, min_periods=window // 2).cov(bm_log_rets)
    var = bm_log_rets.rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0, np.nan)).reindex(ticker_log_rets.index)


# Values a discrete feature may take: binary flags (0/1), signs (-1/0/+1),
# half-step composites, and small integer codes (ict zone priority 0..4).
_DISCRETE_VALUES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0])

# Bounded-by-construction features: ADX/DI live in ~[0,100], so there are no
# outliers to tame — and winsorizing the one-sided pair (adx_bull/adx_bear)
# actively corrupts it: on a date where <1% of stocks sit on one side, the
# lower-percentile clip lifts the structural zeros ("other side in control")
# to a positive value, breaking the mutual exclusivity the split exists for.
_WINSORIZE_EXCLUDE = {
    f"{FEATURE_PREFIX}adx_14",
    f"{FEATURE_PREFIX}plus_di",
    f"{FEATURE_PREFIX}minus_di",
    f"{FEATURE_PREFIX}adx_bull",
    f"{FEATURE_PREFIX}adx_bear",
}
# Pivot streaks/counts/ages/percentile are bounded-by-construction or discrete-
# integer ranges the auto-detector misses (0..60) — no per-date outliers to clip.
_WINSORIZE_EXCLUDE |= PIVOT_WINSORIZE_EXCLUDE


def _is_discrete_feature(values: np.ndarray) -> bool:
    """True for flag/sign/priority columns that must not be winsorized."""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return False
    uniq = np.unique(vals)
    return len(uniq) <= 6 and np.isin(uniq, _DISCRETE_VALUES).all()


def _winsorize_per_date(panel: pd.DataFrame, feature_cols: List[str], lo: float = 1.0, hi: float = 99.0) -> pd.DataFrame:
    """Winsorize continuous features at [lo, hi] percentile per date.

    Discrete columns (binary flags, signs, dummies, zone-priority codes) are
    skipped: clipping a 0/1 column at the per-date 99th percentile ERASES rare
    flags outright — e.g. 10 of 1500 stocks in a yearly SDZ puts the 99th
    percentile at 0, turning all ten 1s into 0s. That destroys the signal
    precisely on the dates it is rarest and most informative. A yes/no answer
    has no outliers to tame.

    Vectorized: one groupby(date) covering ALL continuous columns at once,
    using pandas' Cython quantile transform (string dispatch, no Python
    lambda), instead of the previous per-COLUMN groupby(date) + lambda +
    np.nanpercentile loop. That loop re-grouped the full panel by date once
    per feature (~250 times) — measured 586.6s on a 20-ticker/254-col panel;
    this version measured 2.9s on the identical panel, verified bit-identical
    output (incl. all-NaN-date-group and partial-NaN edge cases) before
    landing. groupby.transform("quantile", q) skips NaN by default, matching
    np.nanpercentile; an all-NaN date-group naturally yields NaN bounds, and
    clip() against NaN bounds leaves values unchanged — same behavior as the
    original's explicit `if x.notna().any() else np.nan` guard.
    """
    cont_cols = [
        c for c in feature_cols
        if c in panel.columns
        and c not in _WINSORIZE_EXCLUDE
        and not _is_discrete_feature(panel[c].values.astype(float))
    ]
    if not cont_cols:
        return panel
    grouped = panel.groupby(level="date")[cont_cols]
    lower_b = grouped.transform("quantile", lo / 100.0)
    upper_b = grouped.transform("quantile", hi / 100.0)
    panel[cont_cols] = panel[cont_cols].clip(lower=lower_b, upper=upper_b)
    return panel


# Low-cardinality string columns (sector: ~11 values; zone_type_*: SDZ/SSZ/DZ/SZ/"")
# stored as plain object dtype cost ~80 bytes/cell (Python string object + pointer)
# on a multi-million-row panel — at 8.1M rows that's ~3.9GB across these 6 columns
# alone. category dtype stores one small integer code per cell into a shared,
# deduplicated category array, cutting that to well under 100MB. Every downstream
# consumer either casts to str before comparing (this file's own zone-score
# blocks: `grp[tf_col].astype(str)...`) or just reads the value for CSV/JSON
# output (portfolio construction, evaluate_forward_performance) — both behave
# identically on category vs object dtype, so this is a pure memory win.
_CATEGORICAL_COLS = ["sector", "zone_type_1d", "zone_type_1wk", "zone_type_1mo",
                     "zone_type_3mo", "zone_type_1y"]


def _downcast_and_defragment(panel: pd.DataFrame) -> pd.DataFrame:
    """Defragment DataFrame and downcast float64 to float32 to save memory."""
    # A single copy defragments the DataFrame (resolving PerformanceWarnings)
    panel = panel.copy()
    
    # Downcast floats
    float_cols = panel.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        # Cast using a dictionary to avoid fragmentation during assignment
        cast_dict = {c: np.float32 for c in float_cols}
        panel = panel.astype(cast_dict)
        
    return panel


def _cast_categorical(panel: pd.DataFrame) -> pd.DataFrame:
    for col in _CATEGORICAL_COLS:
        if col in panel.columns and str(panel[col].dtype) != "category":
            panel[col] = panel[col].astype("category")
    return panel


class FeatureEngineer:
    """
    Computes all features from §5.2.  All per-ticker computations use groupby.

    Parameters
    ----------
    cfg               : MarketConfig
    benchmark_close   : daily benchmark close Series indexed by date
    """

    def __init__(self, cfg: MarketConfig, benchmark_close: pd.Series,
                 skip_ict: bool = False) -> None:
        self.cfg = cfg
        self.benchmark_close = benchmark_close
        self.skip_ict = skip_ict
        self._ict = ICTFeatureEngine()
        self._mtf = MultiTFMerger()
        self._pivot = PivotFeatureEngine()

    def _per_ticker_parallel(self, panel: pd.DataFrame, all_tickers: list,
                             n_workers: int) -> "list[pd.DataFrame]":
        """Fan the per-ticker section out over worker processes.

        Batches are small (~25 tickers) and pulled dynamically, so a slow
        3,500-bar veteran never leaves other cores idle at the end. The input
        slices are OHLCV-only (tiny); the heavy feature frames stream back
        incrementally. A failed batch raises in the parent — fail loud, no
        silently missing tickers.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        _BATCH = 25
        batches = [all_tickers[i:i + _BATCH]
                   for i in range(0, len(all_tickers), _BATCH)]
        log.info(f"Parallel feature build: {len(all_tickers)} tickers | "
                 f"{n_workers} workers | {len(batches)} batches of <= {_BATCH}")
        tick_level = panel.index.get_level_values("ticker")
        frames: "list[pd.DataFrame]" = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_parallel_batch_worker, self.cfg,
                              self.benchmark_close, self.skip_ict,
                              panel[tick_level.isin(set(b))])
                    for b in batches]
            for _done, fut in enumerate(as_completed(futs), start=1):
                frames.append(fut.result())
                if _done % 4 == 0 or _done == len(futs):
                    log.info(f"Parallel feature build: {_done}/{len(futs)} batches done")
        return frames

    def build(self, panel: pd.DataFrame, _per_ticker_only: bool = False) -> pd.DataFrame:
        """
        Compute all features.  Returns panel with features_* columns added.

        _per_ticker_only: internal, used by the parallel build. A worker
        process computes ONLY the per-ticker section and returns right after
        the concat; every cross-sectional step (sector RS, breadth, regime,
        z-scores, MTF merge, winsorize) runs once in the parent, so those
        always see the full cross-section — identical to a serial build.
        """
        import time
        cfg = self.cfg
        log.info("Engineering features...")
        # Sort index without copying entirely if possible
        if not panel.index.is_monotonic_increasing:
            panel = panel.sort_index()

        # ── Per-ticker computations ───────────────────────────────────────
        ticker_frames: list[pd.DataFrame] = []
        bm_log_rets = np.log(self.benchmark_close / self.benchmark_close.shift(1))
        all_tickers = panel.index.get_level_values("ticker").unique().tolist()
        n_total = len(all_tickers)
        t_build_start = time.time()

        # Parallel dispatch (env FEATURE_BUILD_WORKERS, default 1 = serial
        # path below, bit-identical to before). Each ticker is an atomic unit
        # of work: one worker owns it end-to-end across all timeframes, two
        # workers never touch the same ticker, and the sort_index() after the
        # concat makes assembly order irrelevant to the result.
        _pre_built: "list[pd.DataFrame] | None" = None
        if not _per_ticker_only and n_total > 1:
            _w = feature_build_workers()
            if _w > 1:
                _pre_built = self._per_ticker_parallel(panel, all_tickers, _w)

        _ticker_iter = [] if _pre_built is not None else panel.groupby(level="ticker")
        for _ti, (ticker, grp) in enumerate(_ticker_iter):
            grp = grp.droplevel("ticker").sort_index()
            h = grp["high"].values
            l = grp["low"].values
            c = grp["close"].values
            o = grp["open"].values
            v = grp["volume"].values

            # ── ATR (Wilder) ──────────────────────────────────────────────
            atr14 = _wilder_atr(h, l, c, 14)
            # Floor ATR at 5 bps of price — same guard as ict_features.py. On
            # stale/illiquid tickers ATR decays toward 0 and every /ATR feature
            # (price_vs_sma*, sma slopes, returns) explodes to thousands of
            # "ATRs", which also contaminates the per-date winsorize bounds for
            # healthy stocks. A sub-5bps daily range is noise, not signal.
            # Invalid ATR (NaN/<=0) stays NaN — the NaN-native model reads it
            # as "unknown", not as a meaningful value.
            atr_floor = np.abs(c) * 5e-4
            safe_atr = np.where(np.isnan(atr14) | (atr14 <= 0), np.nan,
                                np.where(atr14 > atr_floor, atr14, atr_floor))
            grp["atr_14"] = atr14

            # Percentage ATR = ATR / close — keeps return normalization dimensionless
            # and consistent across stocks regardless of absolute price level.
            # Use this ONLY for normalizing log returns (which are already dimensionless).
            # Price-vs-SMA and SMA-slope features stay on absolute-ATR units (both ₹).
            # safe_atr is floored at 5 bps of price, so pct_atr >= 5e-4 whenever
            # valid; invalid stays NaN (the old 1e-6 pseudo-floor turned unknown
            # ATR into million-fold return explosions).
            pct_atr = safe_atr / np.where(c > 0, c, np.nan)
            safe_pct_atr = pct_atr

            # ATR percentile rank in trailing 252d window
            # Use pandas rolling.rank(pct=True) — vectorised, no per-row Python call
            atr_pct_rank = pd.Series(atr14, index=grp.index).rolling(252, min_periods=126).rank(pct=True)
            grp[f"{FEATURE_PREFIX}atr_pct_rank_252"] = atr_pct_rank.values

            # ATR 60d max for vol_contraction
            atr_60max = pd.Series(atr14).rolling(60, min_periods=30).max().values
            vol_contraction = np.where(atr_60max > 0, atr14 / atr_60max, np.nan)
            grp[f"{FEATURE_PREFIX}vol_contraction"] = vol_contraction

            # compression_score = 1 - vol_contraction
            grp[f"{FEATURE_PREFIX}compression_score"] = 1.0 - vol_contraction

            # ── ADX ───────────────────────────────────────────────────────
            _adx14 = _wilder_adx(h, l, c, 14)
            grp[f"{FEATURE_PREFIX}adx_14"] = _adx14

            # ── Directional ADX (+DI / −DI) ───────────────────────────────
            # Raw ADX is direction-blind: abs(+DI − −DI) gives the SAME value
            # for a strong uptrend and a strong downtrend (NKE: ADX 36 in a
            # falling stock read identically to a 36-ADX breakout). Expose
            # +DI, −DI and their sign so the model (next retrain) and the
            # momentum-bull gate (now) can distinguish "strong up" from
            # "strong down".
            _plus_di, _minus_di = _wilder_di(h, l, c, 14)
            grp[f"{FEATURE_PREFIX}plus_di"]  = _plus_di
            grp[f"{FEATURE_PREFIX}minus_di"] = _minus_di
            # +1 = bulls in control (+DI>−DI), −1 = bears in control (−DI>+DI)
            grp[f"{FEATURE_PREFIX}adx_dir"]  = np.sign(_plus_di - _minus_di)
            # One-sided split — same dialect as sdz/ssz and regime_bull/bear:
            # trend strength owned by each side. NKE-type bars read
            # adx_bull=0 / adx_bear=36, needing no learned interaction.
            grp[f"{FEATURE_PREFIX}adx_bull"] = np.where(_plus_di > _minus_di, _adx14, 0.0)
            grp[f"{FEATURE_PREFIX}adx_bear"] = np.where(_minus_di > _plus_di, _adx14, 0.0)

            # ── Log returns (percentage-ATR-normalized) ───────────────────
            # Log returns are dimensionless; dividing by absolute ATR (₹) creates
            # a price-level dependency — a 10% move in ₹100 stock vs ₹10k stock
            # would get a 100× different feature value with the same % gain.
            # Dividing by percentage ATR (ATR/close) keeps the ratio consistent.
            log_close = np.log(np.where(c > 0, c, np.nan))
            for lag, name in [(1, "return_1d"), (5, "return_5d"), (20, "return_20d"), (60, "return_60d")]:
                shifted = np.roll(log_close, lag)
                shifted[:lag] = np.nan
                raw_ret = log_close - shifted
                grp[f"{FEATURE_PREFIX}{name}"] = raw_ret / safe_pct_atr

            # ── Historical realized volatility (20-day, annualized) ───────
            log_ret_s = pd.Series(log_close, index=grp.index).diff()
            grp[f"{FEATURE_PREFIX}hist_vol_20d"] = (
                log_ret_s.rolling(20, min_periods=10).std() * np.sqrt(252)
            ).values

            # ── Extreme Volatility & Distribution (Phase 4) ───────────────
            if phase4_features_enabled():
                # Garman-Klass Volatility (20-day, annualized)
                log_hl = np.log(np.where((h > 0) & (l > 0) & (h >= l), h / l, 1.0))
                log_co = np.log(np.where((c > 0) & (o > 0), c / o, 1.0))
                gk_daily = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
                # Clip negative values that can arise from float precision before sqrt
                gk_daily = np.clip(gk_daily, 0, None)
                grp[f"{FEATURE_PREFIX}gk_vol_20d"] = np.sqrt(
                    pd.Series(gk_daily).rolling(20, min_periods=10).mean() * 252
                ).values

                # Rolling Skew and Kurtosis (20d)
                grp[f"{FEATURE_PREFIX}ret_skew_20d"] = log_ret_s.rolling(20, min_periods=15).skew().values
                grp[f"{FEATURE_PREFIX}ret_kurt_20d"] = log_ret_s.rolling(20, min_periods=15).kurt().values

            # ── 52-week high distance ─────────────────────────────────────
            rolling_252_high = pd.Series(h).rolling(252, min_periods=126).max().values
            grp[f"{FEATURE_PREFIX}high_52w_dist"] = np.where(
                rolling_252_high > 0,
                (c - rolling_252_high) / rolling_252_high,
                np.nan,
            ).astype(np.float32)

            # ── 52-week low distance  (bear-side symmetric of high_52w_dist) ─
            # Always >= 0.  Near 0 = price hugging 52w low = breakdown risk.
            # High value = price far above 52w low = not at support breakdown.
            rolling_252_low = pd.Series(l).rolling(252, min_periods=126).min().values
            grp[f"{FEATURE_PREFIX}low_52w_dist"] = np.where(
                rolling_252_low > 0,
                (c - rolling_252_low) / rolling_252_low,
                np.nan,
            ).astype(np.float32)

            # ── Breakout flags (N-day range breakout) ─────────────────────
            # Captures structural breakouts at any price level — not just near
            # 52w high. Combined with vol_ratio_5d and atr_expansion, gives the
            # model vocabulary for early institutional breakout detection.
            hs = pd.Series(h)
            rolling_20d_high = hs.rolling(20, min_periods=10).max().shift(1).values
            rolling_50d_high = hs.rolling(50, min_periods=25).max().shift(1).values
            grp[f"{FEATURE_PREFIX}20d_breakout"] = np.where(
                ~np.isnan(rolling_20d_high),
                (c > rolling_20d_high).astype(np.float32),
                np.nan,
            )
            grp[f"{FEATURE_PREFIX}50d_breakout"] = np.where(
                ~np.isnan(rolling_50d_high),
                (c > rolling_50d_high).astype(np.float32),
                np.nan,
            )

            # ── ATR expansion — momentum ignition signal ───────────────────
            # atr_expansion > 1 = volatility expanding vs recent baseline.
            # Filters weak/drifting breakouts from genuine momentum moves.
            atr_ma20 = pd.Series(atr14).rolling(20, min_periods=10).mean().values
            grp[f"{FEATURE_PREFIX}atr_expansion"] = np.where(
                atr_ma20 > 0,
                (atr14 / atr_ma20).astype(np.float32),
                np.nan,
            )

            # ── SMAs ──────────────────────────────────────────────────────
            cs = pd.Series(c)
            sma20 = cs.rolling(20, min_periods=10).mean().values
            sma50 = cs.rolling(50, min_periods=25).mean().values
            sma200 = cs.rolling(200, min_periods=100).mean().values

            # SMA slopes (ATR-per-bar normalized)
            def _sma_slope(sma_arr: np.ndarray, lag: int) -> np.ndarray:
                shifted = np.roll(sma_arr, lag)
                shifted[:lag] = np.nan
                return (sma_arr - shifted) / (safe_atr * lag)

            grp[f"{FEATURE_PREFIX}sma20_slope_5"] = _sma_slope(sma20, 5)
            grp[f"{FEATURE_PREFIX}sma50_slope_5"] = _sma_slope(sma50, 5)
            grp[f"{FEATURE_PREFIX}sma200_slope_10"] = _sma_slope(sma200, 10)

            # Price vs SMA (ATR-normalized)
            grp[f"{FEATURE_PREFIX}price_vs_sma20"] = (c - sma20) / safe_atr
            grp[f"{FEATURE_PREFIX}price_vs_sma50"] = (c - sma50) / safe_atr
            grp[f"{FEATURE_PREFIX}price_vs_sma200"] = (c - sma200) / safe_atr

            # ── Volume ratios ──────────────────────────────────────────────
            vs = pd.Series(v, dtype=float)
            vol_ma20 = vs.rolling(20, min_periods=10).mean().values
            vol_ma5 = vs.rolling(5, min_periods=3).mean().values
            vol_ma60 = vs.rolling(60, min_periods=30).mean().values
            grp[f"{FEATURE_PREFIX}vol_ratio_5d"] = np.where(vol_ma20 > 0, v / vol_ma20, np.nan)
            grp[f"{FEATURE_PREFIX}vol_ratio_20d"] = np.where(vol_ma60 > 0, vol_ma5 / vol_ma60, np.nan)

            # ── Volume Microstructure (Phase 4) ──────────────────────────────────────
            if phase4_features_enabled():
                # 1. VWAP Distance (ATR-normalized)
                tp = (h + l + c) / 3.0
                tp_v = pd.Series(tp * v)
                vwap_20d = tp_v.rolling(20, min_periods=10).sum() / vs.rolling(20, min_periods=10).sum()
                grp[f"{FEATURE_PREFIX}vwap_dist_20d"] = (c - vwap_20d.values) / safe_atr

                # 2. Chaikin Money Flow (CMF 20d)
                range_hl = h - l
                range_hl_safe = np.where(range_hl == 0, 1e-8, range_hl)
                mf_mult = ((c - l) - (h - c)) / range_hl_safe
                mf_vol = pd.Series(mf_mult * v)
                cmf_20d = mf_vol.rolling(20, min_periods=10).sum() / vs.rolling(20, min_periods=10).sum()
                grp[f"{FEATURE_PREFIX}cmf_20d"] = cmf_20d.values

                # 3. Normalized OBV Oscillator
                v_sign = np.sign(np.append([0], np.diff(c)))
                obv_s = pd.Series(np.cumsum(v_sign * v))
                obv_sma20 = obv_s.rolling(20, min_periods=10).mean()
                v_sma20_s = pd.Series(vol_ma20)
                obv_osc = (obv_s - obv_sma20) / np.where(v_sma20_s > 0, v_sma20_s * 20, np.nan)
                grp[f"{FEATURE_PREFIX}obv_osc_20d"] = obv_osc.values

            # ── Rolling beta vs benchmark ──────────────────────────────────
            ticker_log_rets = pd.Series(np.log(np.where(c > 0, c, np.nan))).diff()
            ticker_log_rets.index = grp.index
            bm_aligned = bm_log_rets.reindex(grp.index, method="ffill")
            beta = _rolling_beta(ticker_log_rets, bm_aligned, 60)
            # Winsorize beta at [-2, 4]
            beta = beta.clip(-2, 4)
            grp[f"{FEATURE_PREFIX}rolling_beta_60d"] = beta.values

            # ── Residual Momentum (Phase 4) ──────────────────────────────────
            if phase4_features_enabled():
                residual_rets = ticker_log_rets - beta * bm_aligned
                grp[f"{FEATURE_PREFIX}residual_mom_20d"] = residual_rets.rolling(20, min_periods=10).sum().values
                grp[f"{FEATURE_PREFIX}residual_mom_60d"] = residual_rets.rolling(60, min_periods=30).sum().values

                # ── Trend Persistence & Fractal Dimension (Exp-404) ────────────
                n_chop = 14
                atr_sum_14 = pd.Series(atr14).rolling(n_chop, min_periods=n_chop//2).sum()
                high_max_14 = pd.Series(h).rolling(n_chop, min_periods=n_chop//2).max()
                low_min_14 = pd.Series(l).rolling(n_chop, min_periods=n_chop//2).min()
                chop_range = high_max_14 - low_min_14
                chop_range_safe = np.where(chop_range == 0, 1e-8, chop_range)
                chop_idx = 100.0 * np.log10(atr_sum_14 / chop_range_safe) / np.log10(n_chop)
                grp[f"{FEATURE_PREFIX}chop_idx_14d"] = chop_idx.values

                # Variance Ratio (Hurst proxy): Var(5d_ret) / (5 * Var(1d_ret))
                var_1d = log_ret_s.rolling(20, min_periods=10).var()
                var_5d = pd.Series(log_close, index=grp.index).diff(5).rolling(20, min_periods=10).var()
                vr_5d = var_5d / (5 * var_1d.replace(0, np.nan))
                grp[f"{FEATURE_PREFIX}var_ratio_5d"] = vr_5d.values

            # ── ICT features (skipped when skip_ict=True e.g. feature_set=zone) ──
            if not self.skip_ict:
                grp = self._ict.compute(grp,
                                        implementation_mode=_ICT_IMPL_MODE,
                                        disp_mult=_ICT_DISP_MULT["1d"],
                                        proximity_pct=_ZONE_PROXIMITY_PCT["1d"])

                _n_bob     = int(grp.get("ict_bob_active",     pd.Series(0)).sum())
                _n_sob     = int(grp.get("ict_sob_active",     pd.Series(0)).sum())
                _n_bullbb  = int(grp.get("ict_bullrb_active",  pd.Series(0)).sum())
                _n_bearbb  = int(grp.get("ict_bearrb_active",  pd.Series(0)).sum())
                _n_bullfvg = int(grp.get("ict_bullfvg_active", pd.Series(0)).sum())
                _n_bearfvg = int(grp.get("ict_bearfvg_active", pd.Series(0)).sum())
                _n_bsl     = int(grp.get("ict_bsl_swept",      pd.Series(0)).sum())
                _n_ssl     = int(grp.get("ict_ssl_swept",      pd.Series(0)).sum())
                if (_ti + 1) % 100 == 0 or _n_bob + _n_bullbb + _n_bullfvg == 0:
                    log.info(
                        f"{ticker} ICT | bars={len(grp)} | "
                        f"BullOB={_n_bob} BearOB={_n_sob} | "
                        f"BullBB={_n_bullbb} BearBB={_n_bearbb} | "
                        f"BullFVG={_n_bullfvg} BearFVG={_n_bearfvg} | "
                        f"BSL_swept={_n_bsl} SSL_swept={_n_ssl}"
                    )
                if _n_bob == 0 and _n_bullbb == 0 and _n_bullfvg == 0:
                    log.warning(f"{ticker}: NO bull ICT signals detected in {len(grp)} bars — check OHLCV quality")

                for ict_col in [c for c in grp.columns
                                if c.startswith("ict_") and not c.startswith(FEATURE_PREFIX)]:
                    grp[f"{FEATURE_PREFIX}{ict_col}"] = grp.pop(ict_col)

                try:
                    ohlcv_d   = grp[["open", "high", "low", "close", "volume"]].copy()
                    daily_idx = ohlcv_d.index

                    ict_bull_htf = np.zeros(len(grp), dtype=np.float32)
                    ict_bear_htf = np.zeros(len(grp), dtype=np.float32)

                    for _prio_col, _acc in [
                        (f"{FEATURE_PREFIX}ict_bull_zone_priority", ict_bull_htf),
                        (f"{FEATURE_PREFIX}ict_bear_zone_priority", ict_bear_htf),
                    ]:
                        if _prio_col in grp.columns:
                            _acc += (_ICT_HTF_W["1d"] *
                                     grp[_prio_col].fillna(0).values.astype(np.float32)
                                     / _ICT_PRIORITY_MAX)

                    for tf_label, rule in _ICT_HTF_RESAMPLE.items():
                        try:
                            htf = ohlcv_d.resample(rule).agg({
                                "open": "first", "high": "max",
                                "low": "min",   "close": "last", "volume": "sum",
                            }).dropna(subset=["close"])

                            if len(htf) < 5:
                                continue

                            htf["atr_14"] = _wilder_atr(
                                htf["high"].values.astype(float),
                                htf["low"].values.astype(float),
                                htf["close"].values.astype(float), 14,
                            )

                            htf_ict = self._ict.compute(
                                htf,
                                implementation_mode=_ICT_IMPL_MODE,
                                zone_expiry_bars=_ICT_ZONE_EXPIRY[tf_label],
                                disp_mult=_ICT_DISP_MULT[tf_label],
                                proximity_pct=_ZONE_PROXIMITY_PCT[tf_label],
                            )

                            htf_reset = htf_ict.reset_index()
                            date_col  = htf_reset.columns[0]
                            htf_reset = htf_reset.rename(columns={date_col: "date"}).sort_values("date")
                            carry     = [c for c in _ICT_CARRY_COLS if c in htf_reset.columns]

                            daily_r = pd.DataFrame({"date": daily_idx}).sort_values("date")
                            merged  = pd.merge_asof(
                                daily_r,
                                htf_reset[["date"] + carry],
                                on="date", direction="backward",
                            ).set_index("date")

                            w = _ICT_HTF_W[tf_label]
                            _htf_new = {}
                            for col in carry:
                                vals = merged[col].reindex(daily_idx).fillna(0).values.astype(np.float32)
                                _htf_new[f"{FEATURE_PREFIX}{col}_{tf_label}"] = vals

                                if col == "ict_bull_zone_priority":
                                    ict_bull_htf += w * vals / _ICT_PRIORITY_MAX
                                elif col == "ict_bear_zone_priority":
                                    ict_bear_htf += w * vals / _ICT_PRIORITY_MAX
                            # Single concat per timeframe instead of ~9 one-column
                            # inserts — the insert pattern fragments grp and fired
                            # PerformanceWarnings across the whole build.
                            grp = pd.concat(
                                [grp, pd.DataFrame(_htf_new, index=grp.index)], axis=1
                            )

                        except Exception as _e:
                            log.debug(f"{ticker} ICT HTF {tf_label}: {_e}")

                    grp[f"{FEATURE_PREFIX}ict_bull_htf_score"] = (
                        ict_bull_htf / _ICT_SIGNAL_MAX
                    ).clip(0, 1).astype(np.float32)
                    grp[f"{FEATURE_PREFIX}ict_bear_htf_score"] = (
                        ict_bear_htf / _ICT_SIGNAL_MAX
                    ).clip(0, 1).astype(np.float32)

                except Exception as e:
                    log.warning(f"{ticker}: MTF ICT failed ({e}) — using zeros")
                    grp[f"{FEATURE_PREFIX}ict_bull_htf_score"] = np.float32(0.0)
                    grp[f"{FEATURE_PREFIX}ict_bear_htf_score"] = np.float32(0.0)

            # ── HTF Zone features (via ZoneAnalyzer — full RBR/DBD/SDZ/SSZ pipeline) ──
            # Zones are computed from OHLCV using the same market-vision logic.
            # No Drv CSV files required. Produces zone_type_1d/1wk/1mo/3mo/1y columns.
            # Weights: 1d=1, 1wk=2, 1mo=3, 3mo=4, 1y=5  |  SDZ/SSZ get 2× base weight
            _HTF_W = {
                "zone_type_1d":  1,
                "zone_type_1wk": 2,
                "zone_type_1mo": 3,
                "zone_type_3mo": 4,
                "zone_type_1y":  5,
            }
            _MAX_SCORE = sum(_HTF_W.values()) * 2  # = 30

            try:
                ohlcv = grp[["open", "high", "low", "close", "volume"]].copy()
                zone_df = compute_zone_features(ohlcv)
                for zcol in ["zone_type_1d", "zone_active_1d", "zone_dist_atr_1d",
                             "zone_strength_1d", "zone_type_1wk", "zone_type_1mo",
                             "zone_type_3mo", "zone_type_1y"]:
                    grp[zcol] = zone_df[zcol].values if zcol in zone_df.columns else (
                        "" if "type" in zcol else 0.0
                    )
                # Zone summary rolled into ICT debug line above — no per-TF log here
                # (zone_features is called during every HPO fold recompute; logging there
                #  generates millions of lines and slows HPO significantly)
            except Exception as e:
                log.warning(f"{ticker}: zone computation failed ({e}) — using zeros")
                for zcol in ["zone_type_1d", "zone_type_1wk", "zone_type_1mo",
                             "zone_type_3mo", "zone_type_1y"]:
                    grp[zcol] = ""
                for zcol in ["zone_active_1d", "zone_dist_atr_1d", "zone_strength_1d"]:
                    grp[zcol] = 0.0

            # Daily zone proximity features
            grp[f"{FEATURE_PREFIX}zone_active"]   = grp["zone_active_1d"].astype(np.float32)
            grp[f"{FEATURE_PREFIX}zone_dist_atr"] = grp["zone_dist_atr_1d"].astype(np.float32)
            grp[f"{FEATURE_PREFIX}zone_strength"] = grp["zone_strength_1d"].astype(np.float32)

            # ── BOS/CHoCH market-structure features (toggle) ─────────────────
            # Initial build (no cutoff). Per-fold causal recompute happens in
            # recompute_fold_features. OFF by default → zero baseline impact.
            if getattr(self.cfg, "use_structure_features", False):
                try:
                    sf = structure_feature_frame(
                        grp[["open", "high", "low", "close", "volume"]],
                        cutoff_date=None,
                        prefix=FEATURE_PREFIX,
                        major_swing_length=getattr(self.cfg, "structure_major_swing", 25),
                        minor_swing_length=getattr(self.cfg, "structure_minor_swing", 5),
                    )
                    for scol in sf.columns:
                        grp[scol] = sf[scol].values
                except Exception as e:
                    log.warning(f"{ticker}: structure features failed ({e}) — skipped")

            # HTF composite scores
            sdz_score = np.zeros(len(grp), dtype=np.float32)
            ssz_score = np.zeros(len(grp), dtype=np.float32)
            dz_score  = np.zeros(len(grp), dtype=np.float32)
            sz_score  = np.zeros(len(grp), dtype=np.float32)

            for tf_col, weight in _HTF_W.items():
                if tf_col not in grp.columns:
                    continue
                zt = grp[tf_col].astype(str).str.strip().str.upper()
                sdz_score += weight * 2 * (zt == "SDZ").astype(np.float32).values
                ssz_score += weight * 2 * (zt == "SSZ").astype(np.float32).values
                dz_score  += weight * 1 * (zt == "DZ").astype(np.float32).values
                sz_score  += weight * 1 * (zt == "SZ").astype(np.float32).values
                tf_short = tf_col.replace("zone_type_", "")
                grp[f"{FEATURE_PREFIX}sdz_{tf_short}"] = (zt == "SDZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}ssz_{tf_short}"] = (zt == "SSZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}dz_{tf_short}"]  = (zt == "DZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}sz_{tf_short}"]  = (zt == "SZ").astype(np.float32)

            grp[f"{FEATURE_PREFIX}sdz_raw_score"]  = (sdz_score / _MAX_SCORE).clip(0, 1)
            grp[f"{FEATURE_PREFIX}ssz_raw_score"]  = (ssz_score / _MAX_SCORE).clip(0, 1)
            grp[f"{FEATURE_PREFIX}dz_raw_score"]   = (dz_score  / _MAX_SCORE).clip(0, 1)
            grp[f"{FEATURE_PREFIX}sz_raw_score"]   = (sz_score  / _MAX_SCORE).clip(0, 1)
            grp[f"{FEATURE_PREFIX}any_valid_sdz"]  = (sdz_score > 0).astype(np.float32)
            grp[f"{FEATURE_PREFIX}any_valid_ssz"]  = (ssz_score > 0).astype(np.float32)
            grp[f"{FEATURE_PREFIX}any_valid_zone"] = (
                (sdz_score + ssz_score + dz_score + sz_score) > 0
            ).astype(np.float32)

            # ── Pivot features (floor / CPR / Camarilla) — toggle ────────────
            # OFF by default (env PIVOT_FEATURES unset) → zero baseline impact.
            # Truncation-invariant (pure trailing functions of OHLC), so no
            # per-fold recompute is needed — see recompute_fold_features().
            if pivot_features_enabled():
                try:
                    pv = self._pivot.compute(
                        grp[["open", "high", "low", "close"]], safe_atr=safe_atr
                    )
                    # Add all 69 at once (concat, not per-column insert) to avoid
                    # DataFrame fragmentation; rename raw pivot_* → features_pivot_*.
                    pv = pv.add_prefix(FEATURE_PREFIX)
                    grp = pd.concat([grp, pv], axis=1)
                except Exception as e:
                    log.warning(f"{ticker}: pivot features failed ({e}) — skipped")

            # Re-attach ticker level
            grp.index = pd.MultiIndex.from_arrays(
                [grp.index, [ticker] * len(grp)], names=["date", "ticker"]
            )
            # Scaffolding is consumed by the ICT/zone blocks above — drop it
            # per-ticker so the concat never assembles it panel-wide.
            _drop_scaffold(grp)
            # Downcast early to save memory before concat
            grp = _downcast_and_defragment(grp)
            ticker_frames.append(grp)

            # Progress every 50 tickers
            if (_ti + 1) % 50 == 0 or (_ti + 1) == n_total:
                elapsed = time.time() - t_build_start
                rate = (_ti + 1) / elapsed
                eta = (n_total - _ti - 1) / rate if rate > 0 else 0
                log.info(
                    f"Feature engineering: {_ti+1}/{n_total} tickers done "
                    f"| elapsed={elapsed:.0f}s | ETA={eta:.0f}s"
                )

        if _pre_built is not None:
            ticker_frames = _pre_built

        panel = pd.concat(ticker_frames).sort_index()
        del ticker_frames
        import gc
        gc.collect()

        if _per_ticker_only:
            # Worker mode: hand the per-ticker frame back to the parent, which
            # runs the cross-sectional steps once over the full universe.
            return panel

        # ── Sector relative strength (cross-sectional, not per-ticker) ────
        panel = self._add_sector_rs(panel)

        # ── Market breadth (benchmark constituent SMA50 pct above) ────────
        panel = self._add_market_breadth(panel)

        # ── Regime label ──────────────────────────────────────────────────
        panel = self._add_regime(panel)

        # ── Cross-Sectional Z-Scores (Exp-403) ────────────────────────────
        if phase4_features_enabled():
            panel = self._add_cross_sectional_zscores(panel)

        # ── Multi-timeframe trends ────────────────────────────────────────
        panel = self._mtf.merge(panel)
        # The merge leaves raw (unprefixed) trend/vol columns beside their
        # features_ twins — delete them while they are freshly-added separate
        # blocks (in-place, no panel copy).
        _drop_scaffold(panel)

        # ── Zone × Trend confirmation ────────────────────────────────────
        # SDZ (swap demand) = bullish → confirmed when weekly+monthly trend UP
        # SSZ (swap supply) = bearish → confirmed when weekly+monthly trend DOWN
        sdz = panel.get(f"{FEATURE_PREFIX}sdz_raw_score", pd.Series(0.0, index=panel.index))
        ssz = panel.get(f"{FEATURE_PREFIX}ssz_raw_score", pd.Series(0.0, index=panel.index))
        wt  = panel.get("weekly_trend",   pd.Series(0.0, index=panel.index)).fillna(0)
        mt  = panel.get("monthly_trend",  pd.Series(0.0, index=panel.index)).fillna(0)
        qt  = panel.get("quarterly_trend",pd.Series(0.0, index=panel.index)).fillna(0)
        yt  = panel.get("yearly_trend",   pd.Series(0.0, index=panel.index)).fillna(0)

        # Expose trend as model features (features_ prefix = visible to FeatureSelector)
        # Yearly carries most predictive weight; daily/weekly least
        panel[f"{FEATURE_PREFIX}weekly_trend"]    = wt.astype(np.float32)
        panel[f"{FEATURE_PREFIX}monthly_trend"]   = mt.astype(np.float32)
        panel[f"{FEATURE_PREFIX}quarterly_trend"] = qt.astype(np.float32)
        panel[f"{FEATURE_PREFIX}yearly_trend"]    = yt.astype(np.float32)

        # Trend alignment multiplier: 0.5 (none) → 1.0 (1 TF) → 2.0 (all 4 TFs)
        up_mult = 0.5 + 0.375*wt + 0.375*mt + 0.375*qt + 0.375*yt   # max = 2.0
        dn_mult = 0.5 + 0.375*(1-wt) + 0.375*(1-mt) + 0.375*(1-qt) + 0.375*(1-yt)

        panel[f"{FEATURE_PREFIX}sdz_htf_score"] = (sdz * up_mult).astype(np.float32)
        panel[f"{FEATURE_PREFIX}ssz_htf_score"] = (ssz * dn_mult).astype(np.float32)

        # Net confluence [-1, +1]: positive = bullish zone bias, negative = bearish
        panel[f"{FEATURE_PREFIX}zone_htf_confluence"] = (
            panel[f"{FEATURE_PREFIX}sdz_htf_score"] -
            panel[f"{FEATURE_PREFIX}ssz_htf_score"]
        ).astype(np.float32)

        # ── ICT × trend multiplier (mirrors zone treatment above) ────────────
        ict_bull = panel.get(f"{FEATURE_PREFIX}ict_bull_htf_score",
                             pd.Series(0.0, index=panel.index))
        ict_bear = panel.get(f"{FEATURE_PREFIX}ict_bear_htf_score",
                             pd.Series(0.0, index=panel.index))
        panel[f"{FEATURE_PREFIX}ict_bull_htf_score"] = (
            ict_bull * up_mult).clip(0, 1).astype(np.float32)
        panel[f"{FEATURE_PREFIX}ict_bear_htf_score"] = (
            ict_bear * dn_mult).clip(0, 1).astype(np.float32)
        panel[f"{FEATURE_PREFIX}ict_htf_confluence"] = (
            panel[f"{FEATURE_PREFIX}ict_bull_htf_score"]
            - panel[f"{FEATURE_PREFIX}ict_bear_htf_score"]
        ).astype(np.float32)

        # sdz_premium_setup / ssz_premium_setup intentionally removed.
        # The per-TF zone features (sdz_1y, sdz_3mo, sdz_1mo, sdz_1wk) and the
        # trend features (yearly_trend, quarterly_trend, monthly_trend, weekly_trend)
        # are already individual model features. LGBM learns the interaction weights
        # and TF hierarchy from data — we should not hand-code them.

        # ── Winsorize all features at [1, 99] per date ────────────────────
        feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
        panel = _winsorize_per_date(panel, feat_cols)

        log.info(f"Features computed: {len(feat_cols)} feature columns")
        panel = _cast_categorical(panel)
        # Per-ticker frames were downcast + consolidated BEFORE the concat and
        # the concat output is itself consolidated, so a full-panel defrag here
        # was a redundant ~2x-panel transient (two ~6 GB copies coexisting at
        # production scale). Only the few panel-level columns added after the
        # concat (breadth, regime, sector-RS, z-scores) can still be float64 —
        # downcast those column-wise: each astype copies one column, never the
        # whole panel.
        for _c in panel.select_dtypes(include=["float64"]).columns:
            panel[_c] = panel[_c].astype(np.float32)
        return panel

    def _add_sector_rs(self, panel: pd.DataFrame) -> pd.DataFrame:
        """sector_rs_20d = ticker 20d return / sector median 20d return - 1."""
        # Use groupby.transform on the ticker level — preserves MultiIndex safely
        # (avoids groupby.apply which can drop/corrupt the date level in pandas 2.x)
        ret20 = panel.groupby(level="ticker")["close"].transform(
            lambda x: np.log(x / x.shift(20))
        )
        panel["_ret20"] = ret20

        # Group by (date, sector) for cross-sectional median.
        # Expose the date index level as a temporary column to avoid pandas 2.x
        # ambiguity between index-level names and column names in groupby.
        panel["_date_tmp"] = panel.index.get_level_values("date")
        sector_med = panel.groupby(["_date_tmp", "sector"])["_ret20"].transform("median")
        panel.drop(columns=["_date_tmp"], inplace=True)

        # Simple excess log-return over sector median — symmetric and economically correct.
        # Previous formula (ret/abs_med - sign(med)) was asymmetric: a 0% return in a
        # -5% sector gave +1.0 (100% "outperformance") which is wrong.
        panel[f"{FEATURE_PREFIX}sector_rs_20d"] = panel["_ret20"] - sector_med
        panel.drop(columns=["_ret20"], inplace=True)
        return panel

    def _add_market_breadth(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        market_breadth = % of in_universe tickers trading above their 50d SMA.
        Computed per date across the full cross-section.
        """
        sma50 = panel.groupby(level="ticker", group_keys=False)["close"].transform(
            lambda x: x.rolling(50, min_periods=25).mean()
        )
        above_sma50 = (panel["close"] > sma50) & (panel["in_universe"] == True)

        breadth = above_sma50.groupby(level="date").mean()
        breadth.name = "mb"
        panel[f"{FEATURE_PREFIX}market_breadth"] = (
            panel.index.get_level_values("date").map(breadth).values
        )

        # ── Advance/Decline Thrust (Exp-404) ──────────────────────────────
        if phase4_features_enabled():
            # Sign of the 1d move: reuse the per-ticker return_1d feature
            # (log return / pct-ATR — the strictly-positive denominator
            # preserves sign) instead of a second full-panel groupby pass.
            ret1 = panel[f"{FEATURE_PREFIX}return_1d"]
            advancing = (ret1 > 0) & (panel["in_universe"] == True)
            declining = (ret1 < 0) & (panel["in_universe"] == True)

            adv_count = advancing.groupby(level="date").sum()
            dec_count = declining.groupby(level="date").sum()
            total_ad  = adv_count + dec_count

            # 0/0 → NaN: a date with no known movers is unknown, not "balanced".
            ad_ratio_s = adv_count / total_ad
            ad_thrust_10d = ad_ratio_s.ewm(span=10, adjust=False).mean()
            panel[f"{FEATURE_PREFIX}ad_thrust_10d"] = (
                panel.index.get_level_values("date").map(ad_thrust_10d).values
            )

        return panel

    def _add_regime(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Market regime from cfg.benchmark_ticker:
          0 = bear  (bm close < bm 200d SMA AND bm 20d < bm 50d SMA)
          1 = choppy (neither bull nor bear)
          2 = bull  (bm close > bm 200d SMA AND bm 20d SMA > bm 50d SMA)
        Represented as 3 binary dummy columns: regime_bull, regime_choppy, regime_bear.
        """
        bm = self.benchmark_close.copy().sort_index()
        bm_sma20 = bm.rolling(20, min_periods=10).mean()
        bm_sma50 = bm.rolling(50, min_periods=25).mean()
        bm_sma200 = bm.rolling(200, min_periods=100).mean()

        bull = (bm > bm_sma200) & (bm_sma20 > bm_sma50)
        bear = (bm < bm_sma200) & (bm_sma20 < bm_sma50)
        regime = pd.Series(1, index=bm.index, dtype=int)  # 1=choppy
        regime[bull] = 2
        regime[bear] = 0

        dates = panel.index.get_level_values("date")
        regime_aligned = dates.map(regime).fillna(1)
        regime_arr = np.array(regime_aligned, dtype=int)
        panel[f"{FEATURE_PREFIX}regime_bull"] = (regime_arr == 2).astype(np.float32)
        panel[f"{FEATURE_PREFIX}regime_choppy"] = (regime_arr == 1).astype(np.float32)
        panel[f"{FEATURE_PREFIX}regime_bear"] = (regime_arr == 0).astype(np.float32)
        return panel

    def _add_cross_sectional_zscores(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-Sectional Z-Scores (Exp-403)
        Standardizes selected metrics cross-sectionally per date to isolate pure
        relative edges and strip out shifting market baseline volatility.

        Where the cross-sectional std is undefined — 0, or fewer than 2 members
        in the (date) / (date, sector) group — the z-score is NaN: unknown, not
        "exactly average" (NaN-native convention; 0.0 would fabricate a neutral
        observation, systematically for singleton sectors).
        """
        log.info("Computing cross-sectional Z-scores...")
        cols_to_zscore = [
            f"{FEATURE_PREFIX}return_20d",
            f"{FEATURE_PREFIX}return_60d",
            f"{FEATURE_PREFIX}vol_ratio_20d",
            f"{FEATURE_PREFIX}residual_mom_20d",
            f"{FEATURE_PREFIX}residual_mom_60d"
        ]

        panel["_date_tmp"] = panel.index.get_level_values("date")
        specs = [(["_date_tmp"], "_csz")]                      # market-neutral
        if "sector" in panel.columns:
            specs.append((["_date_tmp", "sector"], "_sec_csz"))  # sector-neutral

        for col in cols_to_zscore:
            if col not in panel.columns:
                continue
            for keys, suffix in specs:
                g     = panel.groupby(keys)[col]
                mean  = g.transform("mean")
                std_  = g.transform("std")
                panel[f"{col}{suffix}"] = np.where(
                    std_ > 0, (panel[col] - mean) / std_, np.nan
                )

        panel.drop(columns=["_date_tmp"], inplace=True)
        return panel

    def recompute_fold_features(
        self,
        panel: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Recompute ICT + zone features per CV fold using only data up to cutoff_date.

        Call this once per fold (passing the fold's train_end date) instead of
        recompute_zones.  Fixes two sources of future leakage:

        ICT (OB / BB / FVG / sweeps)
        ─────────────────────────────
        Computed on the slice [start → cutoff_date].  The last known active/
        inactive state at cutoff_date is forward-filled to test rows so that
        deactivations caused by post-cutoff price action are not visible during
        training.  Multi-timeframe ICT composites are rebuilt the same way.

        Zones (DZ / SZ / SDZ / SSZ)
        ────────────────────────────
        Passed to compute_zone_features with cutoff_date guard (unchanged from
        the former recompute_zones implementation).

        Panel-level composites
        ───────────────────────
        sdz_htf_score / ssz_htf_score (zone × trend multiplier) are rebuilt
        after the ticker loop using trend columns already in the panel.

        Parameters
        ----------
        panel       : fold panel slice with ohlcv + features_* columns.
        cutoff_date : last training date for this fold (fold.train_end).

        Returns
        -------
        panel with ICT + zone feature columns replaced.
        """
        from pipeline.features.zone_features import compute_zone_features, _ZONE_COLS

        panel = panel.copy()
        frames = []

        _HTF_W_Z  = {"zone_type_1d": 1, "zone_type_1wk": 2, "zone_type_1mo": 3,
                     "zone_type_3mo": 4, "zone_type_1y": 5}
        _MAX_Z    = sum(_HTF_W_Z.values()) * 2   # 30

        for ticker, grp in panel.groupby(level="ticker"):
            grp        = grp.droplevel("ticker").sort_index()
            full_index = grp.index
            ohlcv_full = grp[["open", "high", "low", "close", "volume"]].copy()

            # Slice up to cutoff — used for both ICT and MTF ICT
            grp_cut    = grp[grp.index <= cutoff_date].copy()
            ohlcv_cut  = grp_cut[["open", "high", "low", "close", "volume"]].copy()

            # NOTE: features_pivot_* columns are intentionally NOT recomputed here.
            # Every pivot feature is a pure trailing-window function of OHLC through
            # each row's own date (shift(1) levels, trailing rolling windows,
            # backward virgin scan, period-END MTF resample + backward merge_asof
            # with allow_exact_matches=False). Truncating future rows cannot change
            # a past row's value — proven by tests/test_pivot_features.py::
            # test_truncation_invariance — unlike ICT/zone state, which is revised
            # retroactively by post-cutoff price action. So the existing columns on
            # `grp` pass through untouched (this loop only overwrites ICT/zone cols).

            # ── 1. ICT daily recompute ────────────────────────────────────
            # Compute on training slice, ffill last state to test rows.
            # atr_14 already present from engineer.build() — no recompute needed.
            # Skipped when skip_ict=True (zone/pivot panels carry no ICT columns).
            if not self.skip_ict and len(grp_cut) >= 10:
                try:
                    ict_result   = self._ict.compute(grp_cut, disp_mult=_ICT_DISP_MULT["1d"],
                                                     proximity_pct=_ZONE_PROXIMITY_PCT["1d"])
                    raw_ict_cols = [c for c in ict_result.columns if c.startswith("ict_")]
                    for raw_col in raw_ict_cols:
                        feat_col = f"{FEATURE_PREFIX}{raw_col}"
                        if feat_col in grp.columns:
                            series = (
                                ict_result[raw_col]
                                .reindex(full_index, method="ffill")
                                .fillna(0)
                            )
                            grp[feat_col] = series.values.astype(np.float32)
                except Exception as e:
                    log.warning(f"{ticker}: ICT daily recompute failed ({e}) — keeping existing values")

            # ── 2. MTF ICT recompute (skipped when skip_ict=True) ────────
            if not self.skip_ict:
                try:
                    ict_bull_htf = np.zeros(len(grp), dtype=np.float32)
                    ict_bear_htf = np.zeros(len(grp), dtype=np.float32)

                    # 1d contribution (freshly recomputed above)
                    for _prio_col, _acc in [
                        (f"{FEATURE_PREFIX}ict_bull_zone_priority", ict_bull_htf),
                        (f"{FEATURE_PREFIX}ict_bear_zone_priority", ict_bear_htf),
                    ]:
                        if _prio_col in grp.columns:
                            _acc += (
                                _ICT_HTF_W["1d"]
                                * grp[_prio_col].fillna(0).values.astype(np.float32)
                                / _ICT_PRIORITY_MAX
                            )

                    for tf_label, rule in _ICT_HTF_RESAMPLE.items():
                        try:
                            htf = ohlcv_cut.resample(rule).agg({
                                "open": "first", "high": "max",
                                "low": "min",    "close": "last", "volume": "sum",
                            }).dropna(subset=["close"])

                            if len(htf) < 5:
                                continue

                            htf["atr_14"] = _wilder_atr(
                                htf["high"].values.astype(float),
                                htf["low"].values.astype(float),
                                htf["close"].values.astype(float), 14,
                            )

                            htf_ict   = self._ict.compute(
                                htf,
                                zone_expiry_bars=_ICT_ZONE_EXPIRY[tf_label],
                                disp_mult=_ICT_DISP_MULT[tf_label],
                                proximity_pct=_ZONE_PROXIMITY_PCT[tf_label],
                            )
                            htf_reset = htf_ict.reset_index()
                            date_col  = htf_reset.columns[0]
                            htf_reset = (htf_reset
                                         .rename(columns={date_col: "date"})
                                         .sort_values("date"))
                            carry     = [c for c in _ICT_CARRY_COLS if c in htf_reset.columns]

                            daily_r = pd.DataFrame({"date": full_index}).sort_values("date")
                            merged  = pd.merge_asof(
                                daily_r,
                                htf_reset[["date"] + carry],
                                on="date", direction="backward",
                            ).set_index("date")

                            w = _ICT_HTF_W[tf_label]
                            for col in carry:
                                vals     = merged[col].reindex(full_index).fillna(0).values.astype(np.float32)
                                feat_col = f"{FEATURE_PREFIX}{col}_{tf_label}"
                                if feat_col in grp.columns:
                                    grp[feat_col] = vals
                                if col == "ict_bull_zone_priority":
                                    ict_bull_htf += w * vals / _ICT_PRIORITY_MAX
                                elif col == "ict_bear_zone_priority":
                                    ict_bear_htf += w * vals / _ICT_PRIORITY_MAX

                        except Exception as _e:
                            log.debug(f"{ticker} ICT HTF {tf_label} recompute: {_e}")

                    grp[f"{FEATURE_PREFIX}ict_bull_htf_score"] = (
                        ict_bull_htf / _ICT_SIGNAL_MAX
                    ).clip(0, 1).astype(np.float32)
                    grp[f"{FEATURE_PREFIX}ict_bear_htf_score"] = (
                        ict_bear_htf / _ICT_SIGNAL_MAX
                    ).clip(0, 1).astype(np.float32)

                except Exception as e:
                    log.warning(f"{ticker}: MTF ICT recompute failed ({e})")

            # ── 3. Zone recompute (cutoff guard) ─────────────────────────
            try:
                zone_result = compute_zone_features(ohlcv_full, cutoff_date=cutoff_date)
                for col in _ZONE_COLS:
                    if col in zone_result.columns:
                        grp[col] = zone_result[col].values
            except Exception as e:
                log.warning(f"{ticker}: zone recompute failed ({e})")

            # ── 3b. Structure recompute (cutoff guard, toggle) ───────────────
            if getattr(self.cfg, "use_structure_features", False):
                try:
                    sf = structure_feature_frame(
                        ohlcv_full,
                        cutoff_date=cutoff_date,
                        prefix=FEATURE_PREFIX,
                        major_swing_length=getattr(self.cfg, "structure_major_swing", 25),
                        minor_swing_length=getattr(self.cfg, "structure_minor_swing", 5),
                    )
                    for scol in sf.columns:
                        grp[scol] = sf.reindex(full_index)[scol].values
                except Exception as e:
                    log.warning(f"{ticker}: structure recompute failed ({e})")

            # Rebuild zone composite columns
            sdz = np.zeros(len(grp), dtype=np.float32)
            ssz = np.zeros(len(grp), dtype=np.float32)
            dz  = np.zeros(len(grp), dtype=np.float32)
            sz  = np.zeros(len(grp), dtype=np.float32)
            for tf_col, w in _HTF_W_Z.items():
                if tf_col not in grp.columns:
                    continue
                zt = grp[tf_col].astype(str).str.strip().str.upper()
                sdz += w * 2 * (zt == "SDZ").astype(np.float32).values
                ssz += w * 2 * (zt == "SSZ").astype(np.float32).values
                dz  += w * 1 * (zt == "DZ").astype(np.float32).values
                sz  += w * 1 * (zt == "SZ").astype(np.float32).values
                tf_short = tf_col.replace("zone_type_", "")
                grp[f"{FEATURE_PREFIX}sdz_{tf_short}"] = (zt == "SDZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}ssz_{tf_short}"] = (zt == "SSZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}dz_{tf_short}"]  = (zt == "DZ").astype(np.float32)
                grp[f"{FEATURE_PREFIX}sz_{tf_short}"]  = (zt == "SZ").astype(np.float32)

            grp[f"{FEATURE_PREFIX}sdz_raw_score"]  = (sdz / _MAX_Z).clip(0, 1)
            grp[f"{FEATURE_PREFIX}ssz_raw_score"]  = (ssz / _MAX_Z).clip(0, 1)
            grp[f"{FEATURE_PREFIX}dz_raw_score"]   = (dz  / _MAX_Z).clip(0, 1)
            grp[f"{FEATURE_PREFIX}sz_raw_score"]   = (sz  / _MAX_Z).clip(0, 1)
            grp[f"{FEATURE_PREFIX}any_valid_sdz"]  = (sdz > 0).astype(np.float32)
            grp[f"{FEATURE_PREFIX}any_valid_ssz"]  = (ssz > 0).astype(np.float32)
            grp[f"{FEATURE_PREFIX}any_valid_zone"] = ((sdz + ssz + dz + sz) > 0).astype(np.float32)
            grp[f"{FEATURE_PREFIX}zone_active"]    = grp.get("zone_active_1d",   pd.Series(0.0, index=grp.index)).astype(np.float32)
            grp[f"{FEATURE_PREFIX}zone_dist_atr"]  = grp.get("zone_dist_atr_1d", pd.Series(np.nan, index=grp.index)).astype(np.float32)
            grp[f"{FEATURE_PREFIX}zone_strength"]  = grp.get("zone_strength_1d", pd.Series(0.0,  index=grp.index)).astype(np.float32)

            grp["ticker"] = ticker
            grp = grp.set_index("ticker", append=True).reorder_levels(["ticker", "date"])
            frames.append(grp)

        if not frames:
            log.warning("recompute_fold_features: no frames produced — returning panel unchanged.")
            return panel

        result = pd.concat(frames).sort_index()

        # ── 4. Panel-level: rebuild sdz_htf_score × trend multiplier ─────
        # Trend columns are causal (rolling SMAs) — no recompute needed.
        # Rebuild the zone×trend composite so it reflects fresh zone scores.
        sdz_r = result.get(f"{FEATURE_PREFIX}sdz_raw_score", pd.Series(0.0, index=result.index))
        ssz_r = result.get(f"{FEATURE_PREFIX}ssz_raw_score", pd.Series(0.0, index=result.index))
        wt  = result.get("weekly_trend",    pd.Series(0.0, index=result.index)).fillna(0)
        mt  = result.get("monthly_trend",   pd.Series(0.0, index=result.index)).fillna(0)
        qt  = result.get("quarterly_trend", pd.Series(0.0, index=result.index)).fillna(0)
        yt  = result.get("yearly_trend",    pd.Series(0.0, index=result.index)).fillna(0)
        up_mult = 0.5 + 0.375*wt + 0.375*mt + 0.375*qt + 0.375*yt
        dn_mult = 0.5 + 0.375*(1-wt) + 0.375*(1-mt) + 0.375*(1-qt) + 0.375*(1-yt)
        result[f"{FEATURE_PREFIX}sdz_htf_score"]      = (sdz_r * up_mult).astype(np.float32)
        result[f"{FEATURE_PREFIX}ssz_htf_score"]      = (ssz_r * dn_mult).astype(np.float32)
        result[f"{FEATURE_PREFIX}zone_htf_confluence"] = (
            result[f"{FEATURE_PREFIX}sdz_htf_score"] - result[f"{FEATURE_PREFIX}ssz_htf_score"]
        ).astype(np.float32)

        # ── ICT × trend multiplier (mirrors zone treatment above) ────────────
        # ict_bull/bear_htf_score are normalized [0,1] in the ticker loop but
        # carry no trend context. Apply the same up_mult/dn_mult so bullish ICT
        # is amplified in uptrends and bearish ICT in downtrends — identical to
        # how sdz_htf_score and ssz_htf_score are treated.
        # Guarded twice: (a) skip_ict runs leave existing ICT columns untouched —
        # they already carry the build-time trend multiplier, and multiplying
        # again would double-scale them; (b) column presence keeps the
        # no-added-columns invariant (test_critical_invariants) on panels that
        # never had ICT columns.
        if not self.skip_ict and f"{FEATURE_PREFIX}ict_bull_htf_score" in result.columns:
            ict_bull_r = result.get(f"{FEATURE_PREFIX}ict_bull_htf_score",
                                    pd.Series(0.0, index=result.index))
            ict_bear_r = result.get(f"{FEATURE_PREFIX}ict_bear_htf_score",
                                    pd.Series(0.0, index=result.index))
            result[f"{FEATURE_PREFIX}ict_bull_htf_score"] = (
                ict_bull_r * up_mult).clip(0, 1).astype(np.float32)
            result[f"{FEATURE_PREFIX}ict_bear_htf_score"] = (
                ict_bear_r * dn_mult).clip(0, 1).astype(np.float32)
            # Signed confluence: positive = bullish ICT dominates, negative = bearish.
            # Mirrors zone_htf_confluence; gives the model a single sided ICT signal.
            result[f"{FEATURE_PREFIX}ict_htf_confluence"] = (
                result[f"{FEATURE_PREFIX}ict_bull_htf_score"]
                - result[f"{FEATURE_PREFIX}ict_bear_htf_score"]
            ).astype(np.float32)

        result = _cast_categorical(result)
        result = _downcast_and_defragment(result)
        return result

    def recompute_zones(
        self,
        panel: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Deprecated alias — use recompute_fold_features instead."""
        return self.recompute_fold_features(panel, cutoff_date)


