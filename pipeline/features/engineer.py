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

import warnings
from typing import List

import numpy as np
import pandas as pd
from pipeline.config.base import MarketConfig
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr, _wilder_adx
from pipeline.features.multitf_merger import MultiTFMerger
from pipeline.features.zone_features import compute_zone_features
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

FEATURE_PREFIX = "features_"

# ── Multi-timeframe ICT constants ─────────────────────────────────────────────
_ICT_HTF_RESAMPLE = {"1wk": "W-FRI", "1mo": "MS", "3mo": "QS", "1y": "YS"}
_ICT_HTF_W        = {"1d": 1, "1wk": 2, "1mo": 3, "3mo": 4, "1y": 5}
_ICT_SIGNAL_MAX   = float(sum(_ICT_HTF_W.values()))   # 15.0
_ICT_PRIORITY_MAX = 3.0   # max ZonePriority value (BB = 3)

# Columns carried from each HTF ICT run back to the daily index
_ICT_CARRY_COLS = [
    "ict_bob_active",   "ict_bullbb_active",   "ict_bullfvg_active",
    "ict_sob_active",   "ict_bearbb_active",   "ict_bearfvg_active",
    "ict_bsl_swept",    "ict_ssl_swept",
    "ict_bull_zone_priority", "ict_bear_zone_priority",
]


def _rolling_beta(ticker_log_rets: pd.Series, bm_log_rets: pd.Series, window: int = 60) -> pd.Series:
    """60-day rolling OLS beta — vectorised via pandas rolling cov/var."""
    cov = ticker_log_rets.rolling(window, min_periods=window // 2).cov(bm_log_rets)
    var = bm_log_rets.rolling(window, min_periods=window // 2).var()
    return (cov / var.replace(0, np.nan)).reindex(ticker_log_rets.index)


def _winsorize_per_date(panel: pd.DataFrame, feature_cols: List[str], lo: float = 1.0, hi: float = 99.0) -> pd.DataFrame:
    """Winsorize features at [lo, hi] percentile cross-sectionally per date."""
    for col in feature_cols:
        if col not in panel.columns:
            continue
        panel[col] = (
            panel.groupby(level="date")[col].transform(
                lambda x: x.clip(
                    lower=np.nanpercentile(x.values, lo) if x.notna().any() else np.nan,
                    upper=np.nanpercentile(x.values, hi) if x.notna().any() else np.nan,
                )
            )
        )
    return panel


class FeatureEngineer:
    """
    Computes all features from §5.2.  All per-ticker computations use groupby.

    Parameters
    ----------
    cfg               : MarketConfig
    benchmark_close   : daily benchmark close Series indexed by date
    """

    def __init__(self, cfg: MarketConfig, benchmark_close: pd.Series) -> None:
        self.cfg = cfg
        self.benchmark_close = benchmark_close
        self._ict = ICTFeatureEngine()
        self._mtf = MultiTFMerger()

    def build(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features.  Returns panel with features_* columns added.
        """
        import time
        cfg = self.cfg
        log.info("Engineering features...")
        panel = panel.copy()
        panel = panel.sort_index()

        # ── Per-ticker computations ───────────────────────────────────────
        ticker_frames: list[pd.DataFrame] = []
        bm_log_rets = np.log(self.benchmark_close / self.benchmark_close.shift(1))
        all_tickers = panel.index.get_level_values("ticker").unique().tolist()
        n_total = len(all_tickers)
        t_build_start = time.time()

        for _ti, (ticker, grp) in enumerate(panel.groupby(level="ticker")):
            grp = grp.droplevel("ticker").sort_index()
            h = grp["high"].values
            l = grp["low"].values
            c = grp["close"].values
            o = grp["open"].values
            v = grp["volume"].values

            # ── ATR (Wilder) ──────────────────────────────────────────────
            atr14 = _wilder_atr(h, l, c, 14)
            safe_atr = np.where(np.isnan(atr14) | (atr14 == 0), 1e-8, atr14)
            grp["atr_14"] = atr14

            # Percentage ATR = ATR / close — keeps return normalization dimensionless
            # and consistent across stocks regardless of absolute price level.
            # Use this ONLY for normalizing log returns (which are already dimensionless).
            # Price-vs-SMA and SMA-slope features stay on absolute-ATR units (both ₹).
            pct_atr = safe_atr / np.where(c > 0, c, np.nan)
            safe_pct_atr = np.where(np.isnan(pct_atr) | (pct_atr == 0), 1e-6, pct_atr)

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
            grp[f"{FEATURE_PREFIX}adx_14"] = _wilder_adx(h, l, c, 14)

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

            # ── Rolling beta vs benchmark ──────────────────────────────────
            ticker_log_rets = pd.Series(np.log(np.where(c > 0, c, np.nan))).diff()
            ticker_log_rets.index = grp.index
            bm_aligned = bm_log_rets.reindex(grp.index, method="ffill")
            beta = _rolling_beta(ticker_log_rets, bm_aligned, 60)
            # Winsorize beta at [-2, 4]
            beta = beta.clip(-2, 4)
            grp[f"{FEATURE_PREFIX}rolling_beta_60d"] = beta.values

            # ── ICT features ──────────────────────────────────────────────
            grp = self._ict.compute(grp)

            # ── ICT signal counts (debug logging) ────────────────────────
            _n_bob     = int(grp.get("ict_bob_active",     pd.Series(0)).sum())
            _n_sob     = int(grp.get("ict_sob_active",     pd.Series(0)).sum())
            _n_bullbb  = int(grp.get("ict_bullbb_active",  pd.Series(0)).sum())
            _n_bearbb  = int(grp.get("ict_bearbb_active",  pd.Series(0)).sum())
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

            for ict_col in [
                # Order Blocks
                "ict_bob_active",      "ict_bob_dist",       # Bull OB
                "ict_sob_active",      "ict_sob_dist",       # Short (Bear) OB
                # Breaker Blocks (highest-priority ICT signal)
                "ict_bullbb_active",   "ict_bullbb_dist",    # Bull BB
                "ict_bearbb_active",   "ict_bearbb_dist",    # Bear BB
                # Fair Value Gaps
                "ict_bullfvg_active",  "ict_bullfvg_dist",   # Bull FVG
                "ict_bearfvg_active",  "ict_bearfvg_dist",   # Bear FVG
                # Liquidity sweeps
                "ict_bsl_swept",       "ict_ssl_swept",
                # Zone priority metadata
                "ict_bull_zone_priority", "ict_bear_zone_priority",
            ]:
                if ict_col in grp.columns and not ict_col.startswith(FEATURE_PREFIX):
                    grp[f"{FEATURE_PREFIX}{ict_col}"] = grp.pop(ict_col)

            # ── Multi-timeframe ICT (1wk / 1mo / 3mo / 1y) ───────────────────
            # For each HTF: resample OHLCV → compute ATR → run ICT engine →
            # carry active flags + zone priority back to daily via merge_asof.
            # Composite scores weight higher TFs more (same scheme as zones).
            try:
                ohlcv_d   = grp[["open", "high", "low", "close", "volume"]].copy()
                daily_idx = ohlcv_d.index

                # Start HTF composite accumulators (will include 1d below)
                ict_bull_htf = np.zeros(len(grp), dtype=np.float32)
                ict_bear_htf = np.zeros(len(grp), dtype=np.float32)

                # 1d contribution to composite (already computed above)
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

                        # Wilder ATR on the resampled TF
                        htf["atr_14"] = _wilder_atr(
                            htf["high"].values.astype(float),
                            htf["low"].values.astype(float),
                            htf["close"].values.astype(float), 14,
                        )

                        # Run ICT engine on HTF bars
                        htf_ict = self._ict.compute(htf)

                        # Carry active cols back to daily via merge_asof (backward fill)
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
                        for col in carry:
                            vals = merged[col].reindex(daily_idx).fillna(0).values.astype(np.float32)
                            grp[f"{FEATURE_PREFIX}{col}_{tf_label}"] = vals

                            # Accumulate composite using zone priority
                            if col == "ict_bull_zone_priority":
                                ict_bull_htf += w * vals / _ICT_PRIORITY_MAX
                            elif col == "ict_bear_zone_priority":
                                ict_bear_htf += w * vals / _ICT_PRIORITY_MAX

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

            # Re-attach ticker level
            grp.index = pd.MultiIndex.from_arrays(
                [grp.index, [ticker] * len(grp)], names=["date", "ticker"]
            )
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

        panel = pd.concat(ticker_frames).sort_index()

        # ── Sector relative strength (cross-sectional, not per-ticker) ────
        panel = self._add_sector_rs(panel)

        # ── Market breadth (benchmark constituent SMA50 pct above) ────────
        panel = self._add_market_breadth(panel)

        # ── Regime label ──────────────────────────────────────────────────
        panel = self._add_regime(panel)

        # ── Multi-timeframe trends ────────────────────────────────────────
        panel = self._mtf.merge(panel)

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

        # sdz_premium_setup / ssz_premium_setup intentionally removed.
        # The per-TF zone features (sdz_1y, sdz_3mo, sdz_1mo, sdz_1wk) and the
        # trend features (yearly_trend, quarterly_trend, monthly_trend, weekly_trend)
        # are already individual model features. LGBM learns the interaction weights
        # and TF hierarchy from data — we should not hand-code them.

        # ── Winsorize all features at [1, 99] per date ────────────────────
        feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
        panel = _winsorize_per_date(panel, feat_cols)

        log.info(f"Features computed: {len(feat_cols)} feature columns")
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

            # ── 1. ICT daily recompute ────────────────────────────────────
            # Compute on training slice, ffill last state to test rows.
            # atr_14 already present from engineer.build() — no recompute needed.
            if len(grp_cut) >= 10:
                try:
                    ict_result   = self._ict.compute(grp_cut)
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

            # ── 2. MTF ICT recompute ─────────────────────────────────────
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

                        htf_ict   = self._ict.compute(htf)
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

        return result

    def recompute_zones(
        self,
        panel: pd.DataFrame,
        cutoff_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Deprecated alias — use recompute_fold_features instead."""
        return self.recompute_fold_features(panel, cutoff_date)


