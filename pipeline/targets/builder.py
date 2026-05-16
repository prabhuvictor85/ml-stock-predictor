"""
TargetBuilder — computes all target columns from the master panel.

RULE: All targets computed inside groupby('ticker').
RULE: Drop last max_forward_horizon rows per ticker (RULE 3).
RULE: benchmark_20d_return fetched from cfg.benchmark_ticker via DataFetcher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

MAX_FORWARD_HORIZON = 60   # covers longest horizon (60d)
PURGE_HORIZON       = 80   # conservative purge for CV
HORIZONS            = [20, 40, 60]   # trading days


def _hit_target(
    high: np.ndarray,
    low: np.ndarray,
    open_next: np.ndarray,
    profit_target_pct: float,
    stop_loss_pct: float,
) -> np.ndarray:
    """
    Vectorised hit-target scan using rolling window max/min.
    entry = open[t+1]
    hit = 1 if max(high[t+1:t+21]) >= entry*(1+profit) BEFORE min(low) <= entry*(1-stop)
    
    Implementation: for each bar t, build a (n, window) matrix of future highs/lows,
    then find the first bar where each condition triggers.
    """
    n = len(high)
    W = MAX_FORWARD_HORIZON
    result = np.full(n, np.nan)
    if n <= W:
        return result

    # Pad arrays so rolling window always has W elements
    h_pad = np.concatenate([high,  np.full(W, np.nan)])
    l_pad = np.concatenate([low,   np.full(W, np.nan)])

    # Vectorised: for each t, find first hit/stop index in [t+1, t+W]
    valid = n - W  # number of valid bars
    entries = open_next[1:valid + 1]  # entry prices for t=0..valid-1
    target_p = entries * (1 + profit_target_pct)   # (valid,)
    stop_p   = entries * (1 - stop_loss_pct)        # (valid,)

    # Build (valid, W) matrix of future high/low
    idx = np.arange(W)[None, :] + np.arange(1, valid + 1)[:, None]  # (valid, W)
    fut_h = h_pad[idx]  # (valid, W)
    fut_l = l_pad[idx]  # (valid, W)

    # First bar that hits target
    hit_mask  = fut_h >= target_p[:, None]   # (valid, W) bool
    stop_mask = fut_l <= stop_p[:, None]     # (valid, W) bool

    def _first_true(mask: np.ndarray) -> np.ndarray:
        """Row-wise first True index; returns W if none."""
        # argmax on bool returns first True; if all False it returns 0
        has = mask.any(axis=1)
        idx_first = np.where(has, mask.argmax(axis=1), W)
        return idx_first

    hit_idx  = _first_true(hit_mask)
    stop_idx = _first_true(stop_mask)

    # hit=1 if hit comes before stop
    outcome = np.where(hit_idx < stop_idx, 1.0, 0.0)
    # If neither triggers, outcome=0
    valid_entry = ~np.isnan(entries)
    result[:valid] = np.where(valid_entry, outcome, np.nan)
    return result


class TargetBuilder:
    """
    Adds all target columns to the panel in-place.

    Columns added:
      future_20d_return, benchmark_20d_return, future_20d_excess_return,
      hit_target_20d, max_drawdown_20d, future_vol_20d,
      cs_rank_20d, top_quintile, bot_quintile
    """

    def __init__(self, cfg: MarketConfig) -> None:
        self.cfg = cfg

    def build(
        self,
        panel: pd.DataFrame,
        benchmark_close: pd.Series,
    ) -> pd.DataFrame:
        cfg = self.cfg
        panel = panel.copy()
        log.info("Building targets for horizons: 20d, 40d, 60d ...")

        bm = benchmark_close.copy()

        # ── Multi-horizon returns + excess returns ─────────────────────────
        for h in HORIZONS:
            # Stock future return
            panel[f"future_{h}d_return"] = (
                panel.groupby(level="ticker")["close"]
                .transform(lambda x, h=h: x.shift(-h) / x - 1)
            )
            # Benchmark future return for same horizon
            bm_fut = bm.shift(-h) / bm - 1
            panel[f"benchmark_{h}d_return"] = (
                panel.index.get_level_values("date").map(bm_fut)
            )
            # Excess return
            panel[f"future_{h}d_excess_return"] = (
                panel[f"future_{h}d_return"] - panel[f"benchmark_{h}d_return"]
            )

        # ── Aliases for backward compatibility ────────────────────────────
        panel["future_20d_return"]          = panel["future_20d_return"]
        panel["benchmark_20d_return"]       = panel["benchmark_20d_return"]
        panel["future_20d_excess_return"]   = panel["future_20d_excess_return"]

        # ── hit_target_20d (keep on 20d only — entry/exit logic) ──────────
        def _hit(grp: pd.DataFrame) -> pd.Series:
            hits = _hit_target(
                grp["high"].values, grp["low"].values, grp["open"].values,
                cfg.profit_target_pct, cfg.stop_loss_pct,
            )
            return pd.Series(hits, index=grp.index)

        panel["hit_target_20d"] = (
            panel.groupby(level="ticker", group_keys=False).apply(_hit)
        )

        # ── max_drawdown_20d ───────────────────────────────────────────────
        def _max_dd_t(grp: pd.DataFrame) -> pd.Series:
            cv = grp["close"].values.astype(float)
            lv = grp["low"].values.astype(float)
            n  = len(cv); W = 20
            result = np.full(n, np.nan)
            if n <= W:
                return pd.Series(result, index=grp.index)
            valid  = n - W
            lv_pad = np.concatenate([lv, np.full(W, np.nan)])
            idx    = np.arange(1, W + 1)[None, :] + np.arange(valid)[:, None]
            result[:valid] = np.nanmin(lv_pad[idx], axis=1) / cv[:valid] - 1
            return pd.Series(result, index=grp.index)

        panel["max_drawdown_20d"] = (
            panel.groupby(level="ticker", group_keys=False).apply(_max_dd_t)
        )

        # ── future_vol_20d ─────────────────────────────────────────────────
        def _future_vol_t(grp: pd.DataFrame) -> pd.Series:
            cv = grp["close"].values.astype(float)
            n  = len(cv); W = 20
            result = np.full(n, np.nan)
            if n <= W:
                return pd.Series(result, index=grp.index)
            valid  = n - W
            cv_pad = np.concatenate([cv, np.full(W, np.nan)])
            idx    = np.arange(W + 1)[None, :] + np.arange(valid)[:, None]
            wins   = cv_pad[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(wins[:, 1:] / wins[:, :-1])
            mean_r = np.nanmean(lr, axis=1, keepdims=True)
            ss     = np.nansum((lr - mean_r) ** 2, axis=1)
            counts = np.sum(~np.isnan(lr), axis=1)
            result[:valid] = np.where(counts > 1,
                                      np.sqrt(ss / (counts - 1)) * np.sqrt(252), np.nan)
            return pd.Series(result, index=grp.index)

        panel["future_vol_20d"] = (
            panel.groupby(level="ticker", group_keys=False).apply(_future_vol_t)
        )

        # ── cs_rank + quintile labels for ALL horizons ────────────────────
        universe_mask = panel["in_universe"] == True
        for h in HORIZONS:
            exc_col  = f"future_{h}d_excess_return"
            rank_col = f"cs_rank_{h}d"
            exc = panel.loc[universe_mask, exc_col]
            panel.loc[universe_mask, rank_col] = (
                exc.groupby(level="date").rank(pct=True, na_option="keep")
            )
            panel[f"top_quintile_{h}d"] = (panel[rank_col] >= 0.80).astype("Int8")
            panel[f"bot_quintile_{h}d"] = (panel[rank_col] <= 0.20).astype("Int8")
            panel.loc[panel[rank_col].isna(),
                      [f"top_quintile_{h}d", f"bot_quintile_{h}d"]] = pd.NA

        # ── Composite rank: weighted blend of all three horizons ──────────
        # A stock that outperforms at 20d AND 40d AND 60d gets the highest label.
        # A stock that pops quickly but reverses by 60d gets pulled back toward neutral.
        # Rows near the end of history have NaN for longer horizons — we re-normalise
        # the weights so only available horizons contribute (no neutral fill bias).
        _w = {"cs_rank_20d": 0.5, "cs_rank_40d": 0.3, "cs_rank_60d": 0.2}
        _avail_w = sum(
            w * panel[col].notna().astype(float) for col, w in _w.items()
        )
        _weighted_sum = sum(
            w * panel[col].fillna(0.0) for col, w in _w.items()
        )
        panel["cs_rank_composite"] = (_weighted_sum / _avail_w.replace(0, np.nan))

        # ── Backward-compatible aliases ────────────────────────────────────
        panel["cs_rank_20d"]   = panel["cs_rank_20d"]
        panel["top_quintile"]  = panel["top_quintile_20d"]
        panel["bot_quintile"]  = panel["bot_quintile_20d"]

        log.info(
            f"Targets built. "
            f"top_quintile_20d: {panel['top_quintile_20d'].sum()} | "
            f"top_quintile_40d: {panel['top_quintile_40d'].sum()} | "
            f"top_quintile_60d: {panel['top_quintile_60d'].sum()}"
        )
        return panel


