"""
BacktestEngine — walk-forward backtester (§7).

Signal at close T → execution at open T+1.
Tracks: holdings, NAV, gross/net returns, turnover, sector attribution.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.backtest.execution import ExecutionModel, TradeResult
from pipeline.backtest.reporter import PerformanceReporter
from pipeline.config.base import MarketConfig
from pipeline.models.ensemble import EnsembleRanker
from pipeline.portfolio.constructor import PortfolioConstructor
from pipeline.utils.logging import get_logger
from pipeline.utils.types import PerformanceReport

log = get_logger(__name__)


class BacktestEngine:
    """
    Walk-forward backtester.

    Parameters
    ----------
    cfg           : MarketConfig
    ensemble      : trained EnsembleRanker
    port_ctor     : PortfolioConstructor
    execution     : ExecutionModel
    feature_cols  : list of feature column names the ensemble expects
    top_n         : number of stocks to hold per week
    initial_nav   : starting portfolio NAV in base currency
    """

    def __init__(
        self,
        cfg: MarketConfig,
        ensemble: EnsembleRanker,
        port_ctor: "PortfolioConstructor",
        execution: ExecutionModel,
        feature_cols: List[str],
        top_n: int = 10,
        initial_nav: float = 1_000_000,
    ) -> None:
        self.cfg = cfg
        self.ensemble = ensemble
        self.port_ctor = port_ctor
        self.execution = execution
        self.feature_cols = feature_cols
        self.top_n = top_n
        self.initial_nav = initial_nav

    def run(
        self,
        panel: pd.DataFrame,
        benchmark_close: pd.Series,
    ) -> PerformanceReport:
        """
        Run the full backtest over the panel.

        Parameters
        ----------
        panel           : master panel with features + targets, MultiIndex (date, ticker)
        benchmark_close : daily benchmark close Series

        Returns
        -------
        PerformanceReport (gross + net metrics)
        """
        cfg = self.cfg
        group_dates = sorted(panel["group_date"].dropna().unique())

        holdings: Dict[str, float] = {}   # ticker → current weight
        nav = float(self.initial_nav)

        weekly_gross: List[float] = []
        weekly_net: List[float] = []
        weekly_bm: List[float] = []
        weekly_turnover: List[float] = []
        weekly_hit: List[float] = []
        dates_list: List[pd.Timestamp] = []
        sector_weekly: Dict[str, List[float]] = {}

        bm_log_ret = np.log(benchmark_close / benchmark_close.shift(1)).fillna(0)

        for i, gd in enumerate(group_dates):
            # Cross-section for this rebalance date
            cross = panel[
                (panel["group_date"] == gd) & (panel["in_universe"] == True)
            ].copy()

            if len(cross) < 5:
                log.warning(f"Group {gd}: only {len(cross)} tickers, skipping.")
                continue

            missing_feats = [f for f in self.feature_cols if f not in cross.columns]
            if missing_feats:
                log.warning(f"Missing features at {gd}: {missing_feats[:3]}")
                continue

            X = cross[self.feature_cols]
            vol_col = "future_vol_20d" if "future_vol_20d" in cross.columns else None
            vol_series = cross[vol_col] if vol_col else None

            scores = self.ensemble.score(X, vol_series)
            score_series = pd.Series(scores, index=cross.index)

            # Portfolio construction
            selected, weights = self.port_ctor.construct(cross, score_series)

            # ── Compute returns for CURRENT period (holdings → next rebalance) ──
            # We use t+1 open as fill, and measure 20d return in the panel
            gross_ret = 0.0
            sector_rets_this_week: Dict[str, float] = {}
            hit_flags: List[int] = []

            for ticker, weight in holdings.items():
                t_rows = cross.xs(ticker, level="ticker", drop_level=False) if ticker in cross.index.get_level_values("ticker") else pd.DataFrame()
                if t_rows.empty:
                    continue
                row = t_rows.iloc[0]
                ret = row.get("future_20d_return", 0.0)
                if np.isnan(ret):
                    ret = 0.0
                gross_ret += weight * ret
                sec = row.get("sector", "Unknown")
                sector_rets_this_week[sec] = sector_rets_this_week.get(sec, 0.0) + weight * ret
                exc = row.get("future_20d_excess_return", np.nan)
                if not np.isnan(exc):
                    hit_flags.append(1 if exc > 0 else 0)

            # ── Execute trades: compute costs ──────────────────────────────
            trades: List[TradeResult] = []
            for ticker in set(list(selected.keys()) + list(holdings.keys())):
                target_w = selected.get(ticker, 0.0)
                current_w = holdings.get(ticker, 0.0)
                if abs(target_w - current_w) < 1e-6:
                    continue

                cross_row = cross.xs(ticker, level="ticker", drop_level=False) if ticker in cross.index.get_level_values("ticker") else pd.DataFrame()
                if cross_row.empty:
                    adv = cfg.min_adv_usd
                    open_t1 = 1.0
                else:
                    r = cross_row.iloc[0]
                    adv = float(r.get("adv_20d_usd", cfg.min_adv_usd))
                    open_t1 = float(r.get("open", r.get("close", 1.0)))

                tr = self.execution.compute_trade(
                    ticker=ticker,
                    target_weight=target_w,
                    current_weight=current_w,
                    nav=nav,
                    open_price_t1=open_t1,
                    adv_20d_usd=adv,
                )
                trades.append(tr)

            net_ret = self.execution.apply_costs(gross_ret, trades, nav)

            # ── Benchmark weekly return ────────────────────────────────────
            bm_slice = bm_log_ret.loc[
                bm_log_ret.index[(bm_log_ret.index >= gd) & (bm_log_ret.index < (group_dates[i+1] if i+1 < len(group_dates) else gd + pd.Timedelta(days=8)))]
            ]
            bm_ret = float(np.exp(bm_slice.sum()) - 1) if len(bm_slice) > 0 else 0.0

            # ── Two-way turnover ───────────────────────────────────────────
            all_tickers = set(list(selected.keys()) + list(holdings.keys()))
            turnover = sum(
                abs(selected.get(t, 0.0) - holdings.get(t, 0.0)) for t in all_tickers
            ) / 2
            weekly_turnover.append(turnover)

            # ── Update NAV ────────────────────────────────────────────────
            nav *= (1 + net_ret)

            # ── Record ────────────────────────────────────────────────────
            weekly_gross.append(gross_ret)
            weekly_net.append(net_ret)
            weekly_bm.append(bm_ret)
            weekly_hit.append(float(np.mean(hit_flags)) if hit_flags else np.nan)
            dates_list.append(pd.Timestamp(gd))

            for sec, sec_ret in sector_rets_this_week.items():
                if sec not in sector_weekly:
                    sector_weekly[sec] = []
                sector_weekly[sec].append(sec_ret)

            # ── Update holdings ───────────────────────────────────────────
            holdings = {t: w for t, w in selected.items() if w > 0}

        idx = pd.DatetimeIndex(dates_list)
        reporter = PerformanceReporter(
            weekly_gross_returns=pd.Series(weekly_gross, index=idx),
            weekly_net_returns=pd.Series(weekly_net, index=idx),
            weekly_bm_returns=pd.Series(weekly_bm, index=idx),
            sector_returns={
                sec: pd.Series(v, index=idx[:len(v)])
                for sec, v in sector_weekly.items()
            },
            weekly_turnover=pd.Series(weekly_turnover, index=idx),
            weekly_hit_flags=pd.Series([h for h in weekly_hit if not np.isnan(h)]),
        )
        report = reporter.report()
        reporter.print_summary(report)
        return report

