"""
PerformanceReporter — computes gross and net performance tables (§7.3).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from pipeline.utils.types import PerformanceReport
from pipeline.validation.metrics import _max_drawdown
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


class PerformanceReporter:
    """
    Computes both GROSS and NET performance metrics from weekly return series.

    Parameters
    ----------
    weekly_gross_returns : Series of weekly portfolio gross returns, indexed by date
    weekly_net_returns   : Series of weekly portfolio net returns, indexed by date
    weekly_bm_returns    : Series of weekly benchmark returns, indexed by date
    sector_returns       : Dict[sector_name → weekly return Series]
    """

    def __init__(
        self,
        weekly_gross_returns: pd.Series,
        weekly_net_returns: pd.Series,
        weekly_bm_returns: pd.Series,
        sector_returns: Dict[str, pd.Series] | None = None,
        weekly_turnover: pd.Series | None = None,
        weekly_hit_flags: pd.Series | None = None,
        top_decile_excess_series: pd.Series | None = None,
    ) -> None:
        self.gross = weekly_gross_returns.dropna()
        self.net = weekly_net_returns.dropna()
        self.bm = weekly_bm_returns.reindex(self.gross.index).fillna(0)
        self.sector_returns = sector_returns or {}
        self.weekly_turnover = weekly_turnover
        self.weekly_hit_flags = weekly_hit_flags
        self.top_decile_excess = top_decile_excess_series

    def report(self) -> PerformanceReport:
        # ── Equity curves ─────────────────────────────────────────────────
        eq_gross = (1 + self.gross).cumprod()
        eq_net = (1 + self.net).cumprod()
        eq_bm = (1 + self.bm).cumprod()

        n_weeks = len(self.gross)
        ann_factor = 52

        # ── Returns ───────────────────────────────────────────────────────
        gross_ann = float(eq_gross.iloc[-1] ** (ann_factor / n_weeks) - 1) if n_weeks > 0 else 0.0
        net_ann = float(eq_net.iloc[-1] ** (ann_factor / n_weeks) - 1) if n_weeks > 0 else 0.0

        # ── Sharpe ────────────────────────────────────────────────────────
        exc_gross = self.gross - self.bm
        exc_net = self.net - self.bm
        gross_sharpe = float(
            exc_gross.mean() / (exc_gross.std() + 1e-10) * np.sqrt(ann_factor)
        )
        net_sharpe = float(
            exc_net.mean() / (exc_net.std() + 1e-10) * np.sqrt(ann_factor)
        )

        # ── Max drawdown ──────────────────────────────────────────────────
        max_dd = _max_drawdown(eq_net.values)

        # ── Calmar ────────────────────────────────────────────────────────
        calmar = float(net_ann / abs(max_dd)) if max_dd != 0 else 0.0

        # ── Hit ratio ─────────────────────────────────────────────────────
        hit_ratio = 0.0
        if self.weekly_hit_flags is not None and len(self.weekly_hit_flags) > 0:
            hit_ratio = float(self.weekly_hit_flags.mean())

        # ── Top decile excess return ───────────────────────────────────────
        top_dec = 0.0
        if self.top_decile_excess is not None and len(self.top_decile_excess) > 0:
            top_dec = float(self.top_decile_excess.mean() * ann_factor)

        # ── Turnover ──────────────────────────────────────────────────────
        mean_weekly_turnover = 0.0
        ann_turnover = 0.0
        if self.weekly_turnover is not None and len(self.weekly_turnover) > 0:
            mean_weekly_turnover = float(self.weekly_turnover.mean())
            ann_turnover = mean_weekly_turnover * ann_factor

        # ── Sector attribution ────────────────────────────────────────────
        sector_attr: Dict[str, float] = {}
        for sec, sec_rets in self.sector_returns.items():
            aligned = sec_rets.reindex(self.net.index).fillna(0)
            sector_attr[sec] = float(aligned.sum())
        # Check if any sector > 40% of total return
        total_attr = sum(abs(v) for v in sector_attr.values()) + 1e-10
        for sec, val in sector_attr.items():
            if abs(val) / total_attr > 0.40:
                log.warning(f"Sector attribution: '{sec}' accounts for >{40:.0f}% of return.")

        return PerformanceReport(
            gross_annual_return=gross_ann,
            net_annual_return=net_ann,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            hit_ratio=hit_ratio,
            top_decile_excess_return=top_dec,
            mean_weekly_turnover=mean_weekly_turnover,
            annualized_turnover=ann_turnover,
            sector_attribution=sector_attr,
            equity_curve_gross=eq_gross,
            equity_curve_net=eq_net,
            benchmark_curve=eq_bm,
        )

    def print_summary(self, report: PerformanceReport | None = None) -> None:
        if report is None:
            report = self.report()
        print(f"\n{'='*55}")
        print(f"  GROSS Annual Return : {report.gross_annual_return:.2%}")
        print(f"  NET   Annual Return : {report.net_annual_return:.2%}")
        print(f"  Gross Sharpe        : {report.gross_sharpe:.3f}")
        print(f"  Net Sharpe          : {report.net_sharpe:.3f}")
        print(f"  Max Drawdown        : {report.max_drawdown:.2%}")
        print(f"  Calmar Ratio        : {report.calmar_ratio:.3f}")
        print(f"  Hit Ratio           : {report.hit_ratio:.2%}")
        print(f"  Top Decile Excess   : {report.top_decile_excess_return:.2%}")
        print(f"  Mean Weekly Turnover: {report.mean_weekly_turnover:.2%}")
        print(f"  Ann. Turnover       : {report.annualized_turnover:.2%}")
        if report.sector_attribution:
            print(f"  Sector Attribution:")
            for sec, val in sorted(report.sector_attribution.items(), key=lambda x: -abs(x[1])):
                print(f"    {sec:30s}: {val:.4f}")
        print(f"{'='*55}\n")

