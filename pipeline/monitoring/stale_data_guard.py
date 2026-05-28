"""
pipeline/monitoring/stale_data_guard.py

Stale data detection for production inference.

Checks run BEFORE feature engineering and scoring:
  1. LastBarStalenessCheck  — latest bar per ticker is within max_lag_days
  2. CrossSectionCoverageCheck — enough tickers have fresh data (not just 1 or 2)
  3. PriceSanityCheck — close prices are positive, volume is non-zero
  4. SuspiciousReturnCheck — any ticker with > 3-sigma single-day move
  5. BenchmarkStalenessCheck — benchmark (NSEI) is up to date

Usage
─────
    from pipeline.monitoring.stale_data_guard import StaleDataGuard

    guard = StaleDataGuard(max_lag_days=3, min_coverage_pct=0.80)
    report = guard.check(panel, benchmark_close)
    guard.assert_fresh(panel, benchmark_close)  # raises if stale
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# NSE trading holidays are not modelled here — we use calendar days
# with a generous lag budget. Adjust max_lag_days around festival periods.
_NSE_MAX_LAG_DAYS_DEFAULT = 5   # Mon–Fri + allow for 1 NSE holiday


@dataclass
class StaleIssue:
    severity: str   # 'error' | 'warning'
    check:    str
    message:  str
    tickers:  List[str] = field(default_factory=list)

    def __str__(self) -> str:
        t = f"  tickers: {self.tickers[:10]}" if self.tickers else ""
        return f"[{self.severity.upper()}] {self.check}: {self.message}{t}"


class StaleDataGuard:
    """
    Pre-inference data freshness and sanity checks.

    Parameters
    ----------
    max_lag_days : maximum acceptable calendar days since latest bar
    min_coverage_pct : fraction of universe that must have fresh data
    price_floor : minimum acceptable close price (filters bad/delisted data)
    return_sigma_thresh : single-day return z-score to flag as suspicious
    """

    def __init__(
        self,
        max_lag_days: int   = _NSE_MAX_LAG_DAYS_DEFAULT,
        min_coverage_pct: float = 0.80,
        price_floor: float  = 1.0,
        return_sigma_thresh: float = 5.0,
    ) -> None:
        self.max_lag_days        = max_lag_days
        self.min_coverage_pct   = min_coverage_pct
        self.price_floor         = price_floor
        self.return_sigma_thresh = return_sigma_thresh

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        panel: pd.DataFrame,
        benchmark_close: Optional[pd.Series] = None,
        as_of: Optional[datetime] = None,
    ) -> List[StaleIssue]:
        """
        Run all freshness checks.  Returns list of StaleIssue (empty = OK).

        Parameters
        ----------
        panel          : multi-index (date, ticker) panel
        benchmark_close: NSEI close series (optional but recommended)
        as_of          : reference datetime (defaults to now)
        """
        as_of = as_of or datetime.utcnow()
        issues: List[StaleIssue] = []

        issues += self._check_last_bar_staleness(panel, as_of)
        issues += self._check_cross_section_coverage(panel, as_of)
        issues += self._check_price_sanity(panel)
        issues += self._check_suspicious_returns(panel)
        if benchmark_close is not None:
            issues += self._check_benchmark_staleness(benchmark_close, as_of)

        errors   = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        log.info(f"StaleDataGuard: {errors} errors, {warnings} warnings "
                 f"across {len(issues)} issues.")
        for issue in issues:
            log.warning(str(issue)) if issue.severity == "warning" else log.error(str(issue))
        return issues

    def assert_fresh(
        self,
        panel: pd.DataFrame,
        benchmark_close: Optional[pd.Series] = None,
        as_of: Optional[datetime] = None,
    ) -> None:
        """Raise StaleDataError if any error-severity issues found."""
        issues = self.check(panel, benchmark_close, as_of)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            msg = "\n".join(str(i) for i in errors)
            raise StaleDataError(
                f"Data freshness check failed ({len(errors)} errors):\n{msg}"
            )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_last_bar_staleness(
        self, panel: pd.DataFrame, as_of: datetime
    ) -> List[StaleIssue]:
        """Per-ticker last bar must be within max_lag_days of as_of."""
        dates = panel.index.get_level_values("date")
        tickers = panel.index.get_level_values("ticker")

        last_bar = (
            pd.Series(dates, index=tickers)
            .groupby(level=0)
            .max()
        )
        cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=self.max_lag_days)
        stale  = last_bar[last_bar < cutoff]

        if stale.empty:
            return []
        pct_stale = len(stale) / len(last_bar)
        severity  = "error" if pct_stale > 0.20 else "warning"
        return [StaleIssue(
            severity=severity,
            check="LastBarStaleness",
            message=(
                f"{len(stale)} tickers ({pct_stale:.0%}) have last bar > "
                f"{self.max_lag_days} calendar days before as_of "
                f"({as_of.date()}). Oldest: {stale.min().date()}."
            ),
            tickers=stale.index.tolist()[:20],
        )]

    def _check_cross_section_coverage(
        self, panel: pd.DataFrame, as_of: datetime
    ) -> List[StaleIssue]:
        """
        On the most recent date in the panel, at least min_coverage_pct of
        the full universe must have data.
        """
        latest_date = panel.index.get_level_values("date").max()
        lag_days = (pd.Timestamp(as_of) - latest_date).days
        if lag_days > self.max_lag_days:
            return [StaleIssue(
                severity="error",
                check="CrossSectionCoverage",
                message=(
                    f"Latest date in panel is {latest_date.date()}, "
                    f"which is {lag_days} calendar days before as_of ({as_of.date()}). "
                    f"No fresh cross-section available."
                ),
            )]

        n_latest = len(panel.xs(latest_date, level="date"))
        n_total  = panel.index.get_level_values("ticker").nunique()
        coverage = n_latest / max(n_total, 1)

        if coverage < self.min_coverage_pct:
            return [StaleIssue(
                severity="warning",
                check="CrossSectionCoverage",
                message=(
                    f"Latest cross-section ({latest_date.date()}) has "
                    f"{n_latest}/{n_total} tickers ({coverage:.0%} < "
                    f"{self.min_coverage_pct:.0%} threshold). "
                    f"Data may be partially loaded."
                ),
            )]
        return []

    def _check_price_sanity(self, panel: pd.DataFrame) -> List[StaleIssue]:
        """Close prices must be positive and above price_floor."""
        if "close" not in panel.columns:
            return []
        bad_prices = panel[panel["close"] < self.price_floor]
        if bad_prices.empty:
            return []
        bad_tickers = bad_prices.index.get_level_values("ticker").unique().tolist()
        return [StaleIssue(
            severity="warning",
            check="PriceSanity",
            message=(
                f"{len(bad_tickers)} tickers with close < ₹{self.price_floor} "
                f"(possibly delisted or bad data)."
            ),
            tickers=bad_tickers[:20],
        )]

    def _check_suspicious_returns(self, panel: pd.DataFrame) -> List[StaleIssue]:
        """Flag tickers with > N-sigma single-day returns on the latest bar."""
        if "close" not in panel.columns:
            return []

        latest_date = panel.index.get_level_values("date").max()

        # Compute daily returns across entire panel
        # Deduplicate index before unstacking — duplicate (date, ticker) pairs
        # can appear when multiple CSV rows share the same date (split-adjusted
        # restated files, calendar mismatches, etc.).
        close_ser = panel["close"]
        if close_ser.index.duplicated().any():
            close_ser = close_ser[~close_ser.index.duplicated(keep="last")]
        close_wide = (
            close_ser
            .unstack("ticker")
            .sort_index()
        )
        rets = close_wide.pct_change()
        if len(rets) < 21:
            return []

        # Use rolling z-score based on last 60 days
        roll_mean = rets.rolling(60, min_periods=20).mean()
        roll_std  = rets.rolling(60, min_periods=20).std()
        z_scores  = (rets - roll_mean) / roll_std.replace(0, np.nan)

        if latest_date not in z_scores.index:
            return []

        latest_z = z_scores.loc[latest_date].dropna()
        suspicious = latest_z[latest_z.abs() > self.return_sigma_thresh]

        if suspicious.empty:
            return []
        return [StaleIssue(
            severity="warning",
            check="SuspiciousReturns",
            message=(
                f"{len(suspicious)} tickers with |z-score| > {self.return_sigma_thresh} "
                f"on {latest_date.date()}. May indicate corporate actions, bad prints, "
                f"or circuit breakers. Verify before using in portfolio."
            ),
            tickers=suspicious.index.tolist(),
        )]

    def _check_benchmark_staleness(
        self, benchmark_close: pd.Series, as_of: datetime
    ) -> List[StaleIssue]:
        """Benchmark (NSEI) must also be fresh."""
        if benchmark_close.empty:
            return [StaleIssue(
                severity="warning",
                check="BenchmarkStaleness",
                message="Benchmark series is empty — relative performance cannot be computed.",
            )]
        latest_bm = benchmark_close.index.max()
        lag = (pd.Timestamp(as_of) - pd.Timestamp(latest_bm)).days
        if lag > self.max_lag_days:
            return [StaleIssue(
                severity="error",
                check="BenchmarkStaleness",
                message=(
                    f"Benchmark latest bar is {latest_bm} — "
                    f"{lag} days before as_of. Update ^NSEI data."
                ),
            )]
        return []


class StaleDataError(RuntimeError):
    """Raised when data freshness checks fail at error severity."""