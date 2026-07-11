"""
PurgedWalkForwardCV — expanding-window CV with purge + embargo (§4).

Parameters (from spec):
  min_train_window  = 504 trading days
  test_window       = 252 trading days  (1 year)
  purge_window      = 40 trading days
  embargo_window    = 5 trading days
  min_folds         = 5

Fold boundary alignment: fold boundaries must align to group_date values.
Per-stock dynamic inclusion: a stock is only included in a fold's training/test set if it has enough history up to that fold's cutoff date.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger
from pipeline.utils.types import FoldResult, CVResult

log = get_logger(__name__)

MIN_TRAIN_WINDOW     = 504   # ~2 years minimum training data per stock
TEST_WINDOW          = 252   # 1 full year per test fold
PURGE_WINDOW         = 40
EMBARGO_WINDOW       = 5
MIN_FOLDS            = 5
MIN_TRAIN_GROUP_SIZE = 10
MIN_VAL_GROUP_SIZE   = 5


@dataclass
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _stock_first_dates(panel: pd.DataFrame) -> pd.Series:
    """
    Returns a Series indexed by ticker with the first date each stock
    appears in the panel with a valid close price.
    """
    return (
        panel[panel["close"].notna()]
        .groupby(level="ticker")
        .apply(lambda g: g.index.get_level_values("date").min())
    )


class PurgedWalkForwardCV:
    """
    Expanding-window walk-forward CV with purge + embargo.
    Supports per-stock dynamic inclusion: stocks are only added to a fold
    once they have >= min_train_window days of their own history.

    Usage:
        cv = PurgedWalkForwardCV(n_folds=8)
        for fold_spec, train_idx, test_idx in cv.split(panel):
            ...
    """

    def __init__(
        self,
        n_folds: int = 8,
        min_train_window: int = MIN_TRAIN_WINDOW,
        test_window: int = TEST_WINDOW,
        purge_window: int = PURGE_WINDOW,
        embargo_window: int = EMBARGO_WINDOW,
    ) -> None:
        self.n_folds         = max(n_folds, MIN_FOLDS)
        self.min_train_window = min_train_window
        self.test_window     = test_window
        self.purge_window    = purge_window
        self.embargo_window  = embargo_window

    def get_fold_specs(self, panel: pd.DataFrame) -> List[FoldSpec]:
        all_dates   = sorted(panel.index.get_level_values("date").unique())
        group_dates = sorted(panel["group_date"].dropna().unique())

        if len(all_dates) < self.min_train_window + self.test_window:
            raise ValueError(
                f"Panel too short: {len(all_dates)} days < "
                f"{self.min_train_window + self.test_window} required."
            )

        total_required = self.min_train_window + self.n_folds * self.test_window
        possible_folds = (len(all_dates) - self.min_train_window) // self.test_window

        if len(all_dates) < total_required:
            # Panel too short — reduce folds
            log.warning(
                f"Reducing n_folds from {self.n_folds} to {possible_folds} "
                f"due to panel length constraint."
            )
            self.n_folds = max(possible_folds, MIN_FOLDS)

        first_test_start_idx = self.min_train_window + self.purge_window + self.embargo_window
        folds: List[FoldSpec] = []

        for fold_id in range(self.n_folds):
            test_start_idx = first_test_start_idx + fold_id * self.test_window
            test_end_idx   = test_start_idx + self.test_window - 1
            if test_end_idx >= len(all_dates):
                break

            test_start_date = pd.Timestamp(all_dates[test_start_idx])
            test_end_date   = pd.Timestamp(all_dates[test_end_idx])

            # Ensure both sides of the comparison are Timestamps
            gd_pd = [pd.Timestamp(g) for g in group_dates]
            gd_on_or_after = [g for g in gd_pd if g >= test_start_date]
            if not gd_on_or_after:
                break
            aligned_test_start = gd_on_or_after[0]

            gd_on_or_before_end = [g for g in gd_pd if g <= test_end_date]
            if not gd_on_or_before_end:
                break
            aligned_test_end = gd_on_or_before_end[-1]

            # Purge + embargo counted in TRADING days, not calendar days.
            # timedelta(days=45) ≈ 31 trading days — not enough for a 40+5 td purge.
            all_dates_arr = np.array([pd.Timestamp(d) for d in all_dates])
            days_before_test = all_dates_arr[all_dates_arr < aligned_test_start]
            n_remove = self.purge_window + self.embargo_window
            if len(days_before_test) >= n_remove:
                train_end = days_before_test[-n_remove]
            elif len(days_before_test) > 0:
                train_end = days_before_test[0]
            else:
                train_end = aligned_test_start
            train_start = all_dates[0]

            folds.append(FoldSpec(
                fold_id    = fold_id,
                train_start= pd.Timestamp(train_start),
                train_end  = pd.Timestamp(train_end),
                test_start = pd.Timestamp(aligned_test_start),
                test_end   = pd.Timestamp(aligned_test_end),
            ))

        log.info(f"Generated {len(folds)} fold specs.")
        return folds

    # ── Per-stock eligibility ─────────────────────────────────────────────
    def _eligible_tickers(
        self,
        stock_first: pd.Series,
        train_end: pd.Timestamp,
        all_trading_days: np.ndarray,
    ) -> pd.Index:
        """
        Return tickers that have >= min_train_window trading days of data
        strictly before train_end.

        stock_first : Series[ticker → first_date]
        all_trading_days : sorted array of all unique trading dates in the panel
        """
        # Find the exact date that is min_train_window trading days before train_end
        # using the panel's actual calendar — avoids the inaccurate 365/252 approximation
        # that excluded all tickers when the panel started close to fold 0's cutoff.
        days_up_to_end = all_trading_days[all_trading_days <= train_end]
        if len(days_up_to_end) < self.min_train_window:
            # Not enough trading days in panel up to train_end — no ticker is eligible
            return pd.Index([])
        sufficient_start = days_up_to_end[-self.min_train_window]
        eligible = stock_first[stock_first <= sufficient_start].index
        return eligible

    def split(
        self,
        panel: pd.DataFrame,
    ) -> Iterator[Tuple[FoldSpec, pd.Index, pd.Index]]:
        """
        Yield (fold_spec, train_idx, test_idx) for each fold.

        Per-stock dynamic inclusion:
          - Training: stock included only if it has >= min_train_window days
            of data before the fold's train_end date.
          - Test:     stock included only if it was eligible for this fold
            (same rule — ensures model saw it during training).
        """
        specs            = self.get_fold_specs(panel)
        dates            = panel.index.get_level_values("date")
        tickers_idx      = panel.index.get_level_values("ticker")
        stock_first      = _stock_first_dates(panel)
        all_trading_days = np.array(sorted(dates.unique()))

        for spec in specs:
            # ── Eligible tickers for this fold ────────────────────────────
            eligible = self._eligible_tickers(stock_first, spec.train_end,
                                              all_trading_days)
            eligible_mask = tickers_idx.isin(eligible)

            n_total    = len(stock_first)
            n_eligible = len(eligible)
            n_new      = n_total - n_eligible
            log.info(
                f"Fold {spec.fold_id}: {n_eligible}/{n_total} stocks eligible "
                f"({n_new} excluded — insufficient history before {spec.train_end.date()})"
            )

            # ── Training rows ─────────────────────────────────────────────
            train_date_mask = (dates >= spec.train_start) & (dates <= spec.train_end)
            train_idx = np.where(train_date_mask & eligible_mask)[0]

            # ── Test rows ─────────────────────────────────────────────────
            # Only test stocks that were eligible (model was trained on them)
            test_date_mask = (dates >= spec.test_start) & (dates <= spec.test_end)
            test_idx = np.where(test_date_mask & eligible_mask)[0]

            # ── Validate group sizes ──────────────────────────────────────
            train_panel = panel.iloc[train_idx]
            train_group_sizes = (
                train_panel[train_panel["in_universe"] == True]
                .groupby("group_date")["in_universe"].count()
            )
            small_train_groups = (train_group_sizes < MIN_TRAIN_GROUP_SIZE).sum()
            if small_train_groups > 0:
                log.warning(
                    f"Fold {spec.fold_id}: {small_train_groups} training groups "
                    f"have < {MIN_TRAIN_GROUP_SIZE} tickers — these will be dropped."
                )

            log.info(
                f"Fold {spec.fold_id}: "
                f"train [{spec.train_start.date()} -> {spec.train_end.date()}] "
                f"({len(train_idx)} rows), "
                f"test [{spec.test_start.date()} -> {spec.test_end.date()}] "
                f"({len(test_idx)} rows)"
            )
            yield spec, train_idx, test_idx

    def build_group_array(
        self,
        panel: pd.DataFrame,
        min_group_size: int = MIN_TRAIN_GROUP_SIZE,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Build LightGBM group array for the given panel slice.
        Drops groups smaller than min_group_size.
        Returns (filtered_panel, group_sizes_array).
        """
        panel = panel[panel["in_universe"] == True]

        if "ticker" not in panel.columns:
            panel = panel.reset_index(level="ticker")
        panel = panel.sort_values(["group_date", "ticker"])

        group_counts = panel.groupby("group_date")["ticker"].count()
        valid_gd     = group_counts[group_counts >= min_group_size].index
        dropped      = group_counts[group_counts < min_group_size]
        if len(dropped) > 0:
            log.warning(f"Dropping {len(dropped)} groups with < {min_group_size} tickers.")
        panel = panel[panel["group_date"].isin(valid_gd)]

        if "ticker" in panel.columns and "ticker" not in [panel.index.name, *list(panel.index.names)]:
            panel = panel.set_index("ticker", append=True)
            panel.index.names = ["date", "ticker"]

        raw_sizes = (
            panel.index.get_level_values("ticker")
            .to_series()
            .groupby(panel["group_date"].values)
            .count()
            .sort_index()
            .values
        )

        # LightGBM lambdarank hard-limits each query to 10,000 rows.
        # Split any group exceeding 9,900 into equal sub-groups.
        MAX_GROUP = 9900
        group_sizes: list[int] = []
        for sz in raw_sizes:
            if sz <= MAX_GROUP:
                group_sizes.append(sz)
            else:
                n_chunks = int(np.ceil(sz / MAX_GROUP))
                chunk = sz // n_chunks
                remainder = sz - chunk * n_chunks
                group_sizes.extend([chunk + 1] * remainder + [chunk] * (n_chunks - remainder))

        group_sizes_arr = np.array(group_sizes, dtype=np.int64)
        assert group_sizes_arr.sum() == len(panel), "Group array does not match panel length"
        return panel, group_sizes_arr

