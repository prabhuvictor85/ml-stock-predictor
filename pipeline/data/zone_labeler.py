"""
ExpandingWindowZoneLabeler
──────────────────────────
Computes zone labels (SDZ / SSZ / DZ / SZ) with a strict expanding window
so that each row's zone label reflects ONLY information available at that
point in time.

WHY THIS EXISTS
───────────────
market-vision's analyze_zones / IdentifySwapZones converts a plain demand
zone (DZ) into a Swap Demand Zone (SDZ) only AFTER price later breaks back
through it.  If you run analyze_zones on the full 10-year dataset at once,
a zone row dated 2018 gets labeled SDZ because of a 2021 price move.
The ML model would then see SDZ labels it could never have known — look-ahead
leakage that inflates backtest performance.

HOW IT WORKS
────────────
For each "checkpoint" date (default: last trading day of every calendar year),
we run analyze_zones on the OHLCV slice [start → checkpoint].
We then assign each row's zone label from the analysis that was current AS OF
that checkpoint.

Zone labels are stable between checkpoints: a zone's status doesn't change
until the next checkpoint date — consistent with how a real practitioner
would re-run the analysis periodically.

USAGE
─────
    from pipeline.data.zone_labeler import ExpandingWindowZoneLabeler

    labeler = ExpandingWindowZoneLabeler(analyze_zones_fn=your_fn)
    panel = labeler.label(panel, timeframes=["1d", "1wk", "1mo", "3mo", "1y"])

The labeler writes/overwrites columns:  zone_1d, zone_1wk, zone_1mo, zone_3mo, zone_1y
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Default checkpoint frequency: yearly (last calendar date of each year).
# Can be changed to "QS" (quarterly) for more granular, "MS" for monthly.
DEFAULT_CHECKPOINT_FREQ = "YE"    # pandas offset alias for year-end

# Minimum number of trading rows required per ticker to attempt zone analysis.
MIN_ROWS_FOR_ANALYSIS = 60


class ExpandingWindowZoneLabeler:
    """
    Attaches time-honest zone labels to a panel DataFrame.

    Parameters
    ----------
    analyze_zones_fn : Callable[[pd.DataFrame], pd.DataFrame]
        Your market-vision analyze_zones function (or any wrapper around it).
        Must accept a DataFrame with columns: open, high, low, close, volume, Date
        and return a DataFrame with a 'Zone' column and optionally zone-type columns.

    checkpoint_freq  : str
        Pandas offset alias that controls how often zones are recomputed.
        Default "YE" = year-end.  Use "QS" for quarterly.

    timeframes       : list[str]
        The HTF timeframe suffixes to label.  Determines which columns are written.
        Default: ["1d", "1wk", "1mo", "3mo", "1y"]

    min_rows         : int
        Minimum rows per ticker before analysis is attempted.
    """

    _ZONE_TYPES = {"SDZ", "SSZ", "DZ", "SZ"}
    _UNKNOWN = ""

    def __init__(
        self,
        analyze_zones_fn: Callable[[pd.DataFrame], pd.DataFrame],
        checkpoint_freq: str = DEFAULT_CHECKPOINT_FREQ,
        timeframes: Optional[List[str]] = None,
        min_rows: int = MIN_ROWS_FOR_ANALYSIS,
    ) -> None:
        self.analyze_zones_fn = analyze_zones_fn
        self.checkpoint_freq = checkpoint_freq
        self.timeframes = timeframes or ["1d", "1wk", "1mo", "3mo", "1y"]
        self.min_rows = min_rows

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────────────

    def label(
        self,
        panel: pd.DataFrame,
        ohlcv_raw: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compute expanding-window zone labels and attach them to the panel.

        Parameters
        ----------
        panel    : MultiIndex (date, ticker) DataFrame.
        ohlcv_raw: Optional pre-fetched {ticker → OHLCV DataFrame} map.
                   If None, OHLCV is extracted from the panel itself.

        Returns
        -------
        panel with zone_1d / zone_1wk / zone_1mo / zone_3mo / zone_1y columns
        added/overwritten.
        """
        panel = panel.copy()
        tickers = panel.index.get_level_values("ticker").unique().tolist()
        all_dates = panel.index.get_level_values("date").unique().sort_values()

        # Build checkpoint dates (year-end or custom freq)
        checkpoints = self._build_checkpoints(all_dates)
        log.info(
            f"ExpandingWindowZoneLabeler: {len(tickers)} tickers, "
            f"{len(checkpoints)} checkpoints ({self.checkpoint_freq}), "
            f"timeframes={self.timeframes}"
        )

        # Initialise zone columns with empty string
        for tf in self.timeframes:
            col = f"zone_{tf}"
            panel[col] = self._UNKNOWN

        for ticker in tickers:
            # Pull this ticker's full OHLCV slice from the panel
            try:
                ticker_df = (
                    panel.xs(ticker, level="ticker")[["open", "high", "low", "close", "volume"]]
                    .sort_index()
                    .copy()
                )
            except KeyError:
                log.warning(f"Ticker {ticker} not found in panel — skipping zone labeling.")
                continue

            if len(ticker_df) < self.min_rows:
                log.warning(
                    f"{ticker}: only {len(ticker_df)} rows — below min_rows={self.min_rows}. "
                    "Zone labels will remain empty."
                )
                continue

            ticker_labels = self._label_ticker_expanding(ticker, ticker_df, checkpoints)

            # Write back into the panel
            for tf in self.timeframes:
                col = f"zone_{tf}"
                if col in ticker_labels.columns:
                    panel.loc[
                        panel.index.get_level_values("ticker") == ticker, col
                    ] = ticker_labels[col].reindex(
                        panel.loc[
                            panel.index.get_level_values("ticker") == ticker
                        ].index.get_level_values("date")
                    ).fillna(self._UNKNOWN).values

        log.info("ExpandingWindowZoneLabeler: zone labeling complete.")
        return panel

    # ─────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────────────

    def _build_checkpoints(self, all_dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Build the list of expanding-window checkpoint dates.

        Each checkpoint is the last trading date in the period defined by
        checkpoint_freq.  The first checkpoint requires at least min_rows
        from the start.
        """
        period_ends = pd.date_range(
            start=all_dates.min(),
            end=all_dates.max(),
            freq=self.checkpoint_freq,
        )

        checkpoints = []
        for pe in period_ends:
            # Last actual trading date on or before this period-end
            avail = all_dates[all_dates <= pe]
            if len(avail) >= self.min_rows:
                checkpoints.append(avail[-1])

        # Always include the final date so the last partial period is covered
        if len(all_dates) >= self.min_rows and all_dates[-1] not in checkpoints:
            checkpoints.append(all_dates[-1])

        return sorted(set(checkpoints))

    def _label_ticker_expanding(
        self,
        ticker: str,
        ticker_df: pd.DataFrame,
        checkpoints: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        For each checkpoint date, run analyze_zones on data up to that date.
        Each date in the panel receives the zone classification that was
        current AS OF the most recent checkpoint at or before that date.

        Returns a DataFrame indexed by date with zone_{tf} columns.
        """
        result = pd.DataFrame(
            index=ticker_df.index,
            columns=[f"zone_{tf}" for tf in self.timeframes],
            dtype=object,
        ).fillna(self._UNKNOWN)

        prev_checkpoint = None

        for checkpoint in checkpoints:
            # Slice: only data up to and including this checkpoint
            window = ticker_df[ticker_df.index <= checkpoint].copy()

            if len(window) < self.min_rows:
                prev_checkpoint = checkpoint
                continue

            # Call market-vision analyze_zones
            zone_labels = self._run_analyze_zones(ticker, window, checkpoint)

            # Determine which panel dates this checkpoint applies to:
            # from (prev_checkpoint + 1 day) → checkpoint
            if prev_checkpoint is None:
                date_mask = result.index <= checkpoint
            else:
                date_mask = (result.index > prev_checkpoint) & (result.index <= checkpoint)

            if not date_mask.any():
                prev_checkpoint = checkpoint
                continue

            # For each date in the window, apply the zone label from THIS run
            for date in result.index[date_mask]:
                if date in zone_labels.index:
                    for tf in self.timeframes:
                        col = f"zone_{tf}"
                        if col in zone_labels.columns:
                            result.at[date, col] = zone_labels.at[date, col]

            log.debug(
                f"{ticker}: checkpoint={checkpoint.date()} | "
                f"window_rows={len(window)} | dates_labeled={date_mask.sum()}"
            )
            prev_checkpoint = checkpoint

        return result

    def _run_analyze_zones(
        self,
        ticker: str,
        window: pd.DataFrame,
        checkpoint: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Run analyze_zones on the given OHLCV window.

        Returns a DataFrame indexed by date with zone_{tf} columns.
        Falls back to empty labels if analyze_zones raises.
        """
        # Prepare input: analyze_zones expects 'Date' column
        df_input = window.copy().reset_index()
        df_input = df_input.rename(columns={"date": "Date"})
        df_input.columns = [c.capitalize() if c in ("open","high","low","close","volume") else c
                            for c in df_input.columns]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                zone_data = self.analyze_zones_fn(df_input)
        except Exception as e:
            log.warning(
                f"{ticker} @ {checkpoint.date()}: analyze_zones failed — {e}. "
                "Zone labels will be empty for this window."
            )
            return pd.DataFrame(
                index=window.index,
                columns=[f"zone_{tf}" for tf in self.timeframes],
            ).fillna(self._UNKNOWN)

        return self._parse_zone_output(zone_data, window.index)

    def _parse_zone_output(
        self,
        zone_data: pd.DataFrame,
        date_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Parse analyze_zones output into a date-indexed DataFrame of zone_{tf} labels.

        analyze_zones returns a DataFrame indexed by Date (after _base_eliminator
        sets Date as the index).  We normalise this so the date is always accessible
        via the index regardless of whether 'Date' is a column or the index.

        Priority rule: SDZ > SSZ > DZ > SZ > empty (highest significance first).
        If multiple zones map to the same date, keep the highest priority.
        """
        result = pd.DataFrame(
            index=date_index,
            columns=[f"zone_{tf}" for tf in self.timeframes],
            dtype=object,
        ).fillna(self._UNKNOWN)

        if zone_data is None or len(zone_data) == 0:
            return result

        priority = {"SDZ": 4, "SSZ": 3, "DZ": 2, "SZ": 1, "": 0}

        # ── Normalise: ensure Date is a column (analyze_zones sets it as index) ──
        zone_data = zone_data.copy()
        if zone_data.index.name == "Date" or (
            isinstance(zone_data.index, pd.DatetimeIndex)
        ):
            zone_data = zone_data.reset_index()
            zone_data = zone_data.rename(columns={"index": "Date"})

        # Identify which column holds the date
        date_col = next(
            (c for c in ["Date", "date"] if c in zone_data.columns),
            None,
        )
        if date_col is None:
            log.warning("_parse_zone_output: no Date column found in zone_data — returning empty")
            return result

        zone_data[date_col] = pd.to_datetime(zone_data[date_col], errors="coerce")
        zone_data = zone_data.dropna(subset=[date_col])

        # Identify ZoneType column
        zone_type_col = next(
            (c for c in ["ZoneType", "zone_type", "Type"] if c in zone_data.columns),
            None,
        )
        # 'Zone' column holds Valid/Invalid/RBR/DBD etc — not the type itself
        if zone_type_col is None:
            log.warning("_parse_zone_output: no ZoneType column — returning empty")
            return result

        # Identify optional Timeframe column
        tf_col = next(
            (c for c in ["Timeframe", "timeframe", "TF", "tf"] if c in zone_data.columns),
            None,
        )

        # Only keep valid (non-invalidated) zone rows
        if "Zone" in zone_data.columns:
            zone_data = zone_data[zone_data["Zone"].isin(["Valid", "RBR", "DBD", "DBR", "RBD"])]

        for _, row in zone_data.iterrows():
            row_date = row[date_col]
            if pd.isna(row_date) or row_date not in result.index:
                continue

            zone_type = str(row.get(zone_type_col, "")).strip().upper()
            if zone_type not in self._ZONE_TYPES:
                continue

            tf_value = str(row.get(tf_col, "1d")).strip().lower() if tf_col else "1d"
            col = f"zone_{tf_value}"
            if col not in result.columns:
                col = "zone_1d"  # fallback to daily

            current = result.at[row_date, col]
            if priority.get(zone_type, 0) > priority.get(str(current).upper(), 0):
                result.at[row_date, col] = zone_type

        return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def build_time_honest_zones(
    panel: pd.DataFrame,
    analyze_zones_fn: Callable[[pd.DataFrame], pd.DataFrame],
    checkpoint_freq: str = "YE",
    timeframes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    One-shot convenience function.

    Parameters
    ----------
    panel            : MultiIndex (date, ticker) panel.
    analyze_zones_fn : market-vision's analyze_zones (or your wrapper).
    checkpoint_freq  : how often zones are recomputed ('YE', 'QS', 'MS').
    timeframes       : list of TF suffixes, default ["1d","1wk","1mo","3mo","1y"].

    Returns
    -------
    panel with zone columns overwritten with time-honest labels.
    """
    labeler = ExpandingWindowZoneLabeler(
        analyze_zones_fn=analyze_zones_fn,
        checkpoint_freq=checkpoint_freq,
        timeframes=timeframes,
    )
    return labeler.label(panel)

