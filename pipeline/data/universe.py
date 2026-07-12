"""
UniverseBuilder — constructs and maintains the ticker universe.
Applies all eligibility filters, handles delistings, and marks in_universe.
All market-specific thresholds read from cfg: MarketConfig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.calendar import get_first_trading_day_of_month, get_trading_days
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

MIN_HISTORY_DAYS = 252       # minimum trading days required to enter universe
MIN_MARKET_CAP_USD = 50e6   # $50M USD equivalent minimum market cap


@dataclass
class SymbolMaster:
    """Symbol lifecycle metadata. One row per ticker."""
    ticker: str
    list_date: Optional[pd.Timestamp]
    delist_date: Optional[pd.Timestamp]
    successor_ticker: Optional[str]
    sector: str


class UniverseBuilder:
    """
    Manages universe reconstitution and in_universe flag computation.

    Rules (from spec §1.3):
    - Monthly reconstitution on first trading day of each month.
    - Eligible: adv_20d_usd > min_adv_usd AND history ≥ 252 days
                AND market_cap_usd > 50M USD AND not suspended.
    - Delisted tickers: kept in panel with in_universe=False after delist_date.
    """

    def __init__(self, cfg: MarketConfig) -> None:
        self.cfg = cfg
        self._symbol_master: Dict[str, SymbolMaster] = {}

    def load_symbol_master(self, records: List[Dict]) -> None:
        """
        Load symbol master from a list of dicts with keys:
        ticker, list_date, delist_date, successor_ticker, sector.
        """
        for rec in records:
            ld = pd.Timestamp(rec["list_date"]).tz_localize(None) if rec.get("list_date") else None
            dd = pd.Timestamp(rec["delist_date"]).tz_localize(None) if rec.get("delist_date") else None
            self._symbol_master[rec["ticker"]] = SymbolMaster(
                ticker=rec["ticker"],
                list_date=ld,
                delist_date=dd,
                successor_ticker=rec.get("successor_ticker"),
                sector=rec.get("sector", "Unknown"),
            )
        log.info(f"Symbol master loaded: {len(self._symbol_master)} tickers")

    def get_sector(self, ticker: str) -> str:
        """Return sector for ticker, 'Unknown' if not in master."""
        sm = self._symbol_master.get(ticker)
        return sm.sector if sm else "Unknown"

    def build_in_universe_flags(self, panel: pd.DataFrame) -> pd.Series:
        """
        Compute in_universe boolean Series for a full panel DataFrame.
        panel must have MultiIndex (date, ticker) and columns including:
          adv_20d_usd, market_cap_usd.

        Returns a boolean Series aligned to panel.index.

        Algorithm:
        1. Identify all monthly reconstitution dates.
        2. On each reconstitution date, determine eligible set using that date's data.
        3. Forward-fill the in_universe flag until next reconstitution.
        4. Apply delist_date override: set False for dates after delist_date.
        """
        cfg = self.cfg
        panel = panel.copy()
        
        # Ensure naive timestamps for index
        panel_dates = panel.index.get_level_values("date")
        if panel_dates.tz is not None:
             panel_dates = panel_dates.tz_localize(None)
        
        start_date = panel_dates.min()
        end_date = panel_dates.max()

        trading_days = get_trading_days(cfg.exchange_calendar, start_date, end_date)
        if trading_days.tz is not None:
            trading_days = trading_days.tz_localize(None)

        # All reconstitution dates (first trading day of each month)
        recon_dates: List[pd.Timestamp] = []
        for period in pd.period_range(start_date, end_date, freq="M"):
            fd = get_first_trading_day_of_month(
                cfg.exchange_calendar, period.year, period.month
            )
            if fd is not None:
                if fd.tz is not None:
                    fd = fd.tz_localize(None)
                if start_date <= fd <= end_date:
                    recon_dates.append(fd)

        log.info(f"Reconstitution dates: {len(recon_dates)} months from {start_date.date()} to {end_date.date()}")

        # Build ticker history length map: per date, how many trading days back does this ticker have?
        # We compute this as the number of rows in the panel per ticker up to each date.
        ticker_row_counts: Dict[str, pd.Series] = {}
        for ticker, grp in panel.groupby(level="ticker"):
            # Series: date → cumulative count of available bars
            counts = pd.Series(
                range(1, len(grp) + 1),
                index=grp.index.get_level_values("date"),
            )
            ticker_row_counts[ticker] = counts

        # in_universe starts as False everywhere
        in_universe = pd.Series(False, index=panel.index, name="in_universe")

        # For each reconstitution period, mark eligible tickers
        recon_dates_sorted = sorted(recon_dates)
        for i, recon_dt in enumerate(recon_dates_sorted):
            # Determine next recon date to know the validity window
            if i + 1 < len(recon_dates_sorted):
                next_recon = recon_dates_sorted[i + 1]
            else:
                next_recon = end_date + pd.Timedelta(days=1)

            if recon_dt not in panel_dates:
                # Use the nearest available date
                avail = trading_days[trading_days >= recon_dt]
                if len(avail) == 0:
                    continue
                recon_dt = avail[0]

            try:
                snapshot = panel.xs(recon_dt, level="date")
            except KeyError:
                continue

            eligible_tickers = set()
            for ticker, row in snapshot.iterrows():
                # Liquidity check
                if row.get("adv_20d_usd", 0) <= cfg.min_adv_usd:
                    continue
                # Market cap check
                if row.get("market_cap_usd", 0) <= MIN_MARKET_CAP_USD:
                    continue
                # History length check
                hist = ticker_row_counts.get(ticker)
                if hist is None:
                    continue
                hist_before = hist[hist.index <= recon_dt]
                if len(hist_before) == 0 or hist_before.iloc[-1] < MIN_HISTORY_DAYS:
                    continue
                # Delist check
                sm = self._symbol_master.get(ticker)
                if sm and sm.delist_date and sm.delist_date <= recon_dt:
                    continue
                eligible_tickers.add(ticker)

            log.info(
                f"Recon {recon_dt.date()}: {len(eligible_tickers)} eligible tickers "
                f"(window until {next_recon.date()})"
            )

            # Mark in_universe=True for eligible tickers during [recon_dt, next_recon)
            date_level = panel.index.get_level_values("date")
            ticker_level = panel.index.get_level_values("ticker")
            mask = (
                (date_level >= recon_dt)
                & (date_level < next_recon)
                & ticker_level.isin(eligible_tickers)
            )
            in_universe.loc[mask] = True

        # Apply delist overrides: any row after delist_date → False
        for ticker, sm in self._symbol_master.items():
            if sm.delist_date is not None:
                delist_dt = sm.delist_date if sm.delist_date.tz is None else sm.delist_date.tz_localize(None)
                t_mask = (
                    (panel.index.get_level_values("ticker") == ticker)
                    & (panel_dates > delist_dt)
                )
                in_universe.loc[t_mask] = False

        return in_universe

