"""
PanelConstructor — assembles the master panel parquet file.

Schema (§1.1):
  MultiIndex: (date, ticker)
  Columns: open, high, low, close, volume, market_cap_usd, adv_20d_usd,
           sector, in_universe, group_date
  + features_* and target_* added by downstream modules

RULE: No weekend or holiday rows.
RULE: adv_20d_usd computed inside groupby('ticker') — never across tickers.
RULE: Panel is partitioned by year for efficient loading.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.data.fetcher import DataFetcher
from pipeline.data.universe import UniverseBuilder
from pipeline.utils.calendar import assign_group_dates, get_trading_days
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

PANEL_DIR = Path("panel")


class PanelConstructor:
    """
    Builds and persists the master panel.

    Usage:
        pc = PanelConstructor(cfg, fetcher, universe_builder)
        panel = pc.build(tickers, start, end)
        pc.save(panel, output_dir)
    """

    def __init__(
        self,
        cfg: MarketConfig,
        fetcher: DataFetcher,
        universe_builder: UniverseBuilder,
    ) -> None:
        self.cfg = cfg
        self.fetcher = fetcher
        self.universe_builder = universe_builder

    def build(
        self,
        tickers: List[str],
        start: str,
        end: str,
        shares_outstanding: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Build the full master panel DataFrame.

        Parameters
        ----------
        tickers : list of ticker symbols (exchange-native format)
        start   : 'YYYY-MM-DD'
        end     : 'YYYY-MM-DD'
        shares_outstanding : ticker → shares. If None, market_cap_usd will be NaN.

        Returns
        -------
        DataFrame with MultiIndex (date, ticker), columns as per §1.1 schema.
        """
        cfg = self.cfg
        log.info(f"Building panel for {len(tickers)} tickers [{start} → {end}] market={cfg.market_id}")

        # ── 1. Trading calendar — no weekends/holidays ────────────────────
        trading_days = get_trading_days(cfg.exchange_calendar, start, end)
        log.info(f"Trading days in range: {len(trading_days)}")

        # ── 2. FX rate ─────────────────────────────────────────────────────
        fx = self.fetcher.fetch_fx(start, end)
        fx = fx.reindex(trading_days, method="ffill")

        # ── 3. Fetch OHLCV for each ticker ─────────────────────────────────
        all_raw: Dict[str, pd.DataFrame] = self.fetcher.fetch_many(tickers, start, end)

        # ── 4. Align to trading calendar, compute adv_20d_usd ──────────────
        frames: List[pd.DataFrame] = []
        for ticker in tickers:
            df = all_raw.get(ticker, pd.DataFrame())
            if df.empty:
                log.warning(f"No data for {ticker}, skipping.")
                continue

            # Keep only trading days
            df = df.reindex(trading_days).dropna(how="all")
            if df.empty:
                log.warning(f"All NaN after calendar alignment for {ticker}, skipping.")
                continue

            df = df.copy()
            df["ticker"] = ticker

            # FX-adjusted dollar volume for this ticker
            fx_aligned = fx.reindex(df.index, method="ffill").fillna(method="bfill")
            dollar_vol_usd = df["volume"] * df["close"] * fx_aligned

            # adv_20d_usd — computed inside single ticker (RULE 1 compliance)
            df["adv_20d_usd"] = dollar_vol_usd.rolling(20, min_periods=10).mean()

            # market_cap_usd
            shares = (shares_outstanding or {}).get(ticker, np.nan)
            df["market_cap_usd"] = df["close"] * shares * fx_aligned if not np.isnan(shares) else np.nan

            # sector
            df["sector"] = self.universe_builder.get_sector(ticker)

            frames.append(df)

        if not frames:
            raise RuntimeError("No valid ticker data — panel is empty.")

        panel = pd.concat(frames)
        panel.index.name = "date"
        panel = panel.reset_index().set_index(["date", "ticker"])
        panel = panel.sort_index()

        # ── 5. in_universe flag ────────────────────────────────────────────
        log.info("Computing in_universe flags...")
        panel["in_universe"] = self.universe_builder.build_in_universe_flags(panel)

        # ── 6. group_date (last trading day of each ISO week) ─────────────
        dates_series = panel.index.get_level_values("date").to_series().reset_index(drop=True)
        group_dates = assign_group_dates(dates_series, cfg.exchange_calendar)
        panel["group_date"] = group_dates.values

        log.info(
            f"Panel built: {len(panel)} rows, "
            f"{panel.index.get_level_values('ticker').nunique()} tickers, "
            f"{panel['in_universe'].sum()} in-universe rows"
        )
        return panel

    @staticmethod
    def save(panel: pd.DataFrame, output_dir: str | Path = PANEL_DIR) -> None:
        """
        Save panel partitioned by year: {output_dir}/year=YYYY/part-0.parquet
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dates = panel.index.get_level_values("date")
        years = dates.year.unique()

        for year in sorted(years):
            year_dir = output_dir / f"year={year}"
            year_dir.mkdir(parents=True, exist_ok=True)
            part = panel[dates.year == year]
            path = year_dir / "part-0.parquet"
            part.to_parquet(path)
            log.info(f"Saved {len(part)} rows to {path}")

    @staticmethod
    def load(input_dir: str | Path = PANEL_DIR, years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load panel from partitioned parquet directory.
        Optionally filter by list of years for efficient partial loading.
        """
        input_dir = Path(input_dir)
        parts: List[pd.DataFrame] = []

        for year_dir in sorted(input_dir.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            if years is not None and year not in years:
                continue
            path = year_dir / "part-0.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path))
                log.info(f"Loaded {path}")

        if not parts:
            raise FileNotFoundError(f"No panel parquet files found in {input_dir}")

        panel = pd.concat(parts)
        panel = panel.sort_index()
        return panel

