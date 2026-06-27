"""
Point-in-time universe membership.

Fixes the survivorship bias of building the training universe from a CURRENT
constituent snapshot: a stock that cratered and left the index between 2010
and today is otherwise invisible to training, which systematically inflates
backtest results (worst for momentum strategies, where the blowups live).

Data format (membership CSV — e.g. fja05680/sp500 `sp500_ticker_start_end.csv`):

    ticker,start_date,end_date
    TSLA,2020-12-21,            <- blank end = still a member
    SIVB,2018-03-19,2023-03-15  <- removed (SVB collapse)
    AAL,1996-01-02,1997-01-15   <- re-entries appear as multiple rows
    AAL,2015-03-23,2024-09-23

Semantics of `apply_pit_universe`:
  - A (date, ticker) row is in-universe iff the date falls inside ANY of the
    ticker's membership intervals [start_date, end_date).
  - Tickers absent from the membership file were NEVER members -> always out.
  - This is strict PIT mode: enable it for honest survivorship-free runs on
    the index the file describes. (Coverage for SP400/600 requires their own
    membership files — same format, concat the rows.)

Known limitation (documented, not hidden): membership says who was in the
index, but yfinance purges DELISTED tickers' prices, so dead members with no
local CSV still contribute nothing. Membership alone removes anachronistic
members from training labels; the full fix additionally needs dead-ticker
OHLCV (e.g. Norgate / Sharadar).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


def normalize_ticker(t: str) -> str:
    """Membership sources use dots for share classes (BRK.B); local price
    files use dashes (BRK-B-1d.csv, yfinance style). Normalize to dashes."""
    return str(t).strip().upper().replace(".", "-")


def load_membership_intervals(path: str | Path) -> pd.DataFrame:
    """Load membership intervals CSV -> DataFrame[ticker, start_date, end_date].

    end_date is NaT for current members. Tickers are dash-normalized.
    Raises FileNotFoundError with a download hint if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Membership file not found: {path}\n"
            "Download S&P 500 intervals (free):\n"
            "  curl -L https://raw.githubusercontent.com/fja05680/sp500/master/"
            "sp500_ticker_start_end.csv -o <path>"
        )
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    required = {"ticker", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Membership CSV missing columns: {missing}")
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df.dropna(subset=["start_date"])
    log.info(
        f"Membership intervals: {len(df)} rows, {df['ticker'].nunique()} tickers, "
        f"{df['end_date'].isna().sum()} open (current members), "
        f"span {df['start_date'].min().date()} -> "
        f"{max(df['start_date'].max(), df['end_date'].max()).date()}"
    )
    return df


def _interval_map(intervals: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """ticker -> list of (start, end] windows; open end becomes Timestamp.max."""
    out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    far_future = pd.Timestamp.max
    for row in intervals.itertuples(index=False):
        end = row.end_date if pd.notna(row.end_date) else far_future
        out.setdefault(row.ticker, []).append((row.start_date, end))
    return out


def pit_universe_mask(panel: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
    """Boolean Series aligned to panel's (date, ticker) MultiIndex:
    True iff the ticker was an index member on that date."""
    imap = _interval_map(intervals)
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker").map(normalize_ticker)

    mask = np.zeros(len(panel), dtype=bool)
    # Vectorize per ticker — panel is sorted by (date, ticker) but groups are
    # cheap either way; ~1.5k tickers x a few intervals each.
    for tk, idx in pd.Series(np.arange(len(panel)), index=tickers).groupby(level=0):
        windows = imap.get(tk)
        if not windows:
            continue  # never a member
        pos = idx.values
        d = dates[pos]
        m = np.zeros(len(pos), dtype=bool)
        for start, end in windows:
            # member from start (inclusive) until removal date (exclusive):
            # on the removal day the stock is already out / halted (SIVB).
            m |= (d >= start) & (d < end)
        mask[pos] = m
    return pd.Series(mask, index=panel.index)


def apply_pit_universe(panel: pd.DataFrame, membership_csv: str | Path) -> pd.DataFrame:
    """Restrict panel['in_universe'] to point-in-time index membership.

    Logs the damage report: how many rows / tickers the survivorship snapshot
    was wrongly including. Returns the panel (modified in place).
    """
    intervals = load_membership_intervals(membership_csv)
    mask = pit_universe_mask(panel, intervals)

    before = int(panel["in_universe"].sum())
    panel["in_universe"] = panel["in_universe"] & mask.values
    after = int(panel["in_universe"].sum())

    tickers_all = panel.index.get_level_values("ticker").nunique()
    covered = intervals["ticker"].nunique()
    flipped = before - after
    log.info(
        f"PIT universe applied: {before:,} -> {after:,} in-universe rows "
        f"({flipped:,} anachronistic rows removed, {flipped / max(before, 1):.1%}). "
        f"Panel tickers: {tickers_all}, membership-covered tickers: {covered}."
    )
    print(
        f"  [PIT] survivorship correction: removed {flipped:,} of {before:,} "
        f"in-universe rows ({flipped / max(before, 1):.1%}) that were not index "
        f"members on their date."
    )
    return panel
