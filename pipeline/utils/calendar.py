"""Calendar utilities — all exchange-calendar logic lives here."""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import pandas as pd
import pandas_market_calendars as mcal

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


def get_trading_days(
    calendar_name: str,
    start: str | date,
    end: str | date,
) -> pd.DatetimeIndex:
    """Return all trading days for the given exchange calendar between start and end (inclusive)."""
    cal = mcal.get_calendar(calendar_name)
    schedule = cal.schedule(start_date=str(start), end_date=str(end))
    return mcal.date_range(schedule, frequency="1D").normalize()


def get_last_trading_day_of_week(
    calendar_name: str,
    year: int,
    week: int,
) -> pd.Timestamp | None:
    """Return the last trading day of a given ISO year-week."""
    # Monday of that ISO week
    monday = date.fromisocalendar(year, week, 1)
    friday = monday + timedelta(days=4)
    days = get_trading_days(calendar_name, monday, friday)
    if len(days) == 0:
        return None
    return days[-1]


def get_first_trading_day_of_month(
    calendar_name: str,
    year: int,
    month: int,
) -> pd.Timestamp | None:
    """Return the first trading day of a given year-month."""
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    days = get_trading_days(calendar_name, start, end)
    if len(days) == 0:
        return None
    return days[0]


def assign_group_dates(
    dates: pd.Series,
    calendar_name: str,
) -> pd.Series:
    """
    For each date, return the last trading day of that date's ISO week.
    Uses vectorised approach: build a lookup table for the date range then map.
    """
    if dates.empty:
        return dates.copy()

    all_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    start = all_dates[0]
    end = all_dates[-1]
    trading_days = get_trading_days(calendar_name, start, end)

    # Build mapping: each trading day → last trading day of its ISO week
    df_td = pd.DataFrame({"td": trading_days})
    df_td["year"] = df_td["td"].dt.isocalendar().year.astype(int)
    df_td["week"] = df_td["td"].dt.isocalendar().week.astype(int)
    week_last = df_td.groupby(["year", "week"])["td"].max().rename("group_date")
    df_td = df_td.join(week_last, on=["year", "week"])
    lookup = df_td.set_index("td")["group_date"]

    return dates.map(lookup)

