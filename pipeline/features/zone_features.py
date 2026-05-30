"""
ZoneFeatures — per-fold causal zone features using pipeline.utils.zone_analyzer.

LEAKAGE IN ALL ZONE TYPES (DZ/SZ/SDZ/SSZ):
  - DZ/SZ: ZoneAnalyzer's base_eliminator uses FUTURE price to invalidate zones.
           A DZ at bar s may be marked Invalid because a candle at bar b > s overlapped it.
  - SDZ/SSZ: Swap detection looks forward to confirm breakout at bar b > s.
             Bar s would be labeled SDZ even though the breakout hadn't happened yet.

THE ONLY CORRECT FIX: compute zones per fold using only training data.
  compute_zone_features(df, cutoff_date=T) runs ZoneAnalyzer on df[date <= T] only,
  then carries zone labels forward across the full df index via merge_asof.
  This means:
    - In fold training: zones computed on training window → no test data seen
    - In fold test: last zone state as of training cutoff is carried forward
    - Per-fold recompute: a zone valid in fold 1 may be invalidated in fold 2 ✓
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from pipeline.utils.zone_analyzer import ZoneAnalyzer
from pipeline.features.ict_features import _wilder_atr
from pipeline.utils.logging import get_logger

log = get_logger(__name__)
_analyzer = ZoneAnalyzer()

_HTF_RESAMPLE = {
    "1d":  None,
    "1wk": "W-FRI",
    # Period-END anchored (right-labelled). A period-START rule (MS/QS/YS) labels
    # the bar on its first day, so the cutoff filter `index <= cutoff_date` lets a
    # still-incomplete period (whose aggregate includes data PAST the cutoff) into
    # the training window — defeating the cutoff guard. ME/QE/YE label on the last
    # day, so an incomplete current period is correctly excluded until it closes.
    "1mo": "ME",
    "3mo": "QE",
    "1y":  "YE",
}
_MIN_BARS = {"1d": 30, "1wk": 10, "1mo": 6, "3mo": 4, "1y": 2}
_ZONE_COLS = [
    "zone_type_1d", "zone_type_1wk", "zone_type_1mo",
    "zone_type_3mo", "zone_type_1y",
    "zone_active_1d", "zone_dist_atr_1d", "zone_strength_1d",
]


def _run_analyzer(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Prepare OHLCV and run ZoneAnalyzer. Returns zone_df indexed by Date."""
    df = ohlcv.copy().reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={
        date_col: "Date",
        "open": "Open", "high": "High",
        "low":  "Low",  "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return _analyzer.analyze_zones(df)


def _zones_to_daily(
    zone_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    col_name: str,
) -> pd.Series:
    """
    Take ZoneAnalyzer output, keep only Valid zones, carry zone_type
    forward to every daily bar via merge_asof (backward). No lookahead.
    """
    valid = zone_df[
        zone_df["Zone"].astype(str).str.upper() == "VALID"
    ][["ZoneType", "Proximal", "Distal"]].copy()
    valid.index = pd.to_datetime(valid.index)
    valid = valid.sort_index()
    valid["ZoneType"] = valid["ZoneType"].astype(str).str.upper()

    if valid.empty:
        return pd.Series("", index=daily_index, name=col_name, dtype=object)

    daily_r = pd.DataFrame({"date": daily_index})
    daily_r["date"] = pd.to_datetime(daily_r["date"])
    zone_r = valid.reset_index().rename(columns={valid.reset_index().columns[0]: "date"})
    zone_r = zone_r.sort_values("date")

    merged = pd.merge_asof(
        daily_r.sort_values("date"),
        zone_r[["date", "ZoneType", "Proximal", "Distal"]],
        on="date", direction="backward",
    ).set_index("date")

    zt = merged["ZoneType"].reindex(daily_index).fillna("")
    return zt.rename(col_name)


def compute_zone_features(
    df: pd.DataFrame,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute zone features for a single ticker's daily OHLCV DataFrame.

    Parameters
    ----------
    df          : Daily OHLCV, date-indexed, lower-case columns.
    cutoff_date : If given, ZoneAnalyzer is run ONLY on data up to this date.
                  Zone labels are then carried forward to ALL rows in df
                  via merge_asof — so test-period rows see the last zone
                  state as of the cutoff. This eliminates all leakage from
                  base_eliminator and swap zone detection.
                  If None, uses the full df (initial feature build only).

    Returns
    -------
    df copy with zone_type_1d/1wk/1mo/3mo/1y + zone_active_1d /
    zone_dist_atr_1d / zone_strength_1d columns.
    """
    result = df.copy()
    daily_index = pd.to_datetime(result.index)

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    atr = _wilder_atr(h, l, c, 14)
    safe_atr = np.where((atr > 0) & np.isfinite(atr), atr, np.nan)

    proximal_1d = pd.Series(np.nan, index=daily_index)

    for tf_label, resample_rule in _HTF_RESAMPLE.items():
        col = f"zone_type_{tf_label}"
        try:
            # ── Select & optionally resample OHLCV ───────────────────────
            ohlcv_full = df[["open", "high", "low", "close", "volume"]]

            if resample_rule is not None:
                ohlcv_full = ohlcv_full.resample(resample_rule).agg({
                    "open": "first", "high": "max",
                    "low": "min", "close": "last", "volume": "sum",
                }).dropna(subset=["close"])

            # ── Apply cutoff: ZoneAnalyzer only sees training data ────────
            if cutoff_date is not None:
                ohlcv_train = ohlcv_full[ohlcv_full.index <= cutoff_date]
            else:
                ohlcv_train = ohlcv_full

            if len(ohlcv_train) < _MIN_BARS[tf_label]:
                result[col] = ""
                continue

            zone_df = _run_analyzer(ohlcv_train)

            # ── Carry zone labels to FULL daily index (incl. test bars) ──
            zt_series = _zones_to_daily(zone_df, daily_index, col)
            result[col] = zt_series.values

            # Save proximal for 1d distance calc
            if tf_label == "1d":
                valid = zone_df[
                    zone_df["Zone"].astype(str).str.upper() == "VALID"
                ][["Proximal"]].copy()
                valid.index = pd.to_datetime(valid.index)
                if not valid.empty:
                    vr = valid.reset_index().rename(
                        columns={valid.reset_index().columns[0]: "date"}
                    ).sort_values("date")
                    daily_r = pd.DataFrame({"date": daily_index})
                    m = pd.merge_asof(
                        daily_r.sort_values("date"), vr, on="date", direction="backward"
                    ).set_index("date")
                    proximal_1d = m["Proximal"].reindex(daily_index)

        except Exception:
            result[col] = ""

    # ── Zone detail columns (daily TF) ────────────────────────────────────
    zt_1d = pd.Series(
        result.get("zone_type_1d", pd.Series("", index=result.index)).values,
        index=daily_index,
    )
    result["zone_active_1d"] = (zt_1d != "").astype(np.float32)
    result["zone_strength_1d"] = np.where(
        zt_1d.isin(["SDZ", "SSZ"]), 2.0,
        np.where(zt_1d.isin(["DZ", "SZ"]), 1.0, 0.0),
    ).astype(np.float32)

    dist_arr = np.full(len(result), np.nan, dtype=np.float32)
    prox_arr = proximal_1d.values.astype(float)
    zt_arr = zt_1d.values
    for i in range(len(result)):
        if not np.isfinite(prox_arr[i]) or not np.isfinite(safe_atr[i]):
            continue
        zt = zt_arr[i]
        if zt in ("DZ", "SDZ"):
            dist_arr[i] = (c[i] - prox_arr[i]) / safe_atr[i]
        elif zt in ("SZ", "SSZ"):
            dist_arr[i] = (prox_arr[i] - c[i]) / safe_atr[i]
    result["zone_dist_atr_1d"] = dist_arr

    return result
