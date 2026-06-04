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

# Period-END resample aliases.  pandas 2.2 renamed M/Q/Y → ME/QE/YE and emits a
# FutureWarning for the old spellings; pandas < 2.2 ONLY accepts the old ones and
# raises ValueError on the new ones.  Pick the spelling the installed pandas
# accepts so the resample never silently fails (the broad except below would
# otherwise blank every HTF zone column on a version mismatch).
_PD_22_PLUS  = tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2)
_MONTH_END   = "ME" if _PD_22_PLUS else "M"
_QUARTER_END = "QE" if _PD_22_PLUS else "Q"
_YEAR_END    = "YE" if _PD_22_PLUS else "Y"

_HTF_RESAMPLE = {
    "1d":  None,
    "1wk": "W-FRI",
    # Period-END anchored (right-labelled). A period-START rule (MS/QS/YS) labels
    # the bar on its first day, so the cutoff filter `index <= cutoff_date` lets a
    # still-incomplete period (whose aggregate includes data PAST the cutoff) into
    # the training window — defeating the cutoff guard. ME/QE/YE (M/Q/Y on
    # pandas < 2.2) label on the last day, so an incomplete current period is
    # correctly excluded until it closes.
    "1mo": _MONTH_END,
    "3mo": _QUARTER_END,
    "1y":  _YEAR_END,
}
_MIN_BARS = {"1d": 30, "1wk": 10, "1mo": 6, "3mo": 4, "1y": 2}

# Maximum age (calendar days) of the BREAKOUT candle for an SDZ/SSZ to remain
# active.  For DZ/SZ the zone's own date is used.  A recent breakout means
# institutional demand/supply converted the level recently — it stays relevant.
# A breakout older than this window is stale: drop it unless price is near.
_ZONE_MAX_AGE_DAYS = {
    "1d":  90,     #  3 months  — daily SDZ stale after a quarter
    "1wk": 180,    #  6 months  — weekly SDZ stale after half a year
    "1mo": 365,    #  1 year    — monthly SDZ stale after one year
    "3mo": 730,    #  2 years   — quarterly SDZ stale after two years
    "1y":  1825,   #  5 years   — yearly SDZ stale after five years
}

# Price-proximity gate for SDZ/SSZ.  The breakout candle's edge
# (BreakoutProximal = breakout High for SDZ, breakout Low for SSZ) anchors
# relevance.  A swap zone stays active only while price is still within this
# fraction of that edge.  A stock that rallied far past its breakout candle has
# left the zone behind — the level no longer represents nearby institutional
# demand/supply, so the SDZ is dropped even when the breakout is recent.
#   SDZ dropped when close > BreakoutProximal * (1 + pct)
#   SSZ dropped when close < BreakoutProximal * (1 - pct)
# Higher timeframes get wider bands: a quarterly breakout legitimately spans a
# larger price move than a daily one.  This is the second half of the joint
# gate (time AND proximity) — both must pass for the swap zone to remain.
_ZONE_PROXIMITY_PCT = {
    "1d":  0.10,   # 10%
    "1wk": 0.20,   # 20%
    "1mo": 0.35,   # 35%
    "3mo": 0.40,   # 40%
    "1y":  0.50,   # 50%
}
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
    max_age_days: Optional[int] = None,
    daily_close: Optional[pd.Series] = None,
    proximity_pct: Optional[float] = None,
) -> pd.Series:
    """
    Carry zone_type forward to every daily bar via merge_asof (backward).

    Two-pass merge — SDZ/SSZ wins over DZ/SZ when both are recent
    ──────────────────────────────────────────────────────────────────
    Problem: merge_asof uses base-candle date as the sort key.  A Sep-2023
    monthly SZ (base date recent) beats a 2021-base SDZ whose *breakout*
    was May 2023 — erasing the SDZ label even though its breakout is fresh.

    Fix — two independent merges, then SDZ/SSZ takes priority:

    Pass 1 — DZ/SZ zones, merged by base date.
      Recency filter: drop if (bar_date - base_date) > max_age_days.

    Pass 2 — SDZ/SSZ zones, merged by BreakoutDate (the candle that
      confirmed the swap, not the original base).  A JOINT GATE applies:
        (a) Time   — drop if (bar_date - BreakoutDate) > max_age_days.
        (b) Price  — drop if price has rallied/fallen beyond
                     BreakoutProximal * (1 ± proximity_pct).
      BOTH must pass for the swap zone to stay active.  A 7-month-old monthly
      breakout with price still near the breakout High stays active; a recent
      breakout the stock has since left far behind (e.g. +300%) is dropped.

    Final label: SDZ/SSZ (Pass 2) overrides DZ/SZ (Pass 1) when active.
    """
    has_brk = "BreakoutDate" in zone_df.columns
    has_bp  = "BreakoutProximal" in zone_df.columns
    extra   = []
    if has_brk:
        extra.append("BreakoutDate")
    if has_bp:
        extra.append("BreakoutProximal")

    valid = zone_df[
        zone_df["Zone"].astype(str).str.upper() == "VALID"
    ][["ZoneType", "Proximal", "Distal"] + extra].copy()
    valid.index = pd.to_datetime(valid.index)
    valid = valid.sort_index()
    valid["ZoneType"] = valid["ZoneType"].astype(str).str.upper()

    if valid.empty:
        return pd.Series("", index=daily_index, name=col_name, dtype=object)

    daily_r = pd.DataFrame({"date": pd.to_datetime(daily_index)})
    zone_r  = valid.reset_index().rename(columns={valid.reset_index().columns[0]: "date"})
    zone_r  = zone_r.sort_values("date")

    bar_dates = pd.to_datetime(daily_index)

    # Normalise daily dates to a consistent unit to avoid datetime64[us] vs
    # datetime64[ns] MergeError on Python 3.12+ / newer pandas builds.
    _unit = "us"
    daily_r_norm = daily_r.copy()
    daily_r_norm["date"] = daily_r_norm["date"].dt.as_unit(_unit)

    def _merge_and_filter(
        right: pd.DataFrame,
        right_date_col: str,
        apply_proximity: bool = False,
    ) -> pd.Series:
        """
        merge_asof daily_r (left) against right on (date, right_date_col),
        return ZoneType series with optional recency + proximity blanking.
        right_date_col is preserved separately so we can compute age.
        When apply_proximity is True, right must carry a BreakoutProximal
        column and daily_close/proximity_pct must be supplied.
        """
        if right.empty:
            return pd.Series("", index=daily_index)

        right_s = right.copy()
        right_s[right_date_col] = pd.to_datetime(
            right_s[right_date_col]
        ).dt.as_unit(_unit)
        right_s = right_s.sort_values(right_date_col)

        m = pd.merge_asof(
            daily_r_norm.sort_values("date"),
            right_s,
            left_on="date",
            right_on=right_date_col,
            direction="backward",
        )
        # m columns: "date" (bar date), right_date_col (zone ref date), "ZoneType"
        zt_out = m["ZoneType"].fillna("").values

        # ── Time gate ────────────────────────────────────────────────
        if max_age_days is not None:
            ref = pd.to_datetime(m[right_date_col])
            age = (pd.to_datetime(m["date"]) - ref).dt.days.values
            zt_out = np.where(
                pd.isna(ref) | (age > max_age_days), "", zt_out
            )

        # ── Price-proximity gate (SDZ/SSZ only) ──────────────────────
        if (
            apply_proximity
            and proximity_pct is not None
            and daily_close is not None
            and "BreakoutProximal" in m.columns
        ):
            close_vals = (
                daily_close.reindex(pd.to_datetime(m["date"].values))
                .values.astype(float)
            )
            bp_vals = m["BreakoutProximal"].values.astype(float)
            sdz_mask = zt_out == "SDZ"
            ssz_mask = zt_out == "SSZ"
            # SDZ: price must stay at/below breakout-High * (1 + pct)
            too_far_sdz = sdz_mask & (close_vals > bp_vals * (1.0 + proximity_pct))
            # SSZ: price must stay at/above breakout-Low * (1 - pct)
            too_far_ssz = ssz_mask & (close_vals < bp_vals * (1.0 - proximity_pct))
            drop_mask = too_far_sdz | too_far_ssz | (
                (sdz_mask | ssz_mask) & ~np.isfinite(bp_vals)
            )
            zt_out = np.where(drop_mask, "", zt_out)

        result = pd.Series(zt_out, index=pd.to_datetime(m["date"].values))
        return result.reindex(daily_index).fillna("")

    # ── Pass 1: DZ / SZ — merge by base-candle date ───────────────────
    non_swap = zone_r[~zone_r["ZoneType"].isin(["SDZ", "SSZ"])][
        ["date", "ZoneType"]
    ].copy()
    zt_ns = _merge_and_filter(non_swap, "date")

    # ── Pass 2: SDZ / SSZ — merge by BreakoutDate ─────────────────────
    # Recency measured from the breakout candle, not the old SZ base.
    if has_brk:
        swap_cols = ["BreakoutDate", "ZoneType"]
        if has_bp:
            swap_cols.append("BreakoutProximal")
        swap = zone_r[
            zone_r["ZoneType"].isin(["SDZ", "SSZ"]) &
            pd.to_datetime(zone_r.get("BreakoutDate", pd.Series(dtype="datetime64[ns]"))).notna()
        ][swap_cols].copy()
        swap["BreakoutDate"] = pd.to_datetime(swap["BreakoutDate"])
    else:
        swap = pd.DataFrame(columns=["BreakoutDate", "ZoneType"])

    zt_sw = _merge_and_filter(swap, "BreakoutDate", apply_proximity=True)

    # ── SDZ/SSZ takes priority over DZ/SZ ─────────────────────────────
    zt = pd.Series(
        np.where(zt_sw.values != "", zt_sw.values, zt_ns.values),
        index=daily_index,
    )
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
    # Floor the ATR denominator at 5 bps of price. On illiquid/penny names ATR
    # can collapse to ~1e-6 (flat/untraded), and dividing a zone distance by it
    # explodes the feature (std in the hundreds) — destabilising tree splits and
    # any downstream scaling. A sub-5bps daily range is noise, not signal.
    atr_floor = np.abs(c) * 5e-4
    safe_atr = np.where(np.isfinite(atr) & (atr > atr_floor), atr, atr_floor)
    safe_atr = np.where(safe_atr > 0, safe_atr, np.nan)

    proximal_1d = pd.Series(np.nan, index=daily_index)
    # Ungated nearest-daily-zone TYPE, paired with proximal_1d from the SAME
    # merge.  zone_dist_atr_1d is a GEOMETRIC feature ("how far is price from the
    # nearest daily zone edge") and must NOT be silenced by the SDZ relevance
    # gate — that gate governs the zone_type_*/strength SIGNAL, not geometry.
    # Sourcing the distance's type here (ungated) keeps it consistent with the
    # ungated proximal price, so a still-existing zone whose label aged out of
    # the relevance window still yields a valid distance.
    geom_zt_1d = pd.Series("", index=daily_index, dtype=object)

    # Daily close series for the SDZ/SSZ price-proximity gate. Each daily bar's
    # close is compared against the (HTF) breakout candle's edge — same price
    # scale for the same instrument, so direct comparison is valid.
    daily_close_s = pd.Series(c, index=daily_index)

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
            # Recency is measured from BreakoutDate for SDZ/SSZ so that a
            # 7-month-old breakout on a monthly SDZ stays active even when
            # a newer DZ formed above it afterward.
            zt_series = _zones_to_daily(
                zone_df, daily_index, col,
                max_age_days=_ZONE_MAX_AGE_DAYS[tf_label],
                daily_close=daily_close_s,
                proximity_pct=_ZONE_PROXIMITY_PCT[tf_label],
            )
            result[col] = zt_series.values

            # Save proximal + (ungated) type of the nearest daily zone for the
            # geometric distance feature.  Both come from this one merge so the
            # distance numerator (proximal) and its sign/validity (ZoneType) are
            # always the SAME zone — decoupled from the relevance-gated label.
            if tf_label == "1d":
                valid = zone_df[
                    zone_df["Zone"].astype(str).str.upper() == "VALID"
                ][["Proximal", "ZoneType"]].copy()
                valid.index = pd.to_datetime(valid.index)
                if not valid.empty:
                    valid["ZoneType"] = valid["ZoneType"].astype(str).str.upper()
                    vr = valid.reset_index().rename(
                        columns={valid.reset_index().columns[0]: "date"}
                    ).sort_values("date")
                    daily_r = pd.DataFrame({"date": daily_index})
                    m = pd.merge_asof(
                        daily_r.sort_values("date"), vr, on="date", direction="backward"
                    ).set_index("date")
                    proximal_1d = m["Proximal"].reindex(daily_index)
                    geom_zt_1d  = m["ZoneType"].reindex(daily_index).fillna("")

        except Exception:
            result[col] = ""

    # ── Zone detail columns (daily TF) ────────────────────────────────────
    zt_1d = pd.Series(
        result.get("zone_type_1d", pd.Series("", index=result.index)).values,
        index=daily_index,
    )
    result["zone_active_1d"]   = (zt_1d != "").astype(np.float32)
    result["zone_strength_1d"] = np.where(
        zt_1d.isin(["SDZ", "SSZ"]), 2.0,
        np.where(zt_1d.isin(["DZ", "SZ"]), 1.0, 0.0),
    ).astype(np.float32)

    # Distance uses the UNGATED nearest-zone type (geom_zt_1d), not the
    # relevance-gated zt_1d above — geometry is well-defined whether or not the
    # zone is still a "fresh" tradeable signal.
    dist_arr = np.full(len(result), np.nan, dtype=np.float32)
    prox_arr = proximal_1d.values.astype(float)
    geom_zt_arr = geom_zt_1d.values
    for i in range(len(result)):
        if not np.isfinite(prox_arr[i]) or not np.isfinite(safe_atr[i]):
            continue
        zt = geom_zt_arr[i]
        if zt in ("DZ", "SDZ"):
            dist_arr[i] = (c[i] - prox_arr[i]) / safe_atr[i]
        elif zt in ("SZ", "SSZ"):
            dist_arr[i] = (prox_arr[i] - c[i]) / safe_atr[i]
    # Defensive cap: beyond ±20 ATR the "distance" is saturated and meaningless
    # to the model; clipping is standard winsorisation and a hard safety net so
    # no future tiny-ATR edge case can re-introduce an exploding feature.
    result["zone_dist_atr_1d"] = np.clip(dist_arr, -20.0, 20.0)

    return result
