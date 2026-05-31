"""
ZoneAnalyzer
─────────────
Self-contained supply/demand zone engine ported from market-vision.

Identifies the full zone hierarchy:
  DZ  — Demand Zone          (Drop-Base-Rally or Rally-Base-Rally)
  SZ  — Supply Zone          (Rally-Base-Drop or Drop-Base-Drop)
  SDZ — Swap Demand Zone     (SZ broken upward — former supply becomes demand)
  SSZ — Swap Supply Zone     (DZ broken downward — former demand becomes supply)

Pipeline (matches analyze_zones from market-vision
zone_identifier_service.py — base_eliminator runs LAST):
  1. identify_rally_candles   → candle that closes above prior high
  2. identify_drop_candles    → candle that closes below prior low
  3. identify_base_candles    → small consolidation inside prior range
  4. identify_zones           → group consecutive bases, pattern-match RBR/DBD/DBR/RBD
  5. Zone="Valid"             → matched pattern labels promoted to Valid
  6. identify_swap_zones      → SDZ: SZ broken up cleanly
  7. identify_swap_supply     → SSZ: DZ broken down cleanly
  8. base_eliminator          → invalidate bases whose range is later violated (LAST)

No external dependencies beyond numpy and pandas.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Candle type labels ────────────────────────────────────────────────────────
RALLY = "Rally"
DROP  = "Drop"
BASE  = "Base"
NONE  = "None"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Rally candles
# ─────────────────────────────────────────────────────────────────────────────

def _identify_rally_candles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rally: current candle is bullish AND close > prior high.
    Sets Type = SubType = 'Rally'.
    """
    bullish        = data["Open"] < data["Close"]
    above_prev_high = data["Close"] > data["High"].shift(1)
    is_rally       = bullish & above_prev_high

    data["Type"]    = np.where(is_rally, RALLY, NONE)
    data["SubType"] = np.where(is_rally, RALLY, NONE)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Drop candles
# ─────────────────────────────────────────────────────────────────────────────

def _identify_drop_candles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop: current candle is bearish AND close < prior low.
    Sets Type = SubType = 'Drop' (only where not already Rally).
    """
    bearish         = data["Open"] > data["Close"]
    below_prev_low  = data["Close"] < data["Low"].shift(1)
    is_drop         = bearish & below_prev_low

    data["Type"]    = np.where(is_drop, DROP,  data["Type"])
    data["SubType"] = np.where(is_drop, DROP,  data["SubType"])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Base candles
# ─────────────────────────────────────────────────────────────────────────────

def _identify_base_candles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Base: candle body is INSIDE the prior candle's range (high/low).
    Rally-Base: bullish body inside prior range  → Type=Rally, SubType=Base
    Drop-Base:  bearish body inside prior range  → Type=Drop,  SubType=Base
    """
    inside_hi = data["Close"] <= data["High"].shift(1)
    inside_lo = data["Close"] >= data["Low"].shift(1)
    inside    = inside_hi & inside_lo

    bullish   = data["Open"] < data["Close"]
    bearish   = data["Close"] < data["Open"]

    rly_base  = bullish & inside
    drp_base  = bearish & inside

    data["Type"]    = np.where(rly_base, RALLY, np.where(drp_base, DROP,  data["Type"]))
    data["SubType"] = np.where(rly_base, BASE,  np.where(drp_base, BASE,  data["SubType"]))
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Identify zones (RBR / DBD / DBR / RBD)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_zones(data: pd.DataFrame) -> pd.DataFrame:
    """
    Group consecutive base candles, merge them into one representative row,
    then pattern-match:
      RBR (Rally-Base-Rally) → Demand Zone   (DZ)
      DBD (Drop-Base-Drop)   → Supply Zone   (SZ)
      DBR (Drop-Base-Rally)  → Demand Zone   (DZ)  — continuation demand
      RBD (Rally-Base-Drop)  → Supply Zone   (SZ)  — continuation supply

    Proximal = price closest to future action (the zone edge price approaches first)
    Distal   = price furthest from future action (the zone outer boundary)
    """
    df = data.copy()

    # ── Assign group IDs so consecutive same-SubType candles are grouped ──
    df["group_id"] = df["SubType"].ne(df["SubType"].shift()).cumsum()

    non_base = df[df["SubType"] != BASE].copy()
    base_df  = df[df["SubType"] == BASE].copy()

    # Normalise base OHLC so Open ≤ Close (use body min/max)
    body_max = base_df[["Open", "Close"]].max(axis=1)
    body_min = base_df[["Open", "Close"]].min(axis=1)
    base_df["Close"] = body_max
    base_df["Open"]  = body_min

    # Collapse consecutive base groups into one representative row each
    for col, fn in [("Open","min"), ("High","max"), ("Low","min"), ("Close","max")]:
        base_df[col] = base_df.groupby(["SubType", "group_id"])[col].transform(fn)
    base_df = base_df.drop_duplicates(subset=["Open", "High", "Close", "Low"])

    df = pd.concat([non_base, base_df]).sort_values("Date").reset_index(drop=True)

    # Initialise zone columns
    df["ZoneType"] = NONE
    df["Zone"]     = "0"
    df["Distal"]   = 0.0
    df["Proximal"] = 0.0

    sub = df["SubType"]

    # ── RBR: Rally → Base → Rally  →  Demand Zone ─────────────────────────
    fltr_rbr = (sub == BASE) & (sub.shift(1) == RALLY) & (sub.shift(-1) == RALLY)
    df.loc[fltr_rbr, "Zone"]     = "RBR"
    df.loc[fltr_rbr, "ZoneType"] = "DZ"
    # Distal  = min low of the two rally candles surrounding the base
    # Proximal = base close (upper edge of demand)
    df.loc[fltr_rbr, "Distal"]   = (
        df["Low"].shift(-1).rolling(2, min_periods=0).min()[fltr_rbr]
    )
    df.loc[fltr_rbr, "Proximal"] = df.loc[fltr_rbr, "Close"]

    # ── DBD: Drop → Base → Drop  →  Supply Zone ──────────────────────────
    fltr_dbd = (sub == BASE) & (sub.shift(1) == DROP) & (sub.shift(-1) == DROP)
    df.loc[fltr_dbd, "Zone"]     = "DBD"
    df.loc[fltr_dbd, "ZoneType"] = "SZ"
    df.loc[fltr_dbd, "Distal"]   = (
        df["High"].shift(-1).rolling(2, min_periods=0).max()[fltr_dbd]
    )
    df.loc[fltr_dbd, "Proximal"] = df.loc[fltr_dbd, "Open"]

    # ── DBR: Drop → Base → Rally  →  Demand Zone (continuation) ──────────
    fltr_dbr = (sub == BASE) & (sub.shift(1) == DROP) & (sub.shift(-1) == RALLY)
    df.loc[fltr_dbr, "Zone"]     = "DBR"
    df.loc[fltr_dbr, "ZoneType"] = "DZ"
    df.loc[fltr_dbr, "Distal"]   = (
        df["Low"].shift(-1).rolling(3, min_periods=0).min()[fltr_dbr]
    )
    df.loc[fltr_dbr, "Proximal"] = df.loc[fltr_dbr, "Close"]

    # ── RBD: Rally → Base → Drop  →  Supply Zone (continuation) ──────────
    fltr_rbd = (sub == BASE) & (sub.shift(1) == RALLY) & (sub.shift(-1) == DROP)
    df.loc[fltr_rbd, "Zone"]     = "RBD"
    df.loc[fltr_rbd, "ZoneType"] = "SZ"
    df.loc[fltr_rbd, "Distal"]   = (
        df["High"].shift(-1).rolling(3, min_periods=0).max()[fltr_rbd]
    )
    df.loc[fltr_rbd, "Proximal"] = df.loc[fltr_rbd, "Open"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Swap Demand Zones (SDZ)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_swap_demand_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    SDZ: a Supply Zone (SZ) that price later cleanly breaks ABOVE.
    Exact port of market-vision IdentifySwapZones(df, 'SDZ').

    Conversion rule:
      1. Candidate: ZoneType == 'SZ' and Zone == 'Valid'
      2. Breakout : first close > distal after the zone candle
      3. Breakout candle must be Rally (not Drop/Base)
      4. No Drop/Base candle overlapping the zone band between base and breakout
      5. If criteria met → ZoneType = 'SDZ', Distal = candle low, Proximal = candle high
      6. Breach check: if close/open later drops below SDZ low → back to Invalid SZ
    """
    n = len(df)
    if n < 3:
        return df

    close_arr   = pd.to_numeric(df["Close"],    errors="coerce").to_numpy()
    open_arr    = pd.to_numeric(df["Open"],     errors="coerce").to_numpy()
    high_arr    = pd.to_numeric(df["High"],     errors="coerce").to_numpy()
    low_arr     = pd.to_numeric(df["Low"],      errors="coerce").to_numpy()
    subtype_arr = df["SubType"].to_numpy()

    candidates = np.flatnonzero(
        ((df["ZoneType"] == "SZ") & (df["Zone"] == "Valid")).to_numpy()
    )

    for i in candidates:
        # Re-read in case a prior iteration invalidated this row
        if df.iat[i, df.columns.get_loc("ZoneType")] != "SZ":
            continue
        if df.iat[i, df.columns.get_loc("Zone")] != "Valid":
            continue
        if i + 2 >= n:
            continue

        distal   = float(pd.to_numeric(df.iat[i, df.columns.get_loc("Distal")],  errors="coerce"))
        proximal = float(pd.to_numeric(df.iat[i, df.columns.get_loc("Proximal")], errors="coerce"))
        if not np.isfinite(distal) or not np.isfinite(proximal):
            continue

        # First close above distal (breakout up)
        future_closes = close_arr[i + 2:]
        brk_rel = np.flatnonzero(future_closes > distal)
        if brk_rel.size == 0:
            continue

        brk_idx  = int(i + 2 + brk_rel[0])
        brk_type = subtype_arr[brk_idx]

        # Breakout candle must be Rally
        if brk_type in (DROP, BASE):
            df.iat[i, df.columns.get_loc("Zone")] = "Invalid"
            continue

        lo = min(distal, proximal)
        hi = max(distal, proximal)

        # Check for obstructing Drop/Base candles overlapping the zone before breakout
        pre_h   = high_arr  [i + 2: brk_idx]
        pre_l   = low_arr   [i + 2: brk_idx]
        pre_sub = subtype_arr[i + 2: brk_idx]

        overlaps   = (pre_h >= lo) & (pre_l <= hi)
        drop_bases = np.isin(pre_sub, [DROP, BASE])

        if (overlaps & drop_bases).any():
            df.iat[i, df.columns.get_loc("Zone")] = "Invalid"
            continue

        # Convert to SDZ: Distal = candle low, Proximal = candle high
        df.iat[i, df.columns.get_loc("ZoneType")]         = "SDZ"
        df.iat[i, df.columns.get_loc("Distal")]           = low_arr[i]
        df.iat[i, df.columns.get_loc("Proximal")]         = high_arr[i]
        df.iat[i, df.columns.get_loc("BreakoutDate")]     = df["Date"].iat[brk_idx]
        # BreakoutProximal for SDZ = breakout candle High.  The proximity gate
        # checks that current price hasn't moved more than X% above this level.
        df.iat[i, df.columns.get_loc("BreakoutProximal")] = high_arr[brk_idx]

        # Breach check: if price later drops below SDZ low → invalidate
        sdz_low = float(low_arr[i])
        if np.isfinite(sdz_low):
            future = np.concatenate([close_arr[brk_idx + 1:], open_arr[brk_idx + 1:]])
            if (future < sdz_low).any():
                df.iat[i, df.columns.get_loc("ZoneType")]         = "SZ"
                df.iat[i, df.columns.get_loc("Zone")]             = "Invalid"
                df.iat[i, df.columns.get_loc("BreakoutDate")]     = pd.NaT
                df.iat[i, df.columns.get_loc("BreakoutProximal")] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Swap Supply Zones (SSZ)
# ─────────────────────────────────────────────────────────────────────────────

def _identify_swap_supply_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    SSZ: a Demand Zone (DZ) that price later cleanly breaks BELOW.
    Exact reverse of SDZ logic.

    Conversion rule:
      1. Candidate: ZoneType == 'DZ' and Zone == 'Valid'
      2. Breakout : first close < distal after the zone candle
      3. Breakout candle must be Drop (not Rally/Base)
      4. No Rally/Base candle overlapping the zone band between base and breakout
      5. If criteria met → ZoneType = 'SSZ', Distal = candle high, Proximal = candle low
      6. Breach check: if close/open later rallies above SSZ high → back to Invalid DZ
    """
    n = len(df)
    if n < 3:
        return df

    close_arr   = pd.to_numeric(df["Close"],    errors="coerce").to_numpy()
    open_arr    = pd.to_numeric(df["Open"],     errors="coerce").to_numpy()
    high_arr    = pd.to_numeric(df["High"],     errors="coerce").to_numpy()
    low_arr     = pd.to_numeric(df["Low"],      errors="coerce").to_numpy()
    subtype_arr = df["SubType"].to_numpy()

    candidates = np.flatnonzero(
        ((df["ZoneType"] == "DZ") & (df["Zone"] == "Valid")).to_numpy()
    )

    for i in candidates:
        if df.iat[i, df.columns.get_loc("ZoneType")] != "DZ":
            continue
        if df.iat[i, df.columns.get_loc("Zone")] != "Valid":
            continue
        if i + 2 >= n:
            continue

        distal   = float(pd.to_numeric(df.iat[i, df.columns.get_loc("Distal")],  errors="coerce"))
        proximal = float(pd.to_numeric(df.iat[i, df.columns.get_loc("Proximal")], errors="coerce"))
        if not np.isfinite(distal) or not np.isfinite(proximal):
            continue

        # First close below distal (breakout down)
        future_closes = close_arr[i + 2:]
        brk_rel = np.flatnonzero(future_closes < distal)
        if brk_rel.size == 0:
            continue

        brk_idx  = int(i + 2 + brk_rel[0])
        brk_type = subtype_arr[brk_idx]

        # Breakout candle must be Drop
        if brk_type in (RALLY, BASE):
            df.iat[i, df.columns.get_loc("Zone")] = "Invalid"
            continue

        lo = min(distal, proximal)
        hi = max(distal, proximal)

        # Check for obstructing Rally/Base candles overlapping the zone before breakout
        pre_h   = high_arr  [i + 2: brk_idx]
        pre_l   = low_arr   [i + 2: brk_idx]
        pre_sub = subtype_arr[i + 2: brk_idx]

        overlaps    = (pre_h >= lo) & (pre_l <= hi)
        rally_bases = np.isin(pre_sub, [RALLY, BASE])

        if (overlaps & rally_bases).any():
            df.iat[i, df.columns.get_loc("Zone")] = "Invalid"
            continue

        # Convert to SSZ: Distal = candle high (top), Proximal = candle low (bottom)
        df.iat[i, df.columns.get_loc("ZoneType")]         = "SSZ"
        df.iat[i, df.columns.get_loc("Distal")]           = high_arr[i]
        df.iat[i, df.columns.get_loc("Proximal")]         = low_arr[i]
        df.iat[i, df.columns.get_loc("BreakoutDate")]     = df["Date"].iat[brk_idx]
        # BreakoutProximal for SSZ = breakout candle Low.  The proximity gate
        # checks that current price hasn't fallen more than X% below this level.
        df.iat[i, df.columns.get_loc("BreakoutProximal")] = low_arr[brk_idx]

        # Breach check: if price later rallies above SSZ high → invalidate
        ssz_high = float(high_arr[i])
        if np.isfinite(ssz_high):
            future = np.concatenate([close_arr[brk_idx + 1:], open_arr[brk_idx + 1:]])
            if (future > ssz_high).any():
                df.iat[i, df.columns.get_loc("ZoneType")]         = "DZ"
                df.iat[i, df.columns.get_loc("Zone")]             = "Invalid"
                df.iat[i, df.columns.get_loc("BreakoutDate")]     = pd.NaT
                df.iat[i, df.columns.get_loc("BreakoutProximal")] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Base eliminator
# ─────────────────────────────────────────────────────────────────────────────

def _base_eliminator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Invalidate base zones whose [Distal, Proximal] range overlaps with any
    future candle range or future zone range.  Matches market-vision
    base_eliminator.BaseEliminator.

    Called LAST in analyze_zones (after swap detection), matching the
    market-vision zone_identifier_service order:
      identify_zones → Valid → IdentifySwapZones → IdentifySwapSupplyZones →
      BaseEliminator (final).

    Candidate selection mirrors market-vision check_forward_intersections:
      Zone != "0", SubType==Base, not SDZ/SSZ.
    SDZ/SSZ are excluded because they have already passed swap promotion and
    represent confirmed structural zones.
    """
    work = df.reset_index(drop=True).copy()
    n    = len(work)
    if n < 3:
        return work

    lows      = pd.to_numeric(work["Low"],      errors="coerce").to_numpy()
    highs     = pd.to_numeric(work["High"],     errors="coerce").to_numpy()
    distals   = pd.to_numeric(work["Distal"],   errors="coerce").to_numpy()
    proximals = pd.to_numeric(work["Proximal"], errors="coerce").to_numpy()

    # By the time this runs (after Valid conversion + swap detection), Zone
    # values are "Valid" / "Invalid" / "0". `!= "0"` selects the active zone
    # rows, mirroring market-vision's `Zone != 0` candidate filter.
    candidate_mask = (
        (work["Zone"] != "0")
        & (work["SubType"] == BASE)
        & (~work["ZoneType"].isin(["SDZ", "SSZ"]))
    )
    candidates = np.flatnonzero(candidate_mask.to_numpy())

    for i in candidates:
        if i + 2 >= n - 1:
            continue
        lo = min(distals[i], proximals[i])
        hi = max(distals[i], proximals[i])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue

        sl = slice(i + 2, n - 1)
        overlap_hl = (highs[sl] >= lo) & (lows[sl] <= hi)

        f_lo = np.minimum(distals[sl], proximals[sl])
        f_hi = np.maximum(distals[sl], proximals[sl])
        overlap_dp = (f_hi >= lo) & (f_lo <= hi)

        if np.any(overlap_hl | overlap_dp):
            work.at[i, "Zone"] = "Invalid"

    work = work.drop_duplicates().sort_values("Date").set_index("Date")
    return work


# ─────────────────────────────────────────────────────────────────────────────
# Public API — ZoneAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ZoneAnalyzer:
    """
    Self-contained supply/demand zone analyser.

    Pipeline matches market-vision zone_identifier_service.analyze_zones:
      1. RallyIdentifier   → Rally candles
      2. DropIdentifier    → Drop candles
      3. BaseCandleIdentifier → Base candles (Rally-Base / Drop-Base)
      4. identifyZones     → RBR/DBD/DBR/RBD pattern matching, Distal/Proximal assigned
      5. Zone="Valid"      → matched pattern labels converted to Valid
      6. IdentifySwapZones → SZ with clean upward breakout → SDZ
      7. IdentifySwapSupplyZones → DZ with clean downward breakout → SSZ
      8. BaseEliminator    → invalidate zones whose range overlaps future price/zone (LAST)

    base_eliminator runs LAST, exactly as in market-vision; SDZ/SSZ promoted
    in steps 6-7 are excluded from elimination.

    Input DataFrame columns (case-sensitive):
        Date, Open, High, Low, Close, Volume  (Volume optional)

    Output DataFrame (indexed by Date):
        All input columns  +  Type, SubType, ZoneType, Zone, Distal, Proximal
        ZoneType ∈ {DZ, SZ, SDZ, SSZ, NONE}
        Zone     ∈ {Valid, Invalid, 0}
    """

    def analyze_zones(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Full zone analysis pipeline.

        Parameters
        ----------
        data : DataFrame with columns Date, Open, High, Low, Close.
               Must be sorted ascending by Date.

        Returns
        -------
        DataFrame indexed by Date containing all zone rows.
        """
        t0 = time.time()

        df = data.copy()
        df = df.sort_values("Date").reset_index(drop=True)

        # Ensure numeric OHLC
        for col in ("Open", "High", "Low", "Close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Steps 1-3: candle classification ─────────────────────────────
        df = _identify_rally_candles(df)
        logger.debug("Rally step %.4fs", time.time() - t0)

        df = _identify_drop_candles(df)
        df = _identify_base_candles(df)

        # ── Step 4: zone pattern matching ────────────────────────────────
        # Zone column = "RBR"/"DBD"/"DBR"/"RBD" (matched) or "0" (no match)
        zone_df = _identify_zones(df)

        # ── Valid conversion ─────────────────────────────────────────────
        # market-vision blanket-assigns Zone="Valid" to ALL rows here. We
        # convert only the matched pattern labels (RBR/DBD/DBR/RBD) so that
        # non-zone rows keep Zone="0"; the Layer-2 _zones_to_daily adaptation
        # in zone_features.py filters on Zone=="VALID" and would otherwise
        # carry forward spurious non-zone rows.
        zone_df.loc[zone_df["Zone"].isin(["RBR", "DBD", "DBR", "RBD"]), "Zone"] = "Valid"
        zone_df = zone_df.sort_values("Date").reset_index(drop=True)

        # ── Steps 5-6: swap zone detection ───────────────────────────────
        # BreakoutDate     : date of the candle that confirmed the swap.
        # BreakoutProximal : price edge of the breakout candle used for the
        #   proximity gate (how far price has moved since the swap was confirmed).
        #   SDZ → breakout candle High  (price must stay within X% above this)
        #   SSZ → breakout candle Low   (price must stay within X% below this)
        zone_df["BreakoutDate"]     = pd.NaT
        zone_df["BreakoutProximal"] = np.nan
        # Only "Valid" DZ/SZ rows are candidates for SDZ/SSZ promotion.
        t_swap = time.time()
        zone_df = _identify_swap_demand_zones(zone_df)
        zone_df = _identify_swap_supply_zones(zone_df)
        logger.debug("Swap step %.4fs", time.time() - t_swap)

        # ── Step 7: base elimination — runs LAST ─────────────────────────
        # Matches market-vision zone_identifier_service.analyze_zones, where
        # base_eliminator.BaseEliminator(zone_data) is the final call. SDZ/SSZ
        # zones (already promoted above) are excluded from elimination.
        zone_df = _base_eliminator(zone_df)   # → Date-indexed output

        logger.debug("analyze_zones total %.4fs", time.time() - t0)
        return zone_df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (mirrors old market-vision module-level call style)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_zones(data: pd.DataFrame) -> pd.DataFrame:
    """Module-level convenience wrapper around ZoneAnalyzer.analyze_zones."""
    return ZoneAnalyzer().analyze_zones(data)

