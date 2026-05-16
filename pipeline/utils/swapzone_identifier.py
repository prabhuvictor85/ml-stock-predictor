import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"ZoneType", "Zone", "SubType", "Close", "Open", "High", "Low", "Distal", "Proximal"}


def IdentifySwapZones(df: pd.DataFrame, zn: str) -> pd.DataFrame:
    """SDZ: SZ breaks UP → former supply becomes demand (Swap Demand Zone)."""
    if zn != "SDZ":
        return df

    n = len(df)
    if n < 3:
        return df

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        logger.warning("IdentifySwapZones: missing columns %s — skipping", missing)
        return df

    close_arr   = pd.to_numeric(df["Close"],  errors="coerce").to_numpy()
    open_arr    = pd.to_numeric(df["Open"],   errors="coerce").to_numpy()
    high_arr    = pd.to_numeric(df["High"],   errors="coerce").to_numpy()
    low_arr     = pd.to_numeric(df["Low"],    errors="coerce").to_numpy()
    subtype_arr = df["SubType"].to_numpy()

    candidates = np.flatnonzero(((df["ZoneType"] == "SZ") & (df["Zone"] == "Valid")).to_numpy())

    for i in candidates:
        if df.at[i, "ZoneType"] != "SZ" or df.at[i, "Zone"] != "Valid":
            continue
        if i + 2 >= n:
            continue

        distal   = pd.to_numeric(pd.Series([df.at[i, "Distal"]]),   errors="coerce").iat[0]
        proximal = pd.to_numeric(pd.Series([df.at[i, "Proximal"]]), errors="coerce").iat[0]
        if not np.isfinite(distal) or not np.isfinite(proximal):
            continue

        breakout_rel = np.flatnonzero(close_arr[i + 2:] > distal)
        if breakout_rel.size == 0:
            continue

        breakout_idx  = int(i + 2 + breakout_rel[0])
        breakout_type = subtype_arr[breakout_idx]
        if breakout_type in {"Drop", "Base"}:
            df.loc[i, ["ZoneType", "Zone"]] = ["SZ", "Invalid"]
            continue

        lo = min(distal, proximal)
        hi = max(distal, proximal)

        pre_high = high_arr[i + 2: breakout_idx]
        pre_low  = low_arr [i + 2: breakout_idx]
        pre_sub  = subtype_arr[i + 2: breakout_idx]

        overlaps_zone = (pre_high >= lo) & (pre_low <= hi)
        is_drop_base  = np.isin(pre_sub, ["Drop", "Base"])

        if (overlaps_zone & is_drop_base).any():
            df.loc[i, ["ZoneType", "Zone"]] = ["SZ", "Invalid"]
        else:
            df.loc[i, ["ZoneType", "Distal", "Proximal", "cndl_llinecolor", "cndl_bodycolor"]] = [
                "SDZ", low_arr[i], high_arr[i], "blue", "blue"
            ]

        if df.at[i, "ZoneType"] == "SDZ":
            distal_now = float(low_arr[i])
            if (
                np.isfinite(distal_now)
                and (
                    (close_arr[breakout_idx + 1:] < distal_now)
                    | (open_arr[breakout_idx + 1:] < distal_now)
                ).any()
            ):
                # Revert colour to the candle's natural direction (undo the blue)
                c, o = close_arr[i], open_arr[i]
                revert_body = "teal" if c > o else ("red" if c < o else "na")
                df.loc[i, ["ZoneType", "Zone", "cndl_bodycolor", "cndl_llinecolor"]] = [
                    "SZ", "Invalid", revert_body, "black"
                ]

    df.loc[df["ZoneType"] == "SDZ", "cndl_bodycolor"] = "blue"
    return df


def IdentifySwapSupplyZones(df: pd.DataFrame) -> pd.DataFrame:
    """SSZ: DZ breaks DOWN → former demand becomes supply (Swap Supply Zone).

    Exact reverse of IdentifySwapZones/SDZ:
      - Candidates  : DZ zones (demand convention  Distal < Proximal)
      - Trigger     : close < distal  (price drops below DZ bottom)
      - Breakout    : must be a Drop candle (Rally/Base simply invalidates DZ)
      - canSwap     : any Rally/Base overlapping zone between base & breakout blocks swap
      - Conversion  : Distal = candle high (supply top), Proximal = candle low (supply bottom)
      - Breach      : close or open > distal (SSZ high) → back to DZ Invalid
      - Color       : yellow
    """
    n = len(df)
    if n < 3:
        return df

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        logger.warning("IdentifySwapSupplyZones: missing columns %s — skipping", missing)
        return df

    close_arr   = pd.to_numeric(df["Close"],  errors="coerce").to_numpy()
    open_arr    = pd.to_numeric(df["Open"],   errors="coerce").to_numpy()
    high_arr    = pd.to_numeric(df["High"],   errors="coerce").to_numpy()
    low_arr     = pd.to_numeric(df["Low"],    errors="coerce").to_numpy()
    subtype_arr = df["SubType"].to_numpy()

    candidates = np.flatnonzero(((df["ZoneType"] == "DZ") & (df["Zone"] == "Valid")).to_numpy())

    for i in candidates:
        if df.at[i, "ZoneType"] != "DZ" or df.at[i, "Zone"] != "Valid":
            continue
        if i + 2 >= n:
            continue

        distal   = pd.to_numeric(pd.Series([df.at[i, "Distal"]]),   errors="coerce").iat[0]
        proximal = pd.to_numeric(pd.Series([df.at[i, "Proximal"]]), errors="coerce").iat[0]
        if not np.isfinite(distal) or not np.isfinite(proximal):
            continue

        # Breakout DOWN: close falls below DZ bottom (distal)
        breakout_rel = np.flatnonzero(close_arr[i + 2:] < distal)
        if breakout_rel.size == 0:
            continue

        breakout_idx  = int(i + 2 + breakout_rel[0])
        breakout_type = subtype_arr[breakout_idx]
        # Needs a Drop breakout; Rally/Base means DZ was simply invalidated
        if breakout_type in {"Rally", "Base"}:
            df.loc[i, ["ZoneType", "Zone"]] = ["DZ", "Invalid"]
            continue

        lo = min(distal, proximal)
        hi = max(distal, proximal)

        # canSwap: Rally/Base between base candle and breakout that overlaps zone → no swap
        pre_high = high_arr[i + 2: breakout_idx]
        pre_low  = low_arr [i + 2: breakout_idx]
        pre_sub  = subtype_arr[i + 2: breakout_idx]

        overlaps_zone = (pre_high >= lo) & (pre_low <= hi)
        is_rally_base = np.isin(pre_sub, ["Rally", "Base"])

        if (overlaps_zone & is_rally_base).any():
            df.loc[i, ["ZoneType", "Zone"]] = ["DZ", "Invalid"]
        else:
            # Supply convention: Distal = high (top) > Proximal = low (bottom)
            df.loc[i, ["ZoneType", "Distal", "Proximal", "cndl_llinecolor", "cndl_bodycolor"]] = [
                "SSZ", high_arr[i], low_arr[i], "yellow", "yellow"
            ]

        if df.at[i, "ZoneType"] == "SSZ":
            # Breach: close or open rallies back above SSZ top (Distal = high) → Invalid
            distal_now = float(high_arr[i])
            if (
                np.isfinite(distal_now)
                and (
                    (close_arr[breakout_idx + 1:] > distal_now)
                    | (open_arr[breakout_idx + 1:] > distal_now)
                ).any()
            ):
                # Revert colour to the candle's natural direction (undo the yellow)
                c, o = close_arr[i], open_arr[i]
                revert_body = "teal" if c > o else ("red" if c < o else "na")
                df.loc[i, ["ZoneType", "Zone", "cndl_bodycolor", "cndl_llinecolor"]] = [
                    "DZ", "Invalid", revert_body, "black"
                ]

    df.loc[df["ZoneType"] == "SSZ", "cndl_bodycolor"] = "yellow"
    return df

