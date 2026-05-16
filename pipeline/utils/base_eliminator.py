import logging

import numpy as np
import pandas as pd

# Setup logging for the module
logger = logging.getLogger(__name__)


def check_forward_intersections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marks base zones as invalid when their [Distal, Proximal] range overlaps
    with any future candle range or future zone range.
    """
    if not isinstance(df, pd.DataFrame):
        logger.warning("check_forward_intersections received invalid input type: %s", type(df))
        return pd.DataFrame()

    work = df.reset_index(drop=True).copy()
    n = len(work)
    if n < 3:
        return work

    lows = pd.to_numeric(work["Low"], errors="coerce").to_numpy()
    highs = pd.to_numeric(work["High"], errors="coerce").to_numpy()
    distals = pd.to_numeric(work["Distal"], errors="coerce").to_numpy()
    proximals = pd.to_numeric(work["Proximal"], errors="coerce").to_numpy()

    candidate_mask = (
        (work["Zone"] != 0) & (work["SubType"] == "Base") & (~work["ZoneType"].isin(["SDZ", "SSZ"]))
    )
    candidate_indices = np.flatnonzero(candidate_mask.to_numpy())

    for i in candidate_indices:
        if i + 2 >= n - 1:
            continue

        lo = min(distals[i], proximals[i])
        hi = max(distals[i], proximals[i])
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue

        future_slice = slice(i + 2, n - 1)

        overlap_hl = (highs[future_slice] >= lo) & (lows[future_slice] <= hi)

        f_distal = distals[future_slice]
        f_proximal = proximals[future_slice]
        f_lo = np.minimum(f_distal, f_proximal)
        f_hi = np.maximum(f_distal, f_proximal)
        overlap_dp = (f_hi >= lo) & (f_lo <= hi)

        if np.any(overlap_hl | overlap_dp):
            work.at[i, "Zone"] = "Invalid"

    return work


def BaseEliminator(DataSet):
    """
    Eliminates base zones with forward intersections.
    """
    if not isinstance(DataSet, pd.DataFrame):
        logger.warning("BaseEliminator received invalid input type: %s", type(DataSet))
        return pd.DataFrame()

    DataSet = check_forward_intersections(DataSet)
    if DataSet.empty:
        return DataSet

    DataSet = DataSet.drop_duplicates()
    DataSet = DataSet.sort_values(by="Date")
    DataSet = DataSet.set_index("Date")
    logger.info("BaseEliminator completed successfully.")
    return DataSet
