"""
Regression guard: the HTF zone resample must be period-END anchored so the
`cutoff_date` guard in compute_zone_features cannot be defeated by an
incomplete (future-containing) current period leaking into the training window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features.zone_features import _HTF_RESAMPLE


def test_zone_htf_resample_is_period_end_anchored():
    # zone_features picks the alias the installed pandas accepts: M/Q/Y on
    # <2.2, ME/QE/YE on >=2.2. Assert period-END anchoring (the leakage guard),
    # not a literal spelling — the old hardcoded "ME" assertions failed on any
    # pandas <2.2 even though behavior was correct.
    assert _HTF_RESAMPLE["1mo"] in ("M", "ME")
    assert _HTF_RESAMPLE["3mo"] in ("Q", "QE")
    assert _HTF_RESAMPLE["1y"] in ("Y", "YE")
    assert _HTF_RESAMPLE["1wk"] == "W-FRI"
    assert _HTF_RESAMPLE["1d"] is None
    # Semantic check: monthly bars must be labelled at period END, never start.
    idx = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    df = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)
    res = df.resample(_HTF_RESAMPLE["1mo"]).agg({"close": "last"}).dropna()
    assert res.index[0] == pd.Timestamp("2024-01-31"), "monthly bar not end-anchored"


def test_cutoff_guard_excludes_incomplete_period():
    """
    With a mid-May cutoff, the resample+cutoff slice used by compute_zone_features
    must NOT admit any bar whose aggregate window extends past the cutoff.
    """
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="D")
    df = pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)
    cutoff = pd.Timestamp("2024-05-20")

    res = df.resample(_HTF_RESAMPLE["1mo"]).agg({"close": "last"}).dropna()
    train = res[res.index <= cutoff]

    may31_future_value = float(df.loc["2024-05-31", "close"])
    # The incomplete May period must not be present in the training slice.
    assert not (train.index.month == 5).any(), "incomplete May period leaked past cutoff"
    assert may31_future_value not in set(train["close"].tolist())
