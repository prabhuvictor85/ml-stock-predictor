from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# CORE UTILITY (internal only)
# ─────────────────────────────────────────────
def _rank01(series: pd.Series) -> np.ndarray:
    """
    Stable cross-sectional rank transform.

    - NaN safe
    - tie-safe
    - constant-safe
    """
    s = series.astype(float)

    if s.isna().all():
        return np.full(len(s), 0.5)

    if s.nunique(dropna=True) <= 1:
        return np.full(len(s), 0.5)

    ranks = s.rank(method="first", pct=True)
    return np.nan_to_num(ranks.values, nan=0.5)


def _align(series: pd.Series, index: pd.Index) -> pd.Series:
    """Hard alignment guard (prevents silent leakage bugs)."""
    return series.reindex(index).astype(float)


# ─────────────────────────────────────────────
# ENSEMBLE CONFIG
# ─────────────────────────────────────────────
class EnsembleConfig:
    LGBM_WEIGHT = 0.9
    VOL_WEIGHT  = 0.1


# ─────────────────────────────────────────────
# ENSEMBLE RANKER (PUBLIC API)
# ─────────────────────────────────────────────
class EnsembleRanker:
    """
    Cross-sectional ranking ensemble: LightGBM rank (90%) + inverse-vol tilt (10%).

    A second gradient-boosting model (CatBoost) trained on the same features and the
    same target adds very little diversity — the two signals correlate at 0.85+.
    The inverse-vol tilt is a structurally different signal (risk, not alpha) and
    provides meaningful portfolio-level diversification at low weight.
    """

    def __init__(self, lgbm, config: EnsembleConfig = EnsembleConfig):
        self.lgbm = lgbm
        self.cfg  = config

    def score(
        self,
        X: pd.DataFrame,
        hist_vol_20d: Optional[pd.Series] = None,
    ) -> np.ndarray:

        idx = X.index

        # ── LGBM rank ──────────────────────────
        lgbm_raw  = pd.Series(self.lgbm.predict(X), index=idx)
        lgbm_rank = _rank01(lgbm_raw)

        # ── Inverse-vol tilt (historical vol only) ──
        if hist_vol_20d is not None:
            vol = _align(hist_vol_20d, idx)
            vol = vol.fillna(vol.median())
            vol_rank = _rank01(-vol)   # lower vol → higher rank
        else:
            vol_rank = np.full(len(idx), 0.5)

        # ── Final blend ─────────────────────────
        blend = (
            self.cfg.LGBM_WEIGHT * lgbm_rank +
            self.cfg.VOL_WEIGHT  * vol_rank
        )
        return _rank01(pd.Series(blend, index=idx))