"""
SetupMatcher — finds historical similar setups and computes win-rate (§9.2).

Similar setup definition:
  (a) regime_label matches current
  (b) sign pattern of top-5 SHAP features matches historical row

Rules:
  - If n_similar < 30: report 'insufficient_history'. Never extrapolate.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

MIN_SIMILAR_SAMPLES = 30


class SetupMatcher:
    """
    Matches current stock setup to historical occurrences.

    Parameters
    ----------
    historical_panel   : full historical panel with SHAP values and targets
    shap_values        : SHAP value array aligned to historical_panel rows
    feature_names      : list of feature names
    """

    def __init__(
        self,
        historical_panel: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> None:
        self.panel = historical_panel.copy()
        self.shap_values = shap_values
        self.feature_names = feature_names
        # Precompute regime label per row
        if "features_regime_bull" in historical_panel.columns:
            bull = historical_panel["features_regime_bull"].fillna(0).values
            choppy = historical_panel["features_regime_choppy"].fillna(0).values
            bear = historical_panel["features_regime_bear"].fillna(0).values
            self._regimes = np.where(bull > 0.5, "bull", np.where(bear > 0.5, "bear", "choppy"))
        else:
            self._regimes = np.full(len(historical_panel), "unknown")

    def match(
        self,
        current_regime: str,
        top5_shap: List[Tuple[str, float]],
        ticker: str,
    ) -> Dict[str, Any]:
        """
        Find similar historical setups and compute hit rate.

        Parameters
        ----------
        current_regime : 'bull', 'choppy', or 'bear'
        top5_shap      : list of (feature_name, shap_value) for top-5 features
        ticker         : current ticker (for context only)

        Returns
        -------
        dict with keys: n, historical_hit_rate, min_required or 'insufficient_history'
        """
        top5_feature_names = [f for f, v in top5_shap]
        top5_signs = np.array([np.sign(v) for _, v in top5_shap])

        # Filter by regime
        regime_mask = self._regimes == current_regime
        if regime_mask.sum() == 0:
            return {"n": 0, "historical_hit_rate": None, "min_required": MIN_SIMILAR_SAMPLES,
                    "note": "insufficient_history"}

        regime_indices = np.where(regime_mask)[0]
        shap_regime = self.shap_values[regime_mask]
        panel_regime = self.panel.iloc[regime_mask]

        # Get indices of top-5 features
        feat_idx = []
        for fn in top5_feature_names:
            if fn in self.feature_names:
                feat_idx.append(self.feature_names.index(fn))
        if not feat_idx:
            return {"n": 0, "historical_hit_rate": None, "min_required": MIN_SIMILAR_SAMPLES,
                    "note": "insufficient_history"}

        # Sign pattern match
        hist_signs = np.sign(shap_regime[:, feat_idx])   # (n_hist, 5)
        matches = np.all(hist_signs == top5_signs[:len(feat_idx)], axis=1)
        n_similar = int(matches.sum())

        if n_similar < MIN_SIMILAR_SAMPLES:
            return {
                "n": n_similar,
                "historical_hit_rate": None,
                "min_required": MIN_SIMILAR_SAMPLES,
                "note": "insufficient_history",
            }

        # Hit rate: future_20d_excess_return > 0
        exc_rets = panel_regime.loc[matches, "future_20d_excess_return"] if "future_20d_excess_return" in panel_regime.columns else pd.Series(dtype=float)
        exc_rets = exc_rets.dropna()
        if len(exc_rets) == 0:
            return {"n": n_similar, "historical_hit_rate": None, "min_required": MIN_SIMILAR_SAMPLES,
                    "note": "insufficient_history"}

        hit_rate = float((exc_rets > 0).mean())
        return {
            "n": n_similar,
            "historical_hit_rate": round(hit_rate, 4),
            "min_required": MIN_SIMILAR_SAMPLES,
        }

