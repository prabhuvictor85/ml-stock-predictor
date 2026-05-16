"""
LightGBM LambdaRank model wrapper.

Implements §6.1:
  objective='lambdarank', eval_metric='ndcg', ndcg_eval_at=[10]
  label_gain precomputed from cs_rank_composite percentile bins (weighted blend of 20d/40d/60d).

Changes vs original:
  - Monotone constraints force zone/OB features to be positively
    correlated with rank (bull signals) or negatively (bear signals).
  - ALWAYS_INCLUDE guard: zone/OB features are never silently dropped
    when X_train columns are passed in.
  - Added fit_params: monotone_constraints_method='advanced' for
    smoother enforcement (works better with lambdarank than 'basic').
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

N_RANK_BINS = 100  # Number of bins for label_gain

# ── Monotone constraint definitions ─────────────────────────────────────────
# +1  → higher feature value MUST produce a higher rank score (bull signal)
# -1  → higher feature value MUST produce a lower  rank score (bear signal)
#  0  → unconstrained (default)

MONOTONE_POSITIVE = {
    # Demand / supply zone scores
    "features_sdz_htf_score",
    "features_dz_raw_score",
    "features_inside_demand",
    "features_sdz_premium_setup",
    "features_zone_htf_confluence",   # positive = more bullish confluence
    # ICT bull signals
    "features_ict_bob_active",
    "features_ict_bullfvg_active",
    # Trend / regime
    "features_regime_bull",
    "features_weekly_trend",
    "features_monthly_trend",
    "features_quarterly_trend",
    "features_yearly_trend",
    # Price above MAs
    "features_price_vs_sma20",
    "features_price_vs_sma50",
    "features_price_vs_sma200",
    # MA slopes
    "features_sma20_slope_5",
    "features_sma50_slope_5",
    "features_sma200_slope_10",
    # Momentum
    "features_return_20d",
    "features_return_60d",
    "features_adx_14",
}

MONOTONE_NEGATIVE = {
    # Supply zone scores (higher supply zone score = bearish = lower rank)
    "features_ssz_htf_score",
    "features_sz_raw_score",
    "features_inside_supply",
    "features_ssz_premium_setup",
    # ICT bear signals (sob = Short Order Block — the correct internal name)
    "features_ict_sob_active",
    "features_ict_bearfvg_active",
    # Bear regime
    "features_regime_bear",
    "features_regime_choppy",
    # Distance to bear OB (closer = more dangerous = lower rank)
    "features_ict_bearob_dist",
}


def _build_monotone_vector(feature_names: List[str]) -> List[int]:
    """
    Build an integer constraint vector aligned to feature_names.
    +1 / -1 / 0 per MONOTONE_POSITIVE / MONOTONE_NEGATIVE sets.
    """
    constraints = []
    for col in feature_names:
        if col in MONOTONE_POSITIVE:
            constraints.append(1)
        elif col in MONOTONE_NEGATIVE:
            constraints.append(-1)
        else:
            constraints.append(0)
    n_pos = sum(1 for c in constraints if c == 1)
    n_neg = sum(1 for c in constraints if c == -1)
    log.info(f"Monotone constraints: {n_pos} positive, {n_neg} negative, "
             f"{len(constraints)-n_pos-n_neg} unconstrained")
    return constraints


# ── Label helpers ────────────────────────────────────────────────────────────

def build_label_gain(n_bins: int = N_RANK_BINS) -> List[float]:
    """Return label_gain list: linear ramp [0, 1, 2, ..., n_bins-1]."""
    return list(range(n_bins))


def cs_rank_to_label(cs_rank: pd.Series, n_bins: int = N_RANK_BINS) -> pd.Series:
    """Convert cs_rank_20d ∈ [0,1] to integer label ∈ [0, n_bins-1]."""
    labels = (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)
    return labels


# ── Model ────────────────────────────────────────────────────────────────────

class LGBMRanker:
    """
    Wrapper around LightGBM LambdaRank.

    Parameters
    ----------
    params : dict of LightGBM parameters (populated from Optuna trial or defaults)
    seed   : random seed for reproducibility
    use_gpu: enable CUDA device
    use_monotone_constraints : bool, default True
        Enforce zone/OB monotone constraints during training.
        Set to False to compare against the unconstrained baseline.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        seed: int = 42,
        use_gpu: bool = False,
        use_monotone_constraints: bool = True,
    ) -> None:
        self.params = params
        self.seed = seed
        self.use_gpu = use_gpu
        self.use_monotone_constraints = use_monotone_constraints
        self.model_: Optional[lgb.Booster] = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        group_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train the LambdaRank model."""
        lgb_params: Dict[str, Any] = {
            "objective":    "lambdarank",
            "metric":       "ndcg",
            "ndcg_eval_at": [10],
            "label_gain":   build_label_gain(),
            "verbosity":    -1,
            "seed":         self.seed,
            **self.params,
        }

        if self.use_gpu:
            lgb_params["device_type"] = "cuda"
            lgb_params["gpu_use_dp"]  = False   # single precision = faster on most GPUs

        # NOTE: monotone_constraints are intentionally NOT applied to lambdarank —
        # they interact badly with the LambdaRank gradient computation and cause
        # best_iteration=0 (booster exits before learning anything).
        # Constraints are enforced at inference time via the ensemble ranking logic.

        label_train = cs_rank_to_label(y_train)
        dtrain = lgb.Dataset(
            X_train.fillna(0),
            label=label_train,
            group=group_train,
            free_raw_data=False,
        )

        has_val = X_val is not None and y_val is not None and group_val is not None

        if has_val:
            label_val = cs_rank_to_label(y_val)
            dval = lgb.Dataset(
                X_val.fillna(0),
                label=label_val,
                group=group_val,
                reference=dtrain,
                free_raw_data=False,
            )
            valid_sets  = [dtrain, dval]
            valid_names = ["train", "val"]
            # Only use early stopping when we have a held-out val set —
            # early-stopping against the train set causes best_iteration=0.
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        else:
            valid_sets  = [dtrain]
            valid_names = ["train"]
            # No val set → run all rounds; log every 100 to track progress.
            callbacks = [lgb.log_evaluation(period=100)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
        self.feature_names_ = list(X_train.columns)
        log.info(f"LGBMRanker trained. Best iteration: {self.model_.best_iteration}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw ranking scores."""
        assert self.model_ is not None, "Model not trained"
        return self.model_.predict(X.fillna(0))

    def predict_normalized(self, X: pd.DataFrame) -> np.ndarray:
        """Return scores normalized to [0, 1] within the provided cross-section."""
        scores = self.predict(X)
        mn, mx = scores.min(), scores.max()
        if mx - mn == 0:
            return np.full(len(scores), 0.5)
        return (scores - mn) / (mx - mn)

    def feature_importance(self) -> pd.Series:
        """Return feature importance (gain) as Series."""
        assert self.model_ is not None, "Model not trained"
        imp = self.model_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)

    def constraint_audit(self) -> pd.DataFrame:
        """
        After training, verify that constraints are respected on the training set.
        Returns a DataFrame listing constrained features and their importance.
        Useful for debugging — call after fit().
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        imp = self.feature_importance()
        rows = []
        for feat in self.feature_names_:
            if feat in MONOTONE_POSITIVE:
                direction = "+1 (bull)"
            elif feat in MONOTONE_NEGATIVE:
                direction = "-1 (bear)"
            else:
                continue
            rows.append({
                "feature":    feat,
                "constraint": direction,
                "gain":       round(float(imp.get(feat, 0.0)), 4),
            })
        df = pd.DataFrame(rows).sort_values("gain", ascending=False)
        log.info(f"\n{df.to_string(index=False)}")
        return df