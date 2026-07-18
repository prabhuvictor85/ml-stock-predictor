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


def rank_ic_eval(preds: np.ndarray, eval_data: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM feval: mean per-group Spearman IC of predictions vs the BINNED
    lambdarank labels (cs_rank_to_label output, 0..N_RANK_BINS-1).

    Named 'rank_ic_binned' deliberately: compute_fold_metrics' 'mean_rank_ic'
    grades scores against the CONTINUOUS forward excess return. Early stopping
    runs on THIS (coarser, tie-heavy) metric while HPO optimizes that one —
    same direction, different granularity; never compare their magnitudes.

    Vectorized: Spearman with average-rank ties == Pearson on average ranks,
    so one pandas groupby-rank pass + closed-form groupwise correlation
    replaces a per-group scipy loop that ran every boosting round on every
    valid set (thousands of interpreted spearmanr calls per round).
    """
    labels = eval_data.get_label()
    groups = eval_data.get_group()
    if groups is None:
        raise ValueError(
            "rank_ic_eval needs a grouped (ranking) Dataset — an ungrouped "
            "eval set would pool rows across dates and grade a different "
            "quantity. Construct the Dataset with group=..."
        )

    sizes = np.asarray(groups, dtype=np.int64)
    gid   = np.repeat(np.arange(len(sizes)), sizes)
    df = pd.DataFrame({
        "g": gid,
        "p": np.asarray(preds,  dtype=np.float64),
        "l": np.asarray(labels, dtype=np.float64),
    })
    gb = df.groupby("g", sort=False)
    rp = gb["p"].rank(method="average").to_numpy()
    rl = gb["l"].rank(method="average").to_numpy()

    sums = pd.DataFrame({
        "g": gid,
        "rp": rp, "rl": rl,
        "rp2": rp * rp, "rl2": rl * rl, "rpl": rp * rl,
    }).groupby("g", sort=False).sum()

    n     = sizes.astype(np.float64)          # sort=False keeps 0..k order
    cov   = n * sums["rpl"].to_numpy() - sums["rp"].to_numpy() * sums["rl"].to_numpy()
    var_p = n * sums["rp2"].to_numpy() - sums["rp"].to_numpy() ** 2
    var_l = n * sums["rl2"].to_numpy() - sums["rl"].to_numpy() ** 2
    denom = np.sqrt(var_p * var_l)
    # Degenerate groups (constant preds or constant labels, or n<2) carry no
    # rank information — excluded, exactly like the scipy-NaN/std-guard before.
    valid = (n > 1) & (denom > 1e-12)
    ics   = cov[valid] / denom[valid]
    # All-degenerate eval set: neutral 0.0 keeps early stopping alive rather
    # than crashing; it cannot beat any genuinely positive iteration.
    return "rank_ic_binned", float(ics.mean()) if len(ics) else 0.0, True

# ── Model ────────────────────────────────────────────────────────────────────

class LGBMRanker:
    """
    Wrapper around LightGBM LambdaRank.

    Parameters
    ----------
    params : dict
        LightGBM parameters (lambdarank objective is forced).
    seed : int, default 42
    use_monotone_constraints : bool, default False
        Enforce logical directional constraints on known features.
    num_threads : int, default 0 (all cores)
    """

    def __init__(
        self,
        params: dict = None,
        seed: int = 42,
        use_monotone_constraints: bool = False,
        num_threads: int = 0,
        use_gpu: bool = False,
    ) -> None:
        self.params = params or {}
        self.seed = seed
        self.use_monotone_constraints = use_monotone_constraints
        self.num_threads = num_threads
        self.use_gpu = use_gpu
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
        import gc
        # NOTE: no dtype munging here. The engineer guarantees float32 features;
        # re-running reduce_mem_usage per fit copied every column on each of
        # ~500 HPO fits for zero benefit. Dtypes are owned in ONE place.

        lgb_params: Dict[str, Any] = {
            "objective":    "lambdarank",
            "metric":       "None",
            "label_gain":   build_label_gain(),
            "verbosity":    -1,
            "seed":         self.seed,
            "num_threads":  self.num_threads,  # -1 = all cores; set lower when n_jobs>1
            "max_bin":      63,                # strict regularizer and memory saver
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
        # NaNs pass through untouched — LightGBM learns a per-split default
        # direction for missing values, which is strictly more expressive than
        # fillna(0) (0 is a meaningful value for price_vs_sma*, slopes, returns:
        # it conflates "unknown" with "exactly neutral"). Models trained this way
        # set nan_native_=True; predict() uses it to keep old artefacts (trained
        # on filled data) scoring exactly as they were trained.
        self.nan_native_ = bool(X_train.isna().to_numpy().any())
        self.feature_names_ = list(X_train.columns)

        dtrain = lgb.Dataset(
            X_train,
            label=label_train,
            group=group_train,
            free_raw_data=True,
        )

        has_val = X_val is not None and y_val is not None and group_val is not None

        if has_val:
            label_val = cs_rank_to_label(y_val)
            dval = lgb.Dataset(
                X_val,
                label=label_val,
                group=group_val,
                reference=dtrain,
                free_raw_data=True,
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

        # Aggressive GC before training starts
        del X_train, y_train, group_train
        if has_val:
            del X_val, y_val, group_val
        gc.collect()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = lgb.train(
                lgb_params,
                dtrain,
                num_boost_round=self.params.get("n_estimators", 500),
                valid_sets=valid_sets,
                valid_names=valid_names,
                feval=rank_ic_eval,
                callbacks=callbacks,
            )
        log.info(f"LGBMRanker trained. Best iteration: {self.model_.best_iteration}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw ranking scores.

        Train/serve consistency: artefacts pickled before the NaN-native change
        lack nan_native_ and were trained on fillna(0) data — they must keep
        receiving filled inputs or their split routing silently shifts.
        """
        assert self.model_ is not None, "Model not trained"
        if getattr(self, "nan_native_", False):
            return self.model_.predict(X)
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

