"""
CatBoost Classifier — secondary ranking signal (§6.1).
Trained on top_quintile binary target.
Uses calibrated probability as secondary ranking signal.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


class CatBoostModel:
    """
    Wrapper around CatBoostClassifier.

    Parameters
    ----------
    params : dict of CatBoost parameters
    seed   : random seed
    """

    def __init__(self, params: Dict[str, Any], seed: int = 42, use_gpu: bool = False) -> None:
        self.params = params
        self.seed = seed
        self.use_gpu = use_gpu
        self.model_ = None
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Train the CatBoost classifier."""
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            raise ImportError("catboost is required. Install: pip install catboost")

        cb_params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": self.seed,
            "verbose": 0,
            "allow_writing_files": False,
            **self.params,
        }
        if self.use_gpu:
            cb_params["task_type"] = "GPU"
            cb_params.pop("thread_count", None)  # thread_count not supported on GPU
        self.model_ = CatBoostClassifier(**cb_params)
        train_pool = Pool(X_train.fillna(0), label=y_train.values)

        if X_val is not None and y_val is not None:
            eval_pool = Pool(X_val.fillna(0), label=y_val.values)
            self.model_.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
        else:
            self.model_.fit(train_pool)

        self.feature_names_ = list(X_train.columns)
        log.info("CatBoostModel trained.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of top_quintile=1."""
        assert self.model_ is not None, "Model not trained"
        return self.model_.predict_proba(X.fillna(0))[:, 1]

    def predict_normalized(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities normalized to [0, 1] within cross-section."""
        probs = self.predict_proba(X)
        mn, mx = probs.min(), probs.max()
        if mx - mn == 0:
            return np.full(len(probs), 0.5)
        return (probs - mn) / (mx - mn)

