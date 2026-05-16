"""
SHAPExplainer — global and per-stock SHAP explanations (§9.1, §9.2).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.models.lgbm_ranker import LGBMRanker
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

REPORTS_DIR = Path("reports")


class SHAPExplainer:
    """
    Computes SHAP values for the LightGBM Ranker.

    Parameters
    ----------
    lgbm_model : trained LGBMRanker
    """

    def __init__(self, lgbm_model: LGBMRanker) -> None:
        self.lgbm = lgbm_model
        self._shap_values: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    def compute(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values. Returns array (n_samples, n_features)."""
        try:
            import shap
        except ImportError:
            raise ImportError("shap is required. Install: pip install shap")

        assert self.lgbm.model_ is not None, "LGBMRanker not trained."
        explainer = shap.TreeExplainer(self.lgbm.model_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X.fillna(0))
        self._shap_values = shap_values
        self._feature_names = list(X.columns)
        return shap_values

    def global_importance(self, top_k: int = 20) -> pd.DataFrame:
        """Return top-k features by mean |SHAP|."""
        assert self._shap_values is not None, "Call .compute() first."
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self._feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_k)
        return df

    def feature_rank_stability(self, shap_values_per_fold: List[np.ndarray], feature_names: List[str]) -> pd.DataFrame:
        """
        Compute SHAP rank across last 3 CV folds.
        Flag features whose rank std > 5 as 'unstable'.
        """
        all_ranks = []
        for sv in shap_values_per_fold:
            mean_abs = np.abs(sv).mean(axis=0)
            ranks = pd.Series(mean_abs, index=feature_names).rank(ascending=False)
            all_ranks.append(ranks)

        rank_df = pd.concat(all_ranks, axis=1)
        rank_df.columns = [f"fold_{i}" for i in range(len(all_ranks))]
        rank_df["rank_mean"] = rank_df.mean(axis=1)
        rank_df["rank_std"] = rank_df.std(axis=1)
        rank_df["unstable"] = rank_df["rank_std"] > 5
        return rank_df.sort_values("rank_mean")

    def plot_global(self, X: pd.DataFrame, output_path: Optional[Path] = None) -> None:
        """Generate and save beeswarm + bar chart for top-20 features."""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("shap or matplotlib not available for plots.")
            return

        if output_path is None:
            output_path = REPORTS_DIR / "shap_global.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        shap_values = self.compute(X)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        plt.sca(axes[0])
        shap.summary_plot(shap_values, X.fillna(0), show=False, max_display=20)
        axes[0].set_title("SHAP Beeswarm (Top 20)")

        plt.sca(axes[1])
        shap.summary_plot(shap_values, X.fillna(0), plot_type="bar", show=False, max_display=20)
        axes[1].set_title("SHAP Mean |Value| (Top 20)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP global plot saved to {output_path}")

    def explain_stock(
        self,
        ticker: str,
        rank: int,
        rank_score: float,
        X_row: pd.Series,
        shap_row: np.ndarray,
        regime: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Build per-stock explanation dict for §9.2.
        """
        feat_shap = list(zip(self._feature_names, shap_row))
        pos = sorted([(f, v) for f, v in feat_shap if v > 0], key=lambda x: -x[1])[:top_k]
        neg = sorted([(f, v) for f, v in feat_shap if v < 0], key=lambda x: x[1])[:top_k]

        return {
            "ticker": ticker,
            "rank": rank,
            "rank_score": float(rank_score),
            "top_positive_features": [(f, round(v, 4)) for f, v in pos],
            "top_negative_features": [(f, round(v, 4)) for f, v in neg],
            "regime": regime,
        }

