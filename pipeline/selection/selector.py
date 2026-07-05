"""
FeatureSelector — fold-scoped feature selection (§5.3).

Steps:
  1. Missing filter: drop features with >5% NaN.
  2. Correlation clustering: drop lower-importance feature in pairs with |ρ| > 0.92.
  3. Permutation importance: rank features on 20% held-out train slice.
  4. SHAP stability: flag unstable features (rank std > 10 across 3 bootstraps).
  5. Return top-K features (K is a hyperparameter).

RULE: Selection state is fold-local. Feature sets may differ across folds.

Changes vs original:
  - ALWAYS_INCLUDE set: zone/OB features are force-included after selection so
    they are never silently dropped by correlation pruning or permutation rank.
  - IMPORTANCE_BOOST: zone features get a synthetic importance multiplier in
    Step 2 so they win deduplication ties against correlated momentum features.
  - ALWAYS_INCLUDE features are placed at the front of the final list so they
    are visible to monotone constraint enforcement in LGBMRanker.
  - top_k is expanded by the number of forced features so total feature count
    does not shrink (i.e. you still get top_k *non-forced* features on top).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# ── Zone / OB features that must always be selected ─────────────────────────
# Edit this set to control which features are protected from elimination.
ALWAYS_INCLUDE: set = {
    # Zone proximity — #1 feature by importance; has >5% NaN (stocks not near any
    # zone get NaN) so must be protected from the NaN-drop gate in Step 1.
    "features_zone_dist_atr",
    # Demand zone scores
    "features_sdz_htf_score",
    "features_dz_raw_score",
    "features_zone_htf_confluence",
    # Per-TF SDZ flags (weekly+) — protected because user-validated: weekly+ zones trigger runs
    "features_sdz_1wk",
    "features_sdz_1mo",
    "features_sdz_3mo",
    "features_sdz_1y",
    # Supply zone scores
    "features_ssz_htf_score",
    "features_sz_raw_score",
    # Per-TF SSZ flags (weekly+) — symmetric to SDZ
    "features_ssz_1wk",
    "features_ssz_1mo",
    "features_ssz_3mo",
    "features_ssz_1y",
    # sdz_premium_setup / ssz_premium_setup removed from ALWAYS_INCLUDE:
    # redesigned as a continuous tiered score (Y=0.40 > Q=0.30 > M=0.20 > W=0.10)
    # — real variance now, FeatureSelector evaluates them on merit.
    # ICT Bull Order Block (OB)
    "features_ict_bob_active",
    "features_ict_bob_atr_dist",         # ATR-normalised distance from OB midpoint
    # ICT Bear Order Block (SOB — Short Order Block)
    "features_ict_sob_active",
    "features_ict_sob_atr_dist",         # ATR-normalised distance from SOB midpoint
    # ICT Breaker Blocks (highest-priority ICT signals)
    "features_ict_bullrb_active",
    "features_ict_bullrb_atr_dist",
    "features_ict_bearrb_active",
    "features_ict_bearrb_atr_dist",
    # ICT Fair Value Gaps
    "features_ict_bullfvg_active",
    "features_ict_bullfvg_atr_dist",
    "features_ict_bearfvg_active",
    "features_ict_bearfvg_atr_dist",
    # Liquidity sweeps
    "features_ict_bsl_swept",
    "features_ict_ssl_swept",
    # ICT HTF composite scores
    "features_ict_bull_htf_score",
    "features_ict_bear_htf_score",
}

# Importance multiplier applied to ALWAYS_INCLUDE features during correlation
# deduplication (Step 2). Prevents zone features being dropped in favour of
# correlated momentum features.  2.5× is a reasonable starting point.
IMPORTANCE_BOOST: float = 2.5


class FeatureSelector:
    """
    Fold-scoped feature selector.

    Usage:
        selector = FeatureSelector(seed=42)
        selected = selector.select(X_train, y_train, top_k=30)
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.selected_features_: List[str] = []
        self.unstable_features_: List[str] = []
        self.forced_features_: List[str] = []   # zone features that were force-included

    def select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        top_k: int = 30,
        groups: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Run the full 4-step selection on training fold data.

        Returns list of selected feature names with zone/OB features guaranteed
        to appear at the front regardless of their permutation rank.
        """
        features = list(X_train.columns)
        log.info(f"FeatureSelector starting with {len(features)} features, top_k={top_k}")

        # Identify which ALWAYS_INCLUDE features are actually present in X_train
        present_forced = [f for f in ALWAYS_INCLUDE if f in X_train.columns]
        missing_forced = [f for f in ALWAYS_INCLUDE if f not in X_train.columns]
        if missing_forced:
            log.warning(f"ALWAYS_INCLUDE features absent from X_train "
                        f"({len(missing_forced)}): {missing_forced}")

        # ── Step 1: Missing filter ─────────────────────────────────────────
        nan_frac = X_train.isna().mean()
        # Never drop ALWAYS_INCLUDE features on NaN grounds — impute instead
        keep_nan = nan_frac[nan_frac <= 0.05].index.tolist()
        keep_nan_forced = [f for f in present_forced if f not in keep_nan]
        if keep_nan_forced:
            log.warning(f"Step 1: keeping {len(keep_nan_forced)} zone features despite >5% NaN "
                        f"(will be imputed): {keep_nan_forced}")
        keep = list(dict.fromkeys(keep_nan + keep_nan_forced))  # deduplicated, order preserved

        removed_nan = set(features) - set(keep)
        if removed_nan:
            log.info(f"Step 1 removed {len(removed_nan)} features with >5% NaN: "
                     f"{list(removed_nan)[:5]}...")
        features = keep
        # NaN kept (not filled): selection must judge features under the same
        # native-NaN regime the final LGBM trains with. float32 halves memory.
        X = X_train[features].astype("float32")

        # ── Step 2: Correlation clustering ────────────────────────────────
        # Compute quick importance; apply boost to zone/OB features so they
        # win any deduplication contest against correlated momentum features.
        quick_imp = self._quick_importance(X, y_train)
        quick_imp = {
            f: (v * IMPORTANCE_BOOST if f in ALWAYS_INCLUDE else v)
            for f, v in quick_imp.items()
        }

        corr_matrix = X.corr(method="spearman").abs()
        to_drop: set = set()
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if j <= i:
                    continue
                if corr_matrix.loc[f1, f2] > 0.92:
                    imp1 = quick_imp.get(f1, 0)
                    imp2 = quick_imp.get(f2, 0)
                    loser = f2 if imp1 >= imp2 else f1
                    winner = f1 if loser == f2 else f2
                    # Never drop a forced feature regardless of importance.
                    # If both are forced, skip — keep both (correlation accepted).
                    if loser in ALWAYS_INCLUDE:
                        if winner in ALWAYS_INCLUDE:
                            continue
                        loser = winner
                    to_drop.add(loser)

        features = [f for f in features if f not in to_drop]
        if to_drop:
            log.info(f"Step 2 removed {len(to_drop)} correlated features")
        X = X[features]

        # ── Step 3: Permutation importance ────────────────────────────────
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_train, test_size=0.2, shuffle=False
        )
        fast_model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            random_state=self.seed,
            n_jobs=-1,
            verbosity=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fast_model.fit(X_tr, y_tr)

        # Cap validation rows to 50k to avoid OOM in parallel permutation workers.
        # n_jobs=1 prevents multiple copies of X_val being allocated simultaneously.
        _PERM_MAX_ROWS = 50_000
        if len(X_val) > _PERM_MAX_ROWS:
            rng_perm = np.random.default_rng(self.seed)
            perm_idx = rng_perm.choice(len(X_val), size=_PERM_MAX_ROWS, replace=False)
            X_val_perm = X_val.iloc[perm_idx]
            y_val_perm = y_val.iloc[perm_idx]
        else:
            X_val_perm, y_val_perm = X_val, y_val

        perm_result = permutation_importance(
            fast_model, X_val_perm, y_val_perm, n_repeats=3, random_state=self.seed, n_jobs=1
        )
        imp_mean = perm_result.importances_mean

        # Apply boost to zone features in permutation ranking too
        boosted_imp = np.array([
            v * IMPORTANCE_BOOST if features[i] in ALWAYS_INCLUDE else v
            for i, v in enumerate(imp_mean)
        ])

        ranked_indices  = np.argsort(boosted_imp)[::-1]
        ranked_features = [features[i] for i in ranked_indices]

        # ── Step 4: SHAP stability (3 bootstrap resamplings) ──────────────
        shap_ranks = self._bootstrap_shap_ranks(X_tr, y_tr, ranked_features, n_boots=3)
        self.unstable_features_ = []
        for feat in ranked_features:
            ranks = shap_ranks.get(feat, [])
            # Never flag ALWAYS_INCLUDE features as unstable — they stay regardless
            if feat in ALWAYS_INCLUDE:
                continue
            if len(ranks) >= 2 and np.std(ranks) > 10:
                self.unstable_features_.append(feat)
        if self.unstable_features_:
            log.warning(f"Step 4: {len(self.unstable_features_)} unstable features "
                        f"(rank std > 10): {self.unstable_features_[:5]}")

        # ── Select top-K ──────────────────────────────────────────────────
        # Split ranked list: forced vs ordinary
        forced_present   = [f for f in ranked_features if f in ALWAYS_INCLUDE]
        ordinary_ranked  = [f for f in ranked_features if f not in ALWAYS_INCLUDE]

        # Take top_k ordinary features (zone features get their own slots on top)
        selected_ordinary = ordinary_ranked[:top_k]
        self.forced_features_ = forced_present

        # Final list: forced features first, then ordinary top-K
        # deduplicate just in case (shouldn't happen, but safe)
        final = list(dict.fromkeys(forced_present + selected_ordinary))

        self.selected_features_ = final
        log.info(
            f"Selected {len(final)} features  "
            f"({len(forced_present)} zone/OB forced  +  {len(selected_ordinary)} top-{top_k} ordinary)"
        )
        return final

    def _quick_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Fast importance estimate using LightGBM gain."""
        model = lgb.LGBMClassifier(
            n_estimators=50, num_leaves=20, random_state=self.seed, verbosity=-1, n_jobs=-1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        return dict(zip(X.columns, model.feature_importances_))

    def _bootstrap_shap_ranks(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str],
        n_boots: int = 3,
    ) -> Dict[str, List[int]]:
        """Compute SHAP ranks across n_boots bootstrap resamplings."""
        try:
            import shap
        except ImportError:
            log.warning("shap not installed, skipping SHAP stability check")
            return {}

        ranks: Dict[str, List[int]] = {f: [] for f in features}
        rng = np.random.default_rng(self.seed)
        _SHAP_MAX_ROWS = 30_000
        shap_idx_base = rng.choice(len(X), size=min(len(X), _SHAP_MAX_ROWS), replace=False)
        X_shap_base = X.iloc[shap_idx_base]
        for b in range(n_boots):
            idx = rng.choice(len(X_shap_base), size=len(X_shap_base), replace=True)
            X_b = X_shap_base.iloc[idx]
            y_b = y.iloc[shap_idx_base[idx]]
            model = lgb.LGBMClassifier(
                n_estimators=50, num_leaves=20, random_state=self.seed + b, verbosity=-1, n_jobs=-1
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_b, y_b)
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_shap_base)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # positive class
            mean_abs = np.abs(shap_vals).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            for rank_pos, feat_idx in enumerate(order):
                feat = features[feat_idx]
                ranks[feat].append(rank_pos)
        return ranks