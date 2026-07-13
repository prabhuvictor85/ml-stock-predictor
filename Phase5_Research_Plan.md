# Phase 5: Institutional-Grade Feature Selection & Label Research Plan

## Critique of Current Architecture
The LightGBM models currently ingest over 100 features. While tree-based models (like LightGBM) are somewhat robust to noisy features, providing too many correlated or low-information features inherently damages out-of-sample generalization. 
Currently, the pipeline:
1. Keeps all ICT, SMC, and traditional technical features without rigorously pruning redundant factors.
2. Experiences extreme memory pressure (OOMs) during panel builds due to the sheer size of the feature space.
3. Uses a static classification scheme or continuous rank labels without systematically evaluating alternative target constructions (like varying return horizons).

## Objective
Identify the maximal-information, minimal-size subset of orthogonal features that maintains or improves the Rank IC against the `future_20d_excess_return` target. Develop a streamlined feature registry and prepare for the ultimate model tuning phase.

## Prioritized Experiment Backlog

### Exp-501: Feature Redundancy & VIF Pruning
- **Description:** Run a robust collinearity analysis. Compute the Variance Inflation Factor (VIF) and Pairwise Spearman Correlation across the entire cross-section.
- **Hypothesis:** Many standard SMA distance metrics (e.g., `price_vs_sma20`, `sma20_slope_5`) are highly correlated with new phase 4 residual momentum and VWAP distance predictors. Removing the collinear, less predictive counterparts will improve training stability.
- **Action:** 
  1. Correlate Phase 3 (ICT/Zones) with Phase 4 (Residual Mom/Microstructure).
  2. Drop simple technicals (`price_vs_sma`) if they are $>0.8$ correlated with robust replacements.

### Exp-502: Tree-Based Split Gain & SHAP Nullification
- **Description:** Train an ablation LightGBM model utilizing out-of-sample validation folds. Extract normalized split-gain and SHAP importance stability.
- **Hypothesis:** Features that consistently receive zero (or near zero) splits across multiple training folds, or whose SHAP values cluster strictly around 0, impart no continuous gradient for the trees to exploit and only act as noise.
- **Action:** 
  1. Train models across multiple walk-forward, purged, out-of-sample folds.
  2. Extract SHAP values on the out-of-sample test slice for each fold.
  3. Compute OOS SHAP magnitude and stability (mean / std) across folds to accurately identify trailing noise instead of relying on in-sample split gain.
  4. Prune unstable or near-zero SHAP impact features.

### Exp-503: Orthogonalization of Structural Features
- **Description:** Investigate the distribution of binary dummy features (e.g., `ict_bullfvg_active`) vs continuous features (Zone distance/strength).
- **Hypothesis:** LightGBM splits binary sparse flags poorly compared to continuous scaled metrics. If binary indicators like ChoCH/BOS don't provide isolated SHAP impact outside of the continuous zone proximity scores, they can be pruned.
- **Action:** Convert strict boolean state trackers to time-since-activation or exponentially decaying continuous variables, and drop the underlying binaries.

### Exp-504: Label Target Horizon Diagnostics
- **Description:** The current baseline utilizes `target_ret20` (20-day returns) as the primary LightGBM lambda rank target. Evaluate 5-day and 10-day targets.
- **Hypothesis:** Microstructure features (VWAP, Order blocks) have a naturally shorter half-life than macro regime filters. Predicting a 5-day or 10-day horizon might align better with the feature decay profiles, maximizing IC before the signal is swamped by market noise.
- **Action:** Evaluate model performance by swapping the target variable and comparing Walk-Forward Rank-IC across the horizons.

## Success Criteria for Phase 5
1. **Dimensionality Reduction:** Reduce the total feature count by at least 30-40% without heavily degrading the validation Rank IC.
2. **Train Speed:** Prove a tangible reduction in LightGBM Dataset compilation and C++ histogram binning time due to the smaller, cleaner dataset.
3. **Target Selection:** Conclusively select the optimal target horizon (5d, 10d, or 20d) that yields an ICIR > 1.0.

## Next Steps
Once approved, we will begin with **Exp-501** and **Exp-502** to algorithmically extract the SHAP importance distributions and prune the bottom quintile of trailing noise. We will log the pruned subset inside `pipeline/features/schema_lock.py` to freeze the model state for Phase 6.
