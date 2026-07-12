[← Back to index](README.md)

# Explainability

**Level 1.** For every stock on the list, the system can show *why* — which specific signals pushed it up or down in the ranking — so a human reviewer never has to take the model's word for it.

**Level 2.** Like a teacher who doesn't just hand back a grade, but circles exactly which parts of the answer earned or lost points — you can see the reasoning, not just the verdict.

**Level 3 — Technical Deep Dive.**

**SHAP (SHapley Additive exPlanations), `pipeline/explainability/shap_explainer.py`:** computes per-feature contribution to each stock's score using Shapley values (a game-theoretic method for fairly attributing a prediction across contributing features, based on the average marginal contribution of each feature across all possible feature orderings). Provides:
- Global feature importance (beeswarm/bar plots, saved to `reports/shap_global_{market}.png`).
- Feature rank stability across CV folds (does the same feature matter consistently, or is its importance an artifact of one lucky fold).
- Per-stock explanation dictionaries: top positive/negative feature contributors for each ranked ticker.

**Why SHAP over LIME?** SHAP's Shapley-value foundation gives *consistent, additive* attributions (contributions sum exactly to the prediction minus a baseline) and has an efficient, exact implementation for tree ensembles (TreeSHAP) — a strong match for a LightGBM model. LIME's local linear-surrogate approach is more general-purpose (model-agnostic) but less exact for tree models and was not adopted here since TreeSHAP's speed/exactness advantage is directly available.

**Setup Matching (`pipeline/explainability/setup_matcher.py`):** compares a current stock's setup against historical rows sharing the same market regime and the same sign pattern among top SHAP features, returning a historical win rate for genuinely similar situations. **Design decision:** returns `insufficient_history` rather than extrapolating a win rate when fewer than 30 similar examples exist — a deliberate guard against false confidence from small samples.

**Confidence Scores:** the model's rank score itself is the primary confidence signal (higher = more confident in relative outperformance); the setup-matcher's historical win rate provides a complementary, sample-size-aware confidence read.

**Limitations:**
- SHAP explains the *model's* reasoning, not ground truth about markets — a well-explained pick can still be wrong.
- TreeSHAP importance can still reflect spurious/unstable features if selection/correlation pruning missed a redundant pair (mitigated but not eliminated by the SHAP rank-stability check in `FeatureSelector`).
- Setup-matcher win rates are descriptive of the past, not a probability guarantee for the future.

---

**Previous:** [← 09 · Operational Lifecycle](09-operational-lifecycle.md) &nbsp;|&nbsp; **Next:** [11 · Risks →](11-risks.md)
