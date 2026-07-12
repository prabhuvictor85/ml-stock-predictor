[← Back to ML Design overview](README.md) &nbsp;|&nbsp; [← Back to index](../README.md)

# Model Evaluation

**Level 1.** We don't just ask "did the model get every stock right" — we ask "if you'd only acted on its top picks, would you have beaten the market, with enough consistency across time that it wasn't just luck."

**Level 2.** It's less like a school exam graded right/wrong on every question, and more like judging a talent scout: did the people they flagged as "top prospects" actually go on to outperform, across many different scouting seasons, not just one lucky year?

**Level 3 — Technical Deep Dive.**

Because the core task is ranking rather than binary classification, the primary evaluation metrics are **ranking and return-based**, not the classic confusion-matrix family — though a classification-style read is included below for completeness since some auxiliary quintile labels exist.

**Primary metrics (`pipeline/validation/metrics.py`, `scripts/tools/validate_lockbox.py`):**

| Metric | What it measures | Business interpretation |
|---|---|---|
| **Rank-IC (Information Coefficient)** @20d | Spearman correlation between predicted rank and realized forward excess return, per date | "On an average week, how well did the model's ordering track what actually happened next" |
| **IC t-stat (non-overlapping)** | Statistical significance of the mean IC, using non-overlapping observations to avoid autocorrelation inflating confidence | "Is this edge distinguishable from noise, honestly accounting for the fact that overlapping 20-day windows aren't independent samples" |
| **Top-decile excess return** | Average forward excess return of the top 10% ranked names | "If you'd only bought the top decile, how much would you have beaten the benchmark by" |
| **NDCG@10** | Ranking quality metric, weights correctness at the top of the list more heavily | Training-time optimization target and CV fold metric |
| **Precision@k** | Fraction of top-k picks that were actually top-performing in hindsight | Simpler, complementary read alongside NDCG |
| **Net Sharpe / Calmar / max drawdown** | Portfolio-level risk-adjusted return, after simulated execution costs | "Would this have been a good portfolio to actually hold, not just a good ranking" |

**Confusion-matrix-style read (auxiliary, quintile classification):** `top_quintile`/`bot_quintile` binary labels support a traditional precision/recall/F1/ROC-AUC read as a secondary diagnostic, but this is **not** the primary evaluation surface — the ranking metrics above are.

**Confidence thresholds / pass criteria (frozen in [PROTOCOL.md §5](../../../PROTOCOL.md)):**

| Outcome | Criteria (ALL must hold) |
|---|---|
| **PASS** | non-overlap IC t-stat > 2.0 AND mean IC > 0.02 AND top-decile excess 95% CI excludes 0 |
| **MARGINAL** | mean IC > 0 and top-decile excess > 0, but t-stat between 1 and 2 |
| **FAIL** | mean IC ≤ 0.02 or t-stat < 1 or CI includes 0 |

**Failure cases (documented, real, in-repo):**
- MODEL_C (ICT-only feature subset) walk-forward CV: mean IC = **-0.00002**, t = -0.01 — effectively zero signal on its own.
- Full 88-feature ICT v2 decomposition showed a **63% train→lockbox sign-flip rate** vs. **0%** for the 16 zone-core features — a concrete example of a feature family that looks fine in-sample but is unstable out-of-sample.
- The 85/15 `composite_score` blend was found to roughly **halve** the top-decile edge vs. pure `model_score`, with the blend's CI including zero — a documented reason the lockbox's primary verdict metric is `model_score`, not the blend.

**Business interpretation guardrail:** an in-sample pulse check (mean IC +0.0655, t=2.44, top-decile +1.47%) is explicitly labeled optimistic-by-construction in PROTOCOL.md — "the fenced number will be lower... treat the result as an upper bound on the true edge." This document and PROTOCOL.md deliberately avoid quoting the in-sample number as if it were the expected live result.

<Model Training Flow>

```mermaid
flowchart LR
    A[Panel: features + targets] --> B[PurgedWalkForwardCV.split]
    B --> C{Optuna trial}
    C -->|per fold| D[FeatureSelector.select<br/>train fold only]
    D --> E[LGBMRanker.fit<br/>lambdarank]
    E --> F[compute_fold_metrics<br/>NDCG@10, IC, top-decile excess]
    F --> G{Prune?<br/>top-decile excess <= 0}
    G -->|yes| H[Trial pruned]
    G -->|no| I[Score: mean_ndcg - 0.5*std_ndcg]
    I --> C
    C -->|best trial| J[Final FeatureSelector.select<br/>full train]
    J --> K[Final LGBMRanker.fit]
    K --> L[EnsembleRanker assemble]
    L --> M[SHAP global explanations]
    M --> N[FeatureDriftMonitor.fit_baseline]
    N --> O[Save artefacts/market/]
```

---

**Previous:** [← 03 · Learning Strategy](03-learning-strategy.md) &nbsp;|&nbsp; **Next:** [06 · Data Architecture →](../06-data-architecture.md)
