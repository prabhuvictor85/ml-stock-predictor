[← Back to index](../README.md)

# Machine Learning Design

This is the largest cluster in the architecture document, split into four focused files:

| # | Section | Covers |
|---|---|---|
| 1 | [Problem Formulation](01-problem-formulation.md) | Why this is a ranking problem, not classification/regression/detection |
| 2 | [Feature Engineering](02-feature-engineering/README.md) | Raw/derived features split into 10 families (ATR, ADX, SMA, Volume, Zones, ICT, Pivots, Returns/Momentum, Trend, Market Regime & Context), each with candlestick diagrams and exact formulas |
| 3 | [Learning Strategy](03-learning-strategy.md) | Model choice (LightGBM LambdaRank), training pipeline, CV, HPO, regularization |
| 4 | [Model Evaluation](04-model-evaluation.md) | Rank-IC, NDCG, top-decile excess return, documented failure cases, pass/fail bar |

---

**Previous:** [← 04 · Functional Design](../04-functional-design.md) &nbsp;|&nbsp; **Next:** [06 · Data Architecture →](../06-data-architecture.md)
