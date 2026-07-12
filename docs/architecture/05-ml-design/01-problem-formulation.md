[← Back to ML Design overview](README.md) &nbsp;|&nbsp; [← Back to index](../README.md)

# Problem Formulation

**Level 1 — Executive Summary.** This is not "will the stock go up or down" in isolation — it's "which stocks, relative to the others, are most likely to beat the market over the next month." That distinction (ranking vs. yes/no prediction) matches how a real portfolio decision actually works: you can only buy your top choices, not every stock you're moderately optimistic about.

**Level 2 — Plain English.** Instead of grading each stock pass/fail on a fixed test, the model is more like ranking runners in a race — it only needs to get the *order* roughly right, especially at the front and back of the pack, not the exact finishing time of every runner.

**Level 3 — Technical Deep Dive.** The core problem is formulated as **learning-to-rank**, specifically LightGBM's `lambdarank` objective, optimizing NDCG (Normalized Discounted Cumulative Gain) at cutoff 10 (`ndcg_eval_at=[10]`). The label is `cs_rank_20d` — the cross-sectional percentile rank of forward excess return within each weekly `group_date`, converted to integer relevance buckets via `cs_rank_to_label`. A secondary classifier head exists for quintile-style targets (`top_quintile`/`bot_quintile`), and forecasting-flavored auxiliary targets (`future_20d_return`, `max_drawdown_20d`, `future_vol_20d`) support the inverse-volatility ensemble tilt and risk sizing, but are not the primary training objective.

**Why ranking, not classification/regression?**
- **Classification** (e.g., "will this stock beat the benchmark, yes/no") throws away the *magnitude* and *relative* ordering information that portfolio construction actually needs — you can only hold the top N names, so getting the top of the list right matters far more than getting a borderline case's binary label right.
- **Regression** on raw forward returns is dominated by noisy, heavy-tailed absolute return magnitudes and is highly sensitive to market-wide moves that have nothing to do with stock selection skill (a rising tide lifts all boats). Ranking within each date's cross-section factors out the common market move.
- **Ranking (LambdaRank)** directly optimizes for "is the relative order at the top of the list correct," which is exactly what a top-N portfolio construction step consumes.

**Alternatives considered and rejected:**
- Plain regression on `future_20d_excess_return` — tried conceptually but abandoned because raw return regression overweights outlier tail returns and doesn't directly optimize top-of-list ordering.
- Binary quintile classification — retained as an auxiliary label but not the primary objective, for the reason above.
- Anomaly/detection framing — not applicable; this isn't about detecting rare fraud-like events, it's about relative ranking across a full, mostly "normal" population every week.
- Recommendation-system framing (collaborative filtering across "users") — not applicable; there is one "user" (the strategy) and the problem is closer to search ranking (rank documents/stocks by relevance to a query/date).

---

**Previous:** [← ML Design overview](README.md) &nbsp;|&nbsp; **Next:** [02 · Feature Engineering →](02-feature-engineering/README.md)
