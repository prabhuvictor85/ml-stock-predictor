[← Back to index](README.md)

# Glossary

| Term | Plain-English meaning |
|---|---|
| **ADV** | Average Daily (dollar) Volume — how much of a stock trades on a typical day; used to judge if a stock is liquid enough to trade without excessive slippage. |
| **ADX / ±DI** | Average Directional Index / Directional Indicators — a classic technical indicator measuring trend strength and direction. |
| **ATR** | Average True Range — a measure of how much a stock typically moves per day; used to normalize other features so a $5 stock and a $500 stock are comparable. |
| **Backtest** | Simulating a strategy on historical data to see how it would have performed. |
| **CHoCH / MSS (ICT)** | Change of Character / Market Structure Shift — chart-pattern concepts describing a shift in short-term trend direction. |
| **Composite score** | A blended score combining multiple signals (here, a since-questioned 85/15 blend of model score and another factor). |
| **Cross-sectional** | Comparing all stocks *on the same date* to each other, rather than comparing one stock across time. |
| **Drift (PSI)** | Population Stability Index — a statistical measure of how much a feature's distribution has shifted from what the model was trained on. |
| **Embargo window** | A buffer period after a test window, before the next training window resumes, to prevent short-term autocorrelation leakage. |
| **Ensemble** | Combining multiple models/signals into one final score. |
| **Feature** | A derived, model-ready signal computed from raw data (e.g., "20-day return normalized by volatility"). |
| **Feature store** | Where computed features are saved so they don't need to be recomputed from scratch every time. |
| **FVG (Fair Value Gap)** | An ICT concept: a price gap left by a fast, one-sided move, often treated as a level price may return to. |
| **Gradient boosting** | A machine learning technique that builds many small models in sequence, each correcting the errors of the previous ones. |
| **ICT** | "Inner Circle Trader" — a family of order-flow-inspired technical concepts (order blocks, FVGs, liquidity pools, etc.). |
| **IC (Information Coefficient)** | Correlation between predicted rank and actual outcome — the core "is the model's ranking any good" metric. |
| **LambdaRank** | A learning-to-rank training objective that optimizes ranking order (via NDCG), not just individual predictions. |
| **Leakage** | When information that wouldn't have been available at decision time accidentally influences training or evaluation, inflating results dishonestly. |
| **Lockbox** | A pre-registered, fenced holdout test — the recipe is frozen before results are seen, and the test can only be run once. |
| **NDCG** | Normalized Discounted Cumulative Gain — a ranking-quality metric that weights getting the top of the list right more heavily than the bottom. |
| **Order block (ICT)** | A candle/zone associated with a large directional move, treated as a potential support/resistance zone. |
| **Panel** | The master dataset — a table indexed by (date, ticker) holding all prices, features, and targets. |
| **PIT (Point-in-Time)** | Data that reflects what was actually known/true *at that historical moment*, not filtered through hindsight (e.g., who was actually in the index back then). |
| **Purge window** | Training rows removed near a test-window boundary because their labels overlap into the test period. |
| **Quality gate** | A rule-based filter that vetoes model picks failing simple sanity checks (e.g., technically broken trend). |
| **Rank-IC** | Information Coefficient computed on ranks (Spearman-style) rather than raw values. |
| **SHAP** | A method for explaining individual predictions by fairly attributing the outcome across contributing features. |
| **Survivorship bias** | The bias introduced by only looking at entities (stocks) that "survived" to the present, ignoring those that were delisted/failed. |
| **Walk-forward validation** | Testing a model the way it would actually be used over time — train on the past, test on the next period, then roll forward. |
| **Winsorize** | Clipping extreme values to a percentile boundary, so outliers don't dominate a calculation. |

---

**Previous:** [← 13 · Future Roadmap](13-future-roadmap.md) &nbsp;|&nbsp; **Next:** [15 · Appendix →](15-appendix.md)
