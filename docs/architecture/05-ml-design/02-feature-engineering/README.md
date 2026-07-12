[← Back to ML Design overview](../README.md) &nbsp;|&nbsp; [← Back to index](../../README.md)

# Feature Engineering

**Level 1.** The model doesn't look at raw prices directly — it looks at *derived signals* that describe a stock's behavior relative to itself and its peers: how strong its recent trend is, how compressed its volatility is, whether it's near a historically significant price level, and how it compares to its sector and the broader market.

**Level 2.** This is exactly like a chef prepping ingredients before cooking: raw vegetables (prices) aren't thrown in the pot whole — they're washed, chopped, and combined into something the recipe (the model) can actually use. A "$5 stock up $0.50" and a "$500 stock up $50" are the same 10% move, but only after you normalize do they look comparable — that's the chopping step.

**Level 3 — Technical Deep Dive.** All feature computation happens inside `pipeline/features/engineer.py`'s `FeatureEngineer.build()`, per ticker (`groupby(level="ticker")`), so a rolling window can never accidentally span two different stocks. Ten distinct feature *families* are computed, each detailed in its own file below because each has its own mechanics worth understanding on its own terms:

| Family | What it captures | Deep dive |
|---|---|---|
| **ATR** | "How much does this stock normally move in a day" — the ruler every other family divides by | [01 · ATR](01-atr.md) |
| **ADX / DI** | Trend strength, split by direction (bulls vs. bears in control) | [02 · ADX & Directional Movement](02-adx.md) |
| **SMA** | Trend direction/slope, how stretched price is from its moving averages, 52-week range, breakout flags | [03 · Simple Moving Averages](03-sma.md) |
| **Volume** | Is participation confirming the price move, or is it drifting on thin volume | [04 · Volume](04-volume.md) |
| **Zones** | Supply/demand levels from candlestick-pattern base detection (DZ/SZ/SDZ/SSZ) | [05 · Zones](05-zones.md) |
| **ICT** | Order blocks, fair value gaps, breaker blocks, liquidity sweeps, BOS/CHoCH, premium/discount | [06 · ICT (Order Flow)](06-ict.md) |
| **Pivots** | Floor pivots, Central Pivot Range (CPR), Camarilla levels (experimental, OFF by default) | [07 · Pivots](07-pivots.md) |
| **Returns / Momentum** | The stock's own 1d/5d/20d/60d log return, ATR-normalized | [08 · Returns & Momentum](08-returns-momentum.md) |
| **Trend** | Multi-timeframe (weekly/monthly/quarterly/yearly) "above its own moving average" flags | [09 · Trend](09-trend.md) |
| **Market Regime & Context** | Benchmark bull/choppy/bear regime, sector relative strength, market breadth | [10 · Market Regime & Context](10-market-regime-context.md) |

All seven families share the same normalization convention: **every distance-from-a-level feature is expressed in ATR units** (`(price − level) / ATR`), and every log-return feature is normalized by *percentage* ATR (`ATR / close`) rather than absolute ATR — see [ATR](01-atr.md) for why the distinction matters. This is what makes a feature computed on a $5 stock directly comparable to the same feature on a $500 stock.

---

## Cross-Cutting Concerns (apply to all seven families)

### Normalization/Scaling
ATR-normalization (distances/returns divided by Average True Range) makes features comparable across tickers of vastly different price/volatility levels — this is the single most important scaling decision in the pipeline (see [ATR](01-atr.md)). **Cross-sectional winsorization** clips every `features_*` column at the 1st/99th percentile *per date* (not globally) via `_winsorize_per_date`, so a single-day earnings-gap outlier in one stock can't dominate that day's cross-section, while still preserving cross-date regime differences (unlike global winsorization, which would smear a 2020-crash-era distribution across calmer years).

**Discrete features are exempt from winsorizing.** Binary flags, signs, and zone-priority codes (e.g. `adx_bull`/`adx_bear`, `zone_active`, ICT active flags) are detected automatically (`_is_discrete_feature`: ≤6 unique values, all in `{-1, -0.5, 0, 0.5, 1, 2, 3, 4}`) and skipped. Clipping a 0/1 column at the per-date 99th percentile would erase rare-but-informative flags outright — e.g. 10 of 1,500 stocks flagging a yearly SSZ puts the 99th percentile at 0, turning all ten 1s into 0s, destroying the signal precisely on the dates it's rarest and most informative.

### Encoding
Sector is used as a categorical grouping key for relative-strength features and sector-cap portfolio construction rather than one-hot encoded into the tree model directly (tree models split on raw categorical/ordinal values natively via LightGBM's categorical feature support where used).

### Feature Selection (`pipeline/selection/selector.py`)
1. Drop features with >5% missing values.
2. Remove features with Spearman correlation above 0.92 (redundancy pruning).
3. Rank by permutation importance from a small LightGBM classifier.
4. SHAP rank-stability check over bootstrap samples (drop features whose importance rank is unstable).
5. Return top-K, where K is itself an Optuna-tuned hyperparameter (`feature_top_K`).

**Why this selection process?** Fold-scoped (fit only on training data within each CV fold) to prevent selection leakage — a feature "discovered" using test-period information would inflate CV metrics dishonestly. Spearman (rank) correlation, not Pearson, because tree-model splits are order-based, not linear-relationship-based — feature redundancy for a tree model is about rank redundancy, not linear redundancy.

### Missing Data Strategy
LightGBM's native NaN handling for tree splits is used directly (it learns default split directions for missing values) rather than mean/median imputation, avoiding the distortion imputation can introduce into a nonlinear model. Ranker-label NaNs (unknowable forward return) are **dropped from training**, not zero-filled — a fix (2026-06-26) after zero-filling was found to corrupt the LambdaRank label by scoring unknowable-outcome rows as the worst possible rank.

### Outlier Handling
Cross-sectional per-date winsorization (see above) rather than fixed absolute clipping thresholds, because "extreme" is a relative, regime-dependent concept in markets — a 5% single-day move is unremarkable during a volatile regime and extreme during a calm one. Additionally, distance-style features (zone/ICT/pivot) are hard-clipped to **±20 ATR** as a defensive cap: beyond that a "distance" is saturated and meaningless, and it's a safety net against any residual near-zero-ATR explosion on illiquid tickers.

### Design decisions / alternatives / trade-offs
- **Chose:** per-ticker groupby computation for all rolling stats — **why:** prevents accidental cross-ticker leakage (e.g., a rolling window silently spanning two tickers' rows). **Alternative rejected:** vectorized whole-panel rolling ops without groupby boundaries — faster but leak-prone; correctness over raw speed.
- **Chose:** experimental feature families (pivots, structure) gated OFF by default behind env vars, with bit-identical verification tests. **Why:** lets research proceed without risking silent recipe drift in production runs. **Trade-off:** more surface area / knobs to track (see the [PROTOCOL.md](../../../../PROTOCOL.md) changelog §3.1), each logged as a researcher-degrees-of-freedom ledger entry.

### Common Pitfalls
- Forgetting the `features_` prefix convention silently drops a column from model training. **This currently affects `MultiTFMerger`'s `atr_pct`, `weekly_vol`/`monthly_vol`/`quarterly_vol`/`yearly_vol`, and `return_20d`/`return_60d` columns** (`pipeline/features/multitf_merger.py`) — they are computed and joined into the panel but never re-exposed under `features_*`, so the model never sees them. (An older version of this note incorrectly named `weekly_trend`/`monthly_trend` here — `engineer.py` now explicitly re-exposes those four as `features_weekly_trend` etc.; see [Trend](09-trend.md).)
- Enabling an experimental feature family without running its pre-registered CV gate first — this is explicitly forbidden by the ledger process in PROTOCOL.md.
- Using `future_vol_20d` at inference time (forward-looking target column) — could leak if accidentally present; the ensemble correctly falls back to neutral (0.5) when absent, but presence during live inference should never happen.
- **Per-fold recompute asymmetry:** zone and ICT features are recomputed per CV fold with a `cutoff_date` guard (`recompute_fold_features`) because their state is *retroactively revised* by future price action (a zone can be invalidated, a swing can un-confirm). Pivot features are **not** recomputed per fold — they are pure trailing functions of OHLC through each row's own date and are truncation-invariant by construction (`tests/test_pivot_features.py::test_truncation_invariance`). Treating pivots like zones/ICT (recomputing them unnecessarily) would waste compute; treating zones/ICT like pivots (skipping recompute) would leak.

### Best Practices
Run the pre-existing test suite (`tests/test_feature_engineering.py`, `test_ict_features.py`, `test_pivot_features.py`, `test_zone_features.py`, `test_leakage_suite.py`) after any feature change; consult the `feature-gates` skill before touching gate thresholds or `engineer.py`.

---

**Previous:** [← 01 · Problem Formulation](../01-problem-formulation.md) &nbsp;|&nbsp; **Next:** [03 · Learning Strategy →](../03-learning-strategy.md)
