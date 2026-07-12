# ML Stock Predictor Code Explanation

This document explains the structure and behavior of the `ml-stock-predictor` codebase. The project is a market-agnostic machine learning pipeline for ranking stocks, constructing weekly portfolios, and evaluating performance through walk-forward backtests.

The code is organized around three operational workflows:

1. Training: build or load market data, engineer features, build targets, tune models, train the final model, compute explainability outputs, and save artifacts.
2. Inference: fetch a fresh recent snapshot, compute features, score the latest cross-section, construct a portfolio/watchlist, explain selections, and monitor drift.
3. Backtesting: load trained artifacts and a historical panel, score each rebalance group, simulate portfolio changes with execution costs, and generate performance reports.

## Repository Layout

```text
.
|-- main.py                         # PyCharm sample script, not part of the pipeline
|-- pipeline/
|   |-- train.py                    # Training entry point
|   |-- infer.py                    # Weekly inference entry point
|   |-- backtest_run.py             # Backtest entry point
|   |-- config/                     # Market configuration presets
|   |-- data/                       # Data fetching, universe, panel construction
|   |-- features/                   # Technical and market feature engineering
|   |-- targets/                    # Forward-return and classification target creation
|   |-- validation/                 # Walk-forward CV and ranking metrics
|   |-- models/                     # LightGBM ranker + ensemble (LGBM + inverse-vol)
|   |-- portfolio/                  # Portfolio selection and weighting
|   |-- backtest/                   # Backtest engine, execution model, performance reporter
|   |-- explainability/             # SHAP explanations and similar setup matching
|   |-- monitoring/                 # PSI drift monitor and retrain queue
|   |-- reports/                    # HTML report generator
|   `-- utils/                      # Calendar, logging, shared dataclasses
|-- test_integration.py             # Synthetic-data smoke test
|-- bench_speed.py                  # Performance benchmark helper
|-- profile_features.py             # Feature profiling helper
|-- run_nse_local.py                # Local NSE-oriented runner/script
|-- pyproject.toml                  # Package metadata and dependencies
`-- requirements.txt                # Runtime dependencies
```

## Core Concepts

### MarketConfig

`pipeline/config/base.py` defines the central `MarketConfig` dataclass. It is intended to be the single source of truth for market-specific values such as:

- market identifier, exchange calendar, benchmark ticker, and currency
- data source preferences and fallback sources
- liquidity thresholds and slippage tiers
- rebalance schedule
- risk limits
- profit target, stop loss, and drift thresholds
- random seed

The concrete presets are:

- `pipeline/config/nse.py`: NSE India, benchmark `^NSEI`, currency `INR`, calendar `XBOM`
- `pipeline/config/sp500.py`: S&P 500, benchmark `SPY`, currency `USD`, calendar `XNYS`
- `pipeline/config/nasdaq.py`: NASDAQ, benchmark `QQQ`, currency `USD`, calendar `XNAS`

`pipeline/config/__init__.py` exposes `get_config(market)`, which maps `nse`, `sp500`, and `nasdaq` to the relevant config object.

### Panel Data Schema

The pipeline uses a master pandas DataFrame called the panel. It is indexed by:

```text
MultiIndex(date, ticker)
```

The base columns are:

- `open`, `high`, `low`, `close`, `volume`
- `market_cap_usd`
- `adv_20d_usd`
- `sector`
- `in_universe`
- `group_date`

Feature columns are prefixed with:

```text
features_
```

Target columns include:

- `future_20d_return`
- `benchmark_20d_return`
- `future_20d_excess_return`
- `hit_target_20d`
- `max_drawdown_20d`
- `future_vol_20d`
- `cs_rank_20d`
- `top_quintile`
- `bot_quintile`

The `group_date` column is important. It maps each trading row to the last trading day of that ISO week and is used as the weekly ranking/rebalance group.

## Entry Points

### Training: `pipeline/train.py`

The training pipeline is the main model-building workflow.

Typical usage:

```bash
python -m pipeline.train --market sp500 --tickers_file tickers.txt
```

Important arguments:

- `--market`: one of `nse`, `sp500`, `nasdaq`
- `--start`, `--end`: historical data window
- `--n_folds`: walk-forward CV folds
- `--n_trials`: Optuna trials
- `--timeout`: Optuna timeout in seconds
- `--panel_dir`: where parquet panels are saved/loaded
- `--output_dir`: where trained artifacts are saved
- `--skip_fetch`: load an existing panel instead of fetching data
- `--tickers_file`: file containing one ticker per line

Training flow:

1. Load `MarketConfig` using `get_config`.
2. Set random seeds for reproducibility.
3. Build or load a historical panel.
4. Fetch benchmark prices.
5. Run feature engineering.
6. Build forward-looking targets.
7. Run Optuna hyperparameter optimization using purged walk-forward CV.
8. Select final features.
9. Train the final LightGBM LambdaRank model.
10. Assemble the final ensemble (LGBM rank + inverse-volatility tilt).
11. Compute SHAP global explanations where possible.
12. Fit feature drift baselines.
13. Save artifacts under `artefacts/{market}/`.

Saved artifacts include:

- `lgbm_ranker.pkl`
- `ensemble.pkl`
- `drift_monitor.pkl`
- `optuna_study_meta.json`
- `selected_features.txt`

(CatBoost, XGBoost, and the probability calibrator were removed from the
pipeline: their pickled artifacts were never loaded by any consumer, and a
second gradient-boosting model on the same features adds little diversity.
The modules are preserved under `drafts/` for reference.)

The Optuna objective is created by `make_optuna_objective`. It trains a LightGBM ranker inside each fold and optimizes a lower-confidence-bound style metric:

```text
mean_ndcg_at_10 - 0.5 * std_ndcg_at_10
```

Trials are pruned when top-decile excess return is not positive.

### Inference: `pipeline/infer.py`

The inference pipeline creates the weekly watchlist from trained artifacts.

Typical usage:

```bash
python -m pipeline.infer --market sp500 --top_n 10 --tickers_file tickers.txt
```

Inference flow:

1. Load config and artifacts from `artefacts/{market}/`.
2. Load selected feature names.
3. Build a fresh recent data snapshot.
4. Fetch benchmark data for feature engineering.
5. Engineer features on the fresh snapshot.
6. Select the latest available date.
7. Keep only rows with `in_universe == True`.
8. Score stocks with the saved ensemble.
9. Construct portfolio weights.
10. Generate SHAP-based per-stock explanations where possible.
11. Run weekly feature drift checks.
12. Save watchlist CSV and explanations JSON under `output/{market}/`.

Outputs are named with the inference date:

```text
output/{market}/watchlist_YYYYMMDD.csv
output/{market}/explanations_YYYYMMDD.json
```

A key design rule in this file is that inference builds a fresh snapshot instead of taking the tail of the training panel. This reduces the risk of accidentally using stale or training-only data.

### Backtesting: `pipeline/backtest_run.py`

The backtest runner evaluates trained artifacts on a historical panel.

Typical usage:

```bash
python -m pipeline.backtest_run --market sp500 --top_n 10
```

Backtest flow:

1. Load config and trained ensemble.
2. Load selected features.
3. Load historical panel parquet files.
4. Optionally filter the panel by start/end date.
5. Fetch benchmark prices.
6. Engineer features if missing.
7. Build targets if missing.
8. Run `BacktestEngine`.
9. Optionally generate SHAP global plot.
10. Generate HTML performance report.
11. Save equity curve parquet and performance table JSON.

Outputs:

```text
reports/report_{market}.html
reports/equity_curve_{market}.parquet
reports/performance_tables_{market}.json
reports/shap_global_{market}.png
```

## Data Layer

### `pipeline/data/fetcher.py`

`DataFetcher` handles market data retrieval.

Main methods:

- `fetch_single(ticker, start, end)`: fetches OHLCV for one ticker.
- `fetch_many(tickers, start, end)`: loops through tickers and returns a dict of DataFrames.
- `fetch_benchmark(start, end)`: fetches the configured benchmark ticker.
- `fetch_fx(start, end)`: returns a daily USD conversion series.

The module supports:

- Yahoo Finance through `yfinance`
- Polygon.io through REST
- Tiingo through REST

The primary and fallback source are controlled by `MarketConfig`.

For non-USD markets, `_fetch_fx_to_usd` currently handles INR through `USDINR=X` and falls back to a static `1/75` conversion if FX retrieval fails.

### `pipeline/data/universe.py`

`UniverseBuilder` manages the tradable universe.

It can load symbol metadata through `load_symbol_master`, storing:

- ticker
- listing date
- delisting date
- successor ticker
- sector

`build_in_universe_flags(panel)` marks rows as tradable based on:

- monthly reconstitution on the first trading day of each month
- minimum 20-day ADV
- minimum market cap
- minimum trading history
- delisting status

The output is a boolean Series aligned to the panel index.

One practical caveat: if no symbol master is loaded, sectors default to `Unknown`, and market cap may be `NaN` unless shares outstanding are passed to `PanelConstructor.build`. That can prevent securities from entering the universe because the market-cap eligibility check requires a positive value above the threshold.

### `pipeline/data/panel.py`

`PanelConstructor` builds and persists the master panel.

`build(tickers, start, end, shares_outstanding=None)`:

1. Gets official trading days from the configured exchange calendar.
2. Gets FX conversion data.
3. Fetches OHLCV for all tickers.
4. Aligns each ticker to the trading calendar.
5. Computes `adv_20d_usd` per ticker.
6. Computes `market_cap_usd` if shares outstanding are available.
7. Assigns sector.
8. Computes `in_universe`.
9. Assigns weekly `group_date`.

`save(panel, output_dir)` writes partitioned parquet files:

```text
{output_dir}/year=YYYY/part-0.parquet
```

`load(input_dir, years=None)` reads those partitions back into one sorted panel.

## Feature Engineering

### `pipeline/features/engineer.py`

`FeatureEngineer` is the central feature-building class. It receives a `MarketConfig` and benchmark close series.

The code intentionally computes rolling per-security statistics inside `groupby(level="ticker")` to avoid mixing data across tickers.

Main features:

- ATR percentile rank
- volatility contraction and compression score
- ADX
- ATR-normalized log returns over 1, 5, 20, and 60 days
- distance from 52-week high
- SMA slopes and price-vs-SMA distances
- volume ratios
- rolling beta versus benchmark
- ICT-derived order block, fair value gap, and demand/supply features
- sector relative strength
- market breadth
- benchmark regime dummies
- weekly and monthly trend features

After feature creation, all `features_*` columns are winsorized cross-sectionally per date at the 1st and 99th percentiles.

Important helper functions:

- `_rolling_beta`: rolling covariance divided by benchmark variance.
- `_winsorize_per_date`: clips feature values within each date cross-section.

### `pipeline/features/ict_features.py`

`ICTFeatureEngine` computes vectorized technical-pattern features for one ticker at a time.

It includes:

- `_wilder_atr`: Wilder-smoothed ATR.
- `_wilder_adx`: Wilder-smoothed ADX.
- fair value gap detection.
- bullish and bearish order block detection.
- demand/supply zone flags and distances.

Price levels are not exposed directly as raw features. Distances are normalized by ATR.

### `pipeline/features/multitf_merger.py`

`MultiTFMerger` adds multi-timeframe trend/volatility features to the daily panel. **Rewritten since an earlier version of this doc described it** — it no longer resamples to genuine weekly/monthly bars or uses `merge_asof`; its own docstring is explicit about this: *"❌ NO resample-based calendar TFs, ❌ NO merge_asof shifting hacks, ✅ PURE rolling-window features aligned per row."* Instead, `_rolling_trend` computes `close > rolling_SMA(window)` for four trading-day windows (`{"weekly": 20, "monthly": 60, "quarterly": 120, "yearly": 240}`), producing `weekly_trend`, `monthly_trend`, `quarterly_trend`, `yearly_trend` (plus `atr_pct`, `{tf}_vol`, `return_20d`/`return_60d`, none of which are re-exposed under `features_` and are therefore invisible to the model).

`FeatureEngineer.build()` (`pipeline/features/engineer.py`) explicitly re-exposes the four trend columns as `features_weekly_trend`/`features_monthly_trend`/`features_quarterly_trend`/`features_yearly_trend` — **these four ARE visible to the model**, unlike an earlier note in this file claimed. See [docs/architecture/05-ml-design/02-feature-engineering/09-trend.md](architecture/05-ml-design/02-feature-engineering/09-trend.md) for the full explanation, including the columns that remain genuinely unprefixed and invisible (`atr_pct`, `{tf}_vol`, `return_20d`/`return_60d`).

### `pipeline/features/zone_features.py`

This module contains a standalone `identify_zones` function for rolling demand/supply zones based on recent highs/lows and ATR. It is supplementary and is not currently wired into `FeatureEngineer.build`.

## Target Building

### `pipeline/targets/builder.py`

`TargetBuilder` creates supervised learning labels and forward-looking evaluation columns.

Main outputs:

- `future_20d_return`: 20-trading-day forward return per ticker.
- `benchmark_20d_return`: benchmark forward return for the same date.
- `future_20d_excess_return`: stock forward return minus benchmark forward return.
- `hit_target_20d`: whether profit target is hit before stop loss in the next 20 days.
- `max_drawdown_20d`: worst forward low relative to current close.
- `future_vol_20d`: annualized realized volatility over the next 20 days.
- `cs_rank_20d`: cross-sectional percentile rank of forward excess return among in-universe stocks.
- `top_quintile`: binary target for top 20%.
- `bot_quintile`: binary target for bottom 20%.

The target horizon is 20 trading days. The constants are:

```text
MAX_FORWARD_HORIZON = 20
PURGE_HORIZON = 40
```

The implementation leaves trailing rows with unavailable forward data as `NaN`.

## Validation Layer

### `pipeline/validation/cv.py`

`PurgedWalkForwardCV` implements expanding-window cross-validation with purge and embargo windows.

Defaults:

- minimum train window: 504 trading days
- test window: 126 trading days
- purge window: 40 trading days
- embargo window: 5 trading days
- minimum folds: 5

`get_fold_specs(panel)` calculates fold dates and aligns test boundaries to `group_date`.

`split(panel)` yields:

```python
(fold_spec, train_idx, test_idx)
```

`build_group_array(panel)` prepares LightGBM ranking groups by filtering to in-universe rows, sorting by `group_date`, dropping groups that are too small, and returning the group sizes array expected by LightGBM LambdaRank.

### `pipeline/validation/metrics.py`

This module defines ranking and portfolio-like validation metrics:

- `ndcg_at_k`
- `precision_at_k`
- `compute_fold_metrics`
- `_max_drawdown`

`compute_fold_metrics` evaluates each weekly `group_date`, computes ranking metrics, simulates a simple equal-weight top-N return, deducts a two-way cost estimate, and reports net Sharpe and top-decile excess return.

## Model Layer

### `pipeline/models/lgbm_ranker.py`

`LGBMRanker` wraps LightGBM LambdaRank.

It converts `cs_rank_20d` into integer ranking labels using `cs_rank_to_label`, then trains with:

```text
objective = lambdarank
metric = ndcg
ndcg_eval_at = [10]
```

Main methods:

- `fit`
- `predict`
- `predict_normalized`
- `feature_importance`

This is the primary ranking model.

### `pipeline/models/ensemble.py`

`EnsembleRanker` combines two signals:

```text
0.9 * normalized LightGBM rank score
0.1 * normalized inverse volatility signal
```

If volatility is unavailable, the volatility component is neutral at `0.5`.

The final blended score is rank-normalized to `[0, 1]`.

Historical note: earlier versions blended a calibrated CatBoost probability
and trained an XGBoost baseline. Both were removed — their artifacts were
pickled but never loaded, and a second GBM on identical features correlates
0.85+ with LGBM. The retired modules (`catboost_model.py`, `xgb_baseline.py`,
`calibrator.py`) live in `drafts/`.

## Feature Selection

### `pipeline/selection/selector.py`

`FeatureSelector` is fold-scoped and should be fit only on training data.

Selection steps:

1. Drop features with more than 5% missing values.
2. Remove highly correlated features using Spearman correlation above `0.92`.
3. Rank features using permutation importance from a small LightGBM classifier.
4. Run SHAP rank stability checks over bootstrap samples.
5. Return the top K features.

The chosen K is tuned by Optuna through the `feature_top_K` hyperparameter.

## Portfolio Construction

### `pipeline/portfolio/constructor.py`

`PortfolioConstructor` converts ranked scores into selected holdings and weights.

Rules:

- filter out stocks below `cfg.min_adv_usd`
- rank by ensemble score descending
- greedily select up to `top_n`
- enforce approximate sector cap by limiting count per sector
- use equal or inverse-vol weighting
- cap single-stock weights
- reduce high-beta positions if portfolio beta exceeds config limit

`construct(cross_section, scores)` returns:

```python
(ticker_scores, weights)
```

where `ticker_scores` maps ticker to rank score and `weights` maps ticker to final portfolio weight.

## Backtesting and Execution

### `pipeline/backtest/engine.py`

`BacktestEngine` simulates weekly portfolio construction and returns.

At each `group_date`:

1. Filter the in-universe cross-section.
2. Score stocks with the trained ensemble.
3. Build target holdings with `PortfolioConstructor`.
4. Compute returns for current holdings.
5. Generate trades needed to move from current to target weights.
6. Apply execution costs.
7. Update NAV and record weekly metrics.
8. Move target holdings into current holdings for the next period.

The design assumes signal at close T and execution at open T+1.

### `pipeline/backtest/execution.py`

`ExecutionModel` calculates trade costs and liquidity constraints.

It models:

- ADV participation cap
- tiered slippage from `MarketConfig`
- commission
- market impact when trade size exceeds 1% of ADV

`compute_trade` returns a `TradeResult` dataclass.

`apply_costs` subtracts total cost from the gross portfolio return.

### `pipeline/backtest/reporter.py`

`PerformanceReporter` converts weekly gross/net portfolio returns and benchmark returns into a `PerformanceReport`.

Metrics include:

- gross annual return
- net annual return
- gross Sharpe
- net Sharpe
- max drawdown
- Calmar ratio
- hit ratio
- top-decile excess return
- weekly and annualized turnover
- sector attribution
- gross/net/benchmark equity curves

## Explainability

### `pipeline/explainability/shap_explainer.py`

`SHAPExplainer` wraps SHAP for the LightGBM ranker.

It can:

- compute SHAP values
- calculate global feature importance
- measure feature rank stability across folds
- save global beeswarm/bar plots
- build per-stock explanation dictionaries

Per-stock explanations include the top positive and negative feature contributors for a ranked ticker.

### `pipeline/explainability/setup_matcher.py`

`SetupMatcher` compares a current stock setup against historical rows.

Matching logic:

- same market regime
- same sign pattern among the top SHAP features

If fewer than 30 similar examples are found, it returns `insufficient_history` rather than extrapolating a win rate.

## Monitoring and Retraining

### `pipeline/monitoring/drift_monitor.py`

`FeatureDriftMonitor` measures population stability index (PSI) for selected features.

Training time:

- `fit_baseline(train_panel)` computes equal-frequency bins from the training distribution.

Inference time:

- `compute_weekly_drift(current_panel, reference_date)` compares recent feature values against the training baseline.

Alerts:

- PSI above `cfg.psi_alert_threshold` logs a warning.
- PSI above `cfg.psi_retrain_threshold` sets `retrain_flag`.
- If more than 20% of monitored features breach the retrain threshold, the monitor logs a retrain trigger.

`save()` appends results to:

```text
monitoring/feature_drift.parquet
```

### `pipeline/monitoring/retrain_scheduler.py`

`RetrainingScheduler` is a file-backed retraining queue.

It writes jobs to:

```text
monitoring/retrain_queue.json
```

It can queue one pending retrain job per market and optionally process jobs using an injected `retrain_fn`.

## Reports

### `pipeline/reports/generator.py`

`ReportGenerator` builds a standalone HTML report using Chart.js.

It includes:

- metric cards
- equity curve
- drawdown chart
- rolling 26-week Sharpe
- sector attribution
- SHAP global image if available
- per-stock explanations if provided

The HTML is written to `reports/report_{market}.html` by the backtest runner.

## Utilities

### `pipeline/utils/calendar.py`

Calendar utilities centralize exchange calendar behavior through `pandas_market_calendars`.

Functions:

- `get_trading_days`
- `get_last_trading_day_of_week`
- `get_first_trading_day_of_month`
- `assign_group_dates`

### `pipeline/utils/logging.py`

`get_logger(name)` returns a logger that writes JSON-like records to stdout.

### `pipeline/utils/types.py`

Defines shared dataclasses:

- `FoldResult`
- `CVResult`
- `PortfolioSnapshot`
- `PerformanceReport`

## Testing and Diagnostics

### `tests/` (pytest suite)

The maintained test suite lives in `tests/` and runs in CI (leakage suite,
critical invariants, CV splits, feature engineering, regression guards,
zone/ICT features, stale-data guard). Run it with:

```bash
pytest tests/
```

The older root-level scripts `test_integration.py` and `smoke_test.py` were
moved to `drafts/` — they targeted the retired CatBoost/XGBoost ensemble and
are no longer maintained.

### Profiling Helpers

`bench_speed.py`, `profile_features.py`, `profile_out.txt`, and `profile_out2.txt` appear to support feature-performance profiling and benchmark comparisons. They are not required for the core training/inference/backtest workflows.

## End-to-End Data Flow

The high-level data flow is:

```text
tickers + market config
        |
        v
DataFetcher -> raw OHLCV + FX + benchmark
        |
        v
PanelConstructor -> panel indexed by (date, ticker)
        |
        v
UniverseBuilder -> in_universe flags and sectors
        |
        v
FeatureEngineer -> features_* columns
        |
        v
TargetBuilder -> future returns, ranks, quintile labels
        |
        v
PurgedWalkForwardCV + FeatureSelector + Optuna
        |
        v
LGBMRanker
        |
        v
EnsembleRanker (LGBM rank + inverse-vol tilt)
        |
        +--> Inference -> watchlist + explanations + drift checks
        |
        +--> Backtest -> execution costs + performance report
```

## Important Implementation Notes and Caveats

1. `main.py` is not part of the stock-prediction pipeline. It is the default PyCharm sample script.

2. The project relies on live data providers for real runs. For S&P 500 and NASDAQ, Polygon is the configured primary source and Tiingo is fallback. API keys may be required.

3. `PanelConstructor.build` only computes `market_cap_usd` if `shares_outstanding` is provided. Without it, market cap becomes `NaN`, and `UniverseBuilder` may exclude all tickers during eligibility checks.

4. If no `tickers_file` is supplied to training or inference, the code uses only the benchmark ticker as a demo fallback. That is not a realistic equity-selection universe.

5. (Resolved) `weekly_trend`/`monthly_trend`/`quarterly_trend`/`yearly_trend` are explicitly re-exposed with the `features_` prefix in `pipeline/features/engineer.py` and are visible to the model. The columns that genuinely remain unprefixed and excluded are `MultiTFMerger`'s `atr_pct`, `weekly_vol`/`monthly_vol`/`quarterly_vol`/`yearly_vol`, and `return_20d`/`return_60d` — see [docs/architecture/05-ml-design/02-feature-engineering/09-trend.md](architecture/05-ml-design/02-feature-engineering/09-trend.md).

6. The ensemble volatility component uses `future_vol_20d` when available. In live inference this is a forward-looking target and generally should not be available. The code falls back to a neutral volatility component when it is absent, but if the column is present during inference it could introduce leakage.

7. The backtest engine uses `future_20d_return` for holding-period returns while iterating weekly `group_date`s. That means the backtest return horizon and rebalance frequency should be reviewed carefully to avoid overlapping-period interpretation issues.

8. Several comments in source files show mojibake characters caused by an encoding issue. The code still appears readable and executable, but cleaning file encodings would improve maintainability.

9. The report HTML references Chart.js from a CDN. Viewing the report offline without internet access may prevent charts from rendering.

10. (Resolved) Probability calibration was removed from `train.py` along with the CatBoost/XGBoost models — the calibrated artifacts were never consumed by any downstream code.

## How to Add a New Market

The intended extension point is a new `MarketConfig` preset.

Steps:

1. Create `pipeline/config/{market}.py`.
2. Instantiate `MarketConfig` with the new market calendar, benchmark, currency, sources, slippage, liquidity, and risk settings.
3. Import it in `pipeline/config/__init__.py`.
4. Add it to `MARKET_CONFIGS`.
5. Ensure the data provider can fetch ticker formats for that market.
6. Provide a ticker file and, ideally, symbol master/share data for sectors and market cap.

Most downstream modules read from `MarketConfig`, so a new market should not require changes to model, feature, or backtest code unless the data source format is different.

