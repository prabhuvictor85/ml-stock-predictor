[← Back to index](README.md)

# System Vision

### Current State
A Python pipeline (`pipeline/`) plus a family of market-specific "runner" scripts (`run_sp500_local.py`, `run_nse_local.py`, `run_nse_tradingv_local.py`, `run_walkforward_sp500.py`, `run_walkforward_nse.py`) that:
- ingest local CSV price history (primary path today) or live data via yfinance/Polygon/Tiingo,
- engineer ~150+ technical/structural features (classic TA, ICT order-flow concepts, floor pivots),
- train a LightGBM LambdaRank model per market/mode (`momentum` / `reversal`) via Optuna-tuned, purged walk-forward CV,
- score the current cross-section into ranked, gated watchlists (bull/bear × momentum/reversal),
- explain each pick with SHAP,
- backtest historically with an execution-cost model,
- and validate the whole recipe against a fenced, pre-registered holdout (the "lockbox").

It runs locally (Windows dev machine) and on a rented Hetzner server for longer walk-forward jobs. There is no live brokerage integration — output is CSV/JSON watchlists and HTML reports for human review.

### Future Vision
- Point-in-time universe coverage extended beyond SP500 (SP400/600, NSE bhavcopy) to remove remaining survivorship gaps.
- A confirmed, positive lockbox verdict ([PROTOCOL.md](../../PROTOCOL.md)) unlocks scaling from "research signal" to "paper-traded, timestamped live signal" — the renewable, perfectly clean test.
- Possible extension of pivot/structure feature families (currently OFF-by-default experiments) if their pre-registered CV gates pass.
- Tighter execution modeling (t+1 fills) and formal risk-model wiring, contingent on lockbox PASS.

### Scope
- Cross-sectional equity ranking (relative outperformance vs. benchmark), not price forecasting in absolute terms.
- Weekly rebalance cadence; 20-trading-day forward horizon.
- Two markets today (US large/mid-cap via SP500 universe, NSE India), extensible via config.
- Long (bull) and short (bear) candidate lists for two strategy families: momentum and reversal.

### Out of Scope
- Automated order execution / live brokerage connectivity.
- Options, futures, or any non-equity instrument.
- Intraday trading (the model operates on daily bars only).
- Portfolio-level tax/accounting logic.
- Guaranteeing profitability — this system produces a *ranked, evidence-based signal*, not a guarantee.

### Success Criteria
- **Statistical:** lockbox walk-forward non-overlapping IC t-stat > 2.0, mean IC > 0.02, top-decile excess return 95% CI excludes zero (frozen criteria, [PROTOCOL.md §5](../../PROTOCOL.md)).
- **Engineering:** no leakage-suite test failures (`tests/test_leakage_suite.py`), reproducible artifacts, self-monitoring quality gates that alarm on miscalibration rather than silently drifting.
- **Operational:** a new market can be onboarded by adding one `MarketConfig` file, not by touching model/feature/backtest code.

### Assumptions
- Historical technical/structural patterns carry *some* forward-looking information about relative performance, even if the magnitude is modest.
- Daily OHLCV data (plus benchmark and, where available, sector/market-cap metadata) is sufficient input signal — no fundamentals/news/alt-data are currently ingested.
- The user reviewing watchlists understands these are probabilistic rankings, not certainties.

### Constraints
- Local CSV data quality/coverage (dead-ticker prices, NSE bhavcopy) is incomplete — SP500 membership is wired via `--pit_universe`; dead-ticker prices (Norgate), SP400/600 membership intervals, NSE bhavcopy PIT, and an inflation experiment remain open gaps.
- Live data providers (Polygon/Tiingo/yfinance) require API keys and have rate limits; not the primary path for validated runs.
- Compute: HPO Optuna trials are reduced (`n_trials=40`) for tractability on a 2-CPU Hetzner box.
- One-shot lockbox rule: the 2024–2026 holdout window can be spent **once** — no iterative tuning against it ([PROTOCOL.md §6](../../PROTOCOL.md)).

---

**Previous:** [← Index](README.md) &nbsp;|&nbsp; **Next:** [02 · Business Context →](02-business-context.md)
