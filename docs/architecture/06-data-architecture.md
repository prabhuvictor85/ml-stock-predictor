[← Back to index](README.md)

# Data Architecture

**Data Sources:** local CSV OHLCV (`stock_data/{nse_local, nse_tv, us}`), live fallback via yfinance/Polygon/Tiingo, PIT membership CSVs (`stock_lists/membership_sp500.csv`), sector/market-cap symbol master (optional, `load_symbol_master`).

**Schemas:**
- **Panel** (master DataFrame): `MultiIndex(date, ticker)`; base columns `open, high, low, close, volume, market_cap_usd, adv_20d_usd, sector, in_universe, group_date`; feature columns prefixed `features_*`; target columns `future_20d_return, benchmark_20d_return, future_20d_excess_return, hit_target_20d, max_drawdown_20d, future_vol_20d, cs_rank_20d, top_quintile, bot_quintile`.
- **Symbol master:** ticker, listing date, delisting date, successor ticker, sector.
- **PIT membership:** ticker, interval start/end (in-index dates).

**Validation Rules:** monthly universe reconstitution eligibility (ADV, market cap, trading history, delisting status); leakage-suite assertions (`tests/test_leakage_suite.py`); stale-data guard (`tests/test_stale_data_guard.py`); dedup fail-loud above 1% duplicate rows (2026-06-27 addition).

**Lineage:** raw CSV/live fetch → `PanelConstructor.build` → `UniverseBuilder` flags → `FeatureEngineer` → `TargetBuilder` → partitioned parquet panel → CV/selection/HPO → artifacts (`artefacts/{market}/`) tagged implicitly by git commit (PROTOCOL.md requires recording the commit hash per lockbox run for reproducibility).

**Storage:** partitioned parquet (`{output_dir}/year=YYYY/part-0.parquet`) for panels; pickled artifacts for models; JSON for metadata/explanations/drift; CSV for watchlists.

**Versioning:** implicit via git commit hash (recorded manually per PROTOCOL.md run) + `optuna_study_meta.json` capturing the HPO recipe. No automated data/model version registry today — a documented future-roadmap item.

**Data Quality Checks:** stale-data guard, dedup/duplicate-row guard, PSI drift monitor (population-level quality over time), gate calibration self-check (structural veto rate vs. baseline).

**PII Handling:** none — the system processes public market data (prices, volumes, corporate structure metadata) with no personal data. No PII governance requirements apply.

**Governance:** researcher-degrees-of-freedom ledger in [PROTOCOL.md §3.1](../../PROTOCOL.md) — every recipe-affecting change (new knob, threshold, feature family) must be logged with date, rationale, default state, and whether it's tuning-era-only or lockbox-eligible. This is the project's core data/model governance mechanism, standing in for a formal MLOps governance tool.

---

**Previous:** [← 05 · Machine Learning Design](05-ml-design/README.md) &nbsp;|&nbsp; **Next:** [07 · Technical Architecture →](07-technical-architecture.md)
