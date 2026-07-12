[← Back to index](README.md)

# System Design

**Component Diagram**

```mermaid
graph TD
    subgraph pipeline
        config[config/ MarketConfig presets]
        data[data/ Fetcher, Panel, Universe]
        features[features/ Engineer, ICT, Pivots, MultiTF]
        targets[targets/ TargetBuilder]
        selection[selection/ FeatureSelector]
        validation[validation/ CV, metrics]
        models[models/ LGBMRanker, EnsembleRanker]
        portfolio[portfolio/ Constructor]
        backtest[backtest/ Engine, Execution, Reporter]
        explainability[explainability/ SHAPExplainer, SetupMatcher]
        monitoring[monitoring/ DriftMonitor, RetrainingScheduler]
        reports[reports/ Generator]
        gating[gating.py Quality Gate]
        utils[utils/ Calendar, Logging, Types]
    end
    config --> data
    data --> features
    features --> targets
    targets --> validation
    validation --> selection
    selection --> models
    models --> gating
    gating --> portfolio
    portfolio --> backtest
    models --> explainability
    models --> monitoring
    backtest --> reports
    config -.-> features
    config -.-> portfolio
    config -.-> backtest
```

**Class Responsibilities (selected):**

| Class | Responsibility |
|---|---|
| `MarketConfig` | Single source of truth for all market-specific constants |
| `DataFetcher` | Retrieve OHLCV/benchmark/FX from configured provider |
| `PanelConstructor` | Build/persist the master (date, ticker)-indexed panel |
| `UniverseBuilder` | Compute `in_universe` eligibility flags |
| `FeatureEngineer` | Compute all `features_*` columns, per-ticker safe |
| `TargetBuilder` | Compute forward-return labels and ranks |
| `PurgedWalkForwardCV` | Expanding-window, purged, embargoed fold generator |
| `FeatureSelector` | Fold-scoped feature pruning/ranking |
| `LGBMRanker` | LambdaRank model wrapper |
| `EnsembleRanker` | Blend LGBM rank + inverse-vol tilt |
| `momentum_bull_quality_gate` | Rule-based veto of technically-broken momentum-bull picks |
| `PortfolioConstructor` | Convert scores → holdings + weights under risk limits |
| `BacktestEngine` | Simulate weekly rebalance with execution costs |
| `ExecutionModel` | ADV cap, slippage tiers, commission, market impact |
| `PerformanceReporter` | Compute Sharpe/Calmar/drawdown/turnover/attribution |
| `SHAPExplainer` | Global + per-stock explanation |
| `SetupMatcher` | Historical similar-setup matching |
| `FeatureDriftMonitor` | PSI-based drift detection |
| `RetrainingScheduler` | File-backed retrain job queue |

**State Machine — Retraining Trigger**

```mermaid
stateDiagram-v2
    [*] --> Monitoring
    Monitoring --> Monitoring: weekly PSI check, all features below alert threshold
    Monitoring --> Alerted: PSI > psi_alert_threshold (0.20) for some feature
    Alerted --> Monitoring: subsequent weeks return below threshold
    Alerted --> RetrainFlagged: PSI > psi_retrain_threshold (0.25)
    RetrainFlagged --> RetrainTriggered: >20% of monitored features breach retrain threshold
    RetrainTriggered --> Queued: RetrainingScheduler.queue_job(market)
    Queued --> Retraining: retrain_fn invoked (manual or scheduled)
    Retraining --> Monitoring: new artefacts saved, new drift baseline fit
```

**Dependency Graph (module interaction, simplified):** `config` has no internal dependencies (leaf); every other module depends on `config`; `models` depends on `validation` + `selection`; `gating` depends on `models` output columns only (feature-name coupling, not class coupling); `backtest` depends on `models` + `portfolio` + `gating`; `explainability`/`monitoring` depend on `models` but not on each other.

**Event Flow (inference run):** snapshot build → feature engineering → universe filter → scoring → gating → portfolio construction → explanation → drift check → artifact write. Each step is a synchronous, sequential batch stage — there is no async/event-driven architecture, appropriate for a weekly-cadence batch system.

---

**Previous:** [← 07 · Technical Architecture](07-technical-architecture.md) &nbsp;|&nbsp; **Next:** [09 · Operational Lifecycle →](09-operational-lifecycle.md)
