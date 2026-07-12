[← Back to index](README.md)

# Testing Strategy

The maintained suite lives in `tests/` and runs in CI:

| Test file | Purpose |
|---|---|
| `test_leakage_suite.py` | Asserts no forward-looking information crosses fold boundaries |
| `test_critical_invariants.py` | Core invariants that must never break (panel shape, index integrity, etc.) |
| `test_cv_split.py` | `PurgedWalkForwardCV` fold-boundary correctness |
| `test_feature_engineering.py` | Feature computation correctness |
| `test_ict_features.py` | ICT engine correctness |
| `test_pivot_features.py` | Pivot family default-off / truncation-invariance guarantees |
| `test_zone_features.py`, `test_zone_analyzer.py` | Zone-feature correctness |
| `test_quality_gate.py` | `pipeline/gating.py` veto logic |
| `test_regression_guards.py` | Prevents previously-fixed bugs from silently reappearing |
| `test_stale_data_guard.py` | Detects stale local CSV inputs |
| `test_paths_config.py` | `paths.yaml` / env-override resolution |
| `test_save_outputs_regression.py` | Output artifact format stability |
| `test_sources.py`, `test_sources2.py` | Data source adapter behavior |

**Unit tests:** individual feature functions, CV split logic, gating logic.
**Integration tests:** end-to-end panel → feature → target → CV flow (the retired `test_integration.py`/`smoke_test.py` in `drafts/` targeted the old CatBoost/XGBoost ensemble and are no longer maintained — current integration coverage lives in the leakage/invariant/regression suites).
**ML validation tests:** leakage suite, CV split correctness, pivot truncation-invariance (a feature family must produce identical output whether computed on the full history or a truncated prefix through each row's own date — the correctness bar for any causal, no-lookahead feature).
**Performance/load tests:** `bench_speed.py`, `profile_features.py` — informal profiling helpers, not part of the pytest gate.
**Regression tests:** `test_regression_guards.py`, `test_save_outputs_regression.py`.
**Acceptance criteria:** all `tests/` pass in CI before merge; any new experimental feature family must additionally pass its pre-registered CV gate (PROTOCOL.md §3.1 pattern) before being lockbox-eligible.

**Run command:**
```bash
pytest tests/
```

---

**Previous:** [← 11 · Risks](11-risks.md) &nbsp;|&nbsp; **Next:** [13 · Future Roadmap →](13-future-roadmap.md)
