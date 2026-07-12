[← Back to index](README.md)

# Future Roadmap

**Phase 1 (in progress / near-term):**
- Close the SP400/600 PIT membership gap and NSE bhavcopy PIT gap.
- Resolve dead-ticker price coverage (Norgate) so the ~139 dead ex-members currently missing price data are no longer under-represented in the lockbox.
- Complete the MODEL_D pivot-only CV gate and, if passed, its one-shot lockbox diagnostic ([PROTOCOL.md §3.1](../../PROTOCOL.md)).

**Phase 2 (contingent on lockbox PASS):**
- Drop/shrink the composite blend in favor of pure `model_score` in production, per the pre-committed PASS interpretation (PROTOCOL.md §5).
- Fix entry timing to realistic t+1 fills.
- Wire the execution/risk model more tightly to the portfolio constructor.
- Run a PIT-on confirmation pass.

**Phase 3 (longer-term):**
- Begin logging live/paper timestamped picks and re-run the validator quarterly — "the only perfectly clean test," per PROTOCOL.md §6.
- Consider a managed model registry (e.g., MLflow) and containerization if the system moves beyond single-operator research use.
- Consider wiring drift/gate alarms to an external alerting channel.

**Future Improvements:**
- Institutional-mode ICT BOS-gating evaluation (currently frozen at "legacy" pending ≤2023 walk-forward comparison, PROTOCOL.md §3.1).
- `TARGET_TWAP_WINDOW` label-smoothing A/B (currently frozen at window=1, exact-endpoint return).

**Scalability Strategy:** the file-based feature store and filesystem model registry are appropriate at current scale (hundreds of tickers, two markets); if universe size or market count grows substantially, migrate the feature store to a managed store and the registry to a versioned ML registry rather than scaling the current ad hoc file conventions further.

**Research Opportunities:** alternative feature families beyond price/volume-derived TA/ICT/pivots (e.g., fundamentals, alternative data) are explicitly out of current scope but a natural longer-horizon research direction, contingent on first establishing that the existing technical-only signal survives the lockbox.

---

**Previous:** [← 12 · Testing Strategy](12-testing-strategy.md) &nbsp;|&nbsp; **Next:** [14 · Glossary →](14-glossary.md)
