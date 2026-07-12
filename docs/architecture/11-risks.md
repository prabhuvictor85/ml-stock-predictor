[← Back to index](README.md)

# Risks

| Category | Risk | Mitigation |
|---|---|---|
| **Technical** | Feature/label leakage across train/test boundary | Purge (40d) + embargo (5d) windows, dedicated `tests/test_leakage_suite.py` |
| **Technical** | Silent gate/feature-name drift (columns renamed, gate becomes a no-op) | Explicit "GATE INACTIVE" warning + missing-column checks |
| **Technical** | Stale local CSV data | `tests/test_stale_data_guard.py` |
| **Business** | Overfitting mistaken for real edge | Purged walk-forward CV, pre-registered lockbox protocol, one-shot rule |
| **Business** | Composite blend diluting a real signal without anyone noticing | Documented finding (blend ~halves edge, CI includes 0); primary verdict metric is `model_score` |
| **Business** | Survivorship bias inflating backtested performance | PIT universe membership CSVs (SP500 wired via `--pit_universe`; SP400/600 membership intervals and NSE bhavcopy PIT still pending) |
| **Security** | API keys for live data providers | Kept out of source control; environment-variable based |
| **Ethical/Bias** | Universe composition shift causing biased vetoes (e.g., disproportionately filtering certain sectors) | Structural veto-rate self-check with an explicit alarm threshold (>15%) |
| **Failure mode** | Feature family looks good in-sample, flips sign out-of-sample (documented: 63% sign-flip rate for the full ICT v2 decomposition) | Pre-registered CV gates before any new family reaches the lockbox; per-family ledger in PROTOCOL.md |
| **Failure mode** | Researcher degrees of freedom silently accumulating (many small tuning choices each slightly informed by holdout knowledge) | Explicit changelog/ledger (PROTOCOL.md §3.1) logging every recipe-affecting change with rationale and lockbox-eligibility status |
| **Operational** | No automated alerting/paging on drift or gate alarms | Console warnings today; roadmap item to wire to Slack/PagerDuty |

---

**Previous:** [← 10 · Explainability](10-explainability.md) &nbsp;|&nbsp; **Next:** [12 · Testing Strategy →](12-testing-strategy.md)
