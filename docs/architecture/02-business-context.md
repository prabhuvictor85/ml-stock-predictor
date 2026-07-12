[← Back to index](README.md)

# Business Context

**Current manual workflow (before this system):** an analyst opens charts one at a time, applies mental heuristics (moving averages, chart patterns, "does this look strong"), and builds a shortlist by hand. This does not scale past a few dozen names and cannot be back-tested rigorously because the "rules" live in the analyst's head and shift over time (hindsight bias).

**Pain points:**
- Cannot cover a full 500–1,200 stock universe consistently every week.
- No way to prove, after the fact, whether the picks would have worked absent knowledge of what already happened.
- No systematic way to explain *why* a pick was made beyond "it looked good."
- No early warning when market conditions change enough that old rules stop applying.

**Business objectives:**
1. Replace ad-hoc screening with a repeatable, testable process.
2. Produce a defensible, falsifiable claim about the strategy's edge (PASS/MARGINAL/FAIL, not a vague "seems to work").
3. Keep the system explainable enough that a human can sanity-check every pick before acting.

**Expected improvements:** full-universe weekly coverage in minutes instead of hours; an auditable trail from every watchlist entry back to the SHAP feature contributions and gate decisions that produced it.

**KPIs (tracked via the validation layer, not vanity metrics):**

| KPI | Where computed | Target |
|---|---|---|
| Mean rank-IC @ 20d (lockbox) | `scripts/tools/validate_lockbox.py` | > 0.02 |
| IC t-stat (non-overlapping) | same | > 2.0 |
| Top-decile excess return 95% CI | same | excludes 0 |
| Structural quality-gate veto rate | `pipeline/gating.py` self-check | ~5% baseline, alarm > 15% |
| Feature PSI drift | `pipeline/monitoring/drift_monitor.py` | below `psi_alert_threshold` (0.20) |

**ROI expectations:** framed as decision-support time savings (hours → minutes per week) plus the avoided cost of acting on an unvalidated, overfit "edge." No capital-allocation ROI is claimed until a lockbox PASS is recorded — see [PROTOCOL.md §7](../../PROTOCOL.md) for the result placeholder.

---

**Previous:** [← 01 · System Vision](01-system-vision.md) &nbsp;|&nbsp; **Next:** [03 · Conceptual Architecture →](03-conceptual-architecture.md)
