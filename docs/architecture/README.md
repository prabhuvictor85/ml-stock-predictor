# ML Stock Predictor — Living Architecture Document

**Status:** Living document — update the relevant section file whenever the recipe, thresholds, or architecture change. This directory, together with [PROTOCOL.md](../../PROTOCOL.md), is the single source of truth for the system.
**Audience:** Executives, Product, ML Engineers, Software Engineers, QA, and every future contributor.
**Companion documents:** [PROTOCOL.md](../../PROTOCOL.md) (lockbox validation), [CODE_EXPLANATION.md](../CODE_EXPLANATION.md) (module-by-module reference), [FEATURE_REFERENCE.md](../FEATURE_REFERENCE.md), [CV_EXPLANATION.md](../CV_EXPLANATION.md).

> **How to read this document.** Every major technical section is written in three layers:
> **Level 1 (Executive Summary)** — one paragraph, no jargon. **Level 2 (Plain English)** — an analogy a non-engineer can hold in their head. **Level 3 (Technical Deep Dive)** — the actual algorithm, code paths, and numbers. Skip to whichever layer matches your role; each layer stands alone.

---

## Document Map

Each row is its own file — read only what your role needs, or read top to bottom for the full picture.

| # | Section | What's in it |
|---|---|---|
| 01 | [System Vision](01-system-vision.md) | Current state, future vision, scope, success criteria, assumptions, constraints |
| 02 | [Business Context](02-business-context.md) | Manual workflow being replaced, pain points, KPIs, ROI framing |
| 03 | [Conceptual Architecture](03-conceptual-architecture.md) | The full data journey, source-to-feedback-loop, with a stage-by-stage 3-level explanation |
| 04 | [Functional Design](04-functional-design.md) | Personas, use cases, user journey, inputs/outputs, business rules, API contracts |
| 05 | [Machine Learning Design](05-ml-design/README.md) | Problem formulation, feature engineering, learning strategy, model evaluation (4 sub-files) |
| 06 | [Data Architecture](06-data-architecture.md) | Schemas, lineage, storage, versioning, data quality, governance |
| 07 | [Technical Architecture](07-technical-architecture.md) | Tech stack, infra, CI/CD, caching, containerization |
| 08 | [System Design](08-system-design.md) | Component diagram, class responsibilities, state machine, dependency graph |
| 09 | [Operational Lifecycle](09-operational-lifecycle.md) | Deployment, monitoring, drift, retraining, rollback, DR |
| 10 | [Explainability](10-explainability.md) | SHAP, setup matching, confidence, limitations |
| 11 | [Risks](11-risks.md) | Technical/business/security/ethical risks and mitigations |
| 12 | [Testing Strategy](12-testing-strategy.md) | Test suite map, acceptance criteria |
| 13 | [Future Roadmap](13-future-roadmap.md) | Phase 1/2/3, scalability strategy, research opportunities |
| 14 | [Glossary](14-glossary.md) | Every technical term in plain English |
| 15 | [Appendix](15-appendix.md) | Config files, model parameters, example payloads, formulae |

---

## Executive Summary

**What problem are we solving?**
Every week, thousands of publicly traded stocks move for reasons that are individually hard to keep track of — earnings, momentum, technical chart patterns, sector rotation. A human analyst cannot re-evaluate 500–1,200 stocks every week with consistent, unemotional criteria. This system does that job automatically: it reads years of price history for a universe of stocks (S&P 500 and NSE India today), learns statistical patterns that historically preceded stocks outperforming their benchmark over the next ~20 trading days, and produces a ranked, explainable shortlist ("watchlist") every week.

**Why does this matter?**
Manual stock screening is slow, inconsistent, and prone to hindsight bias (a person who already knows a stock went up will always find a reason it "should have" gone up). A systematic pipeline applies the exact same rules to every stock, every week, and — critically — is validated against data it has never seen, so its claimed edge can be trusted (or rejected) on evidence rather than a narrative.

**Who benefits?**
- **The researcher/trader (primary user today):** gets a weekly, ranked, explained shortlist instead of manually scanning hundreds of charts.
- **Future engineers on this project:** inherit a documented, tested, leakage-guarded codebase instead of reverse-engineering scripts.
- **Anyone evaluating the strategy's credibility:** gets a pre-registered, tamper-evident validation protocol ([PROTOCOL.md](../../PROTOCOL.md)) instead of a cherry-picked backtest.

**Expected business impact.**
This is currently a research and personal-decision-support system, not an automated trading system (no capital is deployed automatically). The value is *decision quality and speed*: converting a multi-hour weekly manual screening process into a few-minute automated run, with a documented, self-monitoring quality gate and an honest, pre-registered read on whether the edge is real or an artifact of overfitting.

| Business Need | Technical Solution |
|---|---|
| Screen 500–1,200 stocks weekly without manual effort | Automated feature engineering + LightGBM ranking model scores the full universe in minutes |
| Trust that a signal isn't just curve-fit to history | Purged walk-forward cross-validation + a pre-registered, one-shot "lockbox" holdout test ([PROTOCOL.md](../../PROTOCOL.md)) |
| Avoid picking stocks that are technically "cheap" but structurally broken | Rule-based quality gate (`pipeline/gating.py`) vetoes momentum-bull picks under overhead supply/bearish structure |
| Understand *why* a stock was picked, not just that it was | Per-stock SHAP explanations + similar-historical-setup matching |
| Detect when the market regime has shifted under the model's feet | Population Stability Index (PSI) drift monitor with automatic retrain-flagging |
| Work across multiple markets without rewriting the pipeline | `MarketConfig` dataclass — one new config file adds a market, zero pipeline code changes |
| Avoid survivorship bias (only looking at today's winners) | Point-in-time (PIT) universe membership CSVs — includes delisted/removed constituents |

---

**Next:** [01 · System Vision →](01-system-vision.md)
