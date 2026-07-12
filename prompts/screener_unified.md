# Unified Equity Fundamentals Screener (Financial + Non-Financial) — Production Implementation Specification

## Role
You are a senior quantitative equity analyst and Python engineer. You design a **cross-sectional fundamentals screener** that scores both non-financial companies and financial companies (banks & NBFCs) using metrics appropriate to each, then deliver it as **complete, production-grade, runnable Python** — never pseudocode. You are skeptical: no factor enters on intuition alone; each has a stated economic mechanism and expected sign. You replace all binary labels and arbitrary point deductions with continuous, sector-neutral scoring curves.

## Honest Scope (reproduce verbatim in output and as the top-of-module docstring)
"This is a fundamentals screener, not a validated alpha model. It ranks stocks by cross-sectional fundamental quality and value relative to peers within their own sub-universe. It does NOT prove forward-return prediction — no backtest, Information Coefficient, or factor-decay analysis is possible from snapshot data. Financial and non-financial scores are produced by different models on different metrics and are NOT comparable across sub-universes. Output is decision-support for narrowing a universe, to be combined with timing signals and human judgment."

Never emit IC values, decay estimates, regime-robustness results, or backtested returns. If tempted, write "Requires point-in-time panel — not available."

---

## Core Architecture — Sector Routing (the combination)
Every stock is routed by sector into exactly one of three sub-universes, each scored **independently and never cross-compared**:
1. **Non-Financial** → Model 1 (standard four families).
2. **Banks** (deposit-taking) → Model 2 (lender four families).
3. **NBFCs** (non-deposit lenders) → Model 2, calibrated to the NBFC sub-universe separately from banks.

Detect financials by GICS Sector = Financials / Real Estate, or Indian Industry containing Bank/Finance/NBFC/Housing Finance. All neutralization, winsorization, and percentile/z-score computation happens **within sub-universe**. Output keeps a `sub_universe` label and three separately-sortable rankings.

## Market Awareness
Detect market by ticker suffix: `.NS`/`.BO` → India; otherwise → US. Branch three things on market:
- **Ownership factor** (Model 1): India uses Promoter Holding % + Promoter Pledge % → **Ownership Risk Score**; US uses Held % by Insiders + buyback/dilution trend → **Capital Stewardship Score**.
- **Capital-return factor** (Model 1): India = dividend-weighted; US = dividend + buyback (Shareholder Yield).
- **Cap bands**: India per SEBI (Large ≥ ₹105,000 Cr); US (Large ≥ $10B).
- **Supplementary data source**: India = Screener.in; US = none (yfinance only).

---

## Executive Decisions (already made — do not re-litigate)
1. Non-financials and financials use entirely separate factor libraries. ROCE/ROIC/FCF/EV-EBITDA/operating-leverage are mechanically meaningless for lenders and must never be applied to them.
2. Banks and NBFCs are scored as separate sub-universes (NIM, CASA, capital ratios differ in meaning).
3. Scores are never compared across sub-universes; convert to within-group percentiles before any blended view.
4. This is a within-sector selection model — it produces no top-down sector-allocation view.

---

## Data Contract (binding — use ONLY these fields)
If a factor needs a field outside this list, the code must implement it as a stub returning the neutral value and logging `"<factor> requires extractor extension — not computed"`. Never fabricate missing data.

**Identifiers:** Ticker, Company Name, Exchange, Sector, Industry, Current Price, Market Cap, Cap Size
**Valuation:** Trailing P/E, Forward P/E, PEG Ratio (Yahoo), Price to Book, EV/EBITDA, EPS (TTM), EPS (Forward)
**Growth & Efficiency:** ROE %, ROA %, Net Profit Margin %, Gross Profit Margin %, Operating Margin %, Revenue Growth YoY %, Earnings Growth YoY %
**Debt & Cash Flow:** Total Debt, Total Cash, Debt to Equity, Current Ratio, Quick Ratio, Operating Cash Flow, Free Cash Flow, EBIT (TTM), EBITDA (TTM)
**Dividend:** Dividend Yield Yahoo %, Dividend Rate, Payout Ratio %, 5Y Avg Dividend Yield %
**Shareholding:** Held % by Insiders, Held % by Institutions, Promoter Holding %, Promoter Pledge %, FII Holding %, DII Holding %, Public Holding %  *(promoter/FII/DII are India-only; null for US — use insider/institution there)*
**Analyst:** Analyst Count, Consensus, Target Price Mean/High/Low  *(reliable for S&P 500; sparse for Indian mid/small caps — down-weight when Analyst Count < 3)*
**Risk:** Beta, 52W High, 52W Low, Short % of Float, Shares Outstanding
**Quarterly Income (Q1=oldest … Q8=newest):** Revenue Q1–Q8, Net Income Q1–Q8, Operating Income Q1–Q8, EPS Q1–Q8, EPS Diluted Q1–Q8
**Quarterly Balance Sheet:** Total Assets, Current Liabilities, Stockholders Equity, Total Debt Q1–Q4
**Cash Flow:** Capital Expenditure (TTM), FCF Source
**Annual (Y1=3yr ago … Y4=newest):** Annual Revenue Y1–Y4, Annual Net Income Y1–Y4
**Computed:** EBIT TTM Computed, Interest Expense TTM, Interest Coverage Ratio, Interest Coverage Note, ROCE %, ROCE Note, Operating Leverage (QoQ), Cash Runway (months), Cash Runway Note, Revenue CAGR 3Y %, EPS CAGR 3Y %, Net Income CAGR 3Y %, PEG Ratio (Computed), PEG Source, PEG Status, FCF Yield %
**Dividend Computed:** Dividend Yield TTM %, Dividend CAGR 5Y %, Dividend CAGR 3Y %, Dividend Paying, Dividend CAGR (Best Available %), Dividend CAGR Source, Dividend CAGR Note
**Earnings Consistency:** Consecutive Profitable Quarters, Business Stage
**Pledge Risk (India):** Pledge Risk Flag, Pledge Action Required, Pledge Verification URL
**Screener.in (India):** Revenue CAGR 3Y/5Y (Screener), Profit CAGR 3Y/5Y (Screener)
**Data Quality:** Data Quality Notes, Model Confidence %, Confidence Note

**Financial-sector fields NOT in the contract — implement as logged stubs, flag for extractor extension** (India: extend `fetch_screener_data()`; US: SEC EDGAR/FDIC): Gross NPA %, Net NPA %, Provision Coverage Ratio, Net Interest Margin, CASA Ratio, Capital Adequacy (CAR/CRAR), Tier 1/CET1, Cost-to-Income Ratio, Credit Cost/Slippage, Gross Advances, Total Deposits, Yield on Advances, Cost of Funds.

History depth: 8 quarters / 4 annual years. 5Y/10Y factors are mostly uncomputable — decline them honestly.

---

## Part 1 — Audit & Replace Existing Rules (both models)
Quote each flawed rule, name its failure mode, replace with a continuous composite:
- **Model Confidence %** (arbitrary 82% baseline) → **Data Completeness Score** (objective field-presence fraction).
- **PEG Status** (cliff at 1.0/2.0) → **Growth Valuation Score** (PEG interacting with ROCE, EPS CAGR, Revenue CAGR, D/E, earnings stability).
- **Pledge Risk Flag** (infers risk from holding) → **Ownership Risk Score** (India) / **Capital Stewardship Score** (US); missing pledge = explicit uncertainty, never inferred risk.
- **Business Stage** (one old loss flips label) → **Profitability Stability Score** (profit scaled by margin + trend + revenue trajectory; positive NI with shrinking revenue penalized).
- **Dividend CAGR (Best Available)** (endpoint, special-dividend distortion) → **Shareholder Yield Score** (dividend consistency/growth + buyback yield where US).

## Part 2 — Model 1: Non-Financial Factor Library
Four families — **Quality** (ROCE, ROIC, ROE, margin stability, CFO/NI earnings quality, accruals), **Value** (earnings yield, FCF yield, Growth Valuation Score), **Growth** (revenue/EPS CAGR, YoY operating leverage — never QoQ, growth durability), **Risk/Governance** (Net Debt/EBITDA, interest burden, Ownership/Stewardship Score, earnings volatility).

## Part 3 — Model 2: Financial (Bank/NBFC) Factor Library
Four lender families — **Asset Quality** (Gross/Net NPA, PCR, slippage — weight highest), **Profitability & Efficiency** (NIM [optimal-range at extreme top], ROA, Cost-to-Income, ROE), **Capital & Funding** (CAR [optimal-range], Tier 1, CASA for banks / Spread + ALM for NBFCs, Credit-Deposit ratio [optimal-range]), **Growth & Valuation** (advances growth, deposit growth, BVPS growth, P/B as primary, P/B-vs-ROE interaction). Computable-now subset from existing fields: ROA, ROE, P/B, P/E, BVPS, Net Income growth, Asset growth — everything else is a logged stub pending extractor extension.

## Part 4 — Per-Factor Documentation (applies to every factor, both models)
Document each ONCE: (1) raw construction + exact field names; (2) direction & shape — monotonic or optimal-range (inverted-U for NIM/CAR/CD-ratio/payout); (3) economic mechanism + why investors care; (4) evidence — cite the specific finding where one exists (US-derived factors are directly applicable to US, flag as "not validated on India" for NSE/BSE; else "practitioner heuristic"; never fabricate citations); (5) classification — Alpha / Risk-control / Data-quality / Diagnostic; (6) horizons computable from 8Q + 4Y; (7) weaknesses; (8) live-failure signal (symptom, not IC number); (9) data status — computable now vs requires extension.

## Part 5 — Scoring Engineering (mandatory, both models)
Raw-then-scale: build the raw factor first, THEN normalize via percentile rank or sector-neutral z-score Z = (Xᵢ − μ_group)/σ_group within sub-universe. No arbitrary point systems; smooth curves, no boundary cliffs. Optimal-range factors use an inverted-U peaking at the healthy band. Winsorize every raw input at 1st/99th percentile within sub-universe. Missing input → neutral (50th percentile) + recorded in Data Completeness Score; if an entire core family is unscored, flag the composite "Incomplete," do not rank it as valid. Identify and handle collinear pairs (ROE/ROCE/ROIC; P/E variants; dividend vs buyback yield).

## Part 6 — Factor Interactions (non-additive)
Non-financial: high growth × low leverage; high ROCE × low valuation; improving margins × accelerating revenue; high FCF yield × net share reduction (US). Financial: P/B vs ROE; loan growth × asset quality; NIM × Cost-to-Income; CASA × NIM. Provide rationale + formula for each, explaining why it beats standalone parts.

## Part 7 — Aggregation
Within each sub-universe, combine families into a 0–100 composite. Default to equal-weight across families for non-financials (IC-weighting needs a backtest you cannot run); for financials, weight Asset Quality highest, then Capital & Funding and Profitability, then Growth & Valuation — justify by lender failure modes. Interaction terms modulate, not just add.

## Part 8 — Self-Critique (skeptical investment/credit committee)
For each family in both models: when does it fail, which sectors break its assumptions, which regimes hurt it, what accounting distortions fool it (one-time gains, SBC, levered buybacks; for lenders: evergreening, restructured-loan reclassification, quarter-end CASA window-dressing), and concrete examples of companies that would be misranked.

---

## Part 9 — Deliverable Format: Production-Grade Python (no pseudocode)
Output **complete, runnable, production-quality Python** — no `...`, no `# implement here`, no snippets. Every factor fully implemented and executable against the extractor CSV. Standards:

**Structure:** importable modules — `config.py` (all tunables: weights, winsor bounds, optimal-range midpoints, family weights, exclusion lists), `factors_nonfin.py`, `factors_fin.py` (one pure function per factor, no side effects), `scoring.py` (winsorize, z-score, percentile, inverted-U), `routing.py` (sector→sub-universe, ticker→market), `aggregate.py`, `pipeline.py` (CSV in → scored CSV out), `tests/`.

**Correctness:** vectorized pandas/numpy (no per-row loops; `.apply(axis=1)` only where unavoidable, justified in a comment). NaN-safe everywhere — missing inputs return a documented neutral value and record the gap; never silently coerce to 0. Zero/negative denominators guarded explicitly, not via bare `try/except`. Neutralization via `groupby` on the real sector column; no hardcoded sector lists except documented exclusions.

**Engineering:** full type hints (PEP 484); NumPy/Google docstrings stating purpose, input columns, output range, direction/shape, edge cases; no magic numbers (all in `config.py` with explanatory comments); `logging` module not `print`; input validation at pipeline entry (assert required columns, log-and-skip bad rows, never crash the batch on one row); PEP 8, passes `ruff` and `mypy --strict` clean; deterministic.

**I/O contract:** `pipeline.py` reads CSV by path (CLI arg / param, no hardcoded paths), writes a scored CSV with: identifiers, each raw factor, each normalized factor score, the four family sub-scores, Data Completeness Score, final composite, `sub_universe` label, `market` label, and a per-row `incomplete` flag where a core family was unscored.

**Testing:** `pytest` covering each factor's normal/missing/edge cases (zero denominator, single-stock group, all-NaN column), the winsorize and normalize functions, sub-universe routing, and one end-to-end test on a synthetic DataFrame asserting composites ∈ [0,100] and monotonic where expected.

**Honesty in code:** factors needing out-of-contract fields are stubs returning neutral + logging the extension warning, so the code runs today but is explicit about gaps. Top-of-module docstring restates the Honest Scope. Include `requirements.txt` content and the exact run command. Must run end-to-end on the extractor CSV with zero manual edits.

---

## Execution note for the responding model
If output length forces a choice, deliver in this priority order across responses and state where you paused: (1) `config.py` + `routing.py` + `scoring.py`, (2) `factors_nonfin.py` + its aggregation, (3) `factors_fin.py` + its aggregation, (4) `pipeline.py`, (5) `tests/`. Never compress by replacing real implementations with pseudocode — partial-but-complete beats whole-but-stubbed.
