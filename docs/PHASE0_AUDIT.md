# Phase 0 — Project Audit, Architecture Critique & Research Roadmap

**Date:** 2026-07-11. **Status:** audit only — no implementation. Backlog items
below require approval before execution. Companion docs: `PROTOCOL.md`
(experiment ledger, one-shot lockbox rules), `docs/MODEL_*_PREREGISTRATION.md`.

---

## 1. Executive summary

This is not a greenfield audit. The repository has already executed an
unusually disciplined alpha-research campaign — pre-registered experiments,
frozen gates, one-shot lockbox rules, independent read-only validators, a
researcher-degrees-of-freedom ledger, and four *self-caught* methodology bugs
(zone look-ahead, te_panel HPO leak, ddof=0 t-stats, lambdarank group
misalignment). Most of what a Phase 0–2 program normally has to build already
exists and works.

The honest out-of-sample baseline **exists** and is the central fact of this
audit:

> **Best honest configuration: formation momentum + short sleeve (E2 KERNEL+S),
> mean 20d rank-IC = +0.0168, t = +1.76 (ddof=1), 5/6 folds positive — a
> pre-declared narrow fail against the campaign gate (IC ≥ 0.03, t ≥ 2.0,
> ≥ 4/6 folds positive).**

The mission target (OOS rank-IC 0.03–0.06, ICIR > 1.0) is *numerically
equivalent* to the campaign gate already in force (t ≥ 2.0 over 6 yearly folds
≈ fold-level ICIR ≈ 0.82; IC ≥ 0.03). No goalposts need moving. The honest gap
between +0.0168 and +0.03 **is** the research problem.

This audit found **one new structural defect** (a CV boundary-purge gap, §3.1),
confirms the known integrity gaps, and proposes a prioritized backlog whose top
items complete the missing baseline (MODEL_F, MODEL_A re-run, honest
full-pipeline CV) before any new factor work.

---

## 2. Baseline benchmark — established (do not re-derive)

### 2.1 The honest family ladder

Harness: expanding-window walk-forward CV, yearly folds 2018–2023, LGBM
lambdarank (ndcg@10, num_leaves=31), label = `cs_rank_composite`, metric =
per-date rank-IC vs `future_20d_excess_return`, seeds fixed, date-major groups
asserted, t-stats ddof=1. All numbers from `PROTOCOL.md` §3.1 ledger:

| Family (model) | n feats | mean IC20 | t | folds+ | Verdict |
|---|---|---|---|---|---|
| ICT / SMC only (MODEL_C) | 66 | −0.00002 | −0.009 | — | FAIL (dead) |
| Short-window momentum (MODEL_E) | — | −0.0017 | −0.60 | — | FAIL |
| Zones causal (MODEL_A) | 30 | **SUSPENDED** | — | — | prior +0.0069 VOID (group misalignment); honest IC unknown |
| Pivots / CPR / Camarilla (MODEL_D) | 69 | +0.0092 | +1.00 | 4/6 | FAIL |
| Formation momentum (E2 KERNEL) | 3 | +0.0136 | +1.46 | 5/6 | FAIL |
| **Formation + short sleeve (E2 KERNEL+S)** | 7 | **+0.0168** | **+1.76** | 5/6 | **narrow FAIL** |
| Horizon sweep (E3, TWAP5×60d primary) | 7 | +0.0106 | +0.78 | 4/6 | FAIL; horizon dimension exhausted |

Void / leaked numbers (never cite as benchmarks): zone-core +0.1441, 30-feature
zone baseline +0.1920 (non-causal zone generator); in-sample pulse +0.0655
(t +2.44, top-decile excess +1.47%) is the optimistic upper reference only.

### 2.2 What is still missing from the baseline

1. **Honest full-pipeline CV number.** Every previous full-pipeline CV was
   contaminated by (a) the non-causal zone family and (b) the te_panel
   within-fold recompute leak. Both are now fixed (commits `3389c34`,
   `3f8c91c`) but no clean number has been recorded since.
2. **MODEL_F** (all-causal-features, pre-registered 2026-07-09) — the harness
   script does not exist yet; the experiment has not run.
3. **MODEL_A causal re-run** — zone verdict suspended pending one unchanged
   re-run on the fixed harness.
4. **Reversal mode** — zero honest entries in the ledger. Half the product
   surface (momentum/reversal dual watchlists) has never been audited.
5. **Turnover / capacity** — never measured in the honest campaign (mission
   target: < 20%/day).
6. **Lockbox 2024-01→2026-05: UNSPENT** for the momentum pipeline (PROTOCOL §7
   blank). This is an asset; it must stay unspent until a config clears the
   tuning-era gate.

---

## 3. New findings from this audit

### 3.1 CV boundary purge gap (new, decision-relevant)

The training label `cs_rank_composite` is a 0.5/0.3/0.2 blend of 20/40/60-day
forward-return ranks (`pipeline/targets/builder.py:276`), so a label looks up
to **60 trading days** ahead. Two harnesses under-purge against that horizon:

- **Production CV:** `PurgedWalkForwardCV` defaults `purge_window=40` +
  `embargo=5` → 45 days removed before each test window
  (`pipeline/validation/cv.py:32`). The run scripts construct it without
  overriding (`run_sp500_local.py:855`, and the two NSE runners), even though
  `PURGE_HORIZON = 80` is defined and exported in
  `pipeline/targets/builder.py:29` — **defined but never consumed**. Train rows
  in the last ~15 days before the purge zone have 60d-label components
  realized inside the test window.
- **Standalone experiment harnesses** (C/D/E/E2/E3 — the source of the honest
  ladder): fold split is `year < test_year` vs `year == test_year` with **no
  purge at all** (`scripts/experiments/model_e2_formation_momentum.py:98`).
  Train rows in the last ~60 trading days of December have 20/40/60d label
  windows extending into the test year.

**Impact assessment:** bias is upward (train labels encode a sliver of
test-period cross-sectional return structure) but small — the contaminated
slice is ≲60 days of a ≥2,000-day training window, and only the earliest weeks
of each test year are exposed. **No recorded verdict flips**: every config
FAILED its gate, and the true values are, if anything, slightly lower.
The E2 +0.0168 headline is marginally optimistic. **Required action:** add a
`MAX_FORWARD_HORIZON`-day gap at the fold boundary in the standalone harness
template and pass `purge_window=PURGE_HORIZON` in the runners **before
MODEL_F runs**, and record the fix in the PROTOCOL ledger (it amends the
harness spec of a frozen pre-registration → note it explicitly).

### 3.2 MODEL_F pre-registration §2 is stale

The doc justifies a standalone harness because the te_panel leak was "unfixed."
Commit `3f8c91c` (2026-07-10) fixed it after the doc froze. The standalone
harness is *still* the right choice (comparability with C/D/E/E2 references),
but the ledger should note the rationale change, and the §6 decision tree's
"THEN fix the te_panel leak" step is already done.

### 3.3 Minor / hygiene findings

- **Literal mangled-path directory** at repo root
  (`C:VictorProjectml-stock-predictoroutputus_local…`) plus nested
  `output/us_local/2023-12-07/output/output/…` — a Windows path-join bug in
  some output writer created directories from unexpanded path strings.
  Cleanup is underway per git status; the writer should be found and fixed.
- **Duplicate date rows** in ≥1 local CSV — ingest uniqueness assert still
  pending (known).
- **Anachronistic snapshots:** `cap_tier` (known) and the stock→sector-ETF map
  (constituent CSV `ETF` column, today's snapshot) — acceptable for sector
  *features*, but must be documented as a caveat if used for neutralization
  (sector membership is far stickier than index membership).
- **Repo hygiene:** root-level scratch files (`scratch_nvda_*.py`,
  `debug_pipeline.py`, `test_cv.py`, `diag_folds.py`), a legacy secondary
  entry point (`pipeline/train.py`) that has already drifted from the
  production runners once (HPO feature-set bug). Consolidation candidate, not
  urgent.

---

## 4. Stage-by-stage integrity audit

| Stage | State | Assessment |
|---|---|---|
| **Data ingestion** | Static local CSV snapshots (~1,600 US names), explicit refresh via `download_us_data.py`, `StaleDataGuard`, `--strict_data_check` | Sound design (one adjustment epoch per snapshot). Gaps: dup-date assert pending; recent-dividend staleness is operational discipline. |
| **Universe / survivorship** | PIT SP500 membership wired (`--pit_universe`, 1,202 tickers, intervals 1996–2026); SP400/600 intervals missing; **187 Yahoo-dead ex-members lack prices = 14.5% of 2010–26 membership-days**; 34 "available" names are symbol-reuse traps (never bulk-download) | Best-in-class for a retail-data project, honestly documented as an upper bound. Norgate backfill is the sanctioned unblock for the E2 re-run; deferred by user for cost. |
| **Features** | Truncation-invariance audit: ICT 0/153 changed, pivots 0/68, base 0/37 → causal; **zones 14/33 rewritten → non-causal, quarantined** | The audit methodology itself is a strength. Zones need a v2 causal generator (event-dated invalidation) before any further use. |
| **Labels** | 20/40/60d fwd + excess + `cs_rank_composite` (0.5/0.3/0.2); NaN-native (no rank-0 fill — fixed); TWAP terminal knob frozen at 1 | Sound. Pending: terminal-return blackout before index removal (delisting death spirals) — matters more once dead tickers are backfilled. |
| **Validation** | Purged walk-forward CV + frozen lockbox protocol + one-shot rule + independent validator + pre-registration ledger | Institutional-grade in design. Defect: boundary purge gap (§3.1). Recipe-level leak history honestly documented; lockbox correctly treated as semi-clean upper bound. |
| **Inference / gates** | All ~1,600 names scored daily regardless of membership (by design); gates (ssz>0.6, ict_bear>0.4, ADX prongs) shared via `pipeline/gating.py` with pinned tests | Gate parity resolved. **All gates are prevalence-calibrated, none outcome-validated** — every veto is potential silent alpha loss. The 85/15 composite blend measurably dilutes (pulse check: roughly halves the edge). |
| **Portfolio / risk** | `pipeline/portfolio/` + backtest engine exist with turnover accounting | Disconnected from the honest verdict path; unvalidated. Correctly deprioritized until a signal clears the gate. |

---

## 5. Architecture critique

**Strengths (rare in any codebase, retail or institutional):**
pre-registration culture with commit-hash freeze proofs; independent read-only
validators that recompute returns from raw CSVs; truncation-invariance audit
tooling; NaN-native LGBM paths; shared gating module with behavior-pinning
tests; fold caching + vectorized winsorization (recent 200× speedup);
fail-loud conventions; every methodology bug found so far was found *by this
project's own discipline*.

**Weaknesses:**

1. **Monolithic runners.** `run_sp500_local.py` is 2,879 lines mixing panel
   build, HPO, feature selection, gating, scoring, and I/O. It has already
   caused one recipe bug (HPO feature-set construction) and one leak
   (te_panel) that pure-function modules with tests would have made harder to
   write. *Recommendation: extract only what experiments repeatedly need
   (fold-recompute, HPO objective) — do not refactor for its own sake; alpha
   work outranks architecture work.*
2. **Signal-path complexity exceeds signal strength.** An 85/15 blend, four
   gate prongs, two modes, and per-mode watchlists sit on top of a base signal
   whose honest IC is +0.0168. The blend is known-dilutive; the gates are
   outcome-unvalidated. Complexity here is unpaid-for; the verdict metric
   (`model_score`, pure LGBM) is correctly chosen already.
3. **The ICT engine (1,066 lines, ~184 features) is empirically dead as a
   standalone family** (IC −0.00002, t −0.009; 63% train→lockbox sign-flip on
   the 88-feature v2 decomposition vs 0% for zone-core). The mission brief
   asks to "leverage and heavily scrutinize" it — the scrutiny already
   happened and it failed. The only live hypotheses are (a) incremental
   contribution inside MODEL_F ALL/ALL+SELECT, (b) the frozen
   `ICT_IMPLEMENTATION_MODE=institutional` A/B on tuning-era data. If MODEL_F
   shows no incremental lift, the honest conclusion is that daily-bar SMC
   geometry carries no alpha in this universe, and the engine should be
   frozen (not extended).
4. **No risk-model layer.** Nothing residualizes returns or exposures (beta,
   sector, vol) anywhere in features, labels, or evaluation. This is
   simultaneously the biggest critique and the highest-prior open alpha idea
   (§7, MODEL_G).
5. **Reversal mode is un-audited** despite shipping watchlists.

---

## 6. Limiting-factor analysis — why honest IC 0.03 is hard here

- **Instrument set:** daily OHLCV only, no fundamentals, no intraday, no PIT
  fundamentals. Factor menu = price/volume/membership geometry. The heavily
  arbitraged large-cap US cross-section is the hardest place to find
  daily-bar price-only alpha.
- **Fundamental law (Grinold):** IR ≈ IC × √breadth. At IC ≈ 0.015–0.02 with
  ~500 effective names and a 20d horizon, ICIR > 1 is achievable only via
  (a) stability (reduce fold variance → residualization), (b) breadth (NSE
  port, small caps — where the E2 PIT split already showed the signal is
  *stronger* in the broad ~1,591-name cross-section, +0.0168 vs +0.0117 in
  the SP500 core, consistent with crowding), and (c) combining weakly
  correlated honest families. Not via one magic factor.
- **Regime dependence is the binding constraint, not average level:** 2021
  (momentum crash year) is negative in every momentum config; 2020 dominates
  several buckets. E3 proved longer horizons just buy more regime beta.
  Residualization attacks exactly this.
- **Survivorship ceiling:** 14.5% of membership-days have no prices; the E2
  narrow-fail verdict is explicitly parked on this. Every current number is an
  upper bound of unknown slack until the backfill.

---

## 7. Prioritized experiment backlog

Ordering rule: complete the baseline (P0) before new factor families (P1).
Every new family = ONE pre-registered ledger entry with the standard gate
(mean IC20 ≥ 0.03, t ≥ 2.0 ddof=1, ≥ 4/6 folds+) unless stated. All work on
tuning era (≤ 2023) only; the 2024–26 lockbox stays sealed.

### P0 — finish the baseline

| # | Item | Why / expected outcome | Success criterion | Effort |
|---|---|---|---|---|
| 1 | **Fix the boundary purge gap** (§3.1): 60d gap in standalone harness template; pass `purge_window=PURGE_HORIZON` in runners; ledger note | Integrity of every future verdict; expect ladder numbers to drift *down* slightly | Synthetic test: no train label window overlaps test dates | ~½ day |
| 2 | **Write + run MODEL_F** per frozen pre-registration (harness absent; 4 configs: MOM / ALL ~290 / BASE+MOM / ALL+SELECT top-40 fold-local) | THE open question: do dead-standalone families add incrementally to momentum? Prior: modest (campaign null = noise-drag), but ALL+SELECT vs MOM delta ≥ 0.005 triggers the pre-registered follow-up path | Pre-registered gate + decision tree in `docs/MODEL_F_PREREGISTRATION.md` | ~1 day + compute |
| 3 | **Re-run MODEL_A causal zone CV** (fixed harness, unchanged spec, ONE run) | Closes the suspended verdict; zone funeral or resurrection | Record verdict in ledger, either way | hours (compute) |
| 4 | **Honest production-pipeline CV** (post-`3f8c91c`, post-purge-fix, zones excluded or fence-redrawn) | The official Train/Val Rank-IC of record for the *shipping* pipeline; quantifies gates+blend drag vs MODEL_F's clean harness | Number recorded in ledger with per-fold ICs | 1 overnight run |

### P1 — highest-prior new research (pre-register each)

| # | Item | Hypothesis & evidence | Success criterion | Effort |
|---|---|---|---|---|
| 5 | **MODEL_G: residual momentum** — formation returns residualized vs market beta (rolling ~252d) and sector (SPDR ETF map exists in constituent CSV) | Blitz/Huij/Martens (2011): residual momentum ≈ halves vol, roughly doubles IR vs raw momentum; directly targets the negative 2021 fold. The single best-supported open idea given data constraints | Beats +0.0168 with t ≥ 2.0 AND improves the 2021 fold; else standard gate | 2–3 days |
| 6 | **Outcome-validate the gates**: tuning-era forward returns of gated-out vs gated-in candidates | Each unvalidated veto is potential silent alpha loss; gates that don't separate get removed (reduces d.o.f.) | Per-gate: gated-out cohort underperforms gated-in with t ≥ 2 → keep; else remove | ~1 day |
| 7 | **Reversal-mode honest audit** (same harness/gate as ladder) | Half the product has no honest number; short-horizon reversal is a real literature effect (Jegadeesh 1990) and orthogonal to momentum by construction | Standard gate; also record momentum–reversal score correlation | ~1 day |
| 8 | **Family orthogonality + decay read** (cheap, diagnostic): correlation matrix of family scores (momentum, pivots, zones-if-revived, reversal) + IC decay curves | If pivots (+0.0092) are near-orthogonal to momentum (+0.0168), a 2-family combo can clear the gate even though each fails alone — this quantifies MODEL_F's headroom before/alongside it | Informative, no gate | ~½ day |

### P2 — conditional / infrastructure

| # | Item | Trigger / note |
|---|---|---|
| 9 | **Turnover + capacity measurement** of KERNEL+S top-decile at 14d cadence | Needed for the < 20%/day target; formation momentum at 20d horizon almost certainly passes — measure, don't assume. ~½ day |
| 10 | **Norgate dead-ticker backfill** (~$ paid sub) | BLOCKS the sanctioned E2 re-run (narrow-fail rule already triggered). User-deferred for cost — the decision stands, but it caps how much the momentum verdict can be trusted. Revisit if MODEL_F/G verdicts make momentum the production signal |
| 11 | **Zone v2 causal generator** (event-dated invalidation, timeline-causal eliminator) | Only if the MODEL_A re-run (#3) shows signal worth chasing |
| 12 | **NSE universe port** | Ledger's named next step after MODEL_F; breadth lever per §6; watch the >15% structural-veto alarm → gates must be recalibrated for NSE |
| 13 | **Label hygiene**: terminal-return blackout (~60d pre-removal → NaN) + dup-date ingest assert + fix the mangled-output-path writer | Small effort; blackout matters more post-backfill |
| 14 | **`ICT_IMPLEMENTATION_MODE=institutional` A/B** (tuning era) | Low prior (family dead standalone); only worth compute if MODEL_F shows ICT features being selected |

### Explicitly NOT recommended now

- **Deep learning / TFT / transformers** — no tree-based config has yet proven
  an exploitable stationary signal (the mission's own precondition).
- **New ICT/SMC feature variants** before the MODEL_F read — the family is
  measured dead standalone; more variants = d.o.f. inflation.
- **Horizon extension** — falsified by E3 (monotonic IC decay, t collapse).
- **New gate prongs / blend tweaks** — unpaid-for complexity on a weak base
  signal; gates must first survive #6.
- **Anything that touches 2024–26** — the lockbox stays sealed until a config
  clears the tuning-era gate (one-shot rule, PROTOCOL §6).

---

## 8. Phase mapping (mission phases → repo reality)

| Mission phase | Status here | Remaining work |
|---|---|---|
| 0 Audit & baseline | **This document**; ladder in §2 | P0 items #1–4 complete it |
| 1 Data integrity | Largely done (skills + ledger document universe, splits/dividends epoch argument, survivorship quantified at 14.5%) | #10, #13 |
| 2 Validation framework | Done (purged walk-forward + lockbox + pre-registration) | #1 purge fix |
| 3 Alpha research | Family-by-family ladder done for ICT, zones, pivots, momentum, horizons | #5 residual momentum, #7 reversal, then MODEL_F decision tree |
| 4 Feature selection | ALWAYS_INCLUDE + top-k selector exists; fold-local selection specified in MODEL_F | Orthogonality read #8; VIF/SHAP-stability only after a family passes |
| 5 Label research | 20/40/60d, excess, composite ranks, TWAP knob — done; horizon question CLOSED by E3 | Residual-return label variant rides with #5 |
| 6 Model research | LGBM lambdarank baseline appropriate; simplest-model principle already enforced | Benchmarking CatBoost/XGBoost/ElasticNet is P3 at best — model class is not the binding constraint at IC 0.017 |
| 7 Risk & portfolio | Code exists, unvalidated, correctly parked | Wire only after first gate-passing signal; then Sharpe/DD/capacity after slippage |

**Workflow per factor (unchanged, already in force):** hypothesis →
pre-register (doc + commit hash) → engineer → tuning-era CV → gate → ledger →
only-if-pass: one-shot lockbox consideration.

---

## 9. Bottom line

The platform's methodology is already at or above institutional standard; the
constraint is the data (daily OHLCV, survivorship ceiling) and the universe
(efficient US large caps). The honest edge today is +0.0168 (t +1.76) from
7 momentum features — everything else measured so far is zero or noise. The
credible paths from 0.017 to 0.03+ are, in order of prior: residualization
(MODEL_G), multi-family combination (MODEL_F), breadth (NSE / backfill), and
gate-drag removal (#6). If none of these clear the gate honestly, the correct
institutional conclusion is that daily-bar price-only alpha on this universe
supports paper-trading the momentum sleeve (renewable future lockbox) but not
capital at scale — and the ledger will have proven it the right way.
