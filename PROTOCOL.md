# Lockbox Validation Protocol — SP500 Momentum

**Status:** pre-registration draft. Commit this file (with your final thresholds)
*before* running the fenced test. Once committed, the pass/fail bar is fixed —
the whole point is that you cannot move the goalposts after seeing the result.

---

## 1. What this test answers

Does the SP500 momentum signal have a real edge on data its **configuration**
never saw? The in-sample pulse check was strong but optimistic by construction
(HPO, feature selection, gates were all chosen on history that included
2024–2026). This test re-derives the model with everything fenced at
**2023-12-31** and measures performance on **2024-01-01 → 2026-05-06**, which the
fence never touched.

### Honest scope (read before trusting the result)

This lockbox is **semi-clean, not perfectly clean**:

- **Decontaminated by the fence:** hyperparameters (Optuna), feature selection,
  and model weights — all re-derived using only ≤ 2023-12-31 data.
- **NOT decontaminated:** *design* decisions made in earlier sessions while
  looking at 2024–26 results — which gates exist (ssz>0.6, ict_bear>0.4), the
  85/15 blend, which features are in the pool. Those leaks cannot be undone
  retroactively.

Therefore treat the result as an **upper bound on the true edge**, still better
than any in-sample number. The only perfectly clean holdout is the future
(see §6).

---

## 2. In-sample baseline (the number to beat down from)

Pulse check, momentum/bull/model_score, 2024-01-12 → 2026-05-04, n=53
(`pulse_check_momentum.json`, in-sample / optimistic):

| Metric | In-sample value |
|---|---|
| mean rank-IC @20d | +0.0655 |
| IC t-stat (non-overlap) | +2.44 |
| top-decile excess @20d | +1.47% (t=4.89) |

The fenced number **will be lower**. The question is *how much survives*.

---

## 3. Frozen configuration (the recipe under test)

Recorded here so the run is reproducible and the config is provably fixed:

- **Market / mode:** SP500, `momentum`
- **Primary signal:** `model_score` (pure LGBM rank). *Rationale:* the pulse
  check showed `composite_score` (85/15 blend) roughly halves the edge and its
  top-decile CI includes zero — the blend dilutes. The clean test still records
  composite for completeness but the verdict is on `model_score`.
- **Fence date:** `--train_end 2023-12-31`
- **Train start:** 2010-01-01
- **HPO:** Optuna, `--n_trials 40` (reduced for 2-CPU tractability), seed fixed
  in config. `--n_folds` auto = 12 from the fenced range; may be set lower
  (e.g. 8) to fit the overnight window — record whichever you use.
- **Universe:** **point-in-time** (`--pit_universe` ON) — survivorship-free SP500
  membership from `{data_root}/stock_lists/membership_sp500.csv` (1202 tickers:
  503 current + ~699 ex-members, intervals 1996–2026). Forwarded to the seed AND
  every walk-forward retrain/inference step. Residual caveat: still a *tighter*
  upper bound — ~139 dead ex-members have no price data, so the deepest failures
  remain under-represented. (Prior `current snapshot` runs were survivorship-biased.)
- **Code version:** record the git commit hash of the run here: `__________`

Any change to the above after committing this file invalidates the test.

### 3.1 Configuration changelog (researcher-degrees-of-freedom ledger)

Recipe-affecting code/config changes, dated. The fenced run's git commit (§3)
**must post-date every entry below** — an older binary reproduces the old recipe.

- **2026-06-26 — HPO feature-set construction corrected (production runners).**
  Fix applied to `run_sp500_local.py`, `run_nse_local.py`,
  `run_nse_tradingv_local.py` (the scripts §4 actually invokes). Each Optuna
  trial previously used `pre_selected[:top_k]`. Because `FeatureSelector.select()`
  returns the ~29 `ALWAYS_INCLUDE` features at the *front* of the list, that
  front-slice handed every trial ~20–30 forced features and almost **zero**
  data-ranked ones — so HPO (including the `feature_top_K` choice and all tree
  params) was tuned on the hand-picked set, while the final model shipped
  `forced + top_k ordinary` (~49–59 features). Trials now build
  `_forced_pre + _ordinary_pre[:top_k]`, matching the final model exactly.
  (`pipeline/train.py`, a legacy/secondary entry point not used by §4, was
  corrected the same way earlier but does not affect the lockbox run.)
  *Implication:* any HPO recipe / `ensemble.pkl` / `selected_features.txt`
  produced before this date is from the buggy construction and must be
  regenerated before the lockbox run.

- **2026-06-26 — ICT feature-pool leak fixed (`pipeline/features/engineer.py`).**
  The 1d ICT column prefixer used a hardcoded list that had drifted out of sync,
  so ~50 ICT features (premium/discount, CHoCH/MSS, Breaker Blocks, fill-pct,
  displacement quality, prior-session levels, liquidity-pool stats, OB entry/
  rejection) were emitted **unprefixed** and never reached the model — they sat
  in the panel as dead `ict_*` columns the FeatureSelector ignores. Prefixing is
  now dynamic (all `ict_*` → `features_ict_*`). *Implication:* the available
  feature pool grows materially; the FeatureSelector now sees these ~50 columns
  for the first time. Treat as a recipe change — regenerate artefacts and
  re-run selection before the lockbox.

- **2026-06-26 — Ranker target NaNs dropped, not zero-filled (production runners).**
  `run_sp500_local.py`, `run_nse_local.py`, `run_nse_tradingv_local.py`
  previously did `cs_rank_composite.fillna(0)` when building the ranker label
  (HPO folds + final fit). Rows with an unknowable forward-return rank
  (delisted/halted tickers, last ~20 trading days) were thereby labelled rank 0
  — the *worst* stock in the cross-section — corrupting the LambdaRank target.
  Those rows are now dropped from ranker training and the LightGBM group array
  is recomputed to stay aligned (the classifier head already used its own
  `notna` mask, unchanged). *Implication:* changes the training label set →
  regenerate any pre-this-date model artefacts before the lockbox run.

- **2026-06-27 — New knob `TARGET_TWAP_WINDOW` (target terminal-price smoothing).**
  `TargetBuilder` and `validate_lockbox.py` gained a trailing-average terminal
  window for the forward-return labels (env `TARGET_TWAP_WINDOW`, or
  `terminal_window`/`--twap_window`). **Frozen default = 1 = the exact endpoint
  return** — verified bit-identical to prior behaviour, so the current recipe is
  UNCHANGED. window>1 de-sensitises labels to a single print on day t+h. This is
  a **tuning-era A/B candidate only**: set the SAME value for the builder and the
  validator (the env var does both), evaluate on ≤2023, and if adopted, record
  the chosen window here before the lockbox. Also fixed same-commit: `hit_target`
  now scans 20 bars (was 60; column is `*_20d`) — unused by the model, hygiene
  only. Added `cs_rank_composite_full` (strict all-3-horizon composite) for
  stationary CV/drift; dedup now fails loud above 1% duplicate rows.

  *Eval-side sibling (done 2026-06-27):* `pipeline/validation/metrics.py` no
  longer `fillna(0)`s the relevance label — fold NDCG/precision are now graded
  only on known-outcome stocks (return metrics already dropna). Slightly changes
  the HPO fold objective on the delisting/tail subset → recipe-affecting, lands
  in the same retrain. Also added a fail-loud upper bound on `TARGET_TWAP_WINDOW`
  (must be < shortest horizon). The validator's *primary* IC/top-decile use
  `lag=0` (matches the label's `close[t]` entry); `--fill_lag` drives only the
  separate realistic-fill diagnostic, so the primary ruler stays bit-equal.

- **2026-06-30 — New knob `ICT_IMPLEMENTATION_MODE` (OB/FVG trigger strictness).**
  `ICTFeatureEngine.compute()` accepts `implementation_mode="legacy"|"institutional"`
  (`engineer.py:52`, constant `_ICT_IMPL_MODE`). "institutional" hard-gates OB/FVG
  triggers to require a recent Break-of-Structure event within `bos_lookback`
  bars (`ob_bos_hard_gate`/`fvg_bos_hard_gate=True`), filtering out 2-candle
  patterns not tied to a confirmed structural break. **Frozen default =
  "legacy"** (unchanged from prior behavior). This is a **tuning-era A/B
  candidate only**: institutional mode likely raises OB/FVG signal-to-noise,
  but the gate is still BOS-conditioned — a trend-confirmation concept — so it
  may not fix the regime-dependence found in the MODEL_C ICT-only audit
  (legacy ICT, 66-feature subset: walk-forward mean IC = -0.00002, t = -0.01;
  full 88-feature v2 decomposition: 63% train→lockbox sign-flip rate vs 0% for
  the 16 zone-core features). Evaluate legacy vs institutional on ≤2023
  walk-forward CV before any lockbox use; do not tune this against 2024-2026.

- **2026-07-04 — New knob `PIVOT_FEATURES` (floor-pivot / CPR / Camarilla family).**
  New module `pipeline/features/pivots.py` (`PivotFeatureEngine`) wired into
  `engineer.py` `build()` (per-ticker) and gated by env `PIVOT_FEATURES`
  (call-time read, TWAP pattern; `pivot_features_enabled()`). **Frozen default =
  OFF** — with the knob unset the production panel, `selected_features.txt`, HPO
  recipe and all model artifacts are bit-identical to before (verified: default-off
  build adds zero columns; `tests/test_pivot_features.py::test_default_off_no_pivot_columns`).
  When ON, adds **69** `features_pivot_*` columns (vocabulary frozen in
  `pivots.PIVOT_FEATURE_COLS`): ATR-normalized level distances (floor PP/R1/S1,
  CPR TC/BC, Camarilla H3/H4/H5/L3/L4/L5), CPR width regime, virgin-CPR tracking,
  Camarilla behavior flags, opening relationships, two-day CPR relationship +
  bias confirm/reject, pivot trend side/streak + PP slope, PP acceptance, nearest
  level/support/resistance, and weekly/monthly/yearly pivots. All formulas per
  *Secrets of a Pivot Boss* (Camarilla 1.1/12…1.1/2, H5=(H/L)·C,
  TC=(Pivot−BC)+Pivot). Three deliberate deviations from the source draft, each
  documented in `pivots.py`: (i) TC/BC min/max-normalization applied everywhere
  (the book's own caveat; draft did it in only one place); (ii) trend side fixed
  so inside-band = Neutral; (iii) two-day relationship uses the book's overlap
  definition so all seven states are reachable (draft's if-order left the
  overlapping states unreachable).
  *Recompute:* pivot features are **truncation-invariant** (pure trailing
  functions of OHLC through each row's own date), so per-fold recompute is a
  documented no-op — enforced by
  `tests/test_pivot_features.py::test_truncation_invariance`.
  *Degrees of freedom:* the family's internal parameters (width lookback 60d,
  regime cutoffs 0.25/0.75/0.10, virgin lookback 60d, PP-slope windows 3/5d,
  acceptance windows 5/10d, Camarilla multiplier 1.1) are book/draft defaults —
  **prevalence-style, NOT outcome-validated.** Counted here as ONE family entry in
  the ledger pending the MODEL_D verdict. Also added: runner flag
  `run_sp500_local.py --stop_after_targets` (build features+targets checkpoint,
  run the leakage suite, then exit — runner control only, not recipe-affecting)
  to produce the pivot-enabled panel the experiment consumes.
  **This is a tuning-era experiment candidate only (MODEL_D). Do NOT enable
  `PIVOT_FEATURES` in a lockbox/production run before a positive tuning-era
  verdict is recorded below.**

  *MODEL_D pivot-only pre-registration (criteria fixed BEFORE the run):*
  - Harness: expanding-window walk-forward CV, yearly folds 2018–2023, LGBM
    lambdarank ndcg@10, num_leaves=31, features = all `features_pivot_*`, label =
    `cs_rank_composite`, metric = per-date rank-IC vs `future_20d_excess_return`
    (`scripts/experiments/model_d_pivot_only.py`). Identical to the MODEL_A/C audit.
  - Benchmarks: MODEL_A zone-core CV IC = +0.1441 (t=+8.53); MODEL_C ICT-only
    CV IC = −0.00002 (t=−0.01).
  - **Adopt for further work** iff CV **mean IC ≥ +0.03 AND IC t-stat ≥ 2.0 AND
    ≥ 4 of 6 folds positive** (the script computes and prints this gate).
  - **Lockbox static split** (`model_d_pivot_only_lockbox.py`, train ≤2023-12-31,
    score 2024→panel end) runs **ONCE, only if the CV gate passes.** Consistency
    bar: lockbox IC ≥ 50% of CV IC, same sign. This consumes one look at the
    2024–26 window (one-shot rule, §6) and is a diagnostic peek, not the
    production verdict. Record the read here after running.
  - **Otherwise:** pivots stay OFF (frozen default). Any re-cut of the 69-column
    v1 list is a new ledger entry and re-registration.
  - Result (2026-07-06): CV mean IC `+0.0092` t `+1.10` folds+ `4/6` |
    gate **FAIL** | lockbox: NOT RUN (gate failed — 2024-26 window preserved) |
    results: `/mnt/data/artefacts/experiments/model_d_results.json`.
    Pivots stay OFF. Post-hoc note: the MODEL_A benchmark quoted above
    (+0.1441) was later found to be leaked — see the 2026-07-08 entry; the
    MODEL_D number itself is honest (pivot features are truncation-invariant).

- **2026-07-08 — Zone look-ahead leak: discovery, audit, fixes, causal verdict.**
  *Discovery:* the panel build called `compute_zone_features(ohlcv)` with **no
  cutoff** (`engineer.py:488`), so ZoneAnalyzer saw each ticker's full CSV
  history (~2026-04) and backdated verdicts into every historical row via three
  mechanisms: formation `shift(-1)`, SDZ/SSZ breach scans over all future data,
  and `_base_eliminator` deleting any zone a future candle overlaps. Every
  zone feature therefore encoded "levels the future respected."
  *Audit (truncation-invariance, 5 tickers, cut at 2021, window 2018-2020):*
  ZONE 14/33 columns measurably rewritten (worst `features_zone_dist_atr`
  47.2% of cells; family guilty by generator), **ICT 0/153 exercised columns
  changed, PIVOT 0/68, BASE 0/37** — corruption confined to the zone family.
  MODEL_C and MODEL_D verdicts stand (their features were honest).
  **All previously recorded zone CV numbers (incl. the +0.1441 zone-core
  benchmark cited in the MODEL_D entry and the +0.1920 30-feature baseline of
  the 2026-07 bucket sweep) are VOID — leaked.**
  *Fixes (commit 3389c34):* (i) `--train_end` fence now REDRAWS zone columns
  with cutoff=train_end after row-fencing (row-slicing alone left leaked
  values in pre-fence rows); (ii) `recompute_fold_features` honors `skip_ict`
  (+ guard against double-multiplying ICT×trend scores); (iii) new
  decision-grade harness `scripts/experiments/model_a_causal_cv.py` (per-fold
  zone redraw at each fold's own cutoff; test rows = frozen-carry state →
  conservative lower bound). Separate fix in progress for the HPO fold loop's
  `te_panel` cutoff=test_end leak (task spun off 2026-07-07).
  *MODEL_A causal verdict (one run, 2026-07-08, same harness/gate as C/D):*
  | Config | n | mean IC | t | minIC | folds+ | leaked ref | inflation |
  |---|---|---|---|---|---|---|---|
  | Z zone-only | 30 | **+0.0069** | +1.98 | −0.0098 | 5/6 | +0.1920 | +0.1851 |
  | Z+B1 trend | 34 | +0.0074 | +1.12 | −0.0254 | 4/6 | +0.1969 | +0.1895 |
  | Z+B7 returns | 34 | +0.0090 | +1.19 | −0.0267 | 4/6 | +0.1983 | +0.1893 |
  | Z+B1+B7 | 38 | +0.0078 | +0.97 | −0.0294 | 4/6 | — | — |
  **GATE FAIL, all configs** (bar: IC ≥ 0.03, t ≥ 2.0, ≥ 4/6 folds+). ~96% of
  the leaked zone signal was the eraser. Honest three-family scoreboard:
  ICT −0.00002 (t −0.01) | pivots +0.0092 (t +1.10) | zones +0.0069 (t +1.98).
  *Decision:* zone lockbox CANCELLED (recipe was tuned on leaked numbers; the
  2024-26 window remains unspent). Zones stay out of production candidates
  until a pre-registered v2 (timeline-causal generator, event-dated
  invalidation) earns a new test. Next family: **MODEL_E momentum + base
  features, pre-registered BEFORE results in `docs/MODEL_E_PREREGISTRATION.md`
  (commit ee634df).** Results file:
  `/mnt/data/artefacts/experiments/causal_zone_cv_results.json`.

- **2026-07-08 — MODEL_E (short-window momentum): GATE FAIL.**
  Pre-registered ee634df. Baseline B7 (returns 1d/5d/20d/60d, audited causal):
  **IC20 = −0.0017, t = −0.66, GATE FAIL** — all six folds printed; verdict
  final and deterministic (seeded). Phase-1 bucket deltas (B3 +0.0095,
  B4 +0.0131, B8 +0.0065, B9 +0.0054) are single-fold artifacts — every one
  carried by 2020 alone; no config plausibly near the family gate. First run
  aborted mid-phase-2 by infrastructure (two identical nohup instances raced;
  a chained `poweroff` from the dead twin killed the survivor) — clean rerun
  launched for the complete record (`/tmp/model_e_sweep2.log`; fill summary
  when it lands). *Pre-registration lesson:* the relative KEEP t-guard
  (baseline_t − 1.0) is toothless when baseline t is negative — repaired in
  MODEL_E2 as max(2.0, baseline_t − 1.0). *Interpretation:* sub-month
  formation windows carry no momentum signal here, consistent with the
  literature (short horizon = weak reversal, not momentum).

- **2026-07-08 — MODEL_E2 (formation-window momentum): GATE FAIL, narrow-fail
  rule TRIGGERED.**
  Pre-registered 0414307 (kernel = 3/6/12-month returns skipping 21d,
  computed from close in-script; truncation-invariance verified in a synthetic
  smoke test before the run). One run, as specified:
  | Config | n | IC20 | t | minIC | f+ | IC40 | IC60 | PIT20 | verdict |
  |---|---|---|---|---|---|---|---|---|---|
  | KERNEL | 3 | +0.0136 | +1.60 | −0.0103 | 5 | +0.0106 | +0.0037 | +0.0047 | GATE FAIL |
  | KERNEL+V | 6 | +0.0056 | +0.54 | −0.0344 | 4 | +0.0088 | +0.0113 | +0.0019 | DROP |
  | **KERNEL+S** | 7 | **+0.0168** | **+1.93** | −0.0101 | 5 | +0.0159 | +0.0104 | +0.0117 | MARG / GATE FAIL |
  | KERNEL+V+S | 10 | +0.0106 | +0.98 | −0.0275 | 4 | +0.0142 | +0.0156 | +0.0014 | DROP |
  Fold structure is literature-consistent (2021 momentum-crash year negative
  in every config; post-crash 2020 strongest), vol-scaling hurts at 20d, and
  the PIT split shows the signal is stronger in the broad ~1591-name
  cross-section (+0.0168) than in the S&P-500 core (+0.0117) — crowding-
  consistent. **Honest kernel ladder (same panel/harness):** short-momentum
  −0.0017 < zones +0.0069 < pivots +0.0092 < formation momentum +0.0136 <
  formation+short +0.0168.
  **Decision (rule frozen pre-run, §6 of the E2 doc):** KERNEL+S landed inside
  the pre-declared narrow-fail band (0.015–0.03) → the **dead-ticker (Norgate)
  backfill is now the blocking task**; ONE sanctioned E2 re-run on the
  completed universe follows. No other E2 variant may run before that.
  (2026-07-08 update: backfill DEFERRED by user for cost; E2 re-run parked.
  Scope precisely quantified from stock_lists probe files: 187 Yahoo-dead
  ex-members = 14.5% of 2010-26 membership-days; the 34 "Yahoo-has-data"
  missing names are ALL symbol-reuse traps — never bulk-download them.)
  Results: `/mnt/data/artefacts/experiments/model_e2_results.json`.

- **2026-07-08 — MODEL_A causal verdict SUSPENDED: lambdarank group
  misalignment in the causal harness (user-flagged).**
  LightGBM `group=` arrays require date-major contiguous rows — the pipeline's
  own `cv.build_group_array` re-sorts for this (cv.py:258), but the standalone
  causal harness trained directly on `recompute_fold_features()` output, which
  is TICKER-major (engineer.py reorders to ["ticker","date"] before concat).
  All 24 causal fits therefore trained with garbage query groups. Empirically
  verified on a synthetic end-to-end build: `panel_targets.pkl` is date-major
  (**sweeps, MODEL_A-leaked, C, D, E, E2 groups were all CORRECT — those
  verdicts stand**); only the causal zone run is affected. The **+0.0069 zone
  verdict and the ~96% inflation figure are VOID** pending a re-run: honest
  zone IC is currently UNKNOWN (mis-trained model = loose lower bound only).
  *Fix (this commit):* causal harness reorders to date-major after recompute;
  ALL four experiment harnesses now hard-fail if the train slice is not
  date-major (invariant assertion before every `lgb.train`).
  *Action:* re-run `model_a_causal_cv.py` once, unchanged spec; record the
  corrected verdict here. The zone funeral is postponed, not cancelled —
  the features' non-causality finding (leak, audit, fence fix) is unaffected.

---

## 4. Procedure (all on Hetzner; isolated via ML_ARTEFACTS_ROOT)

`ML_ARTEFACTS_ROOT` redirects *every* output (scores, model artifacts,
checkpoints) so the fenced run cannot touch production artifacts, and forces a
fresh checkpoint build so no stale production panel can leak past the fence.

```bash
cd /root/ml-stock-predictor
git pull origin master                      # get --train_end + validator
export ML_ARTEFACTS_ROOT=/mnt/data/artefacts/us_lockbox

# 1. Seed the FENCED model: HPO + feature selection + weights, all <= 2023-12-31
python3 run_sp500_local.py --mode momentum \
    --train_start 2010-01-01 --train_end 2023-12-31 \
    --as_of 2023-12-29 --n_trials 40

# 2. Walk forward through the holdout, fenced, inference-only.
#    --train_end keeps every retrain fenced; --no_drift_retrain stops
#    mid-walk retrains. The model stays the 2023-frozen recipe throughout.
python3 run_walkforward_sp500.py \
    --start 2024-01-12 --end 2026-05-04 --cadence_days 14 \
    --mode momentum --train_end 2023-12-31 --no_drift_retrain \
    --log_dir /mnt/data/artefacts/us_lockbox/us_local

# 3. Independent verdict (read-only; recomputes returns from price CSVs)
python3 scripts/tools/validate_lockbox.py \
    --scores_dir /mnt/data/artefacts/us_lockbox/us_local/output \
    --data_dir   /mnt/data/Learning_charts/stock_data/us_stocks \
    --mode momentum --side bull --score_field model_score \
    --start 2024-01-01 --end 2026-05-06 \
    --out /mnt/data/artefacts/us_lockbox/lockbox_verdict.json
```

Sanity check during step 1: the log must print
`*** LOCKBOX FENCE ACTIVE: training capped at 2023-12-31`. If it does not, the
fence is off — abort.

---

## 5. Pre-registered pass/fail criteria

Verdict is on **momentum / bull / model_score @ 20d** from step 3.
*(Edit these to your final commitment before committing the file.)*

| Outcome | Criteria (ALL must hold for PASS) |
|---|---|
| **PASS** | non-overlap IC t-stat **> 2.0** AND mean IC **> 0.02** AND top-decile excess 95% CI **excludes 0** |
| **MARGINAL** | mean IC > 0 and top-decile excess > 0, but t-stat between 1 and 2 (real but underpowered) |
| **FAIL** | mean IC ≤ 0.02 or t-stat < 1 or top-decile CI includes 0 |

Interpretation fixed in advance:
- **PASS** → the edge survives fencing; proceed to (a) drop/shrink the composite
  blend, (b) fix entry timing to t+1 fills, (c) wire the execution/risk model,
  then a PIT-on confirmation run.
- **MARGINAL** → real but fragile; do NOT scale capital. Investigate the
  composite dilution and the momentum/reversal split before more work.
- **FAIL** → the in-sample edge was largely overfitting. Stop; rethink features
  before any further tuning.

The expected haircut from the §2 baseline is itself a data point: record
`lockbox_IC / insample_IC` as the overfitting ratio.

---

## 6. The one-shot rule (non-negotiable)

1. **One run, one verdict.** A disappointing result may NOT be answered by
   tweaking a threshold and re-running on this same 2024–26 window — that burns
   the lockbox forever. Go back to ≤2023 data to rethink; the next clean test is
   *future* data.
2. **Criteria are frozen at commit time** (§5). No post-hoc redefinition.
3. **The future is the renewable lockbox.** From today, log live/paper picks
   with committed timestamps and re-run the validator quarterly. That is the
   only perfectly clean test and it costs nothing but patience.

---

## 7. Result (fill in after the single run)

- Run date / git commit: `__________`
- n_folds used / n_trials: `__________`
- mean IC @20d: `______`  |  non-overlap t-stat: `______`  |  positive rate: `____`
- top-decile excess @20d: `______`  CI `[____, ____]`
- Overfitting ratio (lockbox / in-sample IC): `______`
- **Verdict (PASS / MARGINAL / FAIL):** `__________`
- Notes: `__________`
