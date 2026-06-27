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
