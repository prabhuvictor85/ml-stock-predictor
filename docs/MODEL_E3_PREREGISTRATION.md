# MODEL_E3 — Horizon × Terminal-Smoothing Sweep (Pre-registration)

**Status:** FROZEN before first run. Commit hash of this file = freeze proof.
Written 2026-07-08, after MODEL_E2, before any MODEL_E3 configuration has run.

---

## 1. Hypothesis

Two extensions to MODEL_E2, motivated by E2's pre-registered *informational*
40d/60d readings (which were often stronger than 20d — e.g. 2023 IC20 +0.0007
vs IC60 +0.032):

1. **Horizon:** formation-momentum's predictive edge is stronger at longer
   holding horizons (up to ~6 months / 120 trading days).
2. **Terminal smoothing:** grading against a 5-day TWAP exit (instead of the
   single close on day t+h) is more robust to a one-day shock landing on the
   exit date, and may raise measured IC.

> A pre-declared PRIMARY cell — formation-momentum score, 60-day horizon,
> 5-day TWAP exit — predicts forward excess return with IC ≥ 0.03, t ≥ 2.0,
> ≥ 4/6 folds positive.

Using E2's informational results to choose E3's primary is legitimate: that is
exactly what pre-registered exploratory readings are for — generating the next
experiment's hypothesis.

## 2. Fixed elements (no new feature/model degrees of freedom)

- **Features:** E2 `KERNEL+S` = {3m,6m,12m formation returns skip-21d} +
  {return_1d/5d/20d/60d} (E2's best config). Unchanged, audited causal.
- **Training label:** existing `cs_rank_composite` (0.5·20d+0.3·40d+0.2·60d).
  NOT re-weighted to include longer horizons — horizon is a *grading*
  dimension here, not a training knob (adding it would be a new DoF → E4).
- Same panel features, same 6 yearly folds 2018–2023, same seeded LGBM
  lambdarank, same date-major group construction, same ddof=1 t-stat
  (both bugs fixed 2026-07-08: commits 2bad90a, f087ff2).

## 3. Grid (exploratory map)

Horizons **{20, 40, 60, 80, 100, 120}** × terminal windows **{1, 5}** = **12 cells**.

Terminal window changes the label (TWAP smooths future_{h}d_return → changes
cs_rank_composite), so there are **2 target builds** (TWAP=1, TWAP=5); within
each, the model trains ONCE on cs_rank_composite and is graded at all 6
horizons. → 2 trainings, 12 IC readings.

## 4. PRIMARY vs EXPLORATORY (the multiplicity discipline)

**PRIMARY cell — the only one that can claim a pass or trigger further work:**
horizon = 60, terminal window = 5. Judged at the standard campaign gate:
**IC ≥ 0.03 AND t ≥ 2.0 AND ≥ 4/6 folds positive.**

**EXPLORATORY — the other 11 cells:** reported as a map only. To be flagged
even "interesting," a cell must clear a **Bonferroni-corrected** bar:
family-wise one-sided α = 0.05 over 12 tests → per-test α = 0.00417 →
**t ≥ ~4.0** at 5 df. No exploratory cell may claim a pass inline; the most it
can do is justify a NEW pre-registration.

**Honest power caveat (stated before results):** with only 6 folds (5 df), a
12-cell sweep genuinely *cannot* be evaluated at family-wise 0.05 — the
Bonferroni t-threshold (~4.0) is near-unreachable with 6 yearly observations,
and nothing in this entire campaign has reached even t = 2.0. Two consequences,
both accepted up front:
  (a) The 12-cell grid is **hypothesis-generating, not confirmatory.**
  (b) Bonferroni over 12 is conservative because horizons are highly
      correlated (IC60 and IC80 are nearly the same test); the true effective
      number of independent tests is smaller. We still quote the conservative
      bound — the point is moot until something approaches t = 2.

This is *why* the design commits to ONE primary cell rather than "best of 12":
the primary is a single confirmatory test at the normal gate; the grid around
it is a map.

## 5. Known limitations (before results)

- **Survivorship floor** (shared with E2): dead tickers absent → momentum IC
  biased down. A narrow primary result (IC 0.015–0.03) routes to the SAME
  Norgate dead-ticker backfill prerequisite as E2, and the sanctioned re-run
  covers both.
- **Tail loss grows with horizon:** the last ~120 trading days of the usable
  window get NaN labels at h=120 (vs ~60 before), shrinking the gradeable set,
  especially in the latest fold.
- Windows (21d skip, 63/126/252 formation, 5d TWAP) are literature/robustness
  defaults, not tuned — one family entry in the DoF ledger.

## 6. Decision tree (before results)

- **Primary passes** → freeze (horizon, TWAP, KERNEL+S) recipe; lockbox path
  (machinery transfers). Low prior given the campaign, but this is the design
  that gives momentum its best honest shot.
- **Primary narrow-fail (0.015–0.03)** → merge with E2's Norgate-backfill
  prerequisite; one combined re-run on the completed universe.
- **Primary clean fail AND no exploratory cell clears t ≥ 4** → formation
  momentum carries no gate-clearing edge at any tested horizon; horizon
  extension is exhausted. Move to: honest all-features pipeline CV (after the
  te_panel cutoff fix), then NSE universe port.

## 7. References (same panel/harness, ddof=1 corrected)

| Model | best IC20 | t (corrected) | Gate |
|---|---|---|---|
| MODEL_C ICT | −0.00002 | −0.009 | FAIL |
| MODEL_D pivots | +0.0092 | +1.00 | FAIL |
| MODEL_A zones (causal) | pending re-run (group+ddof fixes) | — | — |
| MODEL_E short-momentum | −0.0017 | −0.60 | FAIL |
| MODEL_E2 KERNEL+S | +0.0168 | +1.76 | FAIL |
| MODEL_E3 | _this experiment_ | — | — |
