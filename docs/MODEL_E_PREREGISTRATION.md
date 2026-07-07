# MODEL_E — Momentum + Base-Feature Family (Pre-registration)

**Status:** FROZEN before first run. The commit hash of this file is the freeze proof.
Written 2026-07-08, before any MODEL_E configuration has been trained.

---

## 1. Hypothesis

Cross-sectional momentum (trailing normalized returns) plus basic technical state
(trend flags, breakouts, volatility, volume, market context) carries a positive,
statistically significant edge for ranking S&P-universe stocks by forward excess
return, on the 2018–2023 tuning era.

**Why this family, why now:**
- All three structural families are closed: ICT IC≈0.00 (MODEL_C), pivots +0.0092
  (MODEL_D, gate fail), zones pending causal verdict (MODEL_A causal CV) after the
  look-ahead audit invalidated the leaked +0.19.
- B7 (returns) was the ONLY bucket that added signal in the 2026-07 sweep
  (+0.0062) — and it earned that while competing against a leaked baseline that
  absorbed most available signal. Its features are audited causal.
- Cross-sectional momentum is the most-replicated equity anomaly in the
  literature; this is the highest-prior untested family in the codebase.

## 2. Design (identical harness to MODEL_A/C/D — no new degrees of freedom)

| Element | Value |
|---|---|
| Panel | `us_pivot_v1` panel_targets.pkl (`--pit_universe`), fenced ≤ 2023-12-31 |
| CV | 6 expanding-window yearly folds, test years 2018–2023 |
| Model | LGBM lambdarank, num_leaves=31, n_estimators=400, lr=0.05, all seeds=42 |
| Training label | `cs_rank_composite` (0.5×20d + 0.3×40d + 0.2×60d rank blend) |
| Primary metric | per-date Spearman IC vs `future_20d_excess_return` |
| Causality | all candidate features passed the 2026-07-07 truncation audit — no recompute needed |

## 3. Configurations

**Baseline:** B7 returns only — `features_return_1d/5d/20d/60d` (4 features).

**Buckets tested via the 3-phase sweep** (individual → greedy forward selection →
ALL), each on top of the B7 baseline:
B1 trend (4), B2 regime (3), B3 vol/ADX (11), B4 SMA/breakout (7), B5 volume (2),
B6 context (2), B8 SMA slopes (3), B9 misc (any `features_*` column that is not
zone/ICT/pivot and not in B1–B8 — discovered at runtime and printed, so no base
feature silently escapes the test).

## 4. Pre-registered gates (decided before any result)

**Family gate (on the B7 baseline AND on the greedy-final config):**
mean 20d IC ≥ 0.03 AND t-stat ≥ 2.0 AND ≥ 4/6 folds positive.

**Bucket KEEP rule:** delta IC ≥ +0.005 (≈3× the measured ~0.0015 seeded-rerun
noise floor) AND t-stat ≥ (baseline t-stat − 1.0). The t-guard is relative, not
the zone-era absolute 8.0, because this family's baseline t is unknown a priori.

**Horizon policy:** 40d and 60d ICs are computed and reported for every config
(the same scores correlated against `future_40d/60d_excess_return` — the panel
already carries these targets). They are INFORMATIONAL ONLY. The gate is 20d.
If 20d fails but 40/60d looks strong, that is a NEW hypothesis requiring a fresh
pre-registration — it is NOT a pass of this one.

**One run.** This experiment is run once as specified. Any change to configs,
thresholds, or horizons after seeing results = new experiment, new document.

## 5. Researcher-degrees-of-freedom ledger (additions)

| Knob | Value | Justification |
|---|---|---|
| Baseline choice | B7 (4 returns) | only KEEP bucket in 2026-07 sweep; literature prior |
| KEEP delta threshold | +0.005 | reused from zone sweep (3× noise floor), unchanged |
| KEEP t-guard | baseline_t − 1.0 | relative form of the zone-era rule |
| Horizon reporting | 20d gate, 40/60d info | pre-registered to prevent horizon-shopping |

## 6. Decision tree (written before results)

- **Baseline or greedy-final passes the family gate** → freeze the winning feature
  list here + PROTOCOL.md, then lockbox walk-forward (2024→) with the frozen
  recipe. The lockbox machinery, fence (with zone redraw), and validator transfer
  unchanged.
- **Fails the gate** → record in PROTOCOL.md §3.1 next to MODEL_C/D/A. Next
  pre-registered candidates, in order of cheapness: (a) 40/60d-horizon variant of
  this family (new doc), (b) NSE universe port, (c) zone-v2 with timeline-causal
  generator (event-dated invalidation), (d) fundamental/earnings features.
- **No family ever passes** → the honest product is the screening pipeline +
  the knowledge that these daily-bar features carry no 20d cross-sectional edge
  in this universe. Capital protected is the floor return of this project.

## 7. Reference results (same panel, same harness)

| Model | Features | Mean IC | t | Causality | Gate |
|---|---|---|---|---|---|
| MODEL_C ICT-only | ~115 | −0.00002 | −0.01 | honest (audited) | FAIL |
| MODEL_D pivot-only | 69 | +0.0092 | +1.10 | honest (proven) | FAIL |
| MODEL_A zone-only (leaked) | 30 | +0.1920 | +9.12 | LEAKED | void |
| MODEL_A zone-only (causal) | 30 | +0.0069 | +1.98 | honest | FAIL (2026-07-08) |
| MODEL_E configs | 4–40 | _this experiment_ | — | honest | — |
