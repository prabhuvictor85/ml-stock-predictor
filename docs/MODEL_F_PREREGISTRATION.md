# MODEL_F — All-Causal-Features CV (Pre-registration)

**Status:** FROZEN before first run. Commit hash = freeze proof.
Written 2026-07-09, before any MODEL_F configuration has run.

---

## 1. Hypothesis

Letting the model use the FULL causal feature set (not one family) produces a
higher, gate-clearing IC than the momentum-only baseline — i.e. the dead
standalone families (ICT +0.00, pivots +0.009) contribute *incrementally* on
top of momentum even though each is worthless alone.

Null (the campaign's default expectation): they do not — more features add
noise-drag, and IC stays at or below the momentum floor (~+0.0168).

## 2. Why a standalone harness, not the production pipeline

The production `--full` training's internal HPO/CV recomputes test-fold
features with `cutoff_date=spec.test_end` (run_sp500_local.py:1038) — a
within-fold look-ahead (task_5ebab842, unfixed here) — so its CV number is
contaminated. MODEL_F uses the clean standalone pattern (fold = plain panel
slice, no recompute) proven in C/D/E/E2, sidestepping that leak. The only
non-causal family in the panel is zones; MODEL_F EXCLUDES them.

## 3. Feature set

**All `features_*` columns EXCEPT the zone family** (prefixes sdz/ssz/dz/sz/
zone/any_valid — non-causal in the panel AND proven dead), PLUS the E2
momentum kernel (e2_mom_3m/6m/12m, computed in-script from close).
= base (~37) + ICT (~184) + pivots (~69) + momentum (3) ≈ 290 causal features.
All passed the 2026-07-07 truncation audit (zones were the only failures).

## 4. Configurations

1. **MOM** — KERNEL+S (momentum reference, known ~+0.0168).
2. **ALL** — every causal feature (~290), no selection. Does the kitchen sink
   beat momentum, or does noise-drag win (as it did in the zone sweep's
   66-feature ALL cell)?
3. **BASE+MOM** — base technical state + momentum only (drop ICT+pivots, the
   proven-dead families). Isolates whether *clean* context adds to momentum.
4. **ALL+SELECT** — every causal feature, then LGBM gain-importance top-40 kept
   per fold (a light, leak-free fold-local selection; the lawful "use
   everything then prune"). Selection fit on train fold only.

## 5. Pre-registered gate

Same as the whole campaign: mean 20d IC ≥ 0.03 AND t ≥ 2.0 (ddof=1) AND
≥ 4/6 folds positive. Applied to ALL and ALL+SELECT (the "all-features"
claims). MOM/BASE+MOM are diagnostic references.

**One run.** Config list and gate frozen here. Selection = fold-local LGBM
gain top-40, no peeking at test folds.

## 6. Decision tree

- **ALL or ALL+SELECT passes** → the edge lives in the full set; THEN it's
  worth fixing the te_panel leak and running the real pipeline for a
  lockbox verdict. (First gate-clearing config of the campaign.)
- **Fails but ALL+SELECT > MOM by ≥ 0.005** → selection extracts incremental
  signal; pre-register a focused follow-up on the selected features.
- **Fails, no config beats MOM** → confirms momentum is the ceiling and extra
  families are noise; daily-bar technical features are exhausted on this
  universe. Move to: NSE port, then (if funded) the Norgate backfill re-run.

## 7. References (same panel/harness, ddof=1)

| Config | IC20 | t | Gate |
|---|---|---|---|
| ICT-only (C) | −0.00002 | −0.009 | FAIL |
| pivot-only (D) | +0.0092 | +1.00 | FAIL |
| zone-only causal (A) | pending | — | — |
| momentum KERNEL+S (E2/E3) | +0.0168 | +1.76 | FAIL |
| MODEL_F | _this experiment_ | — | — |

---

## 8. Pre-run amendment (2026-07-11 — before any MODEL_F execution)

Recorded in the PROTOCOL.md ledger (same date). No MODEL_F configuration has
run; amending before first execution preserves the freeze discipline.

1. **Fold-boundary purge.** The train slice for test year Y now ends
   MAX_FORWARD_HORIZON (60) trading days before Y's first trading day, so no
   train label window (`cs_rank_composite` looks up to 60 td ahead) overlaps
   test-period returns. This is the only harness change vs the C/D/E/E2
   pattern. The §7 reference numbers were produced WITHOUT the gap and are
   marginally optimistic — treated as a caveat, not re-run.
2. **§2 rationale update.** The te_panel leak cited there as unfixed was fixed
   in commit 3f8c91c (2026-07-10), after this doc froze. The standalone
   harness is retained regardless, for comparability with the §7 references.
3. **NaN handling pinned.** Train/test rows are dropped only on the momentum
   kernel columns + the required label (matching E2's effective row set, since
   kernel non-null implies short-return non-null); all other features stay
   NaN-native for LGBM. No fillna anywhere.
4. **Selection determinism pinned.** ALL+SELECT ranks by LGBM gain importance
   from a seeded preliminary fit on the train fold only, keeps the top 40,
   then refits on those 40 with the same seeds. Selected lists are recorded
   per fold in the results JSON.
