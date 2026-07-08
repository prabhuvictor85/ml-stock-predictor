# MODEL_E2 — Formation-Window Momentum (Pre-registration)

**Status:** FROZEN before first run. Commit hash of this file = freeze proof.
Written 2026-07-08, after MODEL_E's baseline verdict, before any MODEL_E2
configuration has been trained.

---

## 1. Hypothesis

MODEL_E tested momentum with 1d/5d/20d/60d stopwatches and failed
(baseline IC20 = −0.0017, t = −0.66). The momentum literature is specific:
at sub-month windows the documented effect is weak *reversal*; the anomaly
lives at **3–12 month formation, skipping the most recent month**
(Jegadeesh & Titman 12-1). MODEL_E never measured that. E2 does:

> Stocks ranked by 3/6/12-month trailing return (skipping the last month)
> continue to outperform peers over the next 20–60 trading days.

## 2. Features (computed in-script from panel `close` — trailing arithmetic,
truncation-safe by construction; no FE rebuild)

**Kernel (baseline, 3 features):**
- `e2_mom_3m`  = close[t−21] / close[t−63]  − 1
- `e2_mom_6m`  = close[t−21] / close[t−126] − 1
- `e2_mom_12m` = close[t−21] / close[t−252] − 1

**Bucket V — vol-scaled momentum (3):** each kernel feature divided by the
ticker's trailing 126d daily-return std (documented "momentum quality"
variant; per-stock scaling matters for cross-sectional ranking).

**Bucket S — short windows (4):** the existing `features_return_1d/5d/20d/60d`
(MODEL_E's kernel) — tests whether short-term information adds anything on
top of proper formation windows (incl. short-term-reversal interaction).

Configs (all four run, one pass): `KERNEL`, `KERNEL+V`, `KERNEL+S`, `KERNEL+V+S`.

## 3. Design (identical harness to MODEL_A/C/D/E)

Same panel (us_pivot_v1), same 6 yearly folds 2018–2023, same seeded LGBM
lambdarank, label = `cs_rank_composite`. Universe note: the cross-section is
all ~1,591 tickers (S&P 500 + NASDAQ names), consistent with every prior
experiment.

**Grading:** primary = per-date Spearman IC vs `future_20d_excess_return`,
full cross-section. Informational (pre-registered as such): IC at 40d/60d,
and IC20 within the PIT S&P-500 subset (`in_universe == True`) — same scores,
second grading population, to locate any signal (crowded core vs broader tail).

## 4. Pre-registered gates

**Family gate** (KERNEL and the best config): mean IC20 ≥ 0.03 AND t ≥ 2.0
AND ≥ 4/6 folds positive — full cross-section, 20d only.

**Bucket KEEP rule (repaired after MODEL_E's t-guard degeneracy):**
delta ≥ +0.005 AND config t ≥ **max(2.0, baseline_t − 1.0)**. The absolute
floor of 2.0 prevents the guard going toothless when the baseline t is weak
or negative.

**Horizon/subset policy:** 40d/60d and PIT-subset numbers are informational.
A pass there with a 20d full-universe fail = new hypothesis, new document.
**One run, as specified.**

## 5. Known limitations (stated before results)

- **Survivorship floor:** the panel's ticker set is survivors-to-2026 (dead
  tickers absent — Norgate backfill pending). Momentum's short leg earns from
  losers that keep losing into delisting; without them, measured momentum IC
  is biased DOWN. A pass here is strong evidence; a narrow fail is ambiguous
  and the dead-ticker backfill becomes the blocking task before final verdict.
- 21d skip and 63/126/252d windows are literature defaults, not tuned here.
  Counted as one family entry in the DoF ledger.

## 6. Decision tree (written before results)

- **Pass** → bucket winners frozen; candidate recipe recorded in PROTOCOL.md;
  lockbox machinery transfers unchanged.
- **Narrow fail (IC20 in 0.015–0.03 or PIT/full split divergent)** → dead-ticker
  backfill first, then ONE re-run of this same spec on the completed universe
  (pre-registered here as the only sanctioned repeat).
- **Clean fail** → next in queue: honest all-features pipeline rerun (after
  te_panel fix), NSE universe port, zone-v2 timeline-causal generator.

## 7. References (same panel, same harness)

| Model | Mean IC20 | t | Gate |
|---|---|---|---|
| MODEL_C ICT | −0.00002 | −0.01 | FAIL |
| MODEL_D pivots | +0.0092 | +1.10 | FAIL |
| MODEL_A zones (causal) | +0.0069 | +1.98 | FAIL |
| MODEL_E short-momentum baseline | −0.0017 | −0.66 | FAIL |
| MODEL_E2 | _this experiment_ | — | — |
