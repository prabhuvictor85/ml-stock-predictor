# `pipeline/validation/cv.py` — How Walk-Forward CV Works

## The problem it solves

A naive `train_test_split` on stock data leaks the future into the past in two ways:

1. **Lookahead contamination** — a 20-day forward return computed at time T uses
   prices from T+1 to T+20. If the test fold starts at T and training includes
   T+5, the training label for T+5 *overlaps* the test period's raw prices.
2. **Serial correlation** — adjacent days share almost all their information.
   Splitting them into train/test gives you an artificially clean separation that
   doesn't exist in production.

`PurgedWalkForwardCV` closes both gaps with **purge** and **embargo** gaps
between every training and test block.

---

## The core idea: expanding train, fixed-length test

```
      |<-- min_train_window (504 td) -->|<P+E>|<-- test_window (252 td) -->|
Fold 0: [===========================TRAIN===][   ][==========TEST===========]
Fold 1: [==============================TRAIN======][   ][=====TEST============]
Fold 2: [=================================TRAIN=========][   ][====TEST========]
...

  TRAIN grows (expanding window)
  P+E = purge(40) + embargo(5) trading-day gap
  TEST is always exactly 252 trading days (≈1 year)
```

Each fold sees progressively more training history — this simulates the reality
that a live model accumulates data. The test block never overlaps with training.

---

## Key parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_train_window` | 504 td | Model must see ≥2 years before it may predict |
| `test_window` | 252 td | Each test fold spans exactly 1 year |
| `purge_window` | 40 td | Labels computed at the end of training reach 40 trading days *forward* — these rows would contain future information from the test period. They are removed from training. |
| `embargo_window` | 5 td | 5 extra days of buffer after purge — covers microstructure autocorrelation (bid/ask, price impact) that persists past the label window |
| `n_folds` | 8 (≥5) | Number of test folds. Reduces automatically if the panel is too short. |

### Why 40-day purge?

The longest return label used is `future_60d_return`. But the *training*
objective uses `cs_rank_composite`, which blends 20d/40d/60d. If train_end = T,
the 60d return at T looks *ahead* to T+60 — squarely inside the test period.
So the purge must be ≥60 days.

**But we use 40**, not 60. The composite's dominant weight (0.5) is on 20d, and
the model trains on `cs_rank_composite`, not raw returns. The 40+5=45 trading-day
gap (≈9 calendar weeks) is a deliberate compromise: 45 td is enough to prevent
the 20d component from bleeding, and the 40d/60d tail is only lightly weighted.
`PURGE_HORIZON = 80` in `builder.py` is the *conservative* purge used inside CV
for the label-validity mask — not this CV split purge.

---

## `FoldSpec` — one fold described in four dates

```python
@dataclass
class FoldSpec:
    fold_id    : int
    train_start: pd.Timestamp   # always the first day in the panel
    train_end  : pd.Timestamp   # last safe training day (purge+embargo applied)
    test_start : pd.Timestamp   # first group_date on or after the test window start
    test_end   : pd.Timestamp   # last group_date on or before the test window end
```

`test_start` / `test_end` are **snapped to `group_date` values**, not raw
calendar dates. That matters because NDCG/IC are computed per `group_date` (a
rebalance date every ~20 trading days). Snapping prevents half-empty groups at
the fold edge.

---

## `get_fold_specs(panel)` — building the timeline

1. Collect all unique trading dates from the panel index.
2. Collect all `group_date` values (the rebalance dates).
3. For fold `i`, the test window starts at index `min_train_window + i × test_window`
   in the sorted date list.
4. Snap `test_start` → first `group_date` ≥ raw start; snap `test_end` →
   last `group_date` ≤ raw end.
5. Count back `purge_window + embargo_window` trading days from `test_start` to
   get `train_end` — using the panel's *actual* trading calendar, not a fixed
   number of calendar days (45 calendar days ≠ 45 trading days; the code avoids
   that off-by-one).

If the panel is too short for `n_folds`, it reduces to `possible_folds`
(minimum `MIN_FOLDS = 5`), logs a warning, and continues.

---

## Per-stock dynamic inclusion — the detail that matters most

Stock `AAPL` has data from 2000. Stock `NVDA` (in its current form) only
has enough liquid history from 2015. If fold 3's `train_end` is 2016-12-31,
NVDA only has 2 years of data — it should not be in that fold's training set
because the model hasn't had enough signal from it to rank it reliably.

```
            fold 3 train_end = 2016-12-31
                    ↑
min_train_window = 504 td
                    ↑
sufficient_start = 2014-12-XX  (504 trading days before train_end)
                    ↑
AAPL first date: 2000 → 2000 ≤ 2014-12-XX → ELIGIBLE ✓
NVDA first date: 2015-01 → 2015-01 > 2014-12-XX → EXCLUDED ✗
```

`_eligible_tickers()` implements this by finding the exact trading day that is
exactly `min_train_window` slots before `train_end` in the panel's own calendar
— no approximation with `timedelta(days=N)`.

Both **training** and **test** sets apply the same eligibility mask: if a stock
wasn't eligible for training, it's also excluded from test (you can't evaluate
something the model never saw).

---

## `split(panel)` — what the caller gets

```python
for fold_spec, train_idx, test_idx in cv.split(panel):
    X_train = panel.iloc[train_idx][feature_cols]
    y_train = panel.iloc[train_idx]["cs_rank_composite"]
    X_test  = panel.iloc[test_idx][feature_cols]
    # ... score → metrics.compute_fold_metrics(...)
```

`train_idx` and `test_idx` are **integer positions** into the panel's row order
(not row labels). This is important: LightGBM's `group=` array and the index
slices must stay aligned — positional indexing avoids accidental label/iloc
mismatches.

The fold loop also logs eligibility per fold:
```
Fold 3: 312/1591 stocks eligible (1279 excluded — insufficient history before 2016-12-31)
```

---

## `build_group_array(panel)` — the LightGBM group structure

LightGBM LambdaRank requires a `group=` array that says "these N rows all belong
to the same query". Here, each query is one `group_date` (a cross-section of
stocks on one rebalance day).

```python
panel, groups = cv.build_group_array(train_panel)
ranker.fit(X, y, group=groups)
```

Steps:
1. Filter to `in_universe == True` (non-universe rows have no valid rank label).
2. Sort by `[group_date, ticker]` — LightGBM requires rows to be contiguous
   within each group.
3. Drop any `group_date` with fewer than `MIN_TRAIN_GROUP_SIZE = 10` tickers
   (a degenerate group produces a degenerate ranking loss).
4. Count rows per group → `group_sizes_arr`.
5. **LightGBM hard-limits each query to 10,000 rows.** Any group exceeding
   9,900 is split into equal sub-groups with `ceil(sz / 9900)` chunks. The
   SP500 universe never hits this limit (~500 stocks max), but the NSE universe
   can (~1,600 tickers).

---

## Sequence diagram — one Optuna trial

```
run_sp500_local.py
  └─ PurgedWalkForwardCV.split(panel)
       ├─ Fold 0: train_idx, test_idx
       │    ├─ build_group_array(train_panel) → groups
       │    ├─ lgbm_ranker.fit(X_tr, y_tr, groups)
       │    ├─ lgbm_ranker.predict(X_te) → scores
       │    └─ metrics.compute_fold_metrics(panel_te, scores) → {ndcg, ic, ...}
       ├─ Fold 1 ...
       └─ Fold N → aggregate across folds → trial score (mean top_decile_excess)
```

If `mean(top_decile_excess) ≤ 0` across all folds, the trial is penalised
(`return -1.0`), not pruned — pruning would let Optuna's TPE discard the
data point; a penalty score keeps it as evidence of a bad region.

---

## Numbers at a glance (SP500 lockbox, fence 2023-12-31)

| Item | Value |
|---|---|
| Panel date range | 2010-01-01 → 2023-12-31 (≈3521 trading days) |
| min_train_window | 504 td |
| test_window | 252 td |
| Available folds | (3521 − 504) / 252 = **12.0** |
| Purge + embargo | 45 trading days ≈ 9 calendar weeks |
| Stocks eligible by fold 0 | ~312 / 1591 (rest don't have history back to 2012) |
| Stocks eligible by fold 12 | ~1300+ / 1591 |

---

## What this module does NOT do

- It does **not** compute metrics — that's `pipeline/validation/metrics.py`.
- It does **not** run HPO — Optuna lives in `run_sp500_local.py`.
- It does **not** handle the final (out-of-CV) model fit — that uses the full
  training panel (`train_start → train_end`) without any fold splitting.
- It does **not** implement "nested" CV (no inner HPO loop per fold). HPO runs
  once over the fold aggregate, then the best config is applied to the final fit.
  This is a design choice: nested CV is more rigorous but ~8× slower.
