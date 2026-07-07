# Feature Addition Roadmap — Zone Model Incremental Testing

**Baseline (2026-07-05):** Zone model (MODEL_A) achieves IC=+0.1922, t=+9.11, 6/6 folds positive on 2018-2023 tuning era using 30 zone features.

**Protocol:** Each bucket must be tested via the `model_a_zone_only.py` harness on the same 2018-2023 tuning era. A new bucket is added only after the previous one is evaluated. Record IC, t-stat, and delta vs baseline before moving to the next. Do NOT test on the 2024-2026 lockbox window — that is reserved for the final verdict.

---

## Current Zone Feature Set (30 features — baseline)

| Feature | Description |
|---|---|
| `features_zone_active` | Price inside or touching an active zone |
| `features_zone_dist_atr` | Distance from nearest zone in ATR — **#1 signal** |
| `features_zone_strength` | Strength/confidence of nearest zone |
| `features_zone_htf_confluence` | SDZ minus SSZ (signed zone bias) |
| `features_sdz_1d/1wk/1mo/3mo/1y` | Supply-demand zone active flag per timeframe |
| `features_ssz_1d/1wk/1mo/3mo/1y` | Supply-of-supply zone active flag per timeframe |
| `features_dz_1d/1wk/1mo/3mo/1y` | Demand zone active flag per timeframe |
| `features_sz_1d/1wk/1mo/3mo/1y` | Supply zone active flag per timeframe |
| `features_sdz_raw_score` | Weighted composite across all SDZ timeframes |
| `features_ssz_raw_score` | Weighted composite across all SSZ timeframes |
| `features_dz_raw_score` | Weighted composite across all DZ timeframes |
| `features_sz_raw_score` | Weighted composite across all SZ timeframes |
| `features_sdz_htf_score` | SDZ strength × bullish trend alignment |
| `features_ssz_htf_score` | SSZ strength × bearish trend alignment |

---

## Bucket 1 — Multi-Timeframe Trend (4 features) ⬅ TEST FIRST

**Rationale:** Already computed in every FE run, just excluded by prefix filter. Directly complement zone proximity — a zone trade with trend alignment is higher conviction than a counter-trend zone touch. These already feed into `sdz_htf_score` / `ssz_htf_score` internally, but as explicit features the model can weight them independently.

| Feature | Description |
|---|---|
| `features_weekly_trend` | Binary: close > SMA(20) on weekly bars |
| `features_monthly_trend` | Binary: close > SMA(60) on monthly bars |
| `features_quarterly_trend` | Binary: close > SMA(120) on quarterly bars |
| `features_yearly_trend` | Binary: close > SMA(240) on yearly bars |

**How to test:** Add `"features_weekly_trend", "features_monthly_trend", "features_quarterly_trend", "features_yearly_trend"` to `ZONE_PREFIXES` in `model_a_zone_only.py` (or use an `--add_cols` argument).

**Result:** _(not yet tested)_
| Metric | Baseline | With Bucket 1 | Delta |
|---|---|---|---|
| Mean IC | +0.1922 | — | — |
| t-stat | +9.11 | — | — |
| Folds+ | 6/6 | — | — |

---

## Bucket 2 — Market Regime (3 features)

**Rationale:** Benchmark-level bull/bear/choppy state. A demand zone in a bull regime is more likely to hold than the same zone in a bear regime. Zero-cost to add — always computed from the benchmark close.

| Feature | Description |
|---|---|
| `features_regime_bull` | Binary: benchmark > SMA200 AND SMA20 > SMA50 |
| `features_regime_bear` | Binary: benchmark < SMA200 AND SMA20 < SMA50 |
| `features_regime_choppy` | Binary: neither bull nor bear |

**Result:** _(not yet tested)_
| Metric | Baseline | With B1+B2 | Delta vs B1 |
|---|---|---|---|
| Mean IC | +0.1922 | — | — |
| t-stat | +9.11 | — | — |
| Folds+ | 6/6 | — | — |

---

## Bucket 3 — Volatility & ADX (11 features)

**Rationale:** Vol contraction + zone proximity is the classic squeeze setup — price coiling at a zone before a move. ADX tells whether trend momentum is strong enough to respect or break through the zone. `adx_bull`/`adx_bear` split directional strength.

| Feature | Description |
|---|---|
| `features_atr_pct_rank_252` | ATR percentile rank in 252-day window |
| `features_vol_contraction` | ATR(14) / max(ATR, 60d) — compression ratio |
| `features_compression_score` | 1.0 − vol_contraction |
| `features_adx_14` | Wilder ADX (14-period) |
| `features_plus_di` | Positive directional indicator |
| `features_minus_di` | Negative directional indicator |
| `features_adx_dir` | Sign of (+DI − −DI): +1 bull, −1 bear |
| `features_adx_bull` | ADX where +DI > −DI, else 0 |
| `features_adx_bear` | ADX where −DI > +DI, else 0 |
| `features_hist_vol_20d` | 20d log-return std annualized |
| `features_atr_expansion` | ATR(14) / SMA(ATR, 20) — expansion vs contraction |

**Result:** _(not yet tested)_
| Metric | Baseline | Cumulative | Delta |
|---|---|---|---|
| Mean IC | +0.1922 | — | — |
| t-stat | +9.11 | — | — |
| Folds+ | 6/6 | — | — |

---

## Bucket 4 — Price vs SMA & Breakouts (7 features)

**Rationale:** Where price sits relative to key moving averages tells you whether the zone is a first-touch at support or a re-test after a breakdown. 52-week range position captures the macro trend context.

| Feature | Description |
|---|---|
| `features_price_vs_sma20` | (close − SMA20) / ATR |
| `features_price_vs_sma50` | (close − SMA50) / ATR |
| `features_price_vs_sma200` | (close − SMA200) / ATR |
| `features_high_52w_dist` | (close − 252d high) / 252d high |
| `features_low_52w_dist` | (close − 252d low) / 252d low |
| `features_20d_breakout` | Binary: close > 20-day prior high |
| `features_50d_breakout` | Binary: close > 50-day prior high |

**Result:** _(not yet tested)_

---

## Bucket 5 — Volume (2 features)

**Rationale:** Volume confirmation at a zone touch. High volume at a zone = more conviction. Low volume = weak test, higher chance of zone break.

| Feature | Description |
|---|---|
| `features_vol_ratio_5d` | volume / SMA(volume, 20) |
| `features_vol_ratio_20d` | SMA(volume, 5) / SMA(volume, 60) |

**Result:** _(not yet tested)_

---

## Bucket 6 — Market Context (2 features)

**Rationale:** Sector relative strength tells you if the stock is a leader or laggard in its group — leaders hold zones better. Market breadth gives a macro participation signal.

| Feature | Description |
|---|---|
| `features_sector_rs_20d` | Ticker 20d return − sector median 20d return |
| `features_market_breadth` | % of in_universe tickers above 50d SMA |

**Result:** _(not yet tested)_

---

## Bucket 7 — Momentum / Returns (4 features)

**Rationale:** Prior return momentum as context for zone touches. A zone touch after a 20d decline is different from one after a 20d rally.

| Feature | Description |
|---|---|
| `features_return_1d` | Normalized 1-day log return / ATR% |
| `features_return_5d` | Normalized 5-day log return / ATR% |
| `features_return_20d` | Normalized 20-day log return / ATR% |
| `features_return_60d` | Normalized 60-day log return / ATR% |

**Result:** _(not yet tested)_

---

## Bucket 8 — SMA Slopes (3 features)

**Rationale:** Slope of moving averages — a rising SMA200 while price touches a demand zone is stronger than a flat or falling one.

| Feature | Description |
|---|---|
| `features_sma20_slope_5` | (SMA20 − SMA20.shift(5)) / (ATR × 5) |
| `features_sma50_slope_5` | (SMA50 − SMA50.shift(5)) / (ATR × 5) |
| `features_sma200_slope_10` | (SMA200 − SMA200.shift(10)) / (ATR × 10) |

**Result:** _(not yet tested)_

---

## Bucket 9 — ICT (~115 features)

**Rationale:** ICT standalone was noise (MODEL_C: IC = −0.00002). But ICT order blocks often overlap with zone levels — the question is whether ICT adds incremental information on top of zones at the same price area. This is the highest-risk bucket (many features, noisy standalone signal) — test last.

Sub-families:
- Order blocks & FVGs (20): bull/bear OB, FVG, rejection blocks — active flags, distances, fill %
- Breaker blocks (6): mitigated OBs that flip polarity
- Prior day/week levels (4): PDH, PDL, PWH, PWL distances
- Premium/discount (5): equilibrium position within prior day range
- Liquidity pools BSL/SSL (18): distance, strength, sweep history per side
- Break of structure (8): BOS/CHoCH flags and streaks
- Zone state & confluences (23): OB+FVG confluence, BOS confirmation, macro regime
- Multi-timeframe composites (30+): HTF zone priority carries for 1wk/1mo/3mo/1y
- HTF scores (2): `ict_bull_htf_score`, `ict_bear_htf_score`

**Note:** Requires `skip_ict=False` in FeatureEngineer — add-only test, not the zone-only run.

**Result:** _(not yet tested)_

---

## Bucket 10 — Pivot Features (69 features, requires PIVOT_FEATURES=1)

**Rationale:** Standalone IC = +0.0092 (MODEL_D, GATE FAIL). Incremental on top of zones is unknown. Yearly/monthly pivot distance features dominated MODEL_D — these could act as a second layer of S/R alongside zone levels.

Requires panel rebuilt with `PIVOT_FEATURES=1`.

**Result:** _(not yet tested)_

---

## Testing Protocol

1. Run `model_a_zone_only.py` with expanded feature list (add bucket columns explicitly)
2. Compare mean IC, t-stat, folds+ vs prior best
3. Keep bucket if IC improves and t-stat holds; discard if IC flat or degrades
4. Record result in the table above before moving on
5. Never test on 2024-2026 lockbox window — tuning era (2018-2023) only

**Decision rule:** A bucket is worth keeping if it adds ≥ +0.005 IC without dropping t-stat below 8.0. Noise features will show up as t-stat degradation (higher std_ic) even if mean_ic is flat.

**Bulk sweep:** `scripts/experiments/model_a_bucket_sweep.py` tests all buckets (B1-B8) independently against the zone-only baseline in one unattended run (~90 min). Results saved to `/mnt/data/artefacts/experiments/bucket_sweep_results.json`.

```bash
cd /root/ml-stock-predictor
nohup python3 -u scripts/experiments/model_a_bucket_sweep.py \
    2>&1 | tee /tmp/bucket_sweep.log &
tail -f /tmp/bucket_sweep.log
```

**Current results (as of 2026-07-05):**

| Config | Features | Mean IC | t-stat | Delta vs baseline |
|---|---|---|---|---|
| Zone only (baseline) | 30 | +0.1922 | +9.11 | — |
| Zone + B1 Trend | 34 | +0.1958 | +9.43 | +0.0036 |
| Zone + B1 + ADX/Vol | 38 | +0.1943 | +9.03 | +0.0021 |
| B2–B8 results | — | pending sweep | — | — |
