# Feature Reference — NSE TradingView ML Pipeline

> Detailed documentation of every feature used to train the LightGBM LambdaRank model
> and the per-feature significance, formulas, and composite weights.
>
> **Source files**
> - `pipeline/features/engineer.py` — main feature engineering
> - `pipeline/features/zone_features.py` — Supply/Demand zone feature computation
> - `pipeline/features/ict_features.py` — ICT (Institutional Concepts) feature engine
> - `pipeline/features/multitf_merger.py` — Multi-timeframe trend/vol features
> - `pipeline/utils/zone_analyzer.py` — Core SDZ/SSZ/DZ/SZ zone-drawing logic
> - `signal_weights.yaml` — Composite scoring weights

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Volatility Features](#2-volatility-features)
3. [Trend & Momentum Features](#3-trend--momentum-features)
4. [SMA-Based Features](#4-sma-based-features)
5. [Range / Position Features (52-week)](#5-range--position-features-52-week)
6. [Volume Features](#6-volume-features)
7. [Multi-Timeframe Trend Features](#7-multi-timeframe-trend-features)
8. [Cross-Sectional & Regime Features](#8-cross-sectional--regime-features)
9. [ICT Features (Order Blocks, Breaker Blocks, FVGs)](#9-ict-features)
10. [Supply/Demand Zones — DZ, SZ, SDZ, SSZ](#10-supplydemand-zones)
11. [Zone × Trend Confluence](#11-zone--trend-confluence)
12. [Final Composite Scoring & Weights](#12-final-composite-scoring--weights)

---

## 1. Pipeline Overview

### 1.1 Naming convention

Every feature column is prefixed with `features_` (e.g. `features_adx_14`, `features_sdz_1y`). The selector only considers columns with this prefix.

### 1.2 Data flow

```
Raw OHLCV (TradingView CSV)
        ↓
ATR computation (Wilder's 14-day)
        ↓
Per-ticker feature engineering ──────────────┐
  • Volatility (ATR, hist_vol)               │
  • Returns (1d/5d/20d/60d, ATR-normalized) │
  • SMAs + slopes                           │
  • 52-week distance                        │
  • Volume ratios                           │
  • Rolling beta vs benchmark               │
  • ICT signals (BB/OB/FVG, liquidity)     │
  • Multi-TF ICT (1d/1wk/1mo/3mo/1y)        │
  • Supply/Demand zones (5 timeframes)      │
                                             ↓
Cross-sectional features                      │
  • sector_rs_20d                           │
  • market_breadth                          │
  • regime_bull/choppy/bear                 │
                                             ↓
Multi-TF rolling trends                       │
  • weekly/monthly/quarterly/yearly trend   │
                                             ↓
Zone × Trend confluence ──────────────────────┤
  • sdz_htf_score (zone × bull trends)      │
  • ssz_htf_score (zone × bear trends)      │
  • zone_htf_confluence (net bias)          │
                                             ↓
Winsorization (clip to 1st–99th percentile per date)
                                             ↓
Final panel passed to FeatureSelector ────────┘
                                             ↓
Feature Selection (correlation, permutation, SHAP stability)
   ↓
Final 71 (momentum) / 52 (reversal) features
   ↓
LightGBM LambdaRank training
```

### 1.3 Why ATR-normalize?

Most distance and slope features are divided by **Wilder's 14-day ATR**:
```
feature = (raw_value) / ATR_14
```
This keeps features dimensionless — a stock with ₹10,000 price moves more in absolute ₹ than a stock at ₹100, but ATR-normalized features make their *magnitudes* comparable in standardized "volatility units."

### 1.4 Winsorization

After all features are computed, each feature is **winsorized per date at the 1st and 99th percentile**:
```python
panel[col] = panel.groupby(date)[col].transform(lambda x: x.clip(1pct, 99pct))
```
This caps extreme outliers cross-sectionally (e.g. a tiny micro-cap with a 200% one-day move won't blow up the feature distribution for that date).

---

## 2. Volatility Features

### 2.1 `atr_14` (intermediate, not a feature)
**Definition:** Wilder's 14-day Average True Range.

**Formula:**
```
True Range_t = max(High_t − Low_t, |High_t − Close_(t−1)|, |Low_t − Close_(t−1)|)
ATR_t = α·TR_t + (1−α)·ATR_(t−1),  where α = 1/14
```
Wilder's EMA (α=1/period) gives smoother behavior than a simple rolling mean.

**Purpose:** ATR is the denominator for almost every distance/slope feature. It's a measure of typical daily price range and serves as the volatility unit.

```
  True Range — 3 candidate measurements (largest wins):

  prev Close ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                   ↕ (2) |H_t − prevC|     ↕ (3) |L_t − prevC|
         H_t ──┬──                          │
               │  ← Wick                    │
         Open──┤                            │
               │  ← Body   (1) H_t − L_t   │
         Close─┤                            │
               │  ← Wick                    │
         L_t ──┴── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  TR_t = max( H−L, |H−prevClose|, |L−prevClose| )
  ATR_t = 0.929 × ATR_(t−1) + 0.071 × TR_t      (α = 1/14 ≈ 0.071)
```

---

### 2.2 `features_atr_pct_rank_252`
**Definition:** ATR's rolling 252-day percentile rank (0 = lowest, 1 = highest).

**Formula:**
```
atr_pct_rank_252_t = rank_pct(ATR_t within window [t-252, t])
```

**Significance:**
- **High value (≈ 0.9):** Stock is in an unusually high-volatility regime — recent moves are wider than typical.
- **Low value (≈ 0.1):** Stock is unusually calm — pre-breakout coil or institutional accumulation often shows here.
- **For the model:** A low percentile combined with strong demand-zone signals is a classic "calm before breakout" setup.

---

### 2.3 `features_vol_contraction`
**Definition:** Current ATR divided by the rolling 60-day max ATR.

**Formula:**
```
vol_contraction_t = ATR_t / max(ATR_{t-60..t})
```

**Range:** (0, 1]

**Significance:**
- **Value = 1.0:** Current bar has the widest volatility of the last 60 days (range expansion / explosive move).
- **Value < 0.5:** Volatility has contracted dramatically — pre-breakout pinch / squeeze.
- **For the model:** Classic Bollinger Band Squeeze / Volatility Contraction Pattern (VCP) detector.

```
  Volatility Contraction Pattern (VCP) — the squeeze leading to breakout:

  ATR
  ▲
  │ ▓
  │ ▓▓
  │ ▓▓▓▓                          ← vol_contraction ≈ 1.0 (at peak)
  │ ▓▓▓▓▓▓
  │       ▓▓▓▓▓▓
  │             ▓▓▓▓▓             ← vol_contraction ≈ 0.5
  │                  ▓▓▓▓▓▓▓
  │                         ▓▓▓  ← SQUEEZE  vol_contraction ≈ 0.1–0.2
  └──────────────────────────────────────────────────────── Time
                                   ↑
                            compression_score peaks here
                            → breakout candidate

  compression_score = 1 − vol_contraction
  (high = squeezed = setup pending)
```

---

### 2.4 `features_compression_score`
**Definition:** `1 − vol_contraction`. The inverse — higher means *more compressed*.

**Significance:** Same information as `vol_contraction` but with inverted polarity for intuitive interpretation in scoring (high = compressed = setup pending).

---

### 2.5 `features_hist_vol_20d`
**Definition:** Annualized 20-day realized volatility from log returns.

**Formula:**
```
log_ret_t = ln(Close_t / Close_(t-1))
hist_vol_20d_t = std(log_ret_{t-20..t}) × √252
```

**Significance:**
- Standard volatility measure used by portfolio managers
- Used as the second component of the model ensemble (inverse-vol tilt = lower hist_vol → higher rank)
- Captures recent realized risk; for reversal mode, high vol often signals capitulation

---

## 3. Trend & Momentum Features

### 3.1 `features_adx_14`
**Definition:** Wilder's 14-day Average Directional Index — measures **trend strength** regardless of direction.

**Formula (simplified):**
```
+DI = 100 × Wilder_EMA(+DM) / ATR
−DI = 100 × Wilder_EMA(−DM) / ATR
DX  = 100 × |+DI − −DI| / (+DI + −DI)
ADX = Wilder_EMA(DX, 14)
```
Where +DM = max(0, High_t − High_(t-1)) when up-move > down-move (else 0); −DM = mirror image.

**Range:** 0–100

**Significance:**
- **ADX < 20:** No trend (sideways / choppy market)
- **ADX 20–40:** Trending market
- **ADX > 40:** Strong trend — often near exhaustion
- **For the model:** High ADX confirms breakout strength; low ADX in a demand zone signals base-building.

```
  ADX Interpretation Scale:

     0         10        20        30        40        50+
     ├─────────┼─────────┼─────────┼─────────┼─────────┤
     │  Weak   │Neutral  │Trending │ Strong  │Exhausted│
     │No trend │         │ Market  │  Trend  │(caution)│
     └─────────┴─────────┴─────────┴─────────┴─────────┘

     < 20  → Zones / SMA signals dominate; wait for trend confirmation
     20–40 → Trend-follow mode; momentum features most reliable
     > 40  → Trend may be aging; size down, expect pullback
```

---

### 3.2 Return features: `features_return_1d`, `_5d`, `_20d`, `_60d`

**Definition:** ATR-percentage-normalized log returns over the past 1/5/20/60 bars.

**Formula:**
```
raw_log_ret = ln(Close_t / Close_(t-lag))
pct_atr     = ATR_t / Close_t                      (percentage ATR)
feature     = raw_log_ret / pct_atr
```

**Why divide by *percentage* ATR (not absolute ATR)?**
Log returns are already dimensionless. If we divided by absolute ATR (in ₹), a 10% move in a ₹100 stock vs ₹10,000 stock would produce a 100× difference even though the *relative* move is identical. Percentage ATR keeps the ratio consistent.

**Significance:**
- `return_1d`: Latest momentum signal — overnight/intraday reaction
- `return_5d`: Short-term swing
- `return_20d`: Monthly momentum (the classic "1-month momentum" effect)
- `return_60d`: Quarterly momentum (medium-term trend persistence)

These are the bread-and-butter momentum features that the LGBM ranker uses to discriminate between trending and stagnant names.

---

## 4. SMA-Based Features

### 4.1 `features_price_vs_sma20`, `_sma50`, `_sma200`
**Definition:** Distance of current close above/below the simple moving average, normalized by absolute ATR.

**Formula:**
```
price_vs_sma_t = (Close_t − SMA_N_t) / ATR_t
```
(Note: absolute ATR, not percentage — keeps the feature in "ATR-units" which is natural for "how far am I from this level".)

**Significance:**
- **price_vs_sma20:** Short-term position (≈ 1 month of trading days)
- **price_vs_sma50:** Medium-term position (≈ 2.5 months) — the classic swing-trader baseline
- **price_vs_sma200:** Long-term position (≈ 1 year) — the *Mark Minervini Stage 2* threshold (above SMA200 = bull market for the stock)
- **Positive value:** Stock above the SMA (bullish bias)
- **Negative value:** Stock below the SMA (bearish bias)
- **For the model:** A stock far above SMA200 + recently pulling back to SMA50 is a textbook "buy the dip" setup.

---

### 4.2 `features_sma20_slope_5`, `_sma50_slope_5`, `_sma200_slope_10`
**Definition:** Rate of change of the SMA over the past N bars, normalized by ATR.

**Formula:**
```
sma_slope_t = (SMA_t − SMA_(t-lag)) / (ATR_t × lag)
```

**Significance:**
- **Positive slope:** SMA is rising (uptrend)
- **Negative slope:** SMA is falling (downtrend)
- **Magnitude:** How fast the trend is accelerating (in ATR/bar units)
- **For the model:** Combining `price_vs_sma200 > 0` AND `sma200_slope_10 > 0` is a very strong long-term uptrend signal.

---

## 5. Range / Position Features (52-week)

### 5.1 `features_high_52w_dist`
**Definition:** How far below the 52-week high the current close is (as a fraction).

**Formula:**
```
high_52w_dist_t = (Close_t − rolling_252_high_t) / rolling_252_high_t
```
(Always ≤ 0 — at the high it's 0, 20% below it's -0.20.)

**Significance:**
- **Value ≥ -0.05 (within 5% of 52w high):** Breakout / continuation setup
- **Value ≤ -0.20 (20%+ below high):** Reversal / dip-buy candidate
- **Used as the universe filter for mode selection:**
  - `momentum` mode: `high_52w_dist > -0.15` (within 15% of 52w high)
  - `reversal` mode: `high_52w_dist ≤ -0.20` (more than 20% below)
- **For the model:** Tells the model "where in its historical range" the stock currently sits.

```
  52-Week Range Position Map (high_52w_dist):

  52w Low                                                    52w High
     │                                                           │
     ├────────────────────────────────────────────────────────── ┤
                                                                  ↑ = 0.0
   -1.0      -0.50     -0.30     -0.20     -0.10     -0.05     0.0
     │         │         │         │         │         │         │
  Deep        Off       Fallen   REVERSAL  Near-     Near-    AT HIGH
  dip         high     knife    UNIVERSE  break-     break-  (momentum
                       zone      filter   out zone  out zone  universe)

  Momentum mode picks from: high_52w_dist > -0.15  (right side of chart)
  Reversal  mode picks from: high_52w_dist ≤ -0.20  (left side of chart)
```

---

### 5.2 `features_low_52w_dist`
**Definition:** Bear-side symmetric of high_52w_dist — distance of close *above* the 52-week low.

**Formula:**
```
low_52w_dist_t = (Close_t − rolling_252_low_t) / rolling_252_low_t
```
(Always ≥ 0.)

**Significance:**
- **Near 0:** Price is hugging the 52-week low — breakdown risk
- **High value (e.g. 1.5):** Price is 150% above 52-week low — well off bottom
- **For the model:** Critical for bear-side ranking. Low value + ICT bear signals + zone overhead = high-confidence short setup.

---

## 6. Volume Features

### 6.1 `features_vol_ratio_5d`
**Definition:** Today's volume vs 20-day volume average.

**Formula:**
```
vol_ratio_5d_t = Volume_t / mean(Volume_{t-20..t})
```

**Significance:**
- **Value > 2.0:** Today's volume is 2× the average — confirmed breakout / capitulation
- **Value < 0.5:** Quiet day — drift / coil
- **For the model:** Volume confirmation is essential for distinguishing real breakouts from false ones.

---

### 6.2 `features_vol_ratio_20d`
**Definition:** Recent 5-day average volume vs longer-term 60-day average — captures *sustained* volume changes.

**Formula:**
```
vol_ratio_20d_t = mean(Volume_{t-5..t}) / mean(Volume_{t-60..t})
```

**Significance:**
- **Value > 1.3:** Recent week saw 30%+ more activity than the trailing 3 months — accumulation/distribution
- **Stable around 1.0:** Normal participation
- **For the model:** Smooths out one-day volume spikes and gives a cleaner signal of institutional interest shifts.

---

## 7. Multi-Timeframe Trend Features

### 7.1 `features_weekly_trend`, `_monthly_trend`, `_quarterly_trend`, `_yearly_trend`

**Definition:** Binary indicators (0 or 1) of whether the close is above a rolling SMA over a longer lookback.

**Formula:**
```
weekly_trend_t    = (Close_t > SMA_20_t)
monthly_trend_t   = (Close_t > SMA_60_t)
quarterly_trend_t = (Close_t > SMA_120_t)
yearly_trend_t    = (Close_t > SMA_240_t)
```

**Why rolling instead of resampled?**
Resampled HTF (e.g. true weekly bars) requires merge_asof and introduces alignment quirks. Rolling-window SMAs over daily bars give the same trend-direction information without the bookkeeping complexity, and are leakage-safe by construction.

**Significance:**
- These represent the **four-timeframe trend stack** that institutional traders watch.
- All four = 1 → stock is in confirmed multi-TF uptrend (highest conviction long setup)
- All four = 0 → confirmed multi-TF downtrend
- **For the model:** Yearly trend has the highest predictive weight (institutions only buy stocks in long-term uptrends). The LGBM ranker learns the interaction weights automatically.

```
  Multi-Timeframe Trend Stack — reading the conviction level:

  Timeframe   Feature              SMA Used   Signal
  ─────────── ──────────────────── ─────────  ──────────────────────────────
  Yearly      features_yearly_trend   SMA240   Long-term institutional bias
  Quarterly   features_quarterly_trend SMA120  Medium-term structure intact
  Monthly     features_monthly_trend  SMA60    Swing-trade momentum
  Weekly      features_weekly_trend   SMA20    Short-term pullback complete?

  Reading the stack:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Yearly  Quarterly  Monthly  Weekly  │  Signal                      │
  ├──────────────────────────────────────┼──────────────────────────────┤
  │    ▲         ▲         ▲       ▲     │  Highest conviction LONG      │
  │    ▲         ▲         ▲       ▼     │  Strong long; minor caution   │
  │    ▲         ▲         ▼       ▼     │  Moderate long; needs zones   │
  │    ▲         ▼         ▼       ▼     │  Early pullback; wait         │
  │    ▼         ▼         ▼       ▼     │  Confirmed downtrend          │
  └─────────────────────────────────────────────────────────────────────┘
  ▲ = Close > SMA  |  ▼ = Close < SMA
```

---

## 8. Cross-Sectional & Regime Features

### 8.1 `features_sector_rs_20d`
**Definition:** Stock's 20-day return *relative to the median return of its sector*.

**Formula:**
```
ret_20_t        = ln(Close_t / Close_(t-20))           per ticker
sector_med_t    = median(ret_20 for tickers in same sector on date t)
sector_rs_20d_t = ret_20_t − sector_med_t
```

**Why excess log-return (not ratio)?**
A previous formula `ret/abs(median) − sign(median)` was asymmetric — a 0% return in a -5% sector gave +1.0 ("100% outperformance") which is misleading. Simple excess return is symmetric and economically clean.

**Significance:**
- **Positive:** Stock is outperforming its sector (relative strength leader)
- **Negative:** Stock is lagging — sector tailwind without the stock benefiting
- **For the model:** Classic O'Neil/Minervini RS — institutional money rotates into the strongest names within strong sectors.

---

### 8.2 `features_market_breadth`
**Definition:** Cross-sectional percentage of in-universe tickers trading above their 50-day SMA on a given date.

**Formula:**
```
above_sma50_i,t  = (Close_i,t > SMA_50_i,t) AND (i in universe_t)
market_breadth_t = mean(above_sma50_i,t over all i on date t)
```

**Significance:**
- **> 0.70:** Broad-based strength (most stocks healthy) — bull market
- **0.50–0.70:** Mixed market
- **< 0.30:** Bear market / risk-off
- **For the model:** Macro context. The same stock setup is much more reliable in high-breadth environments.

---

### 8.3 `features_regime_bull` / `_regime_choppy` / `_regime_bear`
**Definition:** Three mutually-exclusive binary dummy columns representing the benchmark's macro regime.

**Formula (using NIFTY for NSE TV):**
```
bm_sma20  = rolling 20-day SMA of NIFTY close
bm_sma50  = rolling 50-day SMA
bm_sma200 = rolling 200-day SMA

regime_bull   = 1  if (NIFTY > bm_sma200) AND (bm_sma20 > bm_sma50)
regime_bear   = 1  if (NIFTY < bm_sma200) AND (bm_sma20 < bm_sma50)
regime_choppy = 1  otherwise
```

```
  Market Regime Decision Tree:

                    Is NIFTY above SMA200?
                     /              \
                   YES               NO
                    │                 │
          Is SMA20 > SMA50?    Is SMA20 < SMA50?
            /      \               /       \
          YES       NO           YES        NO
           │         │            │          │
        ╔═════╗  ╔════════╗  ╔════════╗  ╔═════╗
        ║BULL ║  ║CHOPPY  ║  ║ BEAR  ║  ║CHOPPY║
        ╚═════╝  ╚════════╝  ╚════════╝  ╚══════╝

  Implication for the model:
  BULL   → Trend features dominate; momentum strategies work best
  BEAR   → Reversal / short setups dominate; be cautious on longs
  CHOPPY → Zone confluence essential; tight risk; many false signals
```

**Significance:**
- **Bull regime:** Trends persist longer, momentum strategies work best
- **Bear regime:** Mean reversion + bearish setups dominate
- **Choppy:** Whipsaw — model needs zone confluence + tight risk
- **For the model:** A *regime feature* that gates the importance of all other features. The LGBM tree can branch differently based on regime.

---

### 8.4 `features_rolling_beta_60d`
**Definition:** 60-day rolling OLS beta of the stock's log returns vs benchmark log returns.

**Formula:**
```
beta_t = cov(stock_log_ret, bm_log_ret over window 60) / var(bm_log_ret over window 60)
beta_t = clip(beta_t, -2, 4)
```

**Significance:**
- **Beta ≈ 1:** Moves with the market
- **Beta > 1.5:** High-beta name (amplifies market moves) — leveraged tech / smallcaps
- **Beta < 0.5:** Defensive (utilities, FMCG)
- **Beta < 0 (rare):** Inverse correlation (gold-related, inverse ETFs)
- **For the model:** Helps gauge expected portfolio risk and lets the model differentiate between "stock-specific" alpha and "beta-amplified" market moves.

---

## 9. ICT Features

> **ICT** = "Inner Circle Trader" — a school of price-action analysis that identifies *institutional accumulation/distribution* footprints in candlestick data. The pipeline implements three core ICT signal types: **Order Blocks (OB)**, **Breaker Blocks (BB)**, and **Fair Value Gaps (FVG)**, plus **Liquidity Sweeps (BSL/SSL)**.

### 9.1 Order Block (OB) — Bull (`bob`) and Bear (`sob`)

**Concept:**
An Order Block is the **last opposite-color candle before a strong move** in the new direction. Institutions "load" their positions in this candle, then drive price away.

**Bull OB (BOB):**
The last *bearish* candle before a strong *bullish* breakout. The bullish candle that breaks out must close above the bearish OB's body, and the breakout candle's body must be ≥ 1.2× the OB candle's body (institutional conviction filter).

```
  Bull Order Block (BOB) — candle pattern & zone:

  Bar_(t-1): Bearish (the OB candle)    Bar_t: Bullish (breakout)

          │                                      │
          │ ← wick                        ╔══════╧══════╗ ← Close_t
   ┌──────┤                               ║             ║   body ≥ 1.2×
   │██████│  ← Open_(t-1)        ─────── ╠═════════════╣ ──── BOB Zone High
   │██████│  (OB body)                   ║             ║
   │██████│                     ─────── ╠═════════════╣ ──── BOB Zone Low
   └──────┤  ← Close_(t-1)              ║             ║
          │ ← wick                       ╚══════╤══════╝ ← Open_t
                                                │

  Zone = [ min(Open, Close)_(t-1) ,  max(Open, Close)_(t-1) ]

  When price RETURNS to BOB zone → institutional buyers re-enter → expect ↑
  Zone INVALIDATED if any candle CLOSES below Zone Low.
```

**Mathematical detection:**
```
is_bob = (Close_(t-1) < Open_(t-1))       # prior bar bearish
       & (Close_t > Open_t)                # current bar bullish
       & (Open_t > Close_(t-1))            # opens above prior close (gap up bias)
       & (Close_t > max(Open_(t-1), Close_(t-1)))    # closes above prior body
       & (|Close_t - Open_t| >= 1.2 × |Close_(t-1) - Open_(t-1)|)  # 20% bigger body
```

**Zone boundaries (drawn on chart):**
- **Zone High (zh):** max(Open_(t-1), Close_(t-1)) — top of prior bearish body
- **Zone Low (zl):** min(Open_(t-1), Close_(t-1)) — bottom of prior bearish body

**Forward-fill (still-active):**
The zone remains "active" until price closes below it (`Close < zl` invalidates).

**Distance metric (`ict_bob_dist`):**
```
mid       = (zh + zl) / 2
ict_bob_dist_t = (Close_t - mid) / ATR_14_t × session_weight
```
Positive when price is above the zone midpoint (inside-above), 0 when zone invalidated.

**Bear OB (SOB):** Mirror image — the last bullish candle before a strong bearish breakdown.

**Features produced:**
- `features_ict_bob_active` — 1 if BOB currently active, else 0
- `features_ict_bob_dist` — ATR-normalized distance from zone midpoint
- `features_ict_sob_active`, `features_ict_sob_dist` — mirror for bear OB

---

### 9.2 Breaker Block (BB) — Highest priority ICT signal

**Concept:**
A Breaker Block is a **failed Order Block that flips polarity** after a liquidity sweep. Mark Pickett (ICT) considers this the highest-quality institutional signal because it confirms the OB was a real institutional level that got swept and reclaimed.

**Bull BB:**
1. A previous bearish candle exists (potential OB)
2. A swing low forms (L_(t-1) < L_(t-2), L_(t-3), L_t)
3. **Sell-side liquidity (SSL) is swept** — wick below two prior equal lows that closes back above (stop-hunt)
4. Current candle closes above the prior bearish body (reclaim confirmed)

```
  Bull Breaker Block (BB) — 4 step sequence:

  Step 1: Bearish OB-like candle forms
  Step 2: Swing low confirmed  
  Step 3: SSL sweep (wick below equal lows → grabs retail stop-losses)
  Step 4: Reclaim above OB body = Breaker Block confirmed

                ┌──────────────────────────────────────┐
  Price chart:  │ SWING            SSL SWEEP  RECLAIM  │
                │  HIGH                                 │
                │   │                         ╔═══════╗│
                │   ↓                         ║  BB   ║│
                │ ┌─┐   ┌─┐        ┌─┐   ╔═══╣ZONE   ║│
                │ │█│   │ │        │ │   ║   ╚═══════╝│
  Swing Low →  │ │█│   │ │  ─ ─ ─ │ │   ║   (reclaim)│
                │ └─┘   └─┘   ↓   └─┘   ║            │
  SSL zone →   │           wick       ╔═╝            │
  (retail stops│           sweeps     ║               │
  below lows)  │           below      └──────────────  │
                └──────────────────────────────────────┘

  BB Zone = same body range as the original OB candle
  Priority: BB(3) > OB(2) > FVG(1) — strongest ICT signal
```

```
is_bull_bb = (Close_(t-1) < Open_(t-1))           # OB-like prior candle
           & is_swing_low                          # confirmed swing
           & ssl_swept                             # liquidity took
           & (Close_t > max(Open_(t-1), Close_(t-1)))  # reclaim
```

**Bear BB:** Mirror — swept buy-side liquidity (BSL) above prior swing high, then closes back below.

**Zone priority:**
`BB(3) > OB(2) > FVG(1)` — when multiple ICT signals fire on the same candle, only the highest priority is kept.

**Features:**
- `features_ict_bullbb_active`, `features_ict_bullbb_dist`
- `features_ict_bearbb_active`, `features_ict_bearbb_dist`

---

### 9.3 Fair Value Gap (FVG)

**Concept:**
A 3-candle pattern where there's a *gap* between candle 1's high and candle 3's low (bull FVG) — the middle candle is fully detached. Represents *inefficiency* that price tends to revisit ("price always wants to fill the gap").

**Bull FVG:**
```
is_bull_fvg = (Low_t > High_(t-2))   # current low above 2-bars-ago high
```
- **Zone High (zh):** Low_t
- **Zone Low (zl):** High_(t-2)

```
  Fair Value Gap (Bull FVG) — 3 candle imbalance:

  Bar_(t-2)      Bar_(t-1)      Bar_t
     │               │               │
   ┌─┴─┐           ┌─┴─┐          ╔═╧══╗
   │   │           │   │          ║    ║ ← Low_t  = FVG Zone High (zh)
   │   │ ← High    │   │    ╔════╬════╣ ──────────────────── GAP zone
   │   │   (t-2)   │   │    ║ GAP║    ║ ← High_(t-2) = FVG Zone Low (zl)
   └─┬─┘           │   │    ╚════╬════╣
     │             └─┬─┘          ║    ║
     │               │            ╚═╤══╝
                                    │

  Gap exists because Low_t > High_(t-2)  (no overlap, middle candle is the engine)

  Price revisits the gap → tends to "fill" (revert to midpoint)
  Invalidated when price crosses the gap MIDPOINT (tighter than OB/BB)
```

**Bear FVG:**
```
is_bear_fvg = (High_t < Low_(t-2))   # current high below 2-bars-ago low
```

**Special cancellation rule:**
FVGs use `mid_cancel=True` — they invalidate as soon as price crosses the *midpoint* (not just the edge). Tighter than OB/BB invalidation.

**Features:**
- `features_ict_bullfvg_active`, `features_ict_bullfvg_dist`
- `features_ict_bearfvg_active`, `features_ict_bearfvg_dist`

---

### 9.4 Liquidity Sweeps — `bsl_swept` & `ssl_swept`

**Concept:**
Markets gravitate toward areas of clustered stop orders ("liquidity pools"). When two prior swing highs/lows are equal (within ATR tolerance), retail stops cluster just outside them. Institutional traders engineer a wick to grab those stops before reversing.

**Buy-Side Liquidity (BSL) Sweep — `ict_bsl_swept = 1`:**

```
  Buy-Side Liquidity Sweep (BSL) — Stop Hunt at Equal Highs:

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ←  "Equal highs"
                  ↑ Swing High 1         ↑ Swing High 2           (retail stops
                                                                     cluster here)
     │         │                   │          │
   ┌─┴─┐     ┌─┴─┐               ┌─┴─┐      │  ← Wick wicks ABOVE both highs
   │   │     │   │               │   │      │     (stops triggered)
   │   │     │   │               │   │     ┌┴─┐  CLOSE back below = reversed!
   └─┬─┘     └─┬─┘               └─┬─┘    └─┬┘
     │          │                   │        │
                                         ↓ bsl_swept = 1
                                    Probable reversal DOWN from here
```

```
Two prior swing highs within 0.1 × ATR of each other
AND current bar's High > max(prior two highs)
AND current bar's Close < max(prior two highs)         # reversed back
```

**Sell-Side Liquidity (SSL) Sweep — `ict_ssl_swept = 1`:**
Mirror — wick below two equal lows, closes back above.

**Significance:**
A BSL sweep marks a probable top (longs being trapped); SSL sweep marks a probable bottom (shorts being trapped). These are *contrarian* signals when they appear at HTF zones.

---

### 9.5 Zone Priority Metadata

`features_ict_bull_zone_priority` and `features_ict_bear_zone_priority` are integers `{0, 1, 2, 3}`:
- 0 = no zone
- 1 = FVG only
- 2 = OB only
- 3 = BB only

When two signals overlap on the same candle, the highest priority wins (BB > OB > FVG).

---

### 9.6 Multi-Timeframe ICT — `_1wk`, `_1mo`, `_3mo`, `_1y`

The same ICT detection runs separately on weekly, monthly, quarterly, and yearly resampled bars. Each timeframe's "active" flags are then merged back to the daily index via `merge_asof`, giving features like:

- `features_ict_bob_active_1wk`, `_1mo`, `_3mo`, `_1y`
- `features_ict_bullbb_active_1wk`, `_1mo`, `_3mo`, `_1y`
- `features_ict_bullfvg_active_1wk`, `_1mo`, `_3mo`, `_1y`
- ... and bear-side mirrors
- `features_ict_bull_zone_priority_1wk`, `_1mo`, `_3mo`, `_1y`
- `features_ict_bsl_swept_1mo`, etc.

**HTF weighting:**
```
weight = {1d: 1, 1wk: 2, 1mo: 3, 3mo: 4, 1y: 5}
```
Higher timeframes get more weight in composite scoring — a yearly BB is 5× more important than a daily BB.

---

### 9.7 ICT HTF Composite — `ict_bull_htf_score` & `ict_bear_htf_score`

**Definition:** Weighted sum of zone priorities across all timeframes, normalized to [0, 1].

**Formula:**
```
score = sum( weight_tf × priority_tf / 3 ) / sum(weights)
      = sum( weight_tf × priority_tf / 3 ) / 15

where weights = {1d:1, 1wk:2, 1mo:3, 3mo:4, 1y:5}, max priority = 3 (BB)
```

**Range:** [0, 1]

**Significance:**
- **`ict_bull_htf_score = 1.0`:** Maximum bull confluence — Bull BB active on all 5 timeframes simultaneously (extremely rare, very high conviction)
- **`ict_bull_htf_score ≈ 0.3`:** Daily + weekly BB active, others quiet — moderate bull setup
- **`ict_bull_htf_score = 0`:** No bull ICT signals — neutral or bearish
- **For the model:** A single high-signal feature that summarizes the entire ICT picture. Used by the LGBM model as a "ICT-confluence" gauge.

---

## 10. Supply/Demand Zones

> The most important features for the **bull/bear composite** (these alone drive 30–45% of the final ranking score). Zones are computed by the `ZoneAnalyzer` and represent **institutional supply/demand levels**.

### 10.1 What is a Zone?

A **Supply/Demand Zone** is a price range where institutional traders placed large pending orders. When price returns to this range, those resting orders activate and the zone *acts as support (demand) or resistance (supply)*. Unlike a simple horizontal support line, a zone has a **proximal edge** (the side price hits first) and a **distal edge** (the outer boundary).

### 10.2 The 4 Base Candle Patterns

Zones are formed by **base candles** — small consolidation candles that are *inside the prior candle's range*. The zone classification depends on what's *before* and *after* the base:

| Pattern | Before | After | Zone Type | Bias |
|---------|--------|-------|-----------|------|
| **RBR** | Rally | Rally | **DZ (Demand)** | Bullish |
| **DBD** | Drop  | Drop  | **SZ (Supply)** | Bearish |
| **DBR** | Drop  | Rally | **DZ (Demand)** | Bullish |
| **RBD** | Rally | Drop  | **SZ (Supply)** | Bearish |

```
  RBR → Demand Zone (DZ)            DBD → Supply Zone (SZ)

  Rally     Base    Rally           Drop      Base     Drop
    │                 │
  ┌─┴─┐             ┌─┴─┐         └─┬─┐             └─┬─┐
  │   │    ┌───┐    │   │           │ │    ┌───┐       │ │
  │   │    │   │    │   │           │ │    │   │       │ │
  │   │ ══╪═══╪══  │   │           │ │ ══╪═══╪══     │ │
  └─┬─┘   │DZ │    └─┬─┘         ┌─┴─┘   │SZ │     ┌─┴─┘
           │Zone│                         │Zone│
           └───┘                          └───┘
  Price returns → bounce ↑               Price returns → reject ↓

  DBR → Demand Zone (DZ)            RBD → Supply Zone (SZ)
  (stronger — drop THEN reversal)    (stronger — rally THEN reversal)

  Drop     Base    Rally            Rally     Base     Drop
   │                │
  └┬─┐             ┌┴─┐            ┌┴─┐              └┬─┐
   │ │    ┌───┐    │  │            │  │    ┌───┐       │ │
   │ │    │DZ │    │  │            │  │    │SZ │       │ │
   │ │ ══╪═══╪══  │  │            │  │ ══╪═══╪══     │ │
  ┌┴─┘   └───┘    └┬─┘           └┬─┘    └───┘      ┌┴─┘
```

**Candle type definitions (`zone_analyzer.py`):**
- **Rally:** Bullish bar AND `Close > prior High`
- **Drop:** Bearish bar AND `Close < prior Low`
- **Base:** Bar whose body lies inside the prior bar's range (`Low_(t-1) ≤ body ≤ High_(t-1)`)

---

### 10.3 DZ — Demand Zone (Bullish)

**Definition:** A range where institutional buyers absorbed supply. Drawn after a Rally→Base→Rally (RBR) or Drop→Base→Rally (DBR) pattern.

**Drawing rules:**
- **Proximal (upper edge):** Base candle's *Close*
- **Distal (lower edge):** Lowest *Low* of the surrounding base/rally candles
- **Zone band:** `[Distal, Proximal]`

**Significance:**
- When price returns to a DZ, it tends to **bounce up**
- Strong DZs (RBR formations on yearly TF) are the highest-conviction long entries
- Weak DZs (DBR on daily TF) are weaker — break easily

**Feature column:** Various `features_dz_*` flags exist but the model primarily uses the *swap zone* variants (SDZ) which are stronger.

---

### 10.4 SZ — Supply Zone (Bearish)

**Definition:** Mirror of DZ — a range where institutional sellers absorbed demand. Drawn after a Drop→Base→Drop (DBD) or Rally→Base→Drop (RBD) pattern.

**Drawing rules:**
- **Proximal (lower edge):** Base candle's *Open*
- **Distal (upper edge):** Highest *High* of the surrounding base/drop candles

**Significance:**
- When price returns to an SZ, it tends to **reject down**
- Strong SZs (DBD on yearly TF) often mark major tops

---

### 10.5 SDZ — Swap Demand Zone (Strongest Bullish)

**Definition:** A former Supply Zone that price *cleanly broke above* — old institutional supply has been absorbed and becomes new demand on retest. "Supply-to-Demand flip."

**Conversion rules (`_identify_swap_demand_zones`):**
1. Candidate: An existing `SZ` marked `Valid`
2. First close *above* the SZ's distal (top edge) after the zone candle (the breakout)
3. The breakout candle must be a **Rally** (not Drop or Base) — institutional conviction
4. No Drop/Base candle overlapping the zone band between the base and the breakout (no fakeout)
5. When all 4 conditions hold:
   - `ZoneType` is converted from `SZ` → `SDZ`
   - **New Distal** = the SZ candle's *Low*
   - **New Proximal** = the SZ candle's *High*
6. **Breach check:** If price later closes/opens below the SDZ's low → reverts to invalid SZ

**Why it's the strongest bullish signal:**
- A break of supply (overhead resistance) signals exhausted sellers
- The retest is "buying the breakout pullback" — high-conviction institutional entry
- The clean-break requirement (no obstructing candles) ensures it wasn't a fakeout

```
  SDZ Formation — Supply Zone flips to Swap Demand Zone:

  ─── Phase 1: Supply Zone (SZ) forms ───────────────────────────────────

    Drop        Base (SZ formed here)      Drop
    └──┐
    │  │    ╔═══════════════╗              └──┐
    │  │    ║   SZ  Zone    ║              │  │
    │  │    ║  (resistance) ║              │  │
    ┌──┘    ╚═══════════════╝              ┌──┘

  ─── Phase 2: Price breaks CLEANLY ABOVE the SZ (must be a Rally candle) ─

    ...previous bars...                    Rally (breakout) — MUST be green
                                                 │
    ╔═══════════════╗                        ┌───┘ ← Closes above SZ distal
    ║   SZ  Zone    ║                        │      (top of zone)
    ╚═══════════════╝ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       (no Base/Drop candles overlapping zone between here and breakout)

  ─── Phase 3: SZ flips to SDZ — old resistance becomes new support ──────

                ╔═══════════════════════════╗
                ║  SDZ (Swap Demand Zone)   ║  ← Distal  = SZ candle Low
                ║  "buy the pullback here"  ║  ← Proximal = SZ candle High
                ╚═══════════════════════════╝
                             ↑
                   Price returns → strong bounce expected ↑
                   (Institutional buyers re-enter at broken resistance)

  SDZ invalidated if price CLOSES or OPENS below the new Distal (SZ low).
```

**Significance for the model:**
SDZ on the yearly TF + bullish multi-TF trends = the model's single highest-confidence long signal. This is why `sdz_htf_score` carries the highest weight (3.0) in the bull composite.

---

### 10.6 SSZ — Swap Supply Zone (Strongest Bearish)

**Definition:** A former Demand Zone that price cleanly broke below — old institutional demand has been overwhelmed by selling and becomes new supply on retest.

**Conversion rules:** Exact mirror of SDZ.
1. Candidate: An existing `DZ` marked `Valid`
2. First close *below* the DZ's distal (bottom edge)
3. The breakout candle must be a **Drop** (not Rally or Base)
4. No Rally/Base candle overlapping the zone band between base and breakdown
5. When all 4 conditions hold:
   - `ZoneType` is converted from `DZ` → `SSZ`
   - **New Distal** = the DZ candle's *High*
   - **New Proximal** = the DZ candle's *Low*
6. **Breach check:** If price later rallies above the SSZ's high → reverts to invalid DZ

**Significance:**
SSZ on the yearly TF + bearish multi-TF trends = the model's strongest short signal. Drives `ssz_htf_score`.

---

### 10.7 Per-Timeframe Zone Flags

For each timeframe `{1d, 1wk, 1mo, 3mo, 1y}`, four binary features are produced per zone type:

| Feature | Description |
|---------|-------------|
| `features_sdz_1d` | SDZ active on daily TF (0 or 1) |
| `features_sdz_1wk` | SDZ active on weekly TF |
| `features_sdz_1mo` | SDZ active on monthly TF |
| `features_sdz_3mo` | SDZ active on quarterly TF |
| `features_sdz_1y` | SDZ active on yearly TF |
| `features_ssz_1d`/`_1wk`/`_1mo`/`_3mo`/`_1y` | SSZ mirror set |
| `features_dz_1d`...`_1y`  | DZ mirror set |
| `features_sz_1d`...`_1y`  | SZ mirror set |

These flags are individually exposed so the LGBM ranker can learn the *interaction weights* between TFs from data, rather than us hard-coding them.

---

### 10.8 Composite Zone Scores

For each zone type, a weighted composite is computed:

**Weights:**
```
TF      |  Weight
--------|----------
1d      |  1
1wk     |  2
1mo     |  3
3mo     |  4
1y      |  5
--------|----------
Sum     | 15
```

**SDZ/SSZ multiplier:** 2× (because swap zones are stronger than DZ/SZ)
**Max possible score:** `2 × sum(weights) = 30`

```
  Zone Score Weighting — Higher timeframe = exponentially more important:

  Timeframe  │ Weight │ Share of max score │ Multiplier
  ───────────┼────────┼────────────────────┼──────────────────────
  Daily(1d)  │   1    │ ▓░░░░░░░░░  6.7%  │ SDZ×2=2  │ DZ×1=1
  Weekly(1wk)│   2    │ ▓▓░░░░░░░░ 13.3%  │ SDZ×2=4  │ DZ×1=2
  Monthly    │   3    │ ▓▓▓░░░░░░░ 20.0%  │ SDZ×2=6  │ DZ×1=3
  Quarterly  │   4    │ ▓▓▓▓░░░░░░ 26.7%  │ SDZ×2=8  │ DZ×1=4
  Yearly(1y) │   5    │ ▓▓▓▓▓░░░░░ 33.3%  │ SDZ×2=10 │ DZ×1=5
  ───────────┴────────┴────────────────────┴──────────────────────
  Total weight = 15  │  Max SDZ score = 30  │  Max DZ score = 15
```

**Formulas:**
```
sdz_raw_score = sum(weight_tf × 2 × is_SDZ_tf) / 30        →  range [0, 1]
ssz_raw_score = sum(weight_tf × 2 × is_SSZ_tf) / 30        →  range [0, 1]
dz_raw_score  = sum(weight_tf × 1 × is_DZ_tf)  / 30        →  range [0, 1]
sz_raw_score  = sum(weight_tf × 1 × is_SZ_tf)  / 30        →  range [0, 1]
```

**Features produced:**
- `features_sdz_raw_score`, `features_ssz_raw_score`
- `features_dz_raw_score`, `features_sz_raw_score`
- `features_any_valid_sdz` — 1 if SDZ active on at least one TF
- `features_any_valid_ssz` — mirror
- `features_any_valid_zone` — 1 if ANY zone type active on any TF
- `features_zone_strength` — derived from the 1d zone type (SDZ/SSZ = 2.0, DZ/SZ = 1.0, none = 0)
- `features_zone_active` — 1 if any zone active on the daily TF
- `features_zone_dist_atr` — ATR-normalized distance from the active daily zone proximal

---

## 11. Zone × Trend Confluence

The most important **composite features** that combine zone strength with the multi-timeframe trend stack.

### 11.1 `features_sdz_htf_score` — Strongest Bull Composite

**Definition:** SDZ raw score amplified by trend alignment across the 4 timeframes.

**Formula:**
```
up_mult = 0.5 + 0.375 × weekly_trend
              + 0.375 × monthly_trend
              + 0.375 × quarterly_trend
              + 0.375 × yearly_trend
        # Range: 0.5 (no trends up) → 2.0 (all 4 trends up)

sdz_htf_score = sdz_raw_score × up_mult
```

**Significance:**
- **Max ≈ 2.0:** Yearly SDZ + all 4 TFs trending up — institutional accumulation in a confirmed uptrend
- **Mid (≈ 1.0):** Some SDZ + mixed trend signals
- **Low (< 0.3):** Weak SDZ in a downtrend (often a trap)

**Why this is the dominant bull feature:**
A swap demand zone alone is good, but a swap demand zone in a confirmed multi-TF uptrend is *the* highest-conviction long setup. The trend multiplier ensures the model doesn't give equal weight to a yearly SDZ in a downtrend (weak) vs the same yearly SDZ in an uptrend (institutional gold).

```
  sdz_htf_score — How zone strength and trend alignment combine:

  Trend Multiplier (up_mult) scale:
                                                        up_mult
    No timeframes trending up  (weekly=0,monthly=0,...)  → 0.50
    1 of 4 TFs trending up                               → 0.875
    2 of 4 TFs trending up                               → 1.25
    3 of 4 TFs trending up                               → 1.625
    All 4 TFs trending up      (weekly=monthly=qtrly=yr=1) → 2.00

  sdz_htf_score = sdz_raw_score × up_mult

  Example scenarios (all with yearly SDZ active = sdz_raw_score ≈ 0.33):

  Scenario A: Yearly SDZ + all trends up
    → 0.33 × 2.0 = 0.66  (HIGH confidence — institutional setup)

  Scenario B: Yearly SDZ + no trends up
    → 0.33 × 0.5 = 0.17  (LOW confidence — zone in a downtrend = trap)

  Scenario C: All 5 TF SDZs active + all trends up
    → 1.0  × 2.0 = 2.0   (MAXIMUM — theoretical ceiling, extremely rare)
```

---

### 11.2 `features_ssz_htf_score` — Strongest Bear Composite

**Definition:** Mirror — SSZ raw score amplified by *bearish* trend alignment.

**Formula:**
```
dn_mult = 0.5 + 0.375 × (1 - weekly_trend)
              + 0.375 × (1 - monthly_trend)
              + 0.375 × (1 - quarterly_trend)
              + 0.375 × (1 - yearly_trend)

ssz_htf_score = ssz_raw_score × dn_mult
```

---

### 11.3 `features_zone_htf_confluence` — Net Bias

**Definition:**
```
zone_htf_confluence = sdz_htf_score − ssz_htf_score
```

**Range:** Approximately [-2, +2]

**Significance:**
- **Positive:** Bullish zone bias dominates
- **Negative:** Bearish zone bias dominates
- **Near zero:** Neutral / conflicting zone signals
- **For the model:** Single-column directional zone summary; particularly useful for the LGBM ranker because it can use this one feature as a "direction switch" before branching on detail features.

---

## 12. Final Composite Scoring & Weights

### 12.1 Multi-stage scoring pipeline

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                    FULL SCORING PIPELINE                             │
  └──────────────────────────────────────────────────────────────────────┘

  [71 features / 52 features]          [SDZ/SSZ zone flags per TF]
        (momentum / reversal)                (5 timeframes each)
              │                                       │
              ▼                                       ▼
  ┌───────────────────────┐           ┌───────────────────────────┐
  │  LightGBM LambdaRank  │           │  Zone Composite           │
  │  Objective: lambdarank│           │  bull = Σ(weight × SDZ_tf)│
  │  Eval:      ndcg@10   │           │  bear = Σ(weight × SSZ_tf)│
  │  + 10% inv-vol tilt   │           │  (from signal_weights.yaml│
  └──────────┬────────────┘           └──────────────┬────────────┘
             │  model_score [0,1]                     │  composite [0,1]
             └──────────────────┬─────────────────────┘
                                ▼
               ┌────────────────────────────────────────┐
               │  Mode-specific Weighted Blend          │
               │                                        │
               │  MOMENTUM:  70% model + 30% composite  │
               │  REVERSAL:  55% model + 45% composite  │
               └──────────────────┬─────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
              bull_final [0,1]           bear_final [0,1]
              (long candidates)          (short candidates)
                    │                            │
                    └─────────────┬──────────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Tier Ranking        │
                       │  Tier 1: top decile  │
                       │  Tier 2: next 20%    │
                       │  Tier 3: next 20%    │
                       └──────────────────────┘
```

### 12.2 Stage 1: LGBM Ensemble (in `pipeline/models/ensemble.py`)

- **Model:** LightGBM LambdaRank (`pipeline/models/lgbm_ranker.py`)
  - Objective: `lambdarank`
  - Eval metric: `ndcg@10`
  - Grouped by date (each date = one ranking "query")
- **Inverse-vol tilt:** Inverse rank of `features_hist_vol_20d` — lower vol → higher rank

**Blend:** Fixed at `0.9 × LGBM_rank + 0.1 × vol_rank`, then normalized to [0, 1].

### 12.3 Stage 2: Mode-specific Model + Composite Blend (from `signal_weights.yaml`)

| Mode | Model weight | Composite weight | Composite content |
|------|:------------:|:----------------:|-------------------|
| **Momentum** | **70%** | **30%** | SDZ-weighted (bull) / SSZ-weighted (bear) |
| **Reversal** | **55%** | **45%** | Same, but composite carries more weight |
| Fallback | 60% | 40% | — |

**Why different weights per mode:**
- **Momentum** picks are within 15% of 52w high → the *price action* (LGBM features) is already strong; zones add confirmation.
- **Reversal** picks are 20%+ below 52w high → the *zone* matters more (you need a strong demand zone to justify catching a falling knife); model gets less weight.

### 12.4 Composite formulas

**Bull composite (used for long picks):**
```
bull_composite =   3.5 × sdz_1y           +
                   3.0 × sdz_3mo          +
                   2.5 × sdz_1mo          +
                   2.0 × sdz_1wk          +
                   3.0 × sdz_htf_score
                  ─────────────────────────
                   normalized to [0, 1]
```

**Bear composite (used for short picks):**
```
bear_composite =   3.5 × ssz_1y           +
                   3.0 × ssz_3mo          +
                   2.5 × ssz_1mo          +
                   2.0 × ssz_1wk          +
                   3.0 × ssz_htf_score
                  ─────────────────────────
                   normalized to [0, 1]
```

**Note:** These composite weights come from `signal_weights.yaml`. The composite is intentionally narrow — *only SDZ/SSZ zone signals*. The design principle (per the YAML comment) is:

> "Everything else (ICT, trend, momentum, regime, SMA, volume, ADX) is owned entirely by the ML model — it learns them from data. The composite is SDZ/SSZ zone signals only."

This is because zone signals are **rare, hand-engineered, and rule-based**, while the ML model is better at synthesizing the dozens of correlated technical features.

### 12.5 Final score per ticker per date

```
bull_final_t = normalize_rank(
    model_w × normalize_rank(model_score_t)  +
    composite_w × bull_composite_t
)

bear_final_t = normalize_rank(
    model_w × normalize_rank(1.0 - model_score_t)  +
    composite_w × bear_composite_t
)
```

The `1.0 - model_score` inversion for bear is what makes the same LGBM model usable for both long and short picks — a stock the model ranks "low for going up" is effectively ranked "high for going down."

---

## Appendix A — Final Feature Counts

The full feature panel has **121 columns** with the `features_` prefix. After `FeatureSelector` runs (correlation filtering, permutation importance, SHAP stability bootstrap), the final feature set is:

| Mode | Final features | Selector output |
|------|:--------------:|-----------------|
| **Momentum** | 71 | `artefacts/nse_tradingv/momentum/selected_features.txt` |
| **Reversal** | 52 | `artefacts/nse_tradingv/reversal/selected_features.txt` |

The momentum model favors trend/return/SMA features; the reversal model favors low_52w_dist, hist_vol_20d, and the ICT/zone bear features.

## Appendix B — Why Two Separate Models

The two modes use the *same feature engineering pipeline* but are trained as **separate LightGBM models** with separate hyperparameter searches because:
1. The universes are disjoint (momentum vs reversal stocks have different price-action characteristics)
2. The features that matter differ significantly between continuation and reversal setups
3. Allows independent calibration of the model/composite blend (70/30 vs 55/45)

---

## Appendix C — Quick Cross-Reference: Feature → Source File

| Feature group | Source |
|---------------|--------|
| ATR, returns, SMAs, vol ratios, beta | `pipeline/features/engineer.py` |
| ICT signals (BB/OB/FVG, liquidity, sessions) | `pipeline/features/ict_features.py` |
| Multi-TF trends (weekly/monthly/quarterly/yearly) | `pipeline/features/multitf_merger.py` |
| Zone features (DZ/SZ/SDZ/SSZ per TF) | `pipeline/features/zone_features.py` |
| Zone drawing & RBR/DBD/SDZ/SSZ logic | `pipeline/utils/zone_analyzer.py` |
| Sector RS, market breadth, regime | `pipeline/features/engineer.py` (`_add_*` methods) |
| Composite blends, ensemble | `pipeline/models/ensemble.py`, `signal_weights.yaml` |
| LGBM LambdaRank training | `pipeline/models/lgbm_ranker.py` |
