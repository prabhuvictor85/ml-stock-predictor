[← Back to Feature Engineering](README.md) &nbsp;|&nbsp; [← Back to ML Design overview](../README.md) &nbsp;|&nbsp; [← Back to index](../../README.md)

# Returns / Momentum

## Level 1 — Executive Summary
The most direct question a model can ask is "how has this stock actually performed recently, at a few different timescales?" This family answers exactly that — the stock's own return over the last day, week, month, and quarter — but expressed in a way that's fair to compare across a $5 stock and a $500 stock, and across a calm stock and a wild one.

## Level 2 — Plain English
If someone tells you "the stock moved $2 yesterday," that's meaningless without context — is $2 a huge move or a rounding error? You need to know both the stock's price level (a $2 move on a $10 stock is huge; on a $2,000 stock it's nothing) and its normal daily wiggle (a $2 move on a stock that usually swings $5 a day is unremarkable; on a stock that usually swings $0.20 a day, it's an event). This feature family bakes both corrections in at once, so a "return" reading is directly comparable no matter which stock it came from.

## Level 3 — Technical Deep Dive

### The formula
```python
log_close = log(close)
for lag in [1, 5, 20, 60]:
    raw_log_return = log_close − log_close.shift(lag)
    return_{lag}d   = raw_log_return / pct_atr
```
Four lookback windows are computed: **1 day, 5 days (≈1 week), 20 days (≈1 month), and 60 days (≈1 quarter)** — `return_1d`, `return_5d`, `return_20d`, `return_60d`.

### Why log returns, not simple percentage returns?
`log(close_t / close_{t-lag})` rather than `(close_t − close_{t-lag}) / close_{t-lag}`. Log returns are **additive across time** — the 20-day log return is exactly the sum of the twenty 1-day log returns, which simple percentage returns are not (they compound multiplicatively instead). This makes log returns better-behaved for the kind of rolling-window statistics used throughout this system (e.g. `hist_vol_20d`, see [ATR](01-atr.md#hist_vol_20d--a-volatility-measure-that-deliberately-does-not-use-atr), is built directly on log-return differences for exactly this reason).

### Why divide by *percentage* ATR, not absolute ATR?
This is the second correction, and it's the more important one. A log return is already a dimensionless ratio (it doesn't carry a currency unit). Dividing a dimensionless ratio by an *absolute*-currency-unit ATR (like `price_vs_sma*` does) would introduce a price-level dependency: a 10% move on a ₹100 stock and a 10% move on a ₹10,000 stock are the *same percentage gain*, but if you divided both by their absolute (₹) ATR, the ₹10,000 stock's feature value would come out roughly 100× larger for an identical percentage move — purely an artifact of price level, not a real difference in significance. Dividing by **percentage ATR** (`ATR / close`) instead keeps both dimensionless, so the resulting feature reflects "how many typical daily moves did this return represent," consistently regardless of the stock's price level. See [ATR § Two units](01-atr.md#two-units-of-atr--and-why-both-exist) for the full unit-matching rule this follows.

<details><summary>Worked example — why the percentage-ATR denominator matters</summary>

```text
Stock A: price ₹100,  typical daily range (ATR) = ₹2   → pct_atr = 2%
Stock B: price ₹10,000, typical daily range (ATR) = ₹200 → pct_atr = 2%
(both stocks have the SAME relative volatility — 2% of price, per day)

Both stocks rally 10% over 20 days:
  Stock A: log_return_20d ≈ 0.0953   Stock B: log_return_20d ≈ 0.0953  (identical, as expected)

Dividing by ABSOLUTE ATR (wrong — introduces a price-level artifact):
  Stock A: 0.0953 / ₹2   = 0.048   Stock B: 0.0953 / ₹200 = 0.00048
  → same percentage move, wildly different feature values (100x apart)

Dividing by PERCENTAGE ATR (used in production — correct):
  Stock A: 0.0953 / 0.02 = 4.77    Stock B: 0.0953 / 0.02 = 4.77
  → identical feature values for an identical relative move, regardless
    of absolute price level
```
</details>

### Four horizons, four different jobs
| Feature | Horizon | What it's read as |
|---|---|---|
| `return_1d` | 1 day | Immediate reaction — a single news/earnings/gap day |
| `return_5d` | ~1 week | Short-term momentum, filters out single-day noise |
| `return_20d` | ~1 month | Medium-term trend — the same horizon as the model's own forward-return **label** (`future_20d_return`, see [Problem Formulation](../01-problem-formulation.md)), making this the closest "has this pattern already started" signal available at prediction time |
| `return_60d` | ~1 quarter | Longer-term momentum context, complements the 200-day SMA trend read |

### Design Decisions / Alternatives / Trade-offs
| Decision | Why | Alternative rejected |
|---|---|---|
| Log returns, not simple percentage returns | Additive across time, better-behaved for rolling statistics elsewhere in the system | Simple returns — compound multiplicatively, harder to aggregate consistently across windows |
| Normalize by *percentage* ATR, not absolute ATR | Both numerator and denominator are dimensionless, so the ratio has no hidden price-level dependency | Normalizing by absolute ATR — would silently make the same percentage move look different depending on the stock's price level |
| Four fixed horizons (1/5/20/60d) | Covers immediate, weekly, monthly, and quarterly momentum in one small feature set | A single lookback — would blend genuinely different signals (a 1-day gap and a 60-day trend are not the same phenomenon) into one number |

### Common Pitfalls
- Comparing `return_20d` values across two stocks by eyeballing the raw number as if it were a percentage — it isn't; it's already been divided by percentage ATR, so a value of "4.77" means "this return was about 4.77 typical daily moves," not "4.77% return."
- Forgetting that `return_20d` shares its horizon with the model's own training label (`future_20d_return`/`cs_rank_20d`) — this is the single most label-adjacent input feature in the whole panel, worth extra scrutiny in any feature-importance or leakage review specifically because of how closely its horizon matches the target's.

### Future Improvements
None currently planned. This is a small, foundational family.

---

**Previous:** [← 07 · Pivots](07-pivots.md) &nbsp;|&nbsp; **Next:** [09 · Trend →](09-trend.md)
