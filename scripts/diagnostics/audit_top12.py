"""
Compare pure model ranking vs blended ranking at the top-12 level.
This is the only level that matters for the watchlist.
"""
import json, pathlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

detail_files = sorted(pathlib.Path('output/nse_local').glob('scores_detail_*.json'),
                      key=lambda p: p.stat().st_mtime)
if not detail_files:
    print("No scores_detail JSON found. Run --skip_train first.")
    raise SystemExit

detail = json.loads(detail_files[-1].read_text())

# Extract per-ticker: model score (pure LGBM output), composite score, final blended score
rows = []
for ticker, d in detail.items():
    b = d['bull']
    rows.append({
        'ticker':          ticker,
        'model_score':     b.get('model_score', 0.0),
        'composite_score': b.get('composite_score', 0.0),
        'model_weight':    b.get('model_weight', 0.6),
        'composite_weight':b.get('composite_weight', 0.4),
        'bull_rank':       d.get('bull_rank', 9999),
        'in_bull_wl':      d.get('in_bull_watchlist', False),
    })

df = pd.DataFrame(rows).set_index('ticker')
mw = df['model_weight'].iloc[0]
cw = df['composite_weight'].iloc[0]

# Reconstruct pure model ranking (what the watchlist would look like with model_w=1.0)
df['pure_model_rank']   = df['model_score'].rank(ascending=False).astype(int)
# Blended rank (what actually happened)
df['blended_rank']      = df['bull_rank']
# Composite-only rank (what pure manual would look like)
df['composite_only_rank'] = df['composite_score'].rank(ascending=False).astype(int)

TOP_N = 12

pure_top12     = set(df.nsmallest(TOP_N, 'pure_model_rank').index)
blended_top12  = set(df.nsmallest(TOP_N, 'blended_rank').index)
composite_top12= set(df.nsmallest(TOP_N, 'composite_only_rank').index)

print(f"=== TOP-{TOP_N} WATCHLIST COMPARISON ===")
print(f"Model weight={mw:.0%}, Composite weight={cw:.0%}\n")

print(f"Pure model top-{TOP_N}    vs  Blended top-{TOP_N}:")
print(f"  Overlap     : {len(pure_top12 & blended_top12)}/{TOP_N} stocks identical")
print(f"  Dropped by blend : {pure_top12 - blended_top12}")
print(f"  Added by blend   : {blended_top12 - pure_top12}")

print(f"\nComposite-only top-{TOP_N} vs  Blended top-{TOP_N}:")
print(f"  Overlap          : {len(composite_top12 & blended_top12)}/{TOP_N} stocks identical")

print(f"\nPure model top-{TOP_N} vs Composite-only top-{TOP_N}:")
print(f"  Overlap          : {len(pure_top12 & composite_top12)}/{TOP_N} stocks")
print(f"  Disagreements    : {pure_top12.symmetric_difference(composite_top12)}")

# Rank shifts for the actual watchlist stocks
print(f"\n=== RANK SHIFTS FOR CURRENT WATCHLIST STOCKS ===")
print(f"{'Ticker':<20} {'PureModel':>10} {'Blended':>10} {'CompOnly':>10} {'Shift':>8}")
print("-" * 60)
wl_stocks = df[df['in_bull_wl']].sort_values('blended_rank')
for t, row in wl_stocks.iterrows():
    shift = int(row['blended_rank']) - int(row['pure_model_rank'])
    shift_str = f"{shift:+d}"
    print(f"{t:<20} {int(row['pure_model_rank']):>10} {int(row['blended_rank']):>10} "
          f"{int(row['composite_only_rank']):>10} {shift_str:>8}")

# Spearman at top-50 only (where it matters)
top50 = df.nsmallest(50, 'blended_rank')
r50, _ = spearmanr(top50['model_score'], -top50['blended_rank'])
print(f"\nSpearman r(model, blended) — full universe (457): 0.983  [already computed]")
print(f"Spearman r(model, blended) — top 50 only        : {r50:.3f}")

# How many stocks in top-50 by model are NOT in top-50 by blend?
model_top50   = set(df.nsmallest(50, 'pure_model_rank').index)
blended_top50 = set(df.nsmallest(50, 'blended_rank').index)
print(f"\nTop-50 overlap (model vs blended): {len(model_top50 & blended_top50)}/50")
print(f"  Stocks model ranks top-50 but blend doesn't: {model_top50 - blended_top50}")
print(f"  Stocks blend ranks top-50 but model doesn't: {blended_top50 - model_top50}")
