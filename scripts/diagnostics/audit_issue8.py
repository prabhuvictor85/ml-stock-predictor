"""Audit Issue 8: cross-reference composite signals against selected model features."""
selected = open('artefacts/nse_local/selected_features.txt').read().strip().split()
selected_stripped = [f.replace('features_', '') for f in selected]

composite_bull = {
    'sdz_htf_score': 3.0, 'dz_raw_score': 2.0, 'inside_demand': 1.5,
    'sdz_premium_setup': 3.0, 'ict_bull_htf_score': 2.0, 'ict_bob_active': 1.5,
    'ict_bullfvg_active': 1.0, 'ict_bob_dist': 0.5,
    'price_vs_sma20': 1.5, 'price_vs_sma50': 2.0, 'price_vs_sma200': 2.0,
    'sma20_slope_5': 1.0, 'sma50_slope_5': 1.5, 'sma200_slope_10': 1.5,
    'regime_bull': 2.0, 'weekly_trend': 1.5, 'monthly_trend': 2.0,
    'quarterly_trend': 2.5, 'yearly_trend': 3.0,
    'return_20d': 1.0, 'return_60d': 1.5, 'vol_ratio_5d': 0.5, 'adx_14': 1.0,
}

print('=== BULL COMPOSITE vs MODEL FEATURES ===')
in_model, not_in_model = [], []
for sig, w in composite_bull.items():
    match = any(s == sig or s.startswith(sig) for s in selected_stripped)
    label = 'IN MODEL    ' if match else 'NOT IN MODEL'
    print(f'  {label}  w={w:.1f}  {sig}')
    if match:
        in_model.append((sig, w))
    else:
        not_in_model.append((sig, w))

w_in  = sum(w for _, w in in_model)
w_out = sum(w for _, w in not_in_model)
total = w_in + w_out
print(f'\nSummary: {len(in_model)}/{len(composite_bull)} bull composite signals already in model')
print(f'  Weight already in model : {w_in:.1f} / {total:.1f} = {w_in/total*100:.0f}%')
print(f'  Weight NOT in model     : {w_out:.1f} / {total:.1f} = {w_out/total*100:.0f}%')
print(f'\nSignals NOT in model (potential genuine value-add):')
for sig, w in not_in_model:
    print(f'  w={w:.1f}  {sig}')

# Now measure ranking disruption on the last saved cross-section
print('\n=== RANKING DISRUPTION MEASUREMENT ===')
try:
    import pandas as pd, numpy as np
    from scipy.stats import spearmanr, kendalltau

    bull_csv = sorted(
        (p for p in __import__('pathlib').Path('output/nse_local').glob('watchlist_bull_*.csv')),
        key=lambda p: p.stat().st_mtime
    )[-1]
    scores_json = sorted(
        (p for p in __import__('pathlib').Path('output/nse_local').glob('scores_detail_*.csv')),
        key=lambda p: p.stat().st_mtime
    ) if list(__import__('pathlib').Path('output/nse_local').glob('scores_detail_*.csv')) else []

    # Load scores_detail JSON for all-universe model vs final scores
    import json, pathlib
    detail_files = sorted(pathlib.Path('output/nse_local').glob('scores_detail_*.json'),
                          key=lambda p: p.stat().st_mtime)
    if detail_files:
        detail = json.loads(detail_files[-1].read_text())
        model_scores = {}
        bull_final   = {}
        for t, d in detail.items():
            model_scores[t] = d['bull'].get('model_score', 0)
            bull_final[t]   = d['bull_rank']   # rank (lower = better)

        model_s = pd.Series(model_scores)
        final_r = pd.Series(bull_final)
        # align
        common = model_s.index.intersection(final_r.index)
        r, p = spearmanr(model_s[common], -final_r[common])  # negate rank: lower rank = higher score
        tau, _ = kendalltau(model_s[common].rank(), (-final_r[common]).rank())
        print(f'  Spearman r(model_score, final_rank) = {r:.3f}  (1.0 = identical order)')
        print(f'  Kendall tau                          = {tau:.3f}')
        print(f'  Universe size: {len(common)} stocks')
        print()
        print('  Interpretation:')
        if r > 0.95:
            print('  -> Blending barely changes order. Composite adds negligible disruption.')
        elif r > 0.80:
            print('  -> Moderate disruption. ~10-20% of stocks would rank differently.')
        else:
            print('  -> High disruption. Composite is significantly overriding model ranking.')
    else:
        print('  scores_detail JSON not found — run with --skip_train to generate')
except Exception as e:
    print(f'  Could not compute ranking disruption: {e}')
