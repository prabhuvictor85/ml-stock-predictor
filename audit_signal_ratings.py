"""
Rate each signal in signal_weights.yaml against:
1. Whether it's in the model's selected features
2. Its SHAP importance rank (from the trained model)
3. Its assigned composite weight
"""
import json, pathlib
import pandas as pd
import numpy as np

# Load SHAP importance
shap_files = sorted(pathlib.Path('reports').glob('*.png'))  # just check reports exist

# Load scores_detail to get model scores — use SHAP from explanations
expl_files = sorted(pathlib.Path('output/nse_local').glob('explanations_*.json'),
                    key=lambda p: p.stat().st_mtime)

# Load selected features
selected = open('artefacts/nse_local/selected_features.txt').read().strip().split()
selected_stripped = set(f.replace('features_', '') for f in selected)

# Load SHAP global importance via the saved model
print("Loading SHAP global importance from trained model...")
try:
    import pickle, warnings
    warnings.filterwarnings('ignore')
    with open('artefacts/nse_local/lgbm_ranker.pkl', 'rb') as f:
        ranker = pickle.load(f)

    import shap
    import pandas as pd_
    # Use feature importances from the booster as proxy for SHAP mean abs
    # (faster than recomputing SHAP on full panel)
    booster = ranker.model_
    fi = booster.feature_importance(importance_type='gain')
    feat_names = booster.feature_name()
    shap_df = pd.DataFrame({'feature': feat_names, 'gain': fi})
    shap_df['gain_pct'] = shap_df['gain'] / shap_df['gain'].sum() * 100
    shap_df = shap_df.sort_values('gain', ascending=False).reset_index(drop=True)
    shap_df['rank'] = shap_df.index + 1
    shap_rank = dict(zip(shap_df['feature'].str.replace('features_',''),
                         shap_df['rank']))
    shap_pct  = dict(zip(shap_df['feature'].str.replace('features_',''),
                         shap_df['gain_pct']))
    print(f"  Loaded gain importance for {len(feat_names)} features\n")
except Exception as e:
    print(f"  Could not load model: {e}")
    shap_rank, shap_pct = {}, {}

# Signal weights
bull_signals = {
    'sdz_htf_score': 3.0, 'dz_raw_score': 2.0, 'inside_demand': 1.5,
    'sdz_premium_setup': 3.0, 'ict_bull_htf_score': 2.0, 'ict_bob_active': 1.5,
    'ict_bullfvg_active': 1.0, 'ict_bob_dist': 0.5,
    'price_vs_sma20': 1.5, 'price_vs_sma50': 2.0, 'price_vs_sma200': 2.0,
    'sma20_slope_5': 1.0, 'sma50_slope_5': 1.5, 'sma200_slope_10': 1.5,
    'regime_bull': 2.0, 'weekly_trend': 1.5, 'monthly_trend': 2.0,
    'quarterly_trend': 2.5, 'yearly_trend': 3.0,
    'return_20d': 1.0, 'return_60d': 1.5, 'vol_ratio_5d': 0.5, 'adx_14': 1.0,
}

bear_signals = {
    'ssz_htf_score': 3.0, 'sz_raw_score': 2.0, 'inside_supply': 1.5,
    'ssz_premium_setup': 3.0, 'ict_bear_htf_score': 2.0, 'ict_sob_active': 1.5,
    'ict_bearfvg_active': 1.0, 'ict_sob_dist': 0.5,
    'price_below_sma20': 1.5, 'price_below_sma50': 2.0, 'price_below_sma200': 2.0,
    'sma20_falling': 1.0, 'sma50_falling': 1.5, 'sma200_falling': 1.5,
    'regime_bear': 2.0, 'weekly_down': 1.5, 'monthly_down': 2.0,
    'quarterly_down': 2.5, 'yearly_down': 3.0,
    'return_20d_neg': 1.0, 'return_60d_neg': 1.5, 'vol_ratio_5d': 0.5, 'adx_14': 1.0,
}

# Bear signals map to base feature name
bear_base = {
    'price_below_sma20': 'price_vs_sma20', 'price_below_sma50': 'price_vs_sma50',
    'price_below_sma200': 'price_vs_sma200', 'sma20_falling': 'sma20_slope_5',
    'sma50_falling': 'sma50_slope_5', 'sma200_falling': 'sma200_slope_10',
    'weekly_down': 'weekly_trend', 'monthly_down': 'monthly_trend',
    'quarterly_down': 'quarterly_trend', 'yearly_down': 'yearly_trend',
    'return_20d_neg': 'return_20d', 'return_60d_neg': 'return_60d',
}

def rate(sig, w, is_bear=False):
    base = bear_base.get(sig, sig) if is_bear else sig
    in_model = base in selected_stripped or sig in selected_stripped
    rank = shap_rank.get(base, shap_rank.get(sig, None))
    pct  = shap_pct.get(base, shap_pct.get(sig, 0.0))
    n_feats = len(selected_stripped)

    if not in_model:
        verdict = "NOT IN MODEL"
        stars   = "?"
    elif rank is None:
        verdict = "IN MODEL / no rank"
        stars   = "?"
    elif rank <= 5:
        verdict = "TOP 5"
        stars   = "*****"
    elif rank <= 15:
        verdict = "TOP 15"
        stars   = "****"
    elif rank <= 30:
        verdict = "TOP 30"
        stars   = "***"
    elif rank <= 50:
        verdict = "TOP 50"
        stars   = "**"
    else:
        verdict = f"rank {rank}"
        stars   = "*"

    return in_model, rank, pct, verdict, stars

def print_table(signals, is_bear=False):
    label_w = 24
    print(f"  {'Signal':<24} {'YML_w':>6}  {'InModel':>8}  {'ModelRank':>10}  {'Gain%':>7}  Rating")
    print("  " + "-"*78)
    for sig, w in sorted(signals.items(), key=lambda x: -x[1]):
        in_model, rank, pct, verdict, stars = rate(sig, w, is_bear)
        rank_str = str(rank) if rank else "-"
        pct_str  = f"{pct:.2f}%" if pct else "-"
        in_str   = "YES" if in_model else "NO "
        print(f"  {sig:<24} {w:>6.1f}  {in_str:>8}  {rank_str:>10}  {pct_str:>7}  {stars}  {verdict}")

print("="*82)
print("  BULL SIGNALS — yaml weight vs model importance")
print("="*82)
print_table(bull_signals, is_bear=False)

print()
print("="*82)
print("  BEAR SIGNALS — yaml weight vs model importance")
print("="*82)
print_table(bear_signals, is_bear=True)

print()
print("="*82)
print("  MISMATCH ANALYSIS  (yaml weight vs model rank)")
print("="*82)
print("\n  HIGH yaml weight but LOW model rank (composite overrides model evidence):")
for sig, w in {**bull_signals, **bear_signals}.items():
    base = bear_base.get(sig, sig)
    in_model = base in selected_stripped or sig in selected_stripped
    rank = shap_rank.get(base, shap_rank.get(sig, None))
    if in_model and rank and rank > 30 and w >= 1.5:
        print(f"    {sig:<28} yaml_w={w:.1f}  model_rank={rank}")

print("\n  HIGH model rank but LOW yaml weight (model knows more than yaml gives credit):")
for sig, w in {**bull_signals, **bear_signals}.items():
    base = bear_base.get(sig, sig)
    in_model = base in selected_stripped or sig in selected_stripped
    rank = shap_rank.get(base, shap_rank.get(sig, None))
    if in_model and rank and rank <= 10 and w <= 1.0:
        print(f"    {sig:<28} yaml_w={w:.1f}  model_rank={rank}")
