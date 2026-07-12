import os
import pandas as pd
import numpy as np
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

def main():
    if os.path.exists("panel_features.pkl"):
        print("Loading panel_features.pkl...")
        df = pd.read_pickle("panel_features.pkl")
    elif os.path.exists("artefacts/panel_features.pkl"):
        print("Loading artefacts/panel_features.pkl...")
        df = pd.read_pickle("artefacts/panel_features.pkl")
    else:
        print("Running python scripts/diag_phase4_features.py might have saved something, but I cannot find the pkl.")
        return

    # Panel feature columns are prefixed "features_"
    features = [c for c in df.columns if c.startswith("features_")]
    print(f"Found {len(features)} features.")

    target_col = "future_20d_excess_return"
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found. Cannot compute IC tiebreakers.")
        return

    # 1. Compute Univariate Rank IC globally
    print("Computing out-of-sample (global) Rank IC against target for tiebreakers...")
    ic_df = pd.DataFrame(index=features)
    # df.corr computes pairwise spearman ignoring NaNs correctly without global dropna.
    # To save memory and time, we can compute correlation of features against target directly.
    ic_vals = {}
    for f in features:
        # compute spearman corr between feature and target
        ic_vals[f] = df[f].corr(df[target_col], method="spearman")
        
    ic_series = pd.Series(ic_vals).fillna(0)

    # 2. Compute Feature Correlation Matrix
    # We sample for correlation computation to avoid massive pairwise memory spikes / time
    print("Computing pairwise Feature Correlation Matrix (sampled for speed)...")
    valid_dates = df.index.get_level_values("date").unique()
    # sample cross-sections to preserve relationships
    sample_dates = pd.Series(valid_dates).sample(min(400, len(valid_dates)), random_state=42)
    df_sample = df.loc[df.index.get_level_values("date").isin(sample_dates)]
    
    # corr handles pairwise NaNs automatically
    corr = df_sample[features].corr(method="spearman").abs()
    
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val > 0.8:
                f1 = corr.index[i]
                f2 = corr.columns[j]
                
                ic1 = ic_series[f1]
                ic2 = ic_series[f2]
                
                # Winner is the one with higher absolute IC
                if abs(ic1) >= abs(ic2):
                    winner, loser = f1, f2
                    w_ic, l_ic = ic1, ic2 
                else:
                    winner, loser = f2, f1
                    w_ic, l_ic = ic2, ic1
                    
                high_corr_pairs.append({
                    "f1": f1, "f2": f2, "corr": val,
                    "winner": winner, "winner_ic": w_ic,
                    "loser": loser, "loser_ic": l_ic
                })
                
    high_corr_pairs.sort(key=lambda x: x["corr"], reverse=True)
    print("\nHighly correlated pairs (>0.8) and OOS tiebreakers:")
    for p in high_corr_pairs[:25]:
        print(f"Corr {p['corr']:.3f} | {p['f1']} vs {p['f2']}")
        print(f"  -> Keep: {p['winner']} (IC: {p['winner_ic']:.4f})  Drop: {p['loser']} (IC: {p['loser_ic']:.4f})")

    print("\nComputing VIF (Median-Imputed)...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    
    # Impute missing values with median for VIF computation
    df_vif = df_sample[features].fillna(df_sample[features].median())
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_vif)
    
    vifs = []
    for i in range(X_scaled.shape[1]):
        try:
            v_val = variance_inflation_factor(X_scaled, i)
        except Exception:
            v_val = np.nan
        vifs.append(v_val)
        
    vif_data["VIF"] = vifs
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    print("\nTop 20 VIFs:")
    print(vif_data.head(20))
    
    with open("exp501_results.txt", "w") as f:
        f.write(f"Total features: {len(features)}\n")
        f.write(f"Highly correlated pairs (>0.8): {len(high_corr_pairs)}\n\n")
        f.write("Pairwise Correlations (Winner decided by |IC|):\n")
        for p in high_corr_pairs:
            f.write(f"Corr {p['corr']:.3f} | Keep: {p['winner']} (IC: {p['winner_ic']:.4f}) | Drop: {p['loser']} (IC: {p['loser_ic']:.4f})\n")
        f.write("\n\nVIF:\n")
        f.write(vif_data.to_string())

if __name__ == "__main__":
    main()

