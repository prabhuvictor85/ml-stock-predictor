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

    # Panel feature columns are prefixed "features_" ("feat_" matches nothing:
    # the 5th char of "features_" is 'u', not '_').
    features = [c for c in df.columns if c.startswith("features_")]
    print(f"Found {len(features)} features.")

    # WARNING: dropna() over ALL ~300 features keeps only complete cases —
    # warmup NaNs and NaN-native features (ICT/zones) can shrink this to a
    # tiny, non-representative subset. Check the printed row count before
    # trusting the VIF numbers; consider per-pair dropna or a feature subset.
    df_clean = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Rows after dropping NaNs in features: {len(df_clean)}")
    
    if len(df_clean) > 5000:
        df_sample = df_clean.sample(5000, random_state=42)
    else:
        df_sample = df_clean

    print("Computing Correlation Matrix...")
    corr = df_sample.corr(method="spearman").abs()
    
    high_corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if corr.iloc[i, j] > 0.8:
                high_corr_pairs.append((corr.index[i], corr.columns[j], corr.iloc[i, j]))
                
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    print("\nHighly correlated pairs (>0.8):")
    for p in high_corr_pairs[:20]:
        print(f"{p[0]} - {p[1]}: {p[2]:.3f}")

    print("\nComputing VIF...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_sample.columns
    
    # Normalize features for VIF
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sample)
    
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
        for p in high_corr_pairs:
            f.write(f"{p[0]} - {p[1]}: {p[2]:.3f}\n")
        f.write("\n\nVIF:\n")
        f.write(vif_data.to_string())

if __name__ == "__main__":
    main()

