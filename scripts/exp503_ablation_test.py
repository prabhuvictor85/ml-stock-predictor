import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.targets.builder import PURGE_HORIZON
from pipeline.validation.cv import PurgedWalkForwardCV
from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
from pipeline.validation.metrics import compute_fold_metrics
from pipeline.utils.logging import get_logger

log = get_logger(__name__)

def main():
    print("==========================================================")
    print(" Exp-502b: Ablation Validation Test (Model A vs Model B)")
    print("==========================================================")

    panel_paths = [
        "artefacts/us_local_alpha/checkpoints/panel_targets.pkl" if len(sys.argv) > 1 and sys.argv[1] == "alpha" else "artefacts/us_local/checkpoints/panel_targets.pkl",
        "artefacts/checkpoints/panel_targets.pkl",
        "artefacts/nse_local/checkpoints/panel_targets.pkl",
        "artefacts/nse_tradingv/checkpoints/panel_targets.pkl",
        "panel_features.pkl"
    ]
    
    panel_path = None
    for p in panel_paths:
        if os.path.exists(p):
            panel_path = p
            break
            
    if not panel_path:
        print("Cannot find any panel_targets.pkl. Run builder first.")
        return

    print(f"Loading '{panel_path}'...")
    df = pd.read_pickle(panel_path)
    
    # 1. Assume we have a list of features to test dropping from Exp-502
    drop_candidates = [] # Populate this list with literally 0.0 SHAP features
    
    # Automatically load drop_candidates from exp502_results.csv if it exists
    if os.path.exists("exp502_results.csv"):
        shap_res = pd.read_csv("exp502_results.csv", index_col=0)
        # Find features with EXACTLY 0.0 mean impact
        useless_features = shap_res[shap_res['mean_impact'] == 0.0].index.tolist()
        drop_candidates.extend(useless_features)
        print(f"Auto-loaded {len(useless_features)} drop candidates from exp502_results.csv")
    
    if not drop_candidates:
        print("\n[!] No drop candidates found (no features had exactly 0.0 SHAP impact).")
        print("Nothing to ablate! Baseline is already minimal in terms of dead-weight.")
        return

    all_features = [c for c in df.columns if c.startswith("features_")]
    pruned_features = [c for c in all_features if c not in drop_candidates]

    print(f"\nModel A (All Features): {len(all_features)}")
    print(f"Model B (Pruned):       {len(pruned_features)}")
    
    df_train = df.loc[df.index.get_level_values("date") <= "2023-12-31"].copy()
    cv = PurgedWalkForwardCV(n_folds=5, min_train_window=378, purge_window=PURGE_HORIZON)
    
    # Helper to run cv
    def run_cv(feature_set, name="Model"):
        print(f"\n--- Running Walk-Forward CV for {name} ---")
        model = LGBMRanker(params={"n_estimators": 150, "learning_rate": 0.03, "max_depth": 6})
        fold_metrics = []
        
        for spec, tr_idx, te_idx in cv.split(df_train):
            X_tr = df_train.iloc[tr_idx]
            X_te = df_train.iloc[te_idx]
            
            tr_valid = X_tr[X_tr["in_universe"] == True].dropna(subset=["cs_rank_composite_full", *feature_set])
            te_valid = X_te[X_te["in_universe"] == True].dropna(subset=["cs_rank_composite_full", *feature_set])
            
            if len(tr_valid) == 0 or len(te_valid) == 0:
                continue
                
            y_tr = cs_rank_to_label(tr_valid["cs_rank_composite_full"].values)
            y_te = cs_rank_to_label(te_valid["cs_rank_composite_full"].values)
            
            grp_tr = tr_valid.groupby(level="date").size().values
            grp_te = te_valid.groupby(level="date").size().values
            
            model.fit(
                tr_valid[feature_set], y_tr, grp_tr,
                eval_set=[(te_valid[feature_set], y_te, grp_te)]
            )
            
            preds = pd.Series(model.predict(te_valid[feature_set]), index=te_valid.index)
            
            metrics = compute_fold_metrics(
                panel_test=te_valid,
                scores=preds,
                feature_cols=feature_set,
                benchmark_returns=te_valid["benchmark_20d_return"],
                commission_bps=5.0,
                slippage_bps=10.0,
                top_n=10
            )
            fold_metrics.append(metrics)
            
        mean_ic = np.mean([m['mean_rank_ic'] for m in fold_metrics])
        mean_icir = np.mean([m['icir'] for m in fold_metrics])
        return mean_ic, mean_icir
        
    ic_A, icir_A = run_cv(all_features, "Model A (All Features)")
    ic_B, icir_B = run_cv(pruned_features, "Model B (Pruned Features)")
    
    print("\n==========================================================")
    print(" ABLATION TEST RESULTS")
    print("==========================================================")
    print(f"Model A (All features) - Rank IC: {ic_A:.4f} | ICIR: {icir_A:.4f}")
    print(f"Model B (Pruned list)  - Rank IC: {ic_B:.4f} | ICIR: {icir_B:.4f}")
    
    if ic_B >= ic_A:
        print("\n[SUCCESS] The pruned features were truly noise. Model B holds its ground or improves!")
    else:
        print("\n[WARNING] Rank IC degraded. The dropped features had conditional value. Do NOT prune them.")

if __name__ == "__main__":
    main()

