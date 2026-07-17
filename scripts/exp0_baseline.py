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
    print(" Phase 0: True Walk-Forward Baseline (No HPO / Unpruned)")
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
    
    features = [c for c in df.columns if c.startswith("features_")]
    print(f"Loaded {len(df)} rows.")
    print(f"Total features: {len(features)}")

    # Restrict to <2023 for development baseline (leaving 2024-2026 for pure lockbox)
    df_train = df.loc[df.index.get_level_values("date") <= "2023-12-31"].copy()
    
    cv = PurgedWalkForwardCV(n_folds=5, min_train_window=378, purge_window=PURGE_HORIZON)
    
    print("\nRunning Walk-Forward cross-validation...")
    # Standard realistic parameters
    model = LGBMRanker(params={"n_estimators": 150, "learning_rate": 0.03, "max_depth": 6})

    fold_metrics = []
    
    for spec, tr_idx, te_idx in cv.split(df_train):
        print(f"\n--- Fold {spec.fold_id} ---")
        
        X_tr = df_train.iloc[tr_idx]
        X_te = df_train.iloc[te_idx]
        
        # Filter for universe and non-null target
        tr_valid = X_tr[X_tr["in_universe"] == True].dropna(subset=["cs_rank_composite_full", *features])
        te_valid = X_te[X_te["in_universe"] == True].dropna(subset=["cs_rank_composite_full", *features])
        
        if len(tr_valid) == 0 or len(te_valid) == 0:
            print("Skipping due to insufficient data")
            continue
            
        print(f"Train size: {len(tr_valid):,} | Test size: {len(te_valid):,}")
        
        y_tr = cs_rank_to_label(tr_valid["cs_rank_composite_full"].values)
        y_te = cs_rank_to_label(te_valid["cs_rank_composite_full"].values)
        
        grp_tr = tr_valid.groupby(level="date").size().values
        grp_te = te_valid.groupby(level="date").size().values
        
        model.fit(
            tr_valid[features], y_tr, grp_tr,
            eval_set=[(te_valid[features], y_te, grp_te)]
        )
        
        preds = pd.Series(model.predict(te_valid[features]), index=te_valid.index)
        
        # We compute metrics
        metrics = compute_fold_metrics(
            panel_test=te_valid,
            scores=preds,
            feature_cols=features,
            benchmark_returns=te_valid["benchmark_20d_return"], # Approx mapping
            commission_bps=5.0,
            slippage_bps=10.0,
            top_n=10
        )
        
        print(f"Fold Rank IC:   {metrics['mean_rank_ic']:.4f}")
        print(f"Fold ICIR:      {metrics['icir']:.4f}")
        print(f"Fold NDCG@10:   {metrics['ndcg@10']:.4f}")
        
        fold_metrics.append(metrics)

    if not fold_metrics:
        print("No folds processed.")
        return
        
    print("\n==========================================================")
    print(" FINAL OOS BASELINE RESULTS")
    print("==========================================================")
    
    mean_ic = np.mean([m['mean_rank_ic'] for m in fold_metrics])
    mean_icir = np.mean([m['icir'] for m in fold_metrics])
    mean_ndcg = np.mean([m['ndcg@10'] for m in fold_metrics])
    
    print(f"Mean OOS Rank IC:   {mean_ic:.4f}")
    print(f"Mean OOS ICIR:      {mean_icir:.4f}  (Target > 1.0)")
    print(f"Mean OOS NDCG@10:   {mean_ndcg:.4f}")
    
    with open("phase0_baseline_results.txt", "w") as f:
        f.write("Baseline (All Features, Standard LGBM) Results:\n")
        f.write(f"Mean Rank IC: {mean_ic:.4f}\n")
        f.write(f"Mean ICIR: {mean_icir:.4f}\n")
        f.write(f"Mean NDCG@10: {mean_ndcg:.4f}\n")
        for i, m in enumerate(fold_metrics):
            f.write(f"Fold {i+1} IC: {m['mean_rank_ic']:.4f}\n")

if __name__ == "__main__":
    main()

