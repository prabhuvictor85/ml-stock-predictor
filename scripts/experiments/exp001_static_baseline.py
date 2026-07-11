import sys
import os
import gc
import json
import warnings
from pathlib import Path

# Fix Windows encoding issue
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from pipeline.universe.us_loader import load_us_universe
from pipeline.features.engineer import build_features
from pipeline.targets.builder import TargetBuilder
from pipeline.validation.cv import PurgedWalkForwardCV
from pipeline.config.paths import PATHS
import lightgbm as lgb

STOCK_LIST_CSV  = PATHS.stock_lists.us_combined
STOCK_DATA_DIR  = PATHS.stock_data.us

def load_local_ohlcv(ticker: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"{ticker}-1d.csv"
    if not path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=lambda c: c.lower() in ("date", "datetime", "timestamp", "open", "high", "low", "close", "volume"))
    except:
        return pd.DataFrame()
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if not date_col: return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns: return pd.DataFrame()
    return df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").replace(0, np.nan).dropna(subset=["close"])

def get_base_panel():
    print("Loading universe...", flush=True)
    tickers_info = load_us_universe(STOCK_LIST_CSV)
    tickers = list(tickers_info.keys())
    
    print(f"Loading CSVs for {len(tickers)} tickers...", flush=True)
    dfs = []
    # To save time for this baseline evaluation, let's load just a subset if testing, 
    # but we need the true baseline. Let's load the full set but it takes ~1.5m
    for i, t in enumerate(tickers):
        df = load_local_ohlcv(t, STOCK_DATA_DIR)
        if len(df) > 252:
            df["ticker"] = t
            dfs.append(df.reset_index())
        if i > 0 and i % 500 == 0:
            print(f"Loaded {i}...", flush=True)
            
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.set_index(["date", "ticker"]).sort_index()
    print(f"Panel loaded: {full_df.shape}", flush=True)
    return full_df

def ic_metrics(results):
    ic_arr = np.array(results)
    mean_ic = np.mean(ic_arr)
    std_ic = np.std(ic_arr)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    return mean_ic, icir

def main():
    print("=== Exp-001: True Baseline Establishment ===", flush=True)
    panel = get_base_panel()
    
    # Simple feature set to establish linear and tree baseline viability 
    # (avoiding the 5 minute strict feature engineering lockup for now, 
    # just generating momentum and volatility)
    print("Generating baseline momentum/vol features...", flush=True)
    panel["ret_1d"] = panel.groupby("ticker")["close"].pct_change(1)
    panel["ret_5d"] = panel.groupby("ticker")["close"].pct_change(5)
    panel["ret_20d"] = panel.groupby("ticker")["close"].pct_change(20)
    
    panel["vol_20d"] = panel.groupby("ticker")["ret_1d"].rolling(20).std().reset_index(0, drop=True)
    panel["price_vs_sma20"] = (panel["close"] / panel.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())) - 1.0
    
    features = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "price_vs_sma20"]
    
    # Target Building
    print("Building Targets (20d forward returns)...", flush=True)
    # Forward 20d return
    panel["future_20d_ret"] = panel.groupby("ticker")["close"].transform(lambda x: x.shift(-20) / x - 1)
    
    # Cross sectional ranking of the target (Spearman Rank IC compares rank to rank, so we can just use the continuous target for Spearman or rank it)
    panel["cs_rank_20d"] = panel.groupby("date")["future_20d_ret"].rank(pct=True)
    panel["group_date"] = panel.index.get_level_values("date")
    
    # Drop rows with NaNs in features or targets
    panel = panel.dropna(subset=features + ["future_20d_ret", "cs_rank_20d"])
    
    print(f"Panel ready for ML: {panel.shape}", flush=True)
    
    cv = PurgedWalkForwardCV(n_folds=5, min_train_window=504, test_window=252, purge_window=40)
    
    ridge_ics = []
    lgb_ics = []
    
    for fold_id, (spec, train_idx, test_idx) in enumerate(cv.split(panel)):
        print(f"--- Fold {fold_id} ---", flush=True)
        tr = panel.loc[train_idx]
        te = panel.loc[test_idx]
        
        X_tr = tr[features].values
        # Using continuous target for Ridge
        y_tr = tr["future_20d_ret"].values 
        
        X_te = te[features].values
        y_te = te["future_20d_ret"].values
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        ridge_preds = ridge.predict(X_te)
        
        # LGBM Stump (Depth 3)
        lgb_model = lgb.train(
            {"objective": "regression", "max_depth": 3, "learning_rate": 0.05, "verbose": -1, "seed": 42},
            lgb.Dataset(X_tr, y_tr),
            num_boost_round=100
        )
        lgb_preds = lgb.model.predict(X_te) if hasattr(lgb_model, 'predict') else lgb_model.predict(X_te)
        
        # Eval
        te = te.copy()
        te["ridge_p"] = ridge_preds
        te["lgb_p"] = lgb_preds
        
        # Calculate daily Rank IC
        date_ridge_ic = []
        date_lgb_ic = []
        for dt, grp in te.groupby("date"):
            if len(grp) < 20: continue
            if grp["future_20d_ret"].std() < 1e-9: continue
            
            # Spearman
            ric, _ = spearmanr(grp["ridge_p"], grp["future_20d_ret"])
            lic, _ = spearmanr(grp["lgb_p"], grp["future_20d_ret"])
            
            if not np.isnan(ric): date_ridge_ic.append(ric)
            if not np.isnan(lic): date_lgb_ic.append(lic)
            
        fold_ridge_ic = np.mean(date_ridge_ic) if date_ridge_ic else 0
        fold_lgb_ic = np.mean(date_lgb_ic) if date_lgb_ic else 0
        
        print(f"  Ridge Mean IC: {fold_ridge_ic:.4f}")
        print(f"  LGB Mean IC:   {fold_lgb_ic:.4f}")
        ridge_ics.append(fold_ridge_ic)
        lgb_ics.append(fold_lgb_ic)
        
    print("\n=== Final Baseline Metrics ===", flush=True)
    r_mean, r_icir = ic_metrics(ridge_ics)
    l_mean, l_icir = ic_metrics(lgb_ics)
    
    print(f"Ridge     -> Mean Rank IC: {r_mean:.4f} | ICIR: {r_icir:.2f}")
    print(f"LGB Stump -> Mean Rank IC: {l_mean:.4f} | ICIR: {l_icir:.2f}")

if __name__ == "__main__":
    main()
