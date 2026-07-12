import pandas as pd
import numpy as np
from pathlib import Path
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    try:
        print("Loading test dataset...")
        from pipeline.config.base import MarketConfig
        from pipeline.data.universe import UniverseBuilder
        from pipeline.data.fetcher import DataFetcher
        from pipeline.data.panel import PanelConstructor
        from pipeline.features.engineer import FeatureEngineer
        from pipeline.targets.builder import TargetBuilder

        # We will build a small sample panel dynamically using US stock data via Yahoo Finance!
        cfg = MarketConfig(
            market_id="us",
            exchange_calendar="XNYS",
            benchmark_ticker="SPY",
            currency="USD",
            data_source_primary="yfinance",
            data_source_fallback="yfinance",
            sector_classification="GICS_L1",
            min_adv_usd=100000.0,
        )
        
        # Ensure paths module uses env vars if needed
        print("Building universe...")
        ub = UniverseBuilder(cfg)
        
        # Manually create a small universe of highly liquid stocks
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "JPM", "JNJ", "V"]
        ub.load_symbol_master([
            {"ticker": t, "sector": "Tech"}
            for t in tickers
        ])
        
        print("Fetching data from Yahoo Finance...")
        df = DataFetcher(cfg)
        pc = PanelConstructor(cfg, df, ub)
        # Use a 2 year window
        panel = pc.build(tickers, "2022-01-01", "2023-12-31")

        print(f"Base panel built: {panel.shape} rows.")

        print("Engineering features...")
        try:
            bm_close = panel.loc(axis=0)[:, "SPY"]["close"].droplevel("ticker").sort_index()
        except KeyError:
            bm_close = panel.groupby(level="date")["close"].mean()
            
        fe = FeatureEngineer(cfg, benchmark_close=bm_close)
        panel = fe.build(panel)
        
        print("Building targets...")
        tb = TargetBuilder(cfg, benchmark_close=bm_close)
        panel = tb.build(panel)

        print(f"Feature engineering complete: {panel.shape} rows.")

        # 1. Orthogonality Check
        print("\n--- 1. Orthogonality Check (Spearman Rank Correlation) ---")
        p4_feats = [c for c in panel.columns if "gk_vol_" in c or "ret_skew" in c or "ret_kurt" in c or "vwap_dist" in c or "cmf" in c or "obv_osc" in c or "residual_mom" in c or "chop_idx" in c or "var_ratio" in c or "ad_thrust" in c]
        
        p3_feats = [c for c in panel.columns if ("ict_" in c or "sz_" in c or "dz_" in c or "sdz" in c or "ssz" in c) and c not in p4_feats]
        
        if not p4_feats or not p3_feats:
            print("Features not found in the panel.")
            return
            
        print(f"Found {len(p4_feats)} Phase 4 features.")
        
        # Take a random sample or just compute directly
        sample = panel[p4_feats + p3_feats].dropna().sample(min(100000, len(panel)))
        corr = sample.corr(method="spearman")
        
        high_corr = []
        for f4 in p4_feats:
            for f3 in p3_feats:
                if f4 in corr.index and f3 in corr.columns:
                    val = corr.loc[f4, f3]
                    if abs(val) > 0.5:
                        high_corr.append((f4, f3, val))
                        
        if not high_corr:
            print("PASS: No high correlation (>0.5) between new features and old features.")
        else:
            print("WARN: Found high correlations between new and old features:")
            for f4, f3, val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {f4} <-> {f3} : {val:.2f}")

        # 2. Univariate Rank IC
        print("\n--- 2. Univariate Rank IC (Cross-Sectional) ---")
        target_col = "future_20d_excess_return"

        if not target_col or target_col not in panel.columns:
            print("Target column not found.")
            return
            
        ic_results = []
        for f4 in p4_feats:
            df = panel[["date", f4, target_col]].dropna().reset_index(level="date")
            if df.empty:
                continue
                
            def rank_corr(g):
                if len(g) < 10: return np.nan
                return scipy.stats.spearmanr(g[f4], g[target_col])[0]
                
            daily_ic = df.groupby("date").apply(rank_corr).dropna()
            if daily_ic.empty:
                continue
                
            mean_ic = daily_ic.mean()
            ic_ir = mean_ic / daily_ic.std()
            t_stat = mean_ic / (daily_ic.std() / np.sqrt(len(daily_ic)))
            ic_results.append({
                "Feature": f4,
                "Mean_IC": mean_ic,
                "IC_IR": ic_ir,
                "t-stat": t_stat
            })
            
        if ic_results:
            res_df = pd.DataFrame(ic_results).sort_values("Mean_IC", ascending=False)
            print(res_df.to_string(index=False))
            
        # Also write to file so I can easily read it
        with open("phase4_diag_output.txt", "w") as f:
            f.write("--- Orthogonality Check ---\n")
            if not high_corr:
                f.write("PASS: No high correlation (>0.5)\n")
            else:
                for f4, f3, val in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True):
                    f.write(f"{f4} <-> {f3} : {val:.2f}\n")
            f.write("\n--- Rank IC ---\n")
            if ic_results:
                f.write(res_df.to_string(index=False) + "\n")
    except Exception as exp:
        import traceback
        with open("phase4_diag_output.txt", "w") as f:
            f.write("ERROR:\n" + traceback.format_exc())
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
