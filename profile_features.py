"""Profile the feature engineering step on 10 real tickers to find the bottleneck."""
import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

from run_nse_local import load_local_ohlcv, build_panel_from_local

DATA_DIR = Path(r'C:\Victor\Learning_charts\stock_data')
LIST_CSV = Path(r'C:\Victor\Learning_charts\stock_lists\constituentsi.csv')

ticker_df = pd.read_csv(LIST_CSV)
tickers10 = ticker_df["Symbol"].str.strip().tolist()[:20]

print("Building panel for 20 tickers...")
t0 = time.time()
panel = build_panel_from_local(tickers10, DATA_DIR, min_history_days=252)
print(f"  Panel built: {len(panel)} rows in {time.time()-t0:.1f}s")

# Fake benchmark
dates = panel.index.get_level_values("date").unique().sort_values()
bm = pd.Series(100 * np.cumprod(np.exp(np.random.normal(0.0002, 0.012, len(dates)))),
               index=dates, name="benchmark_close")

from pipeline.config.nse import NSE_CONFIG as cfg

print("Feature engineering (20 tickers)...")
from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
t_steps = {}

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
fe = FeatureEngineer(cfg, bm)
panel = fe.build(panel)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())

feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
print(f"Feature cols: {len(feat_cols)}")
print(f"Total time: {time.time()-t0:.1f}s for 20 tickers => est {(time.time()-t0)/20*457:.0f}s for 457")

