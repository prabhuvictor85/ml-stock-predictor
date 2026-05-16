"""Quick benchmark for ICT features and hit_target vectorised speed."""
import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path

from run_nse_local import load_local_ohlcv
from pipeline.features.ict_features import _wilder_atr, ICTFeatureEngine
from pipeline.targets.builder import _hit_target

df = load_local_ohlcv('360ONE.NS', Path(r'C:\Victor\Learning_charts\stock_data'))
print(f'Ticker rows: {len(df)}')

atr = _wilder_atr(df.high.values, df.low.values, df.close.values, 14)
df['atr_14'] = atr

t0 = time.time()
ict = ICTFeatureEngine()
out = ict.compute(df)
ict_cols = [c for c in out.columns if 'ict' in c or 'demand' in c or 'supply' in c or 'sdz' in c]
print(f'ICT features: {time.time()-t0:.2f}s  cols={ict_cols}')

t0 = time.time()
hits = _hit_target(df.high.values, df.low.values, df.open.values, 0.08, 0.04)
print(f'hit_target:   {time.time()-t0:.3f}s  hit_rate={np.nanmean(hits):.3f}')

# Extrapolate to 457 tickers
avg_rows = len(df)
est_ict   = (time.time() - t0) * 457 / len(df) * avg_rows   # rough
print(f'Est. 457 tickers ICT+targets: ~{(0.05 + 0.01) * 457:.0f}s')
print('BENCHMARK OK')

