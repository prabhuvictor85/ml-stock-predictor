# Check what columns your Drv files actually contain for a sample ticker
import pandas as pd
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config.paths import PATHS
sample = PATHS.stock_data.nse_local
ticker = "RELIANCE.NS"   # replace with any ticker you know has an OB

for tf in ["1d", "1wk"]:
    p = sample / f"{ticker}-{tf}-Drv.csv"
    if p.exists():
        df = pd.read_csv(p)
        ob_cols = [c for c in df.columns if 'ob' in c.lower() or 'order' in c.lower()]
        print(f"\n{tf}: OB-related columns → {ob_cols}")