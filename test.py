# Check what columns your Drv files actually contain for a sample ticker
import pandas as pd
from pathlib import Path

sample = Path(r"C:\Victor\Learning_charts\stock_data")
ticker = "RELIANCE.NS"   # replace with any ticker you know has an OB

for tf in ["1d", "1wk"]:
    p = sample / f"{ticker}-{tf}-Drv.csv"
    if p.exists():
        df = pd.read_csv(p)
        ob_cols = [c for c in df.columns if 'ob' in c.lower() or 'order' in c.lower()]
        print(f"\n{tf}: OB-related columns → {ob_cols}")