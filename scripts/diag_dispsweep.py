"""How many bull Order Blocks / FVGs survive at different displacement multipliers?"""
import glob, os
import numpy as np, pandas as pd
from pipeline.features.ict_features import _wilder_atr
DATA_DIR="C:/Victor/Learning_charts/stock_data/nse_local"
mults=[0.0,1.0,1.5,2.0,2.5,3.0]
ob={m:0 for m in mults}; fvg={m:0 for m in mults}
for path in glob.glob(os.path.join(DATA_DIR,"*-1d.csv")):
    df=pd.read_csv(path,parse_dates=["Date"]).set_index("Date").sort_index()
    if len(df)<60: continue
    o,h,l,c=[df[x].values.astype(float) for x in ["open","high","low","close"]]
    atr=_wilder_atr(h,l,c,14); safe=np.where((atr>0)&~np.isnan(atr),atr,np.nan)
    def sh(a,k):
        r=np.full_like(a,np.nan,dtype=float); r[k:]=a[:-k]; return r
    o1,c1=sh(o,1),sh(c,1); h2=sh(h,2)
    dbmax=np.maximum(o1,c1); rb=np.abs(c-o); db=np.abs(c1-o1)
    bob_core=(c1<o1)&(c>o)&(o>c1)&(c>dbmax)&(rb>=1.2*db)
    fvg_core=(l:=df["low"].values.astype(float))>h2
    for m in mults:
        ob[m]+=np.nansum(bob_core&(rb>m*safe))
        fvg[m]+=np.nansum(fvg_core&(db>m*sh(safe,1)))
print(f"{'mult':>5} {'bull_OB':>10} {'bull_FVG':>10}")
for m in mults: print(f"{m:5.1f} {ob[m]:10.0f} {fvg[m]:10.0f}")
