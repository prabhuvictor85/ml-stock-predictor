"""Post-sync: do OB/FVG now form like the Pine indicator, and does the
forward-filled active flag saturate (the reason the gate was added)?"""
import glob, os
import numpy as np, pandas as pd
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr
DATA_DIR="C:/Victor/Learning_charts/stock_data/nse_local"
eng=ICTFeatureEngine()
tot={"bob":0.0,"bbb":0.0,"bfvg":0.0}      # active-bar sums (ffill'd)
trig={"bob":0,"bbb":0,"bfvg":0}           # trigger counts proxy via active>prev
bars=0
last_prio=[]; live_frac=[]
for path in glob.glob(os.path.join(DATA_DIR,"*-1d.csv")):
    df=pd.read_csv(path,parse_dates=["Date"]).set_index("Date").sort_index()
    if len(df)<60: continue
    g=df[["open","high","low","close"]].astype(float).copy()
    g["atr_14"]=_wilder_atr(g["high"].values,g["low"].values,g["close"].values,14)
    out=eng.compute(g)                    # disp_mult default = 0.0 (gate off)
    bars+=len(out)
    tot["bob"]+=out["ict_bob_active"].sum()
    tot["bbb"]+=out["ict_bullbb_active"].sum()
    tot["bfvg"]+=out["ict_bullfvg_active"].sum()
    bp=out["ict_bull_zone_priority"]
    last_prio.append(float(bp.iloc[-1]))
    live_frac.append(float((bp>0).mean()))
print(f"tickers scanned, total bars: {bars}")
print("\n-- forward-filled ACTIVE-flag fraction (saturation check) --")
for k in ["bob","bbb","bfvg"]:
    print(f"  ict_{k:4s}_active live fraction: {tot[k]/bars*100:5.1f}% of bars")
lp=np.array(last_prio)
print("\n-- last-bar ict_bull_zone_priority cross-section --")
print(f"  value counts: {dict(zip(*np.unique(lp,return_counts=True)))}")
print(f"  nunique={len(np.unique(lp))}  %zero={(lp==0).mean()*100:.1f}  std={lp.std():.3f}")
print(f"  mean per-ticker live fraction: {np.mean(live_frac)*100:.1f}%")
