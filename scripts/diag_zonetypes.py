"""
Why does ict_bull_zone_priority only ever show {0, 3}? Count, across all local
tickers, how many bars each bull zone type TRIGGERS — raw vs after the
displacement gate vs after priority dedup — to find where OB(2)/FVG(1) die.
"""
import glob, os
import numpy as np, pandas as pd
from pipeline.features.ict_features import _wilder_atr

DATA_DIR = "C:/Victor/Learning_charts/stock_data/nse_local"
tot = {k:0 for k in ["raw_bob","raw_bbb","raw_bfvg","disp_bob","disp_bbb","disp_bfvg",
                      "kept_bob","kept_bbb","kept_bfvg"]}
bars = 0
for path in glob.glob(os.path.join(DATA_DIR,"*-1d.csv")):
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    if len(df) < 60: continue
    o,h,l,c = [df[x].values.astype(float) for x in ["open","high","low","close"]]
    atr = _wilder_atr(h,l,c,14)
    n=len(c); bars+=n
    safe_atr = np.where((atr>0)&~np.isnan(atr), atr, np.nan)
    def sh(a,k):
        r=np.full_like(a,np.nan,dtype=float); r[k:]=a[:-k]; return r
    o1,c1,h1,l1=sh(o,1),sh(c,1),sh(h,1),sh(l,1)
    h2,l2=sh(h,2),sh(l,2); h3,l3=sh(h,3),sh(l,3)
    d_body_max=np.maximum(o1,c1); d_body_min=np.minimum(o1,c1)
    r_blen=np.abs(c-o); d_blen=np.abs(c1-o1); mult=1.2
    disp = r_blen > 3.0*safe_atr
    disp1 = d_blen > 3.0*sh(safe_atr,1)
    # raw (no displacement)
    bob_core = (c1<o1)&(c>o)&(o>c1)&(c>d_body_max)&(r_blen>=mult*d_blen)
    is_sl=(l1<l2)&(l1<l3)&(l1<l)
    # bb needs ssl sweep — approximate with swing-low + close>body for raw count
    bbb_core = (c1<o1)&is_sl&(c>d_body_max)
    bfvg_core = (l>h2)
    tot["raw_bob"]+=np.nansum(bob_core); tot["raw_bbb"]+=np.nansum(bbb_core); tot["raw_bfvg"]+=np.nansum(bfvg_core)
    # after displacement gate
    bob_d=bob_core&disp; bfvg_d=bfvg_core&disp1; bbb_d=bbb_core  # bb not disp-gated
    tot["disp_bob"]+=np.nansum(bob_d); tot["disp_bbb"]+=np.nansum(bbb_d); tot["disp_bfvg"]+=np.nansum(bfvg_d)
    # priority dedup: bb(3)>ob(2)>fvg(1)
    prio=np.maximum(np.maximum(bbb_d.astype(int)*3, bob_d.astype(int)*2), bfvg_d.astype(int)*1)
    tot["kept_bob"]+=np.nansum(bob_d&(prio==2))
    tot["kept_bfvg"]+=np.nansum(bfvg_d&(prio==1))
    tot["kept_bbb"]+=np.nansum(bbb_d&(prio==3))

print(f"total bars: {bars}")
for stage in ["raw","disp","kept"]:
    print(f"{stage:5s}  BOB={tot[stage+'_bob']:8.0f}  BBB={tot[stage+'_bbb']:8.0f}  BFVG={tot[stage+'_bfvg']:8.0f}")
