"""Trace NVDA monthly SDZ: show base candle vs breakout candle dates and prices."""
import numpy as np
import pandas as pd
from pipeline.utils.zone_analyzer import (
    _identify_rally_candles, _identify_drop_candles,
    _identify_base_candles, _identify_zones,
    RALLY, DROP, BASE,
)

CSV = r"C:\Victor\Learning_charts\stock_data\us_stocks\NVDA-1d.csv"

raw = pd.read_csv(CSV)
raw["Date"] = pd.to_datetime(raw["Date"])
raw = raw.sort_values("Date").set_index("Date")[["Open","High","Low","Close","Volume"]]

# Monthly resample
mo = raw.resample("ME").agg({
    "Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"
}).dropna(subset=["Close"])

df = mo.reset_index().rename(columns={"index":"Date"})
df["Date"] = pd.to_datetime(df["Date"])
df = _identify_rally_candles(df)
df = _identify_drop_candles(df)
df = _identify_base_candles(df)
zone_df = _identify_zones(df)
zone_df.loc[zone_df["Zone"].isin(["RBR","DBD","DBR","RBD"]), "Zone"] = "Valid"
zone_df = zone_df.sort_values("Date").reset_index(drop=True)

close_arr   = zone_df["Close"].to_numpy(dtype=float)
open_arr    = zone_df["Open"].to_numpy(dtype=float)
high_arr    = zone_df["High"].to_numpy(dtype=float)
low_arr     = zone_df["Low"].to_numpy(dtype=float)
subtype_arr = zone_df["SubType"].to_numpy()
date_arr    = zone_df["Date"].to_numpy()
n = len(zone_df)

# Find SZ candidates (valid SZ zones) — trace breakout candle
sz_mask = (zone_df["ZoneType"] == "SZ") & (zone_df["Zone"] == "Valid")
sz_idx  = np.flatnonzero(sz_mask.to_numpy())

print(f"Monthly bars: {n}  ({pd.Timestamp(date_arr[0]).date()} → {pd.Timestamp(date_arr[-1]).date()})\n")
print(f"{'Base date':<14} {'Base Lo':>8} {'Base Hi':>8}  {'Distal':>8}  "
      f"{'Brk date':<14} {'Brk Lo':>8} {'Brk Hi':>8}  {'Outcome'}")
print("-"*95)

for i in sz_idx:
    distal   = float(zone_df.at[i, "Distal"])
    proximal = float(zone_df.at[i, "Proximal"])
    if i + 2 >= n:
        continue

    future_closes = close_arr[i+2:]
    brk_rel = np.flatnonzero(future_closes > distal)
    if brk_rel.size == 0:
        print(f"{str(pd.Timestamp(date_arr[i]).date()):<14} {low_arr[i]:>8.2f} {high_arr[i]:>8.2f}  "
              f"{distal:>8.2f}  {'no breakout':<14}")
        continue

    brk_idx  = int(i + 2 + brk_rel[0])
    brk_type = subtype_arr[brk_idx]
    brk_date = pd.Timestamp(date_arr[brk_idx]).date()

    # clean-path check
    lo = min(distal, proximal)
    hi = max(distal, proximal)
    pre_h   = high_arr  [i+2:brk_idx]
    pre_l   = low_arr   [i+2:brk_idx]
    pre_sub = subtype_arr[i+2:brk_idx]
    overlaps   = (pre_h >= lo) & (pre_l <= hi)
    drop_bases = np.isin(pre_sub, [DROP, BASE])
    clean = not (overlaps & drop_bases).any()

    if brk_type in (DROP, BASE) or not clean:
        outcome = f"Invalid (brk={brk_type}, clean={clean})"
    else:
        # breach check using base-candle boundaries
        sdz_low = low_arr[i]
        future_after = np.concatenate([close_arr[brk_idx+1:], open_arr[brk_idx+1:]])
        breached = (future_after < sdz_low).any() if len(future_after) else False
        outcome = "SDZ" if not breached else "Breached→Invalid"

    print(f"{str(pd.Timestamp(date_arr[i]).date()):<14} {low_arr[i]:>8.2f} {high_arr[i]:>8.2f}  "
          f"{distal:>8.2f}  {str(brk_date):<14} {low_arr[brk_idx]:>8.2f} {high_arr[brk_idx]:>8.2f}  {outcome}")
