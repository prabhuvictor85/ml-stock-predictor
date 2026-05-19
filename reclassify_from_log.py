"""
reclassify_from_log.py
Parses nse_mcap_download.log, extracts symbol -> market_cap_crore,
applies the correct 4-tier SEBI classification and rebuilds nse_cap_tiers.csv.
"""

import re
import pandas as pd
from pathlib import Path

LOG_FILE  = Path(r"C:\Victor\Project\ml-stock-predictor\artefacts\us_local\logs\nse_mcap_download.log")
TV_CSV    = Path(r"C:\Victor\Learning_charts\stock_lists\constituents_nse_tradingv.csv")
OUT_CSV   = Path(r"C:\Victor\Learning_charts\stock_lists\nse_cap_tiers.csv")

# Thresholds
LARGE = 105000
MID   =  34700
SMALL =   5000

def classify(cr: float) -> str:
    if cr >= LARGE: return "large"
    if cr >= MID:   return "mid"
    if cr >= SMALL: return "small"
    return "micro"

# ── Parse log ─────────────────────────────────────────────────────────────────
# Line formats:
#   [   1/2517] 20MICRONS                     603 cr  Small
#   [  20/2517] ATCENERGY            no data
pattern_data    = re.compile(r"\[\s*\d+/\d+\]\s+(\S+)\s+([\d,]+)\s+cr")
pattern_nodata  = re.compile(r"\[\s*\d+/\d+\]\s+(\S+)\s+no data")

log_mcap: dict[str, float | None] = {}

with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern_data.search(line)
        if m:
            sym  = m.group(1).strip()
            cr   = float(m.group(2).replace(",", ""))
            log_mcap[sym] = cr
            continue
        m2 = pattern_nodata.search(line)
        if m2:
            sym = m2.group(1).strip()
            if sym not in log_mcap:        # don't overwrite a valid value
                log_mcap[sym] = None

print(f"Parsed {len(log_mcap)} symbols from log")
print(f"  With market cap : {sum(1 for v in log_mcap.values() if v is not None)}")
print(f"  No data         : {sum(1 for v in log_mcap.values() if v is None)}")

# ── Load TV constituent list for TV_ticker mapping ─────────────────────────────
tv_df     = pd.read_csv(TV_CSV)
symbols   = tv_df["Symbol"].str.strip().tolist()
tv_ticker = dict(zip(tv_df["Symbol"].str.strip(), tv_df["TV_ticker"].str.strip()))

# ── Build rows ─────────────────────────────────────────────────────────────────
rows = []
missing_from_log = []

for sym in symbols:
    mcap = log_mcap.get(sym)
    if mcap and mcap > 0:
        rows.append({
            "Symbol":           sym,
            "TV_ticker":        tv_ticker.get(sym, sym),
            "market_cap_crore": mcap,
            "cap_tier":         classify(mcap),
        })
    else:
        missing_from_log.append(sym)
        rows.append({
            "Symbol":           sym,
            "TV_ticker":        tv_ticker.get(sym, sym),
            "market_cap_crore": None,
            "cap_tier":         "micro",
        })

# ── Sort: large/mid/small by mcap desc, then micro-with-value, then no-data ────
df_all    = pd.DataFrame(rows)
df_named  = df_all[df_all["cap_tier"].isin(["large","mid","small"])].sort_values("market_cap_crore", ascending=False)
df_micv   = df_all[(df_all["cap_tier"] == "micro") & df_all["market_cap_crore"].notna()].sort_values("market_cap_crore", ascending=False)
df_micn   = df_all[(df_all["cap_tier"] == "micro") & df_all["market_cap_crore"].isna()]
df        = pd.concat([df_named, df_micv, df_micn], ignore_index=True)

df.to_csv(OUT_CSV, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
large = (df["cap_tier"] == "large").sum()
mid   = (df["cap_tier"] == "mid").sum()
small = (df["cap_tier"] == "small").sum()
micro = (df["cap_tier"] == "micro").sum()

print()
print("=" * 60)
print(f"  Saved {len(df)} tickers -> {OUT_CSV.name}")
print()
print(f"  Classification (as of today):")
print(f"  Large Cap  (>= 1,05,000 cr)  : {large:>4}")
print(f"  Mid Cap    (34,700-1,05,000)  : {mid:>4}")
print(f"  Small Cap  (5,000-34,700 cr)  : {small:>4}")
print(f"  Micro Cap  (<  5,000 cr)      : {micro:>4}")
print("=" * 60)
print()
print("Top 10 Large Cap:")
print(df[df["cap_tier"] == "large"][["Symbol","market_cap_crore"]].head(10).to_string(index=False))
print()
print("Top 10 Mid Cap:")
print(df[df["cap_tier"] == "mid"][["Symbol","market_cap_crore"]].head(10).to_string(index=False))
print()
print("Top 10 Small Cap:")
print(df[df["cap_tier"] == "small"][["Symbol","market_cap_crore"]].head(10).to_string(index=False))
print()
print("Top 10 Micro Cap (by market cap):")
print(df_micv[["Symbol","market_cap_crore"]].head(10).to_string(index=False))
if missing_from_log:
    print()
    print(f"Symbols not found in log ({len(missing_from_log)}): {missing_from_log[:10]}")
