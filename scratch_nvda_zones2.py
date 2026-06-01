"""Smoke-test SDZ persistence: show NVDA zone_type cols around Dec 2023."""
import pandas as pd
from pipeline.features.zone_features import compute_zone_features

CSV = r"C:\Victor\Learning_charts\stock_data\us_stocks\NVDA-1d.csv"

raw = pd.read_csv(CSV)
raw["Date"] = pd.to_datetime(raw["Date"])
raw = raw.sort_values("Date").set_index("Date")
raw.columns = [c.lower() for c in raw.columns]
# rename to expected lowercase cols
raw = raw.rename(columns={"close":"close","high":"high","low":"low","open":"open","volume":"volume"})

# compute without cutoff (full history, no fold simulation)
out = compute_zone_features(raw)

# Show last 20 bars
cols = ["close","zone_type_1d","zone_type_1wk","zone_type_1mo","zone_type_3mo",
        "zone_active_1d","zone_dist_atr_1d","zone_strength_1d"]
print(out[cols].tail(20).to_string())

# Summary counts
print("\nZone type counts (last 60 bars):")
tail60 = out.tail(60)
for col in ["zone_type_1d","zone_type_1wk","zone_type_1mo","zone_type_3mo"]:
    vc = tail60[col].value_counts()
    print(f"  {col}: {vc.to_dict()}")
