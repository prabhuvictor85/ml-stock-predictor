import requests, time, json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nseindia.com/",
}

session = requests.Session()
session.headers.update(HEADERS)
session.get("https://www.nseindia.com/", timeout=10)
time.sleep(1)

# ── NSE Total Market: check meta field and ffmc for known stocks ───────────────
print("=== NSE Total Market Index — ffmc as market cap proxy ===")
r = session.get(
    "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY+TOTAL+MARKET",
    timeout=15
)
data = r.json().get("data", [])

# Find RELIANCE, JKCEMENT in the list
targets = {d["symbol"]: d for d in data if d["symbol"] in ("RELIANCE","JKCEMENT","INFY","HDFCBANK")}
for sym, d in targets.items():
    ffmc   = d.get("ffmc", 0)
    price  = d.get("lastPrice", 0)
    meta   = d.get("meta", {})
    ffmc_cr = round(ffmc / 1e7, 0) if ffmc else None
    print(f"{sym:<12} ffmc={ffmc_cr:>10,.0f} cr  price={price}  meta_keys={list(meta.keys()) if meta else 'empty'}")

print(f"\nTotal stocks in NIFTY TOTAL MARKET: {len(data)}")
print()

# ── Check if ffmc = total market cap (compare with our known values) ──────────
# RELIANCE actual mcap ~ 18,06,594 cr (from quote-equity)
# JKCEMENT actual mcap ~ 41,586 cr
print("Comparison (ffmc vs known total mcap):")
known = {"RELIANCE": 1806594, "JKCEMENT": 41586, "INFY": 464612, "HDFCBANK": 1183609}
for sym, actual in known.items():
    if sym in targets:
        ffmc_cr = round(targets[sym].get("ffmc", 0) / 1e7, 0)
        ratio   = round(ffmc_cr / actual, 2) if actual else 0
        print(f"  {sym:<12} ffmc={ffmc_cr:>10,.0f} cr  actual={actual:>10,.0f} cr  ratio={ratio}")

time.sleep(1)

# ── BSE India: dig into CompResp ──────────────────────────────────────────────
print("\n=== BSE India scrip header (RELIANCE=500325) ===")
h3 = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.bseindia.com/",
}
r2 = requests.get(
    "https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w?Debtflag=&scripcode=500325&seriesid=",
    headers=h3, timeout=10
)
d2 = r2.json()
comp = d2.get("CompResp", {})
header = d2.get("Header", {})
print("CompResp keys:", list(comp.keys()))
print("Header keys:  ", list(header.keys()))
for k, v in list(comp.items())[:15]:
    print(f"  CompResp.{k} = {v}")
