import requests, time, json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Referer":    "https://www.nseindia.com/",
}

# ── 1. NSE Equity Master CSV ───────────────────────────────────────────────────
print("=== 1. NSE Equity Master CSV (no market cap, but all symbols) ===")
try:
    r = requests.get("https://archives.nseindia.com/content/equities/EQUITY_L.csv", timeout=10)
    lines = r.text.strip().split("\n")
    print(f"Status={r.status_code}  Rows={len(lines)}")
    print("Header:", lines[0])
    print("Sample:", lines[1])
except Exception as e:
    print(f"ERROR: {e}")

time.sleep(1)

# ── 2. NSE Total Market Index ─────────────────────────────────────────────────
print("\n=== 2. NSE Total Market Index (/api/equity-stockIndices) ===")
try:
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com/", timeout=10)
    time.sleep(1)
    r = session.get(
        "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY+TOTAL+MARKET",
        timeout=15
    )
    print(f"Status={r.status_code}")
    if r.status_code == 200:
        data = r.json().get("data", [])
        print(f"Stocks returned: {len(data)}")
        sample = data[1] if len(data) > 1 else {}
        print("Fields:", list(sample.keys()))
        for key in ["symbol", "marketCap", "ffmc", "totalTradedVolume"]:
            print(f"  {key} = {sample.get(key)}")
except Exception as e:
    print(f"ERROR: {e}")

time.sleep(1)

# ── 3. Screener.in ────────────────────────────────────────────────────────────
print("\n=== 3. Screener.in API ===")
try:
    h2 = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.screener.in/",
    }
    r = requests.get("https://www.screener.in/api/company/RELIANCE/", headers=h2, timeout=10)
    print(f"Status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        print("Keys:", list(d.keys())[:10])
        print("market_cap:", d.get("market_cap"))
        print("name:", d.get("name"))
except Exception as e:
    print(f"ERROR: {e}")

time.sleep(1)

# ── 4. BSE India ──────────────────────────────────────────────────────────────
print("\n=== 4. BSE India quote API ===")
try:
    h3 = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.bseindia.com/",
    }
    # BSE uses scrip code; RELIANCE = 500325
    r = requests.get(
        "https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w?Debtflag=&scripcode=500325&seriesid=",
        headers=h3, timeout=10
    )
    print(f"Status={r.status_code}")
    if r.status_code == 200:
        d = r.json()
        print("Keys:", list(d.keys())[:10])
        for k in ["MarketCapFull", "Mktcap", "mktCap", "CMP"]:
            if k in d:
                print(f"  {k} = {d[k]}")
except Exception as e:
    print(f"ERROR: {e}")
