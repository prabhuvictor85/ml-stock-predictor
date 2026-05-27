"""
download_nse_marketcap.py
Downloads market cap for all NSE TradingView tickers via NSE India API
and classifies them using the official SEBI criteria:

  Large-cap : market cap >= 1,05,000 crore
  Mid-cap   : market cap  34,700 to 1,05,000 crore
  Small-cap : market cap  5,000 to 34,700 crore
  Micro-cap : market cap <  5,000 crore  (or no market cap data)

Strategy:
  Primary  -- NSE India API  /api/quote-equity?symbol={Symbol}
               Market cap = issuedSize x lastPrice / 1e7
  Fallback -- yfinance {Symbol}.NS then {Symbol}.BO
               (used only if NSE API returns no data)
  Session is refreshed every REFRESH_EVERY tickers to keep cookies alive.

Output:
  C:/Victor/Learning_charts/stock_lists/nse_cap_tiers.csv
  Columns: Symbol, TV_ticker, market_cap_crore, cap_tier

Usage (no prompts):
  python download_nse_marketcap.py
"""

import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS
TV_CSV   = PATHS.stock_lists.nse_tv
OUT_CSV  = PATHS.stock_lists.nse_cap_tiers
CACHE    = PATHS.stock_lists.lists_dir / "nse_mcap_cache.csv"

# SEBI thresholds (in crore INR)
LARGE_THRESHOLD = 105000   # >= 1,05,000 crore
MID_THRESHOLD   =  34700   # >= 34,700 crore
SMALL_THRESHOLD =   5000   # >= 5,000 crore  (below = micro)

DELAY_PER_TICKER = 0.4     # seconds between NSE API calls
SAVE_EVERY       = 100     # save cache every N tickers
REFRESH_EVERY    = 200     # re-init NSE session every N tickers (keep cookies alive)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


# ── Classification ─────────────────────────────────────────────────────────────
def classify(crore: float) -> str:
    if crore >= LARGE_THRESHOLD:  return "large"
    if crore >= MID_THRESHOLD:    return "mid"
    if crore >= SMALL_THRESHOLD:  return "small"
    return "micro"


# ── NSE Session ────────────────────────────────────────────────────────────────
def make_nse_session() -> requests.Session:
    """Create a requests session with NSE India cookies."""
    session = requests.Session()
    session.headers.update(NSE_HEADERS)
    try:
        r = session.get("https://www.nseindia.com/", timeout=15)
        cookies = list(session.cookies.keys())
        print(f"  NSE session ready  (status={r.status_code}, cookies={cookies})",
              flush=True)
    except Exception as e:
        print(f"  NSE session warning: {e}", flush=True)
    time.sleep(1)
    return session


# ── Primary fetch: NSE India API ───────────────────────────────────────────────
def fetch_mcap_nse(symbol: str, session: requests.Session) -> float | None:
    """
    Fetch market cap from NSE India quote-equity API.
    Computes: issuedSize (shares) x lastPrice / 1e7  = crore INR
    """
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        r   = session.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data         = r.json()
        issued_size  = data.get("securityInfo", {}).get("issuedSize")
        last_price   = data.get("priceInfo",    {}).get("lastPrice")
        if issued_size and last_price and issued_size > 0 and last_price > 0:
            return round(issued_size * last_price / 1e7, 2)
    except Exception:
        pass
    return None


# ── Fallback fetch: yfinance ───────────────────────────────────────────────────
def fetch_mcap_yf(symbol: str) -> float | None:
    """
    Fallback: try yfinance {symbol}.NS then {symbol}.BO.
    Only called when NSE API returns nothing (e.g. SME / unlisted on NSE main board).
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    for suffix in (".NS", ".BO"):
        try:
            info = yf.Ticker(f"{symbol}{suffix}").info
            mcap = info.get("marketCap")
            if mcap and mcap > 0:
                return round(mcap / 1e7, 2)
        except Exception:
            pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load constituent list
    tv_df     = pd.read_csv(TV_CSV)
    symbols   = tv_df["Symbol"].str.strip().tolist()
    tv_ticker = dict(zip(tv_df["Symbol"].str.strip(),
                         tv_df["TV_ticker"].str.strip()))
    print(f"Loaded {len(symbols)} symbols from {TV_CSV.name}")

    # Load cache (resume if interrupted)
    done: dict[str, float | None] = {}
    if CACHE.exists():
        cache_df = pd.read_csv(CACHE)
        for _, row in cache_df.iterrows():
            val = None if pd.isna(row["market_cap_crore"]) else float(row["market_cap_crore"])
            done[row["Symbol"]] = val
        classified = sum(1 for v in done.values() if v is not None)
        no_data    = sum(1 for v in done.values() if v is None)
        print(f"Resuming from cache: {len(done)} entries "
              f"({classified} classified, {no_data} no-data -> will retry)")

    # Retry any previously None entries plus all uncached
    remaining = [s for s in symbols if done.get(s) is None or s not in done]
    print(f"Remaining to fetch : {len(remaining)}")
    print(f"Estimated time     : ~{len(remaining) * DELAY_PER_TICKER / 60:.1f} minutes")
    print()

    # Init NSE session
    print("[NSE session]")
    session = make_nse_session()
    print()

    for i, sym in enumerate(remaining, 1):

        # Refresh session periodically
        if i > 1 and (i - 1) % REFRESH_EVERY == 0:
            print(f"  --- refreshing NSE session at ticker {i} ---", flush=True)
            session = make_nse_session()

        # Primary: NSE India API
        mcap = fetch_mcap_nse(sym, session)

        # Fallback: yfinance (for SME / off-board tickers)
        if mcap is None:
            mcap = fetch_mcap_yf(sym)

        done[sym] = mcap

        if mcap:
            tier  = classify(mcap)
            label = {"large": "Large", "mid": "Mid", "small": "Small", "micro": "Micro"}[tier]
            print(f"  [{i:>4}/{len(remaining)}] {sym:<20} {mcap:>12,.0f} cr  {label}",
                  flush=True)
        else:
            print(f"  [{i:>4}/{len(remaining)}] {sym:<20} no data", flush=True)

        # Save cache periodically
        if i % SAVE_EVERY == 0:
            _save_cache(done, symbols)
            print(f"  --- cache saved ({len(done)} tickers) ---", flush=True)

        time.sleep(DELAY_PER_TICKER)

    # Final save
    _save_cache(done, symbols)
    _save_final(done, symbols, tv_ticker)


def _save_cache(done: dict, symbols: list):
    rows = [{"Symbol": s, "market_cap_crore": done.get(s)} for s in symbols if s in done]
    pd.DataFrame(rows).to_csv(CACHE, index=False)


def _save_final(done: dict, symbols: list, tv_ticker: dict):
    rows      = []
    micro_syms = []
    for sym in symbols:
        mcap = done.get(sym)
        if mcap and mcap > 0:
            rows.append({
                "Symbol":           sym,
                "TV_ticker":        tv_ticker.get(sym, sym),
                "market_cap_crore": mcap,
                "cap_tier":         classify(mcap),
            })
        else:
            micro_syms.append(sym)
            rows.append({
                "Symbol":           sym,
                "TV_ticker":        tv_ticker.get(sym, sym),
                "market_cap_crore": None,
                "cap_tier":         "micro",
            })

    # Sort: large/mid/small by market cap desc, then micro (with mcap) by mcap desc, then no-data micro at bottom
    df_all = pd.DataFrame(rows)
    df_named   = df_all[df_all["cap_tier"].isin(["large","mid","small"])].sort_values("market_cap_crore", ascending=False)
    df_micro_v = df_all[(df_all["cap_tier"] == "micro") & df_all["market_cap_crore"].notna()].sort_values("market_cap_crore", ascending=False)
    df_micro_n = df_all[(df_all["cap_tier"] == "micro") & df_all["market_cap_crore"].isna()]
    df = pd.concat([df_named, df_micro_v, df_micro_n], ignore_index=True)
    df.to_csv(OUT_CSV, index=False)

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
    print("Top 15 Large Cap:")
    print(df[df["cap_tier"] == "large"][["Symbol", "market_cap_crore"]].head(15).to_string(index=False))
    print()
    print("Top 10 Mid Cap:")
    print(df[df["cap_tier"] == "mid"][["Symbol", "market_cap_crore"]].head(10).to_string(index=False))
    print()
    print(f"First 10 Micro Cap tickers: {micro_syms[:10]}")

    if CACHE.exists():
        CACHE.unlink()
        print("Cache file removed.")


if __name__ == "__main__":
    main()
