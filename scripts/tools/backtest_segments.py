"""
backtest_segments.py — Segmented Backtesting Framework
Works for: sp500 | nse | nse_tradingv

Runs 3 diagnostic metrics per model × side × cap-tier:
  1. Hit Rate       — Did top-N picks beat the 70th-pct of their tier universe?
  2. IC Score       — Does model_score correlate with actual returns? (Spearman)
  3. Missed Opp     — High-return stocks buried too low? (Near / Moderate / Deep)

Decision Engine per segment:
  All 3 healthy       → ✅  TRUST MODEL
  Only missed opp bad → ⚠️  INVESTIGATE FEATURE BIAS
  2 out of 3 bad      → ⚠️  WATCH CLOSELY
  All 3 bad           → ❌  RETRAIN IMMEDIATELY

Forward prices:
  - NSE / NSE TradingView : read from local CSV files (data is current)
  - SP500                 : try local CSV first; download via yfinance if out of range
  - Cached in output/evaluation/fwd_prices_{market}_{entry}_{exit}.csv

Usage:
    python backtest_segments.py --market sp500       --date 2024-04-30
    python backtest_segments.py --market nse         --date 2025-01-31
    python backtest_segments.py --market nse_tradingv --date 2025-01-31
    python backtest_segments.py --market sp500 --date 2024-04-30 --forward_months 6 --top_n 10
"""

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from dateutil.relativedelta import relativedelta
except ImportError:
    print("Installing python-dateutil...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "python-dateutil", "-q"])
    from dateutil.relativedelta import relativedelta

# ── Metric thresholds ──────────────────────────────────────────────────────────
HIT_RATE_HEALTHY = 60.0
HIT_RATE_WATCH   = 40.0
IC_HEALTHY       = 0.20
IC_WATCH         = 0.10

# ══════════════════════════════════════════════════════════════════════════════
# MARKET CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.config.paths import PATHS

MARKET_CONFIGS = {
    "sp500": {
        "label":          "SP500 (US Stocks)",
        "output_dir":     Path("output/us_local"),
        "eval_dir":       Path("output/evaluation"),
        "stock_data_dir": PATHS.stock_data.us,
        "stock_list_csv": PATHS.stock_lists.us_combined,
        "cap_tier_source":"indices_column",   # Indices col in stock_list_csv
        "cap_tier_csv":   None,
        "ticker_col":     "Symbol",           # col used as ticker key in scores_detail
        "yf_suffix":      "",                 # plain ticker for yfinance
        "price_format":   "date_close",       # local CSV: Date/Close columns
        "file_pattern":   "{ticker}-1d.csv",
    },
    "nse": {
        "label":          "NSE (India)",
        "output_dir":     Path("output/nse_local"),
        "eval_dir":       Path("output/evaluation"),
        "stock_data_dir": PATHS.stock_data.nse_local,
        "stock_list_csv": PATHS.stock_lists.nse_local,
        "cap_tier_source":"nse_cap_tiers_csv",
        "cap_tier_csv":   PATHS.stock_lists.nse_cap_tiers,
        # NSE local scores_detail uses Symbol (with .NS); strip .NS for cap tier lookup
        "ticker_col":     "Symbol",           # col with .NS suffix (RELIANCE.NS)
        "ticker_plain_col": "Symbol1",        # plain symbol (RELIANCE) for file + tier lookup
        "yf_suffix":      ".NS",
        "price_format":   "date_close",
        "file_pattern":   "{ticker}-1d.csv",  # uses Symbol1 (plain)
    },
    "nse_tradingv": {
        "label":          "NSE TradingView",
        "output_dir":     Path("output/nse_tradingv"),
        "eval_dir":       Path("output/evaluation"),
        "stock_data_dir": PATHS.stock_data.nse_tv,
        "stock_list_csv": PATHS.stock_lists.nse_tv,
        "cap_tier_source":"nse_cap_tiers_csv",
        "cap_tier_csv":   PATHS.stock_lists.nse_cap_tiers,
        "ticker_col":     "TV_ticker",        # TV_ticker used in scores_detail
        "symbol_col":     "Symbol",           # plain NSE symbol for cap tier lookup
        "yf_suffix":      ".NS",
        "price_format":   "tv_ts_close",      # ts / c columns (Unix epoch)
        "file_pattern":   "NSE_{ticker}_1D_TV_div_adj.csv",
    },
}

TIER_LABELS = {"large": "Large Cap", "mid": "Mid Cap", "small": "Small Cap (5k-35k cr)", "micro": "Micro Cap (< 5k cr)"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. CAP TIER MAP
# ══════════════════════════════════════════════════════════════════════════════

def build_cap_tier_map(cfg: dict) -> dict[str, str]:
    """
    Returns {ticker_as_used_in_scores_detail: "large"|"mid"|"small"}
    Handles all three markets.
    """
    source = cfg["cap_tier_source"]
    tier_map: dict[str, str] = {}

    if source == "indices_column":
        # SP500: Indices column in stock_list_csv
        df = pd.read_csv(cfg["stock_list_csv"])
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).strip()
            idx = str(row.get("Indices", "")).strip()
            if "SPX" in idx or "NDX" in idx:
                tier_map[sym] = "large"
            elif idx == "MID":
                tier_map[sym] = "mid"
            elif idx == "SML":
                tier_map[sym] = "small"

    elif source == "nse_cap_tiers_csv":
        # NSE / NSE TV: nse_cap_tiers.csv (plain Symbol → cap_tier)
        tier_csv = cfg.get("cap_tier_csv")
        if not tier_csv or not Path(tier_csv).exists():
            print("  WARNING: nse_cap_tiers.csv not found. "
                  "Run: python download_nse_index_constituents.py")
            return tier_map

        tier_df = pd.read_csv(tier_csv)
        # Build plain_symbol → tier dict
        plain_to_tier = {
            str(r["Symbol"]).strip(): str(r["cap_tier"]).strip()
            for _, r in tier_df.iterrows()
        }

        if cfg.get("market") == "nse":
            # scores_detail uses ticker WITH .NS — map via Symbol1 (plain) → tier
            list_df = pd.read_csv(cfg["stock_list_csv"])
            for _, row in list_df.iterrows():
                plain  = str(row.get("Symbol1", "")).strip()
                ns_sym = str(row.get("Symbol",  "")).strip()  # RELIANCE.NS
                tier   = plain_to_tier.get(plain)
                if tier:
                    tier_map[ns_sym] = tier   # key = RELIANCE.NS
        else:
            # nse_tradingv: scores_detail uses TV_ticker (plain)
            # TV_ticker ≈ plain Symbol (with rare exceptions)
            list_df = pd.read_csv(cfg["stock_list_csv"])
            tv_to_sym = {
                str(r["TV_ticker"]).strip(): str(r["Symbol"]).strip()
                for _, r in list_df.iterrows()
            }
            for tv_ticker, plain_sym in tv_to_sym.items():
                tier = plain_to_tier.get(plain_sym)
                if tier:
                    tier_map[tv_ticker] = tier

    return tier_map


# ══════════════════════════════════════════════════════════════════════════════
# 2. FORWARD PRICES
# ══════════════════════════════════════════════════════════════════════════════

def _exit_date(entry: date, forward_months: int) -> date:
    d = entry + relativedelta(months=forward_months)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _read_local_price(filepath: Path, target_date: date,
                      price_format: str) -> Optional[float]:
    """Read the closing price on or before target_date from a local stock file."""
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if price_format == "tv_ts_close":
            df["date"] = pd.to_datetime(df["ts"], unit="s").dt.date
            df = df.rename(columns={"c": "Close"})
        else:
            df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df[df["date"] <= target_date].sort_values("date")
        if df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def _yf_batch_close(tickers: list[str], target_date: date,
                    window_days: int = 5) -> pd.Series:
    try:
        import yfinance as yf
    except ImportError:
        print("  ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    start = target_date - timedelta(days=window_days)
    end   = target_date + timedelta(days=2)
    CHUNK = 200
    prices: dict[str, float] = {}
    total_chunks = (len(tickers) - 1) // CHUNK + 1

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i: i + CHUNK]
        chunk_num = i // CHUNK + 1
        print(f"    yfinance chunk {chunk_num}/{total_chunks} "
              f"({len(chunk)} tickers)...", end=" ", flush=True)
        try:
            raw = yf.download(chunk, start=str(start), end=str(end),
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                print("empty")
                continue
            close = raw["Close"]
            if isinstance(close, pd.Series):
                close = close.to_frame(name=chunk[0])
            close = close[pd.to_datetime(close.index).date <= target_date]
            if close.empty:
                print("no data")
                continue
            last = close.iloc[-1]
            for t in chunk:
                if t in last.index and pd.notna(last[t]):
                    prices[t] = float(last[t])
            print(f"got {sum(1 for t in chunk if t in prices)}/{len(chunk)}")
        except Exception as e:
            print(f"error: {e}")
        time.sleep(0.8)

    return pd.Series(prices)


def load_or_build_fwd_prices(tickers: list[str],
                              entry_date: date,
                              exit_date: date,
                              cfg: dict) -> pd.DataFrame:
    """
    For each ticker, get close price at entry_date and exit_date.
    Strategy:
      1. Try local CSV files (all markets)
      2. Fall back to yfinance for any missing (SP500 / NSE)
    Returns DataFrame: ticker | entry_date | entry_price | exit_date | exit_price
    """
    market    = cfg["market"]
    eval_dir  = Path(cfg["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    cache_path = eval_dir / f"fwd_prices_{market}_{entry_date}_{exit_date}.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"  Loaded cached forward prices: {cache_path.name}  ({len(df)} tickers)")
        return df

    print(f"  Building forward prices for {len(tickers)} tickers "
          f"({entry_date} -> {exit_date})...")

    data_dir     = Path(cfg["stock_data_dir"])
    price_fmt    = cfg["price_format"]
    file_pattern = cfg["file_pattern"]
    yf_suffix    = cfg.get("yf_suffix", "")

    entry_prices: dict[str, float] = {}
    exit_prices:  dict[str, float] = {}
    need_yf: list[str] = []

    print(f"  Reading local CSV files...", end=" ", flush=True)
    for ticker in tickers:
        fpath = data_dir / file_pattern.format(ticker=ticker)
        ep    = _read_local_price(fpath, entry_date, price_fmt)
        xp    = _read_local_price(fpath, exit_date,  price_fmt)
        if ep:
            entry_prices[ticker] = ep
        if xp:
            exit_prices[ticker] = xp
        if not ep or not xp:
            need_yf.append(ticker)

    print(f"local hits: {len(entry_prices)} entry / {len(exit_prices)} exit  "
          f"| need yfinance: {len(need_yf)}")

    # yfinance fallback for missing prices
    if need_yf:
        # Build yfinance ticker list (add suffix e.g. .NS)
        yf_map = {t: t + yf_suffix for t in need_yf}   # ticker → yf_ticker
        yf_tickers = list(yf_map.values())

        need_entry = [t for t in need_yf if t not in entry_prices]
        need_exit  = [t for t in need_yf if t not in exit_prices]

        if need_entry:
            print(f"  Downloading ENTRY prices via yfinance ({len(need_entry)} tickers)...")
            yf_entry_tickers = [yf_map[t] for t in need_entry]
            raw = _yf_batch_close(yf_entry_tickers, entry_date)
            for t in need_entry:
                yt = yf_map[t]
                if yt in raw.index:
                    entry_prices[t] = float(raw[yt])

        if need_exit:
            print(f"  Downloading EXIT prices via yfinance ({len(need_exit)} tickers)...")
            yf_exit_tickers = [yf_map[t] for t in need_exit]
            raw = _yf_batch_close(yf_exit_tickers, exit_date)
            for t in need_exit:
                yt = yf_map[t]
                if yt in raw.index:
                    exit_prices[t] = float(raw[yt])

    # Combine
    rows = []
    for t in tickers:
        ep = entry_prices.get(t)
        xp = exit_prices.get(t)
        if ep and xp and ep > 0:
            rows.append({
                "ticker":      t,
                "entry_date":  str(entry_date),
                "entry_price": round(ep, 4),
                "exit_date":   str(exit_date),
                "exit_price":  round(xp, 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  Saved {len(df)} forward prices -> {cache_path.name}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD MASTER DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def build_master_df(scores: dict, fwd_prices: pd.DataFrame,
                    tier_map: dict[str, str],
                    model: str, side: str) -> pd.DataFrame:
    rank_key   = f"{side}_rank"
    price_lkp  = fwd_prices.set_index("ticker")
    rows = []

    for ticker, data in scores.items():
        if ticker not in price_lkp.index:
            continue
        tier = tier_map.get(ticker)
        if not tier:
            continue

        sd      = data.get(side, {})
        rank    = data.get(rank_key, 9999)
        ms      = sd.get("model_score", 0.0)
        cs      = sd.get("composite_score", 0.0)
        mw      = sd.get("model_weight", 0.7)
        cw      = sd.get("composite_weight", 0.3)
        final   = ms * mw + cs * cw

        p       = price_lkp.loc[ticker]
        ep, xp  = float(p["entry_price"]), float(p["exit_price"])
        ret     = (xp / ep - 1) * 100 if ep > 0 else np.nan

        rows.append({
            "ticker":          ticker,
            "cap_tier":        tier,
            "rank":            rank,
            "model_score":     round(ms, 4),
            "composite_score": round(cs, 4),
            "final_score":     round(final, 4),
            "entry_price":     ep,
            "exit_price":      xp,
            "fwd_return_pct":  round(ret, 2) if pd.notna(ret) else np.nan,
            "model":           model,
            "side":            side,
        })

    return (pd.DataFrame(rows)
            .dropna(subset=["fwd_return_pct"])
            .sort_values("rank")
            .reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRIC 1 — HIT RATE
# ══════════════════════════════════════════════════════════════════════════════

def metric_hit_rate(df_tier: pd.DataFrame, top_n: int) -> dict:
    if len(df_tier) < top_n:
        return {"error": "insufficient data"}

    bench     = float(df_tier["fwd_return_pct"].quantile(0.70))
    top_picks = df_tier.nsmallest(top_n, "rank").copy()

    picks = []
    for _, row in top_picks.iterrows():
        picks.append({
            "ticker":      row["ticker"],
            "rank":        int(row["rank"]),
            "model_score": row["model_score"],
            "fwd_return":  row["fwd_return_pct"],
            "hit":         row["fwd_return_pct"] > bench,
        })

    n_hits   = sum(1 for p in picks if p["hit"])
    hit_rate = n_hits / top_n * 100

    return {
        "hit_rate_pct":   round(hit_rate, 1),
        "n_hits":         n_hits,
        "top_n":          top_n,
        "benchmark_70th": round(bench, 2),
        "universe_size":  len(df_tier),
        "health": ("HEALTHY" if hit_rate >= HIT_RATE_HEALTHY
                   else "WATCH" if hit_rate >= HIT_RATE_WATCH else "TROUBLE"),
        "picks": picks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. METRIC 2 — IC SCORE
# ══════════════════════════════════════════════════════════════════════════════

def metric_ic_score(df_tier: pd.DataFrame, top_n: int) -> dict:
    if len(df_tier) < 10:
        return {"error": "insufficient data"}

    ic, pv = stats.spearmanr(df_tier["model_score"].values,
                              df_tier["fwd_return_pct"].values)
    ic = round(float(ic), 4)

    # Sample concordance
    scores, returns = df_tier["model_score"].values, df_tier["fwd_return_pct"].values
    conc = tot = 0
    for i in range(len(scores)):
        for j in range(i + 1, min(i + 50, len(scores))):
            tot += 1
            if (scores[i] > scores[j]) == (returns[i] > returns[j]):
                conc += 1
    concordance = round(conc / tot * 100, 1) if tot else 0

    top5 = df_tier.nlargest(5, "model_score")[
        ["ticker", "model_score", "fwd_return_pct", "rank"]].to_dict("records")

    return {
        "ic":              ic,
        "pvalue":          round(float(pv), 4),
        "universe_size":   len(df_tier),
        "concordance_pct": concordance,
        "health": ("HEALTHY" if abs(ic) >= IC_HEALTHY
                   else "WATCH" if abs(ic) >= IC_WATCH else "TROUBLE"),
        "top5_picks": top5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. METRIC 3 — MISSED OPPORTUNITY
# ══════════════════════════════════════════════════════════════════════════════

def _miss_category(rank: int, top_n: int) -> str:
    if top_n < rank <= top_n + 10:   return "Near Miss"
    elif rank <= top_n + 40:          return "Moderate"
    else:                             return "Deep Miss"


def _miss_reason(row: pd.Series) -> str:
    ms, cs = row["model_score"], row["composite_score"]
    if ms >= 0.70 and cs <= 0.05:  return "composite drag  (high ML, near-zero composite)"
    if ms >= 0.70 and cs > 0.05:   return "rank cutoff     (strong score, just outside top-N)"
    if ms < 0.40:                   return "model blind spot (low model score — event-driven?)"
    return                                  "moderate score  (model under-weighted this setup)"


def metric_missed_opportunity(df_tier: pd.DataFrame, top_n: int,
                               threshold_pct: float) -> dict:
    if df_tier.empty:
        return {"error": "insufficient data"}

    top_rank_cutoff = df_tier.nsmallest(top_n, "rank")["rank"].max()
    outside = df_tier[df_tier["rank"] > top_rank_cutoff]
    missed  = outside[outside["fwd_return_pct"] >= threshold_pct].copy()
    missed  = missed.sort_values("fwd_return_pct", ascending=False)

    misses = []
    for _, row in missed.iterrows():
        misses.append({
            "ticker":          row["ticker"],
            "rank":            int(row["rank"]),
            "fwd_return":      row["fwd_return_pct"],
            "model_score":     row["model_score"],
            "composite_score": row["composite_score"],
            "category":        _miss_category(int(row["rank"]), top_n),
            "reason":          _miss_reason(row),
        })

    n_near = sum(1 for m in misses if m["category"] == "Near Miss")
    n_mod  = sum(1 for m in misses if m["category"] == "Moderate")
    n_deep = sum(1 for m in misses if m["category"] == "Deep Miss")

    health = ("TROUBLE" if n_deep >= 3
              else "WATCH"   if n_deep >= 1 or n_mod >= 3
              else "HEALTHY")

    return {
        "n_missed":      len(misses),
        "n_near":        n_near,
        "n_moderate":    n_mod,
        "n_deep":        n_deep,
        "threshold_pct": threshold_pct,
        "health":        health,
        "misses":        misses,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. DECISION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_HR = {"HEALTHY": 0, "WATCH": 1, "TROUBLE": 2}

def decision_engine(hit: dict, ic: dict, missed: dict) -> tuple[str, str]:
    ranks = [_HR[hit.get("health","TROUBLE")],
             _HR[ic.get("health","TROUBLE")],
             _HR[missed.get("health","TROUBLE")]]
    n_trouble, n_healthy = ranks.count(2), ranks.count(0)

    if n_healthy == 3:
        return "TRUST",       "All 3 metrics healthy — model performing well"
    if n_trouble == 0 and missed.get("health") != "HEALTHY":
        return "INVESTIGATE", "Hit & IC healthy but missing high-return stocks — check feature bias"
    if n_trouble == 0:
        return "WATCH",       "Minor issues — monitor next scoring cycle"
    if n_trouble == 3:
        return "RETRAIN",     "All 3 metrics in trouble — model has degraded significantly"
    if n_trouble >= 2:
        return "RETRAIN",     "2+ metrics in trouble — retrain recommended"
    return "WATCH",           f"Mixed signals — {n_healthy}/3 healthy"


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

_H_ICON = {"HEALTHY": "✅", "WATCH": "⚠️ ", "TROUBLE": "❌"}
_V_ICON = {"TRUST": "✅  TRUST MODEL", "INVESTIGATE": "⚠️  INVESTIGATE FEATURE BIAS",
           "WATCH": "⚠️  WATCH CLOSELY", "RETRAIN": "❌  RETRAIN IMMEDIATELY"}


def print_report(all_results: list, entry_date: date, exit_date: date,
                 market_label: str) -> None:
    W = 72
    print("\n" + "═" * W)
    print(f"  BACKTEST REPORT  |  {market_label}")
    print(f"  Period : {entry_date}  ->  {exit_date}")
    print("═" * W)

    for r in all_results:
        tier   = TIER_LABELS[r["tier"]]
        hit    = r["hit_rate"]
        ic     = r["ic_score"]
        missed = r["missed_opp"]
        top_n  = hit.get("top_n", 10)

        print(f"\n{'─'*W}")
        print(f"  {r['model'].upper()} | {r['side'].upper()} | {tier}"
              f"  ({hit.get('universe_size','?')} tickers)")
        print(f"{'─'*W}")

        # Hit Rate
        print(f"\n  [1] HIT RATE")
        bench = hit.get("benchmark_70th", 0)
        print(f"      70th-pct of {tier} universe returns: {bench:+.1f}%")
        print(f"      {'Rank':<5} {'Ticker':<8} {'ModelScore':<12} {'Return':>8}   Result")
        print(f"      {'─'*50}")
        for p in hit.get("picks", []):
            icon = "✅ HIT " if p["hit"] else "❌ MISS"
            print(f"      {p['rank']:<5} {p['ticker']:<8} {p['model_score']:.4f}"
                  f"       {p['fwd_return']:>+7.1f}%   {icon}")
        hr = hit.get("hit_rate_pct", 0)
        nh = hit.get("n_hits", 0)
        print(f"      {'─'*50}")
        print(f"      Hit Rate: {nh}/{top_n} = {hr:.1f}%  "
              f"  {_H_ICON[hit.get('health','TROUBLE')]} {hit.get('health','')}")

        # IC Score
        print(f"\n  [2] IC SCORE  (Spearman: model_score vs actual return)")
        ic_v = ic.get("ic", 0)
        pv   = ic.get("pvalue", 1)
        conc = ic.get("concordance_pct", 0)
        sig  = "statistically significant" if pv < 0.05 else "NOT significant"
        print(f"      IC = {ic_v:+.4f}   p-value = {pv:.4f}  ({sig})")
        print(f"      Concordance (higher score -> higher return): {conc:.1f}%")
        print(f"      Status: {_H_ICON[ic.get('health','TROUBLE')]} {ic.get('health','')}")
        top5 = ic.get("top5_picks", [])
        if top5:
            print(f"      Top-5 model picks and their returns:")
            for p in top5:
                print(f"        {p['ticker']:<8} score={p['model_score']:.4f}  "
                      f"rank={p['rank']:<6}  return={p['fwd_return_pct']:+.1f}%")

        # Missed Opportunity
        thr = missed.get("threshold_pct", 40)
        print(f"\n  [3] MISSED OPPORTUNITY  "
              f"(return > +{thr:.0f}%, outside top-{top_n})")
        nm = missed.get("n_missed", 0)
        if nm == 0:
            print(f"      No high-return stocks missed  "
                  f"{_H_ICON[missed.get('health','HEALTHY')]} HEALTHY")
        else:
            print(f"      {'Ticker':<8} {'Rank':<6} {'Return':>8}   "
                  f"{'Category':<14}  Why")
            print(f"      {'─'*68}")
            cat_icon = {"Near Miss": "🟡", "Moderate": "⚠️ ", "Deep Miss": "🔴"}
            for m in missed.get("misses", []):
                ci = cat_icon.get(m["category"], "  ")
                print(f"      {m['ticker']:<8} {m['rank']:<6} "
                      f"{m['fwd_return']:>+7.1f}%   "
                      f"{ci}{m['category']:<13}  {m['reason']}")
            nn  = missed.get("n_near", 0)
            nmd = missed.get("n_moderate", 0)
            ndp = missed.get("n_deep", 0)
            print(f"      {'─'*68}")
            print(f"      Near Miss(top {top_n+1}-{top_n+10}): {nn}  |  "
                  f"Moderate: {nmd}  |  Deep Miss(50+): {ndp}")
            print(f"      Status: {_H_ICON[missed.get('health','TROUBLE')]} "
                  f"{missed.get('health','')}")

        # Verdict
        verdict, expl = r["verdict"], r["explanation"]
        print(f"\n  {'═'*60}")
        print(f"  VERDICT: {_V_ICON.get(verdict, verdict)}")
        print(f"  {expl}")
        print(f"  {'═'*60}")

    # Summary table
    print(f"\n\n{'═'*W}")
    print(f"  SUMMARY  —  {market_label}  |  {entry_date} -> {exit_date}")
    print(f"{'═'*W}")
    print(f"  {'Model':<12} {'Side':<6} {'Tier':<12} {'HitRate':>8} "
          f"{'IC':>7} {'Missed':>7}  Verdict")
    print(f"  {'─'*W}")
    for r in all_results:
        tl  = TIER_LABELS[r["tier"]]
        hr  = r["hit_rate"].get("hit_rate_pct", 0)
        icv = r["ic_score"].get("ic", 0)
        nm  = r["missed_opp"].get("n_missed", 0)
        vi  = {"TRUST":"✅","INVESTIGATE":"⚠️ ","WATCH":"⚠️ ","RETRAIN":"❌"}.get(r["verdict"],"?")
        print(f"  {r['model']:<12} {r['side']:<6} {tl:<12} "
              f"{hr:>7.1f}% {icv:>+7.4f} {nm:>7}  {vi} {r['verdict']}")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(all_results: list, master_dfs: dict,
                 scoring_date: str, eval_dir: Path, market: str) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    rows = []
    for r in all_results:
        hit, ic, missed = r["hit_rate"], r["ic_score"], r["missed_opp"]
        rows.append({
            "market": market, "scoring_date": scoring_date,
            "model": r["model"], "side": r["side"],
            "cap_tier": r["tier"], "tier_label": TIER_LABELS[r["tier"]],
            "universe_size":      hit.get("universe_size", 0),
            "hit_rate_pct":       hit.get("hit_rate_pct", np.nan),
            "hit_n_hits":         hit.get("n_hits", 0),
            "hit_benchmark_70th": hit.get("benchmark_70th", np.nan),
            "hit_health":         hit.get("health", ""),
            "ic_score":           ic.get("ic", np.nan),
            "ic_pvalue":          ic.get("pvalue", np.nan),
            "ic_concordance_pct": ic.get("concordance_pct", np.nan),
            "ic_health":          ic.get("health", ""),
            "missed_n_total":     missed.get("n_missed", 0),
            "missed_n_near":      missed.get("n_near", 0),
            "missed_n_moderate":  missed.get("n_moderate", 0),
            "missed_n_deep":      missed.get("n_deep", 0),
            "missed_health":      missed.get("health", ""),
            "verdict":            r["verdict"],
            "explanation":        r["explanation"],
        })
    summary_df = pd.DataFrame(rows)
    spath = eval_dir / f"backtest_summary_{market}_{scoring_date}.csv"
    summary_df.to_csv(spath, index=False)
    print(f"  Saved summary : {spath.name}")

    # Detail CSV (per-ticker)
    detail_rows = []
    for r in all_results:
        key     = (r["model"], r["side"], r["tier"])
        df_tier = master_dfs.get(key, pd.DataFrame())
        if df_tier.empty:
            continue
        top_n   = r["hit_rate"].get("top_n", 10)
        bench   = r["hit_rate"].get("benchmark_70th", 0)
        top_tks = set(df_tier.nsmallest(top_n, "rank")["ticker"])
        thr     = r["missed_opp"].get("threshold_pct", 40)
        for _, row in df_tier.iterrows():
            in_top = row["ticker"] in top_tks
            mo_cat = ""
            if not in_top and row["fwd_return_pct"] >= thr:
                mo_cat = _miss_category(int(row["rank"]), top_n)
            detail_rows.append({
                "market": market, "scoring_date": scoring_date,
                "model": r["model"], "side": r["side"], "cap_tier": r["tier"],
                "ticker": row["ticker"], "rank": row["rank"],
                "model_score": row["model_score"],
                "composite_score": row["composite_score"],
                "final_score": row["final_score"],
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "fwd_return_pct": row["fwd_return_pct"],
                "in_top_n": in_top,
                "hit": (row["fwd_return_pct"] > bench) if in_top else np.nan,
                "missed_opp_cat": mo_cat,
                "miss_reason": _miss_reason(row) if mo_cat else "",
            })

    detail_df = pd.DataFrame(detail_rows)
    dpath = eval_dir / f"backtest_detail_{market}_{scoring_date}.csv"
    detail_df.to_csv(dpath, index=False)
    print(f"  Saved detail  : {dpath.name}  ({len(detail_df)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(market: str, scoring_date_str: str,
                 forward_months: int = 6, top_n: int = 10,
                 miss_threshold: float = 40.0) -> None:

    if market not in MARKET_CONFIGS:
        print(f"ERROR: Unknown market '{market}'. "
              f"Choose from: {list(MARKET_CONFIGS.keys())}")
        sys.exit(1)

    cfg = dict(MARKET_CONFIGS[market])
    cfg["market"] = market   # inject for use in helper functions

    entry_date = date.fromisoformat(scoring_date_str)
    exit_date  = _exit_date(entry_date, forward_months)
    output_dir = Path(cfg["output_dir"])
    eval_dir   = Path(cfg["eval_dir"])

    print("=" * 72)
    print(f"  BACKTEST SEGMENTS  |  {cfg['label']}")
    print(f"  Scoring date   : {entry_date}  ->  exit: {exit_date} "
          f"({forward_months}m forward)")
    print(f"  Top-N per tier : {top_n}  |  Miss threshold: +{miss_threshold:.0f}%")
    print("=" * 72)

    # Cap tier map
    print("\n[1] Building cap tier map...")
    tier_map = build_cap_tier_map(cfg)
    for t in ("large", "mid", "small", "micro"):
        n = sum(1 for v in tier_map.values() if v == t)
        if n:
            print(f"  {TIER_LABELS[t]}: {n}")
    if not tier_map:
        print("ERROR: Empty cap tier map — cannot segment by tier.")
        return

    # Load scores_detail
    print("\n[2] Loading scores_detail JSONs...")
    scores_by_model: dict[str, dict] = {}
    all_tickers: set[str] = set()
    for model in ("momentum", "reversal"):
        path = output_dir / f"scores_detail_{model}_{scoring_date_str}.json"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping {model}")
            continue
        with open(path) as f:
            scores_by_model[model] = json.load(f)
        all_tickers |= set(scores_by_model[model].keys())
        print(f"  {model}: {len(scores_by_model[model])} tickers")
    if not scores_by_model:
        print("ERROR: No scores_detail files found.")
        return

    # Forward prices
    print("\n[3] Forward prices...")
    fwd_prices = load_or_build_fwd_prices(
        sorted(all_tickers), entry_date, exit_date, cfg)
    if fwd_prices.empty:
        print("ERROR: Could not obtain forward prices.")
        return

    # Run metrics
    SIDES  = ["bull", "bear"]
    TIERS  = ["large", "mid", "small", "micro"]
    all_results: list[dict] = []
    master_dfs:  dict       = {}

    print(f"\n[4] Computing metrics "
          f"({len(scores_by_model)} models x 2 sides x 4 tiers)...")
    for model, scores in scores_by_model.items():
        for side in SIDES:
            df_all = build_master_df(scores, fwd_prices, tier_map, model, side)
            for tier in TIERS:
                label   = f"{model.upper()} | {side.upper()} | {TIER_LABELS[tier]}"
                df_tier = df_all[df_all["cap_tier"] == tier].copy()
                if len(df_tier) < 10:
                    print(f"  SKIP {label} — {len(df_tier)} tickers (need ≥10)")
                    continue

                hit    = metric_hit_rate(df_tier, top_n)
                ic     = metric_ic_score(df_tier, top_n)
                missed = metric_missed_opportunity(df_tier, top_n, miss_threshold)
                verdict, expl = decision_engine(hit, ic, missed)

                master_dfs[(model, side, tier)] = df_tier
                all_results.append({
                    "model": model, "side": side, "tier": tier,
                    "hit_rate": hit, "ic_score": ic, "missed_opp": missed,
                    "verdict": verdict, "explanation": expl,
                })
                vi = {"TRUST":"✅","INVESTIGATE":"⚠️","WATCH":"⚠️","RETRAIN":"❌"}.get(verdict,"?")
                print(f"  {vi} {label}")

    if not all_results:
        print("No results computed — check data availability.")
        return

    # Print full report
    print_report(all_results, entry_date, exit_date, cfg["label"])

    # Save
    print("[5] Saving output files...")
    save_outputs(all_results, master_dfs, scoring_date_str, eval_dir, market)
    print("\nBacktest complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Segmented backtesting — Hit Rate, IC Score, Missed Opportunity"
    )
    parser.add_argument("--market", required=True,
                        choices=["sp500", "nse", "nse_tradingv"],
                        help="Market to backtest")
    parser.add_argument("--date",           required=True,
                        help="Scoring date YYYY-MM-DD")
    parser.add_argument("--forward_months", type=int,   default=6,
                        help="Calendar months forward (default: 6)")
    parser.add_argument("--top_n",          type=int,   default=10,
                        help="Top-N picks per tier (default: 10)")
    parser.add_argument("--miss_threshold", type=float, default=40.0,
                        help="Return threshold for missed-opp %% (default: 40)")
    args = parser.parse_args()

    run_backtest(
        market           = args.market,
        scoring_date_str = args.date,
        forward_months   = args.forward_months,
        top_n            = args.top_n,
        miss_threshold   = args.miss_threshold,
    )


if __name__ == "__main__":
    main()
