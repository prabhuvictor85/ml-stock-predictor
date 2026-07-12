#!/usr/bin/env python3
"""
Independent watchlist forward-return validator (self-contained, read-only).
===========================================================================
Grades point-in-time watchlists by what actually happened next. Designed to
run LOCALLY with a one-time Yahoo download, so it does NOT depend on the
pipeline's (possibly stale) price CSVs.

Pipeline:
  1. Scan every watchlist file, collect all stock symbols, DEDUPLICATE into
     one master list.
  2. Download daily history from Yahoo ONCE per unique symbol, up to --cutoff
     (default 2026-06-30), cached to --cache_dir. Re-runs skip cached symbols.
  3. Re-iterate each watchlist; for every pick compute forward returns from the
     cached data.
  4. NEAR-CUTOFF RULE: if a pick lacks a full forward window (e.g. a watchlist
     close to the cutoff), the record is NOT discarded — the return is computed
     over ALL available future bars up to the last downloaded trading day, and
     the actual holding length is reported.

Returns are recomputed from raw adjusted closes (never pipeline code), and the
benchmark (default ^GSPC) gives EXCESS return. ddof=1 t-stats.

Run locally:
    python3 scripts/tools/validate_watchlist_forward.py
    python3 scripts/tools/validate_watchlist_forward.py --force_download   # refresh cache
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

DEF_WL    = r"C:\Victor\Project\ml-stock-predictor\output\us_local"
DEF_CACHE = r"C:\Victor\Project\ml-stock-predictor\output\us_local\_validation_prices"
HORIZONS  = [20, 40, 60, 90]
MAX_WIN   = 252
CUTOFF    = "2026-06-30"
_DATE_RE  = re.compile(r"(\d{4}-\d{2}-\d{2})")


# ── Phase 1: collect + dedupe symbols ────────────────────────────────────────
def collect_symbols(files: list) -> list:
    syms: set = set()
    for f in files:
        try:
            wl = pd.read_csv(f, usecols=lambda c: c.lower() == "ticker")
        except Exception:
            wl = pd.read_csv(f)
        col = next((c for c in wl.columns if c.lower() == "ticker"), None)
        if col:
            syms.update(str(t).strip().upper() for t in wl[col].dropna())
    return sorted(syms)


def yf_symbol(t: str) -> str:
    # yfinance uses '-' for share classes: BRK.B -> BRK-B
    return t.replace(".", "-")


# ── Phase 2: download once per symbol, cached ────────────────────────────────
def download_symbols(symbols: list, cache_dir: str, cutoff: str,
                     force: bool) -> dict:
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    prices: dict = {}
    to_fetch = []
    for s in symbols:
        cp = os.path.join(cache_dir, f"{s}.csv")
        if os.path.exists(cp) and not force:
            try:
                ser = pd.read_csv(cp, index_col=0, parse_dates=True)["close"]
                prices[s] = ser.astype(float)
                continue
            except Exception:
                pass
        to_fetch.append(s)

    print(f"  {len(symbols)} unique symbols; {len(prices)} cached, "
          f"{len(to_fetch)} to download (cutoff {cutoff}) ...")
    end_excl = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    CHUNK = 50
    for i in range(0, len(to_fetch), CHUNK):
        chunk = to_fetch[i:i + CHUNK]
        ymap = {yf_symbol(s): s for s in chunk}
        try:
            raw = yf.download(list(ymap.keys()), start="2015-01-01", end=end_excl,
                              auto_adjust=True, progress=False, threads=True,
                              group_by="ticker")
        except Exception as e:
            print(f"    chunk {i//CHUNK} download error: {e}")
            raw = None
        for ys, s in ymap.items():
            try:
                sub = raw[ys] if (raw is not None and ys in raw.columns.get_level_values(0)) else None
                if sub is None or sub["Close"].dropna().empty:
                    prices[s] = None
                    continue
                ser = sub["Close"].dropna().astype(float)
                ser.index = pd.to_datetime(ser.index).tz_localize(None)
                ser.name = "close"
                ser.to_csv(os.path.join(cache_dir, f"{s}.csv"))
                prices[s] = ser
            except Exception:
                prices[s] = None
        print(f"    downloaded {min(i+CHUNK, len(to_fetch))}/{len(to_fetch)}", flush=True)
        time.sleep(0.5)
    missing = [s for s, v in prices.items() if v is None]
    if missing:
        print(f"  [warn] no data for {len(missing)} symbols: {missing[:12]}"
              f"{' ...' if len(missing) > 12 else ''}")
    return prices


# ── Phase 3: forward returns with near-cutoff fallback ───────────────────────
def fwd_returns(close, date: pd.Timestamp) -> dict:
    if close is None or len(close) == 0:
        return {}
    pos = close.index.searchsorted(date, side="right") - 1     # last bar <= date
    if pos < 0:
        return {}
    entry = float(close.iloc[pos])
    if not np.isfinite(entry) or entry <= 0:
        return {}
    n_fwd = len(close) - 1 - pos           # forward bars available
    out = {"fwd_days": n_fwd}
    for h in HORIZONS:
        j = pos + h
        out[f"h{h}"] = (float(close.iloc[j]) / entry - 1.0) if j < len(close) else np.nan
    # NEAR-CUTOFF RULE: return over ALL available future bars (never discarded)
    if n_fwd >= 1:
        out["ret_avail"] = float(close.iloc[-1]) / entry - 1.0
        w = close.iloc[pos + 1: pos + 1 + MAX_WIN]
        out["max_avail"] = float(w.max()) / entry - 1.0
    else:
        out["ret_avail"] = np.nan
        out["max_avail"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist_dir", default=DEF_WL)
    ap.add_argument("--pattern", default="watchlist_momentum_pureml_bull_large_*.csv")
    ap.add_argument("--cache_dir", default=DEF_CACHE)
    ap.add_argument("--cutoff", default=CUTOFF)
    ap.add_argument("--benchmark", default="^GSPC")
    ap.add_argument("--force_download", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.watchlist_dir, args.pattern)))
    if not files:
        sys.exit(f"No watchlist files match {args.pattern} in {args.watchlist_dir}")
    print(f"Found {len(files)} watchlist files matching {args.pattern}")

    # Phase 1
    symbols = collect_symbols(files)
    dl_list = symbols + ([args.benchmark] if args.benchmark else [])
    # Phase 2
    prices = download_symbols(dl_list, args.cache_dir, args.cutoff, args.force_download)
    bench = prices.get(args.benchmark) if args.benchmark else None
    if args.benchmark and bench is None:
        print(f"  [warn] benchmark {args.benchmark} unavailable — absolute returns only.")

    # Phase 3
    per_date, rank_bucket, skipped = [], {}, []
    for f in files:
        m = _DATE_RE.search(os.path.basename(f))
        if not m:
            continue
        date = pd.Timestamp(m.group(1))
        wl = pd.read_csv(f)
        tcol = next((c for c in wl.columns if c.lower() == "ticker"), None)
        if tcol is None:
            continue
        scol = "score" if "score" in wl.columns else None
        rcol = "rank" if "rank" in wl.columns else None
        bf = fwd_returns(bench, date) if bench is not None else {}

        rows = []
        for _, r in wl.iterrows():
            fr = fwd_returns(prices.get(str(r[tcol]).strip().upper()), date)
            if not fr:
                continue
            rec = {"ticker": r[tcol],
                   "rank": int(r[rcol]) if rcol else 0,
                   "score": float(r[scol]) if scol else np.nan, **fr}
            if bench is not None and bf:
                for h in HORIZONS:
                    rec[f"exc_h{h}"] = rec[f"h{h}"] - bf.get(f"h{h}", np.nan)
                rec["exc_avail"] = rec["ret_avail"] - bf.get("ret_avail", np.nan)
            rows.append(rec)
            rank_bucket.setdefault(rec["rank"], []).append(rec.get("ret_avail", np.nan))

        if not rows:
            skipped.append(m.group(1))
            continue
        d = pd.DataFrame(rows)
        rec = {"date": m.group(1), "n_picks": len(d),
               "min_fwd_days": int(d["fwd_days"].min())}
        for h in HORIZONS:
            rec[f"h{h}"] = d[f"h{h}"].mean()
            if bench is not None:
                rec[f"exc_h{h}"] = d[f"exc_h{h}"].mean()
        rec["ret_avail"] = d["ret_avail"].mean()
        rec["max_avail"] = d["max_avail"].mean()
        if bench is not None:
            rec["exc_avail"] = d["exc_avail"].mean()
        if scol and d["score"].std() > 0 and len(d) >= 4:
            from scipy.stats import spearmanr
            rec["rank_ic"] = spearmanr(d["score"], d["ret_avail"]).statistic
        else:
            rec["rank_ic"] = np.nan
        base = d["exc_avail"] if bench is not None else d["ret_avail"]
        rec["hit_rate"] = float((base > 0).mean())
        per_date.append(rec)

    if not per_date:
        sys.exit("No watchlist produced gradeable rows (no price data matched).")

    res = pd.DataFrame(per_date)
    exc = bench is not None
    lbl = "excess" if exc else "abs"
    M = (lambda h: f"exc_h{h}") if exc else (lambda h: f"h{h}")
    print("\n" + "=" * 92)
    print(f"  WATCHLIST FORWARD VALIDATION — {args.pattern}")
    print(f"  {len(res)} dates graded  |  benchmark: {args.benchmark or 'none'}  "
          f"|  cutoff: {args.cutoff}")
    print("=" * 92)
    print(f"\n  Per-date mean pick return ({lbl} vs benchmark unless noted):")
    print(f"  {'date':<12}{'20d':>8}{'40d':>8}{'60d':>8}{'90d':>8}"
          f"{'avail':>8}{'maxAvl':>8}{'rankIC':>7}{'hit%':>6}{'fdays':>6}")
    for _, r in res.iterrows():
        print(f"  {r['date']:<12}"
              + "".join(f"{r.get(M(h), np.nan):>+8.3f}" for h in HORIZONS)
              + f"{r.get('exc_avail', r['ret_avail']):>+8.3f}{r['max_avail']:>+8.3f}"
                f"{r.get('rank_ic', np.nan):>+7.2f}{r['hit_rate']*100:>5.0f}%"
                f"{r['min_fwd_days']:>6}")

    def agg(col):
        v = res[col].dropna().values
        k = len(v)
        if k < 2:
            return (v.mean() if k else float("nan")), float("nan"), k
        m, s = v.mean(), v.std(ddof=1)
        return m, (m / (s / np.sqrt(k)) if s > 0 else 0.0), k

    print("\n  " + "-" * 88)
    print("  AGGREGATE (mean of per-date means; ddof=1 t across dates):")
    for h in HORIZONS:
        m, t, k = agg(M(h))
        print(f"    {lbl} {h:>3}d : {m:>+.4f}  t={t:>+.2f}  (n={k} full-window dates)")
    am, at, ak = agg("exc_avail" if exc else "ret_avail")
    print(f"    {lbl} avail-window (ALL dates, variable horizon): "
          f"{am:>+.4f}  t={at:>+.2f}  (n={ak})")
    mm, _, mk = agg("max_avail")
    print(f"    max-in-window (abs, best exit) : {mm:>+.4f}  (n={mk})")
    ric, rict, rick = agg("rank_ic")
    print(f"    within-list rank-IC (score vs avail ret): {ric:>+.3f} t={rict:>+.2f} (n={rick})")
    print(f"    dates the list beat benchmark (avail window): "
          f"{(res.get('exc_avail', res['ret_avail']) > 0).mean()*100:.0f}%")

    if rank_bucket:
        print("\n  RANK QUALITY — mean avail-window return by rank position (1=top):")
        for rk in sorted(k for k in rank_bucket if k):
            vv = [x for x in rank_bucket[rk] if np.isfinite(x)]
            if vv:
                print(f"    rank {rk:>2}: {np.mean(vv):>+.4f}  (n={len(vv)})")
    if skipped:
        print(f"\n  Skipped {len(skipped)} dates (no price data): {', '.join(skipped[:8])}")

    if args.out:
        res.to_json(args.out, orient="records", indent=2)
        print(f"\n  Per-date results -> {args.out}")


if __name__ == "__main__":
    main()
