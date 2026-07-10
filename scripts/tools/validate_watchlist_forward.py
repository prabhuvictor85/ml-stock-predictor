#!/usr/bin/env python3
"""
Independent watchlist forward-return validator (read-only, frozen).
===================================================================
Grades point-in-time watchlists by what actually happened next, computed
DIRECTLY from raw price CSVs — never from pipeline code, so a pipeline bug
cannot grade its own homework (model-validation skill, validator design rules).

For each watchlist file (default: momentum_pureml_bull_large_YYYY-MM-DD.csv):
  * entry = close on the watchlist date (nearest trading day <= date)
  * forward return at h in {20,40,60,90} trading days: close[t+h]/close[t]-1
  * max return within 252 trading days: max(close[t+1..t+252])/close[t]-1
  * benchmark (default ^GSPC) same horizons -> EXCESS return
Two independent questions:
  A. LIST quality  — did the 10 picks beat the benchmark? (mean excess, hit rate)
  B. RANK quality  — did the score's ordering predict return? (within-list
     Spearman(score, fwd_return) per date, + mean return by rank bucket)
Dates without enough forward data in the CSVs are skipped and reported.

Usage (server, fresh data):
    python3 scripts/tools/validate_watchlist_forward.py \
      --watchlist_dir /mnt/data/artefacts/us_local/output \
      --price_dir <STOCK_DATA_DIR> --benchmark ^GSPC
Usage (local, stale data — early dates only):
    python3 scripts/tools/validate_watchlist_forward.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEF_WL   = r"C:\Victor\Project\ml-stock-predictor\output\us_local"
DEF_PX   = r"C:\Victor\Learning_charts\stock_data\us_stocks"
HORIZONS = [20, 40, 60, 90]
MAX_WIN  = 252
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_prices(px_dir: str) -> dict:
    cache: dict = {}
    def get(ticker: str):
        if ticker in cache:
            return cache[ticker]
        path = os.path.join(px_dir, f"{ticker}-1d.csv")
        if not os.path.exists(path):
            cache[ticker] = None
            return None
        df = pd.read_csv(path)
        dc = df.columns[0]
        df[dc] = pd.to_datetime(df[dc], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=[dc]).set_index(dc).sort_index()
        df.columns = [c.lower() for c in df.columns]
        cache[ticker] = df["close"].astype(float)
        return cache[ticker]
    return get


def fwd_returns(close: pd.Series, date: pd.Timestamp) -> dict:
    """Trading-day forward returns from entry (nearest close <= date)."""
    if close is None or close.empty:
        return {}
    pos = close.index.searchsorted(date, side="right") - 1   # last bar <= date
    if pos < 0:
        return {}
    entry = close.iloc[pos]
    if not np.isfinite(entry) or entry <= 0:
        return {}
    out = {}
    for h in HORIZONS:
        j = pos + h
        out[f"h{h}"] = (close.iloc[j] / entry - 1.0) if j < len(close) else np.nan
    tail = close.iloc[pos + 1: pos + 1 + MAX_WIN]
    out["max252"] = (tail.max() / entry - 1.0) if len(tail) else np.nan
    out["n_fwd_bars"] = len(close) - 1 - pos      # forward bars available
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist_dir", default=DEF_WL)
    ap.add_argument("--price_dir", default=DEF_PX)
    ap.add_argument("--pattern", default="watchlist_momentum_pureml_bull_large_*.csv")
    ap.add_argument("--benchmark", default="^GSPC",
                    help="benchmark ticker CSV for excess return; '' to skip")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    get_px = load_prices(args.price_dir)
    bench = get_px(args.benchmark) if args.benchmark else None
    if args.benchmark and bench is None:
        print(f"  [warn] benchmark {args.benchmark} CSV not found — "
              f"reporting ABSOLUTE returns only (no excess).")

    files = sorted(glob.glob(os.path.join(args.watchlist_dir, args.pattern)))
    if not files:
        sys.exit(f"No watchlist files match {args.pattern} in {args.watchlist_dir}")

    per_date = []
    rank_bucket: dict = {}     # rank -> list of h20 returns (rank quality)
    skipped = []
    for f in files:
        m = _DATE_RE.search(os.path.basename(f))
        if not m:
            continue
        date = pd.Timestamp(m.group(1))
        wl = pd.read_csv(f)
        if "ticker" not in wl.columns:
            continue
        score_col = "score" if "score" in wl.columns else None

        rows = []
        for _, r in wl.iterrows():
            fr = fwd_returns(get_px(str(r["ticker"])), date)
            if not fr or not np.isfinite(fr.get("h20", np.nan)):
                continue
            rec = {"ticker": r["ticker"], "rank": int(r.get("rank", 0)),
                   "score": float(r[score_col]) if score_col else np.nan, **fr}
            if bench is not None:
                bf = fwd_returns(bench, date)
                for h in HORIZONS:
                    rec[f"exc_h{h}"] = rec[f"h{h}"] - bf.get(f"h{h}", np.nan)
            rows.append(rec)
            rank_bucket.setdefault(rec["rank"], []).append(rec["h20"])

        if not rows:
            skipped.append((m.group(1), "no forward data"))
            continue
        d = pd.DataFrame(rows)
        rec = {"date": m.group(1), "n_picks": len(d)}
        for h in HORIZONS:
            rec[f"h{h}"] = d[f"h{h}"].mean()
            if bench is not None:
                rec[f"exc_h{h}"] = d[f"exc_h{h}"].mean()
        rec["max252"] = d["max252"].mean()
        # rank quality: does higher score predict higher 20d return?
        if score_col and d["score"].std() > 0 and len(d) >= 4:
            rec["rank_ic_h20"] = spearmanr(d["score"], d["h20"]).statistic
        else:
            rec["rank_ic_h20"] = np.nan
        # hit rate vs benchmark (or vs 0 if no benchmark) at 20d
        base = d["exc_h20"] if bench is not None else d["h20"]
        rec["hit_rate_h20"] = float((base > 0).mean())
        per_date.append(rec)

    if not per_date:
        print("No watchlist had sufficient forward price data in the CSVs.")
        print(f"  (price data may be stale — check {args.price_dir})")
        for dt, why in skipped[:5]:
            print(f"    skipped {dt}: {why}")
        return

    res = pd.DataFrame(per_date)
    n = len(res)
    print("=" * 84)
    print(f"  WATCHLIST FORWARD VALIDATION — {os.path.basename(args.pattern)}")
    print(f"  {n} gradeable dates  |  benchmark: {args.benchmark or 'none (absolute)'}")
    print("=" * 84)
    metric = (lambda h: f"exc_h{h}") if bench is not None else (lambda h: f"h{h}")
    lbl = "excess" if bench is not None else "absolute"
    print(f"\n  Per-date mean pick {lbl} return:")
    print(f"  {'date':<12} {'20d':>8} {'40d':>8} {'60d':>8} {'90d':>8} "
          f"{'max1y':>8} {'rankIC':>7} {'hit%':>6}")
    for _, r in res.iterrows():
        print(f"  {r['date']:<12} " + " ".join(
            f"{r.get(metric(h), np.nan):>+8.3f}" for h in HORIZONS)
            + f" {r['max252']:>+8.3f} {r.get('rank_ic_h20', np.nan):>+7.2f} "
              f"{r['hit_rate_h20']*100:>5.0f}%")

    print("\n  " + "-" * 80)
    print("  AGGREGATE (mean of per-date means; t-stat uses ddof=1 across dates):")
    def agg(col):
        v = res[col].dropna().values
        k = len(v)
        if k < 2:
            return v.mean() if k else float("nan"), float("nan"), k
        m, s = v.mean(), v.std(ddof=1)
        return m, (m / (s / np.sqrt(k)) if s > 0 else 0.0), k
    for h in HORIZONS:
        m, t, k = agg(metric(h))
        print(f"    {lbl} {h:>3}d return: {m:>+.4f}  t={t:>+.2f}  (n={k})")
    mm, mt, mk = agg("max252")
    print(f"    max-1y return  : {mm:>+.4f}  (n={mk})  [absolute, not excess]")
    ric, rict, rick = agg("rank_ic_h20")
    print(f"    within-list rank-IC (score vs 20d ret): {ric:>+.3f}  t={rict:>+.2f}  (n={rick})")
    print(f"    dates the list beat benchmark at 20d: "
          f"{(res[metric(20)] > 0).mean()*100:.0f}%")

    if rank_bucket:
        print("\n  RANK QUALITY — mean 20d return by rank position (1=top score):")
        for rk in sorted(rank_bucket):
            vals = rank_bucket[rk]
            print(f"    rank {rk:>2}: {np.mean(vals):>+.4f}  (n={len(vals)})")

    if skipped:
        print(f"\n  Skipped {len(skipped)} dates (insufficient forward data): "
              f"{', '.join(d for d,_ in skipped[:8])}{' ...' if len(skipped)>8 else ''}")

    if args.out:
        res.to_json(args.out, orient="records", indent=2)
        print(f"\n  Per-date results -> {args.out}")


if __name__ == "__main__":
    main()
