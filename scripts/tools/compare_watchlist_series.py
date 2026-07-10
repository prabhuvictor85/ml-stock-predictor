#!/usr/bin/env python3
"""
Head-to-head watchlist grader — compare N watchlist series on identical logic.
=============================================================================
Grades any number of labeled watchlist folders (e.g. frozen-recipe vs causal
walk-forward, or pure-ML vs composite) with the SAME forward-return math:
excess over a benchmark, computed from raw adjusted closes (never pipeline
code). Downloads each unique symbol once (cached). Reports per-year excess,
hit rate, and an exact-date overlap comparison.

Usage:
    python3 scripts/tools/compare_watchlist_series.py \
      --series frozen  output/us_local \
      --series causal  /mnt/data/artefacts/us_local/output \
      --pattern "watchlist_momentum_pureml_bull_large_*.csv" \
      --benchmark ^GSPC
"""
from __future__ import annotations

import argparse, glob, os, re, sys, time
import numpy as np, pandas as pd

CACHE_DEF = r"C:\Victor\Project\ml-stock-predictor\output\us_local\_validation_prices"
HZ = [20, 40, 60]; MAXW = 252; CUTOFF = "2026-06-30"
_RX = re.compile(r"(\d{4}-\d{2}-\d{2})")


def load_cached(cache, t):
    cp = os.path.join(cache, f"{t}.csv")
    if os.path.exists(cp):
        try:
            return pd.read_csv(cp, index_col=0, parse_dates=True)["close"].astype(float)
        except Exception:
            pass
    return None


def download(symbols, cache, cutoff, force):
    import yfinance as yf
    os.makedirs(cache, exist_ok=True)
    prices, fetch = {}, []
    for s in symbols:
        c = None if force else load_cached(cache, s)
        if c is not None:
            prices[s] = c
        else:
            fetch.append(s)
    if fetch:
        print(f"  downloading {len(fetch)} new symbols (of {len(symbols)}) ...")
        end = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for i in range(0, len(fetch), 50):
            chunk = fetch[i:i+50]; ymap = {s.replace(".", "-"): s for s in chunk}
            try:
                raw = yf.download(list(ymap.keys()), start="2015-01-01", end=end,
                                  auto_adjust=True, progress=False, threads=True,
                                  group_by="ticker")
            except Exception:
                raw = None
            for ys, s in ymap.items():
                try:
                    sub = raw[ys] if (raw is not None and ys in raw.columns.get_level_values(0)) else None
                    if sub is None or sub["Close"].dropna().empty:
                        prices[s] = None; continue
                    ser = sub["Close"].dropna().astype(float)
                    ser.index = pd.to_datetime(ser.index).tz_localize(None); ser.name = "close"
                    ser.to_csv(os.path.join(cache, f"{s}.csv")); prices[s] = ser
                except Exception:
                    prices[s] = None
            time.sleep(0.4)
    return prices


def exc(close, bench, date, h):
    def r(c):
        if c is None or len(c) == 0: return np.nan
        p = c.index.searchsorted(date, side="right") - 1
        if p < 0 or p + h >= len(c): return np.nan
        return float(c.iloc[p+h]) / float(c.iloc[p]) - 1
    pr, br = r(close), r(bench)
    return (pr - br) if (np.isfinite(pr) and np.isfinite(br)) else np.nan


def grade_series(name, wl_dir, pattern, prices, bench):
    files = sorted(glob.glob(os.path.join(wl_dir, pattern)))
    recs = []
    for f in files:
        m = _RX.search(os.path.basename(f))
        if not m: continue
        D = pd.Timestamp(m.group(1)); w = pd.read_csv(f)
        tcol = next((c for c in w.columns if c.lower() == "ticker"), None)
        if not tcol: continue
        for _, rr in w.iterrows():
            c = prices.get(str(rr[tcol]).strip().upper())
            for h in HZ:
                e = exc(c, bench, D, h)
                if np.isfinite(e):
                    recs.append({"date": m.group(1), "yr": D.year, "h": h, "exc": e})
    return pd.DataFrame(recs), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", action="append",
                    metavar="LABEL DIR [PATTERN]", required=True,
                    help="Repeatable. 2 args (LABEL DIR) uses --pattern; "
                         "3 args (LABEL DIR PATTERN) overrides per-series.")
    ap.add_argument("--pattern", default="watchlist_momentum_pureml_bull_large_*.csv")
    ap.add_argument("--cache_dir", default=CACHE_DEF)
    ap.add_argument("--benchmark", default="^GSPC")
    ap.add_argument("--cutoff", default=CUTOFF)
    ap.add_argument("--force_download", action="store_true")
    args = ap.parse_args()

    # normalize each --series into (label, dir, pattern)
    series = []
    for entry in args.series:
        if len(entry) == 2:
            series.append((entry[0], entry[1], args.pattern))
        elif len(entry) == 3:
            series.append((entry[0], entry[1], entry[2]))
        else:
            sys.exit(f"--series takes 2 or 3 args, got {entry}")

    # collect symbols across all series
    syms = set()
    for _, d, pat in series:
        for f in glob.glob(os.path.join(d, pat)):
            try:
                w = pd.read_csv(f)
                tc = next((c for c in w.columns if c.lower() == "ticker"), None)
                if tc: syms.update(str(t).strip().upper() for t in w[tc].dropna())
            except Exception:
                pass
    if not syms:
        sys.exit("No watchlists matched in any series dir.")
    prices = download(sorted(syms) + [args.benchmark], args.cache_dir, args.cutoff,
                      args.force_download)
    bench = prices.get(args.benchmark)

    graded = {}
    for label, d, pat in series:
        df, nf = grade_series(label, d, pat, prices, bench)
        graded[label] = df
        print(f"  series '{label}': {nf} files, {df['date'].nunique() if len(df) else 0} graded dates")

    print("\n" + "=" * 78)
    print(f"  HEAD-TO-HEAD — mean EXCESS vs {args.benchmark}, by year")
    print("=" * 78)
    years = sorted({y for df in graded.values() for y in (df['yr'].unique() if len(df) else [])})
    for h in HZ:
        print(f"\n  -- {h}d excess --")
        print(f"  {'year':>6} " + "".join(f"{lab:>22}" for lab, _, _p in series))
        for y in years:
            cells = []
            for lab, _, _p in series:
                df = graded[lab]; s = df[(df.yr == y) & (df.h == h)]
                cells.append(f"{s.exc.mean():+.2%} ({s.date.nunique()}d)" if len(s) else "—")
            print(f"  {y:>6} " + "".join(f"{c:>22}" for c in cells))

    # exact-date overlap on 20d
    print("\n" + "-" * 78)
    print("  EXACT-DATE OVERLAP (20d excess on dates present in ALL series):")
    date_sets = [set(graded[lab][graded[lab].h == 20]["date"]) for lab, _, _p in series]
    common = sorted(set.intersection(*date_sets)) if date_sets and all(date_sets) else []
    if not common:
        print("    (no dates common to all series yet — expected until the causal run emits watchlists)")
    else:
        print(f"  {'date':<12} " + "".join(f"{lab:>14}" for lab, _, _p in series))
        for dt in common:
            row = []
            for lab, _, _p in series:
                df = graded[lab]; v = df[(df.date == dt) & (df.h == 20)]["exc"].mean()
                row.append(f"{v:>+13.2%}")
            print(f"  {dt:<12} " + "".join(f"{c:>14}" for c in row))
        print(f"\n  {len(common)} common dates.")


if __name__ == "__main__":
    main()
