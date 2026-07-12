#!/usr/bin/env python
"""
rank_bucket_analysis.py
=======================
Computes rank-bucket hit-rate analysis for a given watchlist date.

Inputs
------
  --data_dir   : path to public/data/us_local/{date}/ (HF JSON files)
  --eval_xlsx  : path to the 6M forward-eval Excel downloaded from dashboard
  --date       : watchlist date  e.g. 2024-02-09
  --benchmark  : benchmark ticker for excess-return calc (default: SPY)
  --save       : write results to CSV

Usage
-----
  python scripts/tools/rank_bucket_analysis.py \\
      --data_dir   C:/Victor/Projects/ml-stock-dashboard/public/data/us_local/2024-02-09 \\
      --eval_xlsx  output/evaluation/forward_eval_sp500_2024-02-09_2024-08-09.xlsx \\
      --date       2024-02-09

What it produces
----------------
  1. Rank bucket table  : Rank 1-5, 6-15, 16-30 — avg excess return + hit rate
  2. Mode comparison    : Momentum vs Reversal
  3. Model type         : PureML vs Composite vs Both
  4. Random baseline    : random 30-stock sample hit rate for comparison
  5. SDZ tier analysis  : does sdz_1y presence improve outcomes?
  6. Failed pick flags  : low-score picks, marginal signals, stale zones
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _fmt(v, pct=True, decimals=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  N/A "
    suffix = "%" if pct else ""
    sign   = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}{suffix}"


# ── data loading ──────────────────────────────────────────────────────────────

def _load_watchlist_json(data_dir: Path) -> pd.DataFrame:
    """Load bull + bear all-tier JSON, infer cap_tier from tier sub-files."""
    rows = []
    for side in ("bull", "bear"):
        f = data_dir / f"{side}.json"
        if not f.exists():
            print(f"  [warn] {f.name} not found")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for r in data:
            r["side_label"] = side
            rows.append(r)

    if not rows:
        sys.exit(f"No bull.json / bear.json found in {data_dir}")

    df = pd.DataFrame(rows)

    # ── Infer cap_tier from tier-specific JSON files ──────────────────────────
    tier_map: dict[str, str] = {}
    for side in ("bull", "bear"):
        for tier in ("large", "mid", "small", "micro"):
            tf = data_dir / f"{side}_{tier}.json"
            if tf.exists():
                try:
                    tdata = json.loads(tf.read_text(encoding="utf-8"))
                    for r in tdata:
                        t = r.get("ticker", "")
                        if t and t not in tier_map:
                            tier_map[t] = tier
                except Exception:
                    pass

    df["cap_tier"] = df["ticker"].map(tier_map).fillna("unclassified")
    return df


def _load_eval_excel(xlsx_path: Path) -> pd.DataFrame:
    """Load the 6M forward-eval Excel from the dashboard."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="All Tickers")
    except Exception:
        df = pd.read_excel(xlsx_path)   # fallback: first sheet
    # normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # ensure we have pct change column
    for candidate in ("close_pct_change", "pct_change", "change_pct"):
        if candidate in df.columns:
            df["pct_change"] = df[candidate]
            break
    if "pct_change" not in df.columns:
        sys.exit("Could not find a pct_change column in the Excel. "
                 "Expected 'close_pct_change'.")
    return df[["ticker", "base_date", "fwd_date",
               "base_close", "fwd_close", "pct_change"]].dropna(subset=["ticker"])


def _get_benchmark_return(ticker: str, base_date: str, fwd_date: str) -> float:
    """Fetch benchmark 6M return via yfinance."""
    try:
        import yfinance as yf
        start = pd.Timestamp(base_date) - pd.Timedelta(days=5)
        end   = pd.Timestamp(fwd_date)  + pd.Timedelta(days=5)
        df = yf.Ticker(ticker).history(start=str(start.date()),
                                       end=str(end.date()),
                                       auto_adjust=True)
        if df.empty:
            return np.nan
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        # base price: nearest day on or after base_date
        b_candidates = df[df.index >= pd.Timestamp(base_date)]
        f_candidates = df[df.index <= pd.Timestamp(fwd_date)]
        if b_candidates.empty or f_candidates.empty:
            return np.nan
        b_price = float(b_candidates.iloc[0]["Close"])
        f_price = float(f_candidates.iloc[-1]["Close"])
        return (f_price - b_price) / b_price * 100
    except Exception as e:
        print(f"  [warn] Could not fetch {ticker}: {e}")
        return np.nan


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(data_dir: Path, eval_xlsx: Path, date: str,
            benchmark: str, save: bool) -> None:

    # ── 1. Load watchlist metadata ────────────────────────────────────────────
    print(f"\n  Loading watchlist JSON from : {data_dir}")
    wl = _load_watchlist_json(data_dir)
    print(f"  Loaded {len(wl)} rows  "
          f"({wl[wl.side_label=='bull'].shape[0]} bull, "
          f"{wl[wl.side_label=='bear'].shape[0]} bear)")

    # ── 2. Load forward-eval results ─────────────────────────────────────────
    print(f"  Loading eval Excel         : {eval_xlsx}")
    ev = _load_eval_excel(eval_xlsx)
    print(f"  Loaded {len(ev)} tickers with price data")

    # ── 3. Merge ──────────────────────────────────────────────────────────────
    df = wl.merge(ev[["ticker", "pct_change", "base_date", "fwd_date"]],
                  on="ticker", how="left")
    missing = df["pct_change"].isna().sum()
    if missing:
        print(f"  [warn] {missing} watchlist tickers had no price data in Excel")

    df = df.dropna(subset=["pct_change"])

    # ── 4. Benchmark return ───────────────────────────────────────────────────
    base_d = df["base_date"].dropna().iloc[0] if "base_date" in df.columns else date
    fwd_d  = df["fwd_date"].dropna().iloc[0]  if "fwd_date"  in df.columns else ""
    print(f"\n  Fetching benchmark {benchmark} return ({base_d} → {fwd_d}) ...")
    bm_ret = _get_benchmark_return(benchmark, str(base_d), str(fwd_d))
    if np.isnan(bm_ret):
        print(f"  [warn] Could not fetch {benchmark}. Using 0% as benchmark.")
        bm_ret = 0.0
    else:
        print(f"  {benchmark} 6M return: {bm_ret:+.2f}%")

    df["excess_return"] = df["pct_change"] - bm_ret

    # ── 5. Rank buckets ───────────────────────────────────────────────────────
    def bucket_label(r):
        if r <= 5:   return "Rank 01–05  (Top 5)"
        if r <= 15:  return "Rank 06–15  (Mid)"
        if r <= 30:  return "Rank 16–30  (Bottom)"
        return "Rank 31+"

    df["rank_bucket"] = df["rank"].apply(bucket_label)

    _section(f"RANK BUCKET ANALYSIS — {date}  |  benchmark: {benchmark} {bm_ret:+.1f}%")
    print(f"\n  {'Bucket':<28} {'N':>4}  {'Avg Excess':>11}  "
          f"{'Hit Rate':>9}  {'Avg Abs%':>9}  {'Best':>8}  {'Worst':>8}")
    print("  " + "-"*82)

    bucket_order = ["Rank 01–05  (Top 5)", "Rank 06–15  (Mid)",
                    "Rank 16–30  (Bottom)"]
    for side_grp in ("bull", "bear"):
        label = "BULL" if side_grp == "bull" else "BEAR"
        sign  = 1 if side_grp == "bull" else -1   # bear: negative excess = good
        print(f"\n  ── {label} picks ──")
        sub = df[df.side_label == side_grp]
        for bkt in bucket_order:
            g = sub[sub.rank_bucket == bkt]
            if g.empty:
                continue
            exc = g["excess_return"] * sign
            hit = (exc > 0).mean() * 100
            print(f"  {bkt:<28} {len(g):>4}  "
                  f"{_fmt(exc.mean()):>11}  "
                  f"{hit:>8.1f}%  "
                  f"{_fmt(g['pct_change'].mean()):>9}  "
                  f"{_fmt(g['pct_change'].max()):>8}  "
                  f"{_fmt(g['pct_change'].min()):>8}")

    # monotonicity check
    print(f"\n  Skill check (bull): does avg excess drop from top → bottom bucket?")
    bull = df[df.side_label == "bull"]
    for bkt in bucket_order:
        g = bull[bull.rank_bucket == bkt]
        if not g.empty:
            print(f"    {bkt:<28}  avg excess = {_fmt(g['excess_return'].mean())}")

    # ── 6. Mode comparison ────────────────────────────────────────────────────
    _section("MODE COMPARISON — Momentum vs Reversal")
    print(f"\n  {'Mode':<15} {'Side':<6} {'N':>4}  {'Avg Excess':>11}  "
          f"{'Hit Rate':>9}  {'Avg Abs%':>9}")
    print("  " + "-"*58)
    for mode_val in ("momentum", "reversal"):
        for side_grp in ("bull", "bear"):
            sign = 1 if side_grp == "bull" else -1
            g = df[(df.get("mode", pd.Series("", index=df.index)) == mode_val)
                   & (df.side_label == side_grp)] if "mode" in df.columns else pd.DataFrame()
            if g.empty:
                continue
            exc = g["excess_return"] * sign
            hit = (exc > 0).mean() * 100
            print(f"  {mode_val:<15} {side_grp.upper():<6} {len(g):>4}  "
                  f"{_fmt(exc.mean()):>11}  {hit:>8.1f}%  "
                  f"{_fmt(g['pct_change'].mean()):>9}")

    # ── 7. Model type comparison ──────────────────────────────────────────────
    _section("MODEL TYPE — PureML vs Composite vs Both")
    print(f"\n  {'Type':<25} {'Side':<6} {'N':>4}  {'Avg Excess':>11}  {'Hit Rate':>9}")
    print("  " + "-"*60)
    if "model_type" in df.columns:
        for mt in sorted(df["model_type"].dropna().unique()):
            for side_grp in ("bull", "bear"):
                sign = 1 if side_grp == "bull" else -1
                g = df[(df["model_type"] == mt) & (df.side_label == side_grp)]
                if g.empty:
                    continue
                exc = g["excess_return"] * sign
                hit = (exc > 0).mean() * 100
                print(f"  {str(mt):<25} {side_grp.upper():<6} {len(g):>4}  "
                      f"{_fmt(exc.mean()):>11}  {hit:>8.1f}%")

    # ── 8. Cap tier ───────────────────────────────────────────────────────────
    _section("CAP TIER — Large / Mid / Small")
    print(f"\n  {'Tier':<14} {'Side':<6} {'N':>4}  {'Avg Excess':>11}  {'Hit Rate':>9}")
    print("  " + "-"*50)
    for tier in ("large", "mid", "small", "micro", "unclassified"):
        for side_grp in ("bull", "bear"):
            sign = 1 if side_grp == "bull" else -1
            g = df[(df.cap_tier == tier) & (df.side_label == side_grp)]
            if g.empty:
                continue
            exc = g["excess_return"] * sign
            hit = (exc > 0).mean() * 100
            print(f"  {tier:<14} {side_grp.upper():<6} {len(g):>4}  "
                  f"{_fmt(exc.mean()):>11}  {hit:>8.1f}%")

    # ── 9. SDZ tier analysis ──────────────────────────────────────────────────
    _section("SDZ ZONE QUALITY — does higher zone signal improve outcome?")
    bull = df[df.side_label == "bull"]
    if "sdz_htf_score" in bull.columns:
        sdz_thresh = [
            ("sdz_1y active  (score >= 0.75)", bull["sdz_htf_score"] >= 0.75),
            ("sdz_1mo active (0.25–0.75)",     (bull["sdz_htf_score"] >= 0.25)
                                              & (bull["sdz_htf_score"] < 0.75)),
            ("No SDZ        (score < 0.25)",   bull["sdz_htf_score"] < 0.25),
        ]
        print(f"\n  {'SDZ tier':<30} {'N':>4}  {'Avg Excess':>11}  {'Hit Rate':>9}")
        print("  " + "-"*58)
        for label, mask in sdz_thresh:
            g = bull[mask]
            if g.empty:
                continue
            exc = g["excess_return"]
            hit = (exc > 0).mean() * 100
            print(f"  {label:<30} {len(g):>4}  "
                  f"{_fmt(exc.mean()):>11}  {hit:>8.1f}%")

    # ── 10. Random baseline ───────────────────────────────────────────────────
    _section("RANDOM BASELINE — 5 random samples of 30 stocks from full eval")
    np.random.seed(42)
    ev_clean = ev.dropna(subset=["pct_change"])
    for i in range(5):
        sample = ev_clean.sample(30, random_state=i)
        exc = sample["pct_change"] - bm_ret
        hit = (exc > 0).mean() * 100
        print(f"  Sample {i+1}:  avg excess = {_fmt(exc.mean()):>8}  "
              f"hit rate = {hit:.1f}%")

    # ── 11. Failed picks ──────────────────────────────────────────────────────
    _section("FAILED PICKS — bull stocks that underperformed benchmark")
    bull = df[df.side_label == "bull"].copy()
    failed = bull[bull["excess_return"] < 0].sort_values("excess_return")
    print(f"\n  {len(failed)} of {len(bull)} bull picks underperformed {benchmark}")
    print(f"\n  {'#':<4} {'Ticker':<8} {'Rank':>5} {'Score':>7} "
          f"{'Abs%':>8} {'Excess%':>9} {'sdz_htf':>8} {'model_type'}")
    print("  " + "-"*72)
    for i, (_, r) in enumerate(failed.iterrows(), 1):
        flags = []
        if r.get("score", 1) < 0.70:
            flags.append("LOW-SCORE")
        if r.get("sdz_htf_score", 1) < 0.1:
            flags.append("NO-SDZ")
        flag_str = " ".join(flags)
        print(f"  {i:<4} {r['ticker']:<8} {int(r['rank']):>5} "
              f"{r.get('score', 0):>7.3f} "
              f"{_fmt(r['pct_change']):>8} "
              f"{_fmt(r['excess_return']):>9} "
              f"{r.get('sdz_htf_score', 0):>8.3f} "
              f"{str(r.get('model_type','')):<20} {flag_str}")

    # ── 12. Save ──────────────────────────────────────────────────────────────
    if save:
        out = eval_xlsx.parent / f"rank_analysis_{date}.csv"
        save_cols = [c for c in ["ticker", "rank", "rank_bucket", "side_label",
                                  "mode", "model_type", "cap_tier", "score",
                                  "pct_change", "excess_return",
                                  "sdz_htf_score", "ssz_htf_score"] if c in df.columns]
        df[save_cols].sort_values(["side_label", "rank"]).to_csv(out, index=False)
        print(f"\n  Saved enriched CSV: {out}")

    print(f"\n{'='*72}")
    print(f"  Analysis complete — {date}  |  {len(df)} picks  |  "
          f"benchmark {benchmark} = {bm_ret:+.1f}%")
    print(f"{'='*72}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir",  required=True,
                    help="Path to public/data/us_local/{date}/ directory")
    ap.add_argument("--eval_xlsx", required=True,
                    help="Path to 6M forward-eval Excel downloaded from dashboard")
    ap.add_argument("--date",      required=True,
                    help="Watchlist date YYYY-MM-DD")
    ap.add_argument("--benchmark", default="SPY",
                    help="Benchmark ticker (default: SPY)")
    ap.add_argument("--save", action="store_true",
                    help="Save enriched CSV alongside the Excel")
    args = ap.parse_args()

    analyse(
        data_dir  = Path(args.data_dir),
        eval_xlsx = Path(args.eval_xlsx),
        date      = args.date,
        benchmark = args.benchmark,
        save      = args.save,
    )


if __name__ == "__main__":
    main()
