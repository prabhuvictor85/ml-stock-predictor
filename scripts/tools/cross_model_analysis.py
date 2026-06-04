#!/usr/bin/env python
"""
cross_model_analysis.py — cross-reference momentum and reversal pure-ML scores
to surface high-conviction signals and model disagreements.

Both models score ALL ~1550 tickers but filter watchlists differently:
  momentum watchlist : stocks within 40% of 52w high
  reversal watchlist : stocks 40%+ below 52w high

Since both models score the full universe, we can compare their raw model_score
for every ticker to find:
  1. High-conviction BULL  — both models independently rank the stock bullish
  2. High-conviction BEAR  — both models independently rank the stock bearish
  3. Divergent signals     — momentum bull + reversal bear (or vice versa)
  4. Watchlist enrichment  — each watchlist pick ranked by the OTHER model's view

Usage
-----
  # auto-detect latest date from output dir
  python scripts/tools/cross_model_analysis.py \
      --output_dir /mnt/data/artefacts/us_local/output

  # specific date
  python scripts/tools/cross_model_analysis.py \
      --output_dir /mnt/data/artefacts/us_local/output \
      --date 2024-01-12

  # save results to CSV
  python scripts/tools/cross_model_analysis.py \
      --output_dir /mnt/data/artefacts/us_local/output \
      --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_scores(path: Path) -> pd.DataFrame:
    """Load scores_detail JSON → DataFrame with one row per ticker."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for ticker, entry in raw.items():
        rows.append({
            "ticker":               ticker,
            "bull_model_score":     entry["bull"]["model_score"],
            "bear_model_score":     entry["bear"]["model_score"],
            "bull_composite_score": entry["bull"]["composite_score"],
            "bear_composite_score": entry["bear"]["composite_score"],
            "bull_rank":            entry["bull_rank"],
            "bear_rank":            entry["bear_rank"],
            "in_bull_watchlist":    entry["in_bull_watchlist"],
            "in_bear_watchlist":    entry["in_bear_watchlist"],
            "universe_size":        entry["bull"]["universe_size"],
        })
    return pd.DataFrame(rows).set_index("ticker")


def _load_watchlist(path: Path) -> pd.DataFrame:
    """Load watchlist CSV, return ticker-indexed with sector + cap_tier."""
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ["ticker", "sector", "cap_tier", "model_score",
                         "composite_score", "rank"] if c in df.columns]
    return df[keep].set_index("ticker")


def _pct(score: float) -> str:
    return f"{score*100:5.1f}%"


def _rank_label(rank: int, universe: int) -> str:
    pct = rank / max(universe, 1) * 100
    return f"#{rank:4d} / {universe} ({pct:.1f}%)"


def _section(title: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def _table(df: pd.DataFrame, cols: list[str], col_headers: list[str],
           top: int = 20) -> None:
    df = df.head(top)
    widths = [max(len(h), 6) for h in col_headers]
    header = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(col_headers, widths))
    print(header)
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for ticker, row in df.iterrows():
        vals = []
        for c, w in zip(cols, widths):
            v = row.get(c, "")
            vals.append(f"{str(v):<{w}}")
        print("  " + "  ".join(vals))


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse(output_dir: Path, date: str | None, top: int, save: bool) -> None:
    # ── Resolve date ─────────────────────────────────────────────────────────
    if date is None:
        candidates = sorted(output_dir.glob("scores_detail_momentum_*.json"))
        if not candidates:
            sys.exit(f"No scores_detail_momentum_*.json found in {output_dir}")
        date = candidates[-1].stem.split("_")[-1]
        print(f"  Auto-detected date: {date}")

    mom_path = output_dir / f"scores_detail_momentum_{date}.json"
    rev_path = output_dir / f"scores_detail_reversal_{date}.json"

    for p in [mom_path, rev_path]:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    print(f"  Loading momentum scores : {mom_path.name}")
    mom = _load_scores(mom_path).add_prefix("mom_")

    print(f"  Loading reversal scores : {rev_path.name}")
    rev = _load_scores(rev_path).add_prefix("rev_")

    # ── Merge on ticker ──────────────────────────────────────────────────────
    df = mom.join(rev, how="inner")
    n = len(df)
    print(f"  Tickers scored by both models: {n}")

    # ── Load sector / cap_tier from watchlist CSVs ───────────────────────────
    meta = pd.DataFrame(index=df.index)
    for variant in ["pureml", "composite"]:
        for side in ["bull", "bear"]:
            for mode in ["momentum", "reversal"]:
                wl_path = output_dir / f"watchlist_{mode}_{variant}_{side}_{date}.csv"
                if wl_path.exists():
                    wl = _load_watchlist(wl_path)
                    for col in ["sector", "cap_tier"]:
                        if col in wl.columns and col not in meta.columns:
                            meta[col] = wl[col]
                    break
            if "sector" in meta.columns:
                break

    df = df.join(meta, how="left")

    # Ensure optional meta columns always exist so downstream selects don't crash
    for _col in ["sector", "cap_tier"]:
        if _col not in df.columns:
            df[_col] = ""

    # ── Combined conviction scores ───────────────────────────────────────────
    # Both models independently rank the same stock — average their raw scores.
    # bull conviction: high when BOTH models like it long
    # bear conviction: high when BOTH models dislike it (bear_model_score = 1 - bull)
    df["conv_bull"]  = (df["mom_bull_model_score"] + df["rev_bull_model_score"]) / 2
    df["conv_bear"]  = (df["mom_bear_model_score"] + df["rev_bear_model_score"]) / 2

    # Divergence: momentum model bullish but reversal model bearish (or vice versa)
    # Positive = momentum says bull more than reversal
    # Negative = reversal says bull more than momentum
    df["divergence"] = df["mom_bull_model_score"] - df["rev_bull_model_score"]

    # Universe label (which model's watchlist this stock was eligible for)
    df["universe"] = "—"
    df.loc[df["mom_in_bull_watchlist"] | df["mom_in_bear_watchlist"], "universe"] = "MOM"
    df.loc[df["rev_in_bull_watchlist"] | df["rev_in_bear_watchlist"], "universe"] = "REV"
    df.loc[(df["mom_in_bull_watchlist"] | df["mom_in_bear_watchlist"]) &
           (df["rev_in_bull_watchlist"] | df["rev_in_bear_watchlist"]), "universe"] = "BOTH"

    uni = df["universe"].value_counts()
    print(f"\n  Watchlist coverage: MOM={uni.get('MOM',0)}  REV={uni.get('REV',0)}  BOTH={uni.get('BOTH',0)}")

    # ── 1. High-conviction BULL ───────────────────────────────────────────────
    _section(f"HIGH-CONVICTION BULL — top {top} (both models agree: long)")
    bull_top = (df.sort_values("conv_bull", ascending=False)
                  .head(top)
                  [["conv_bull", "mom_bull_model_score", "rev_bull_model_score",
                    "mom_in_bull_watchlist", "rev_in_bull_watchlist",
                    "sector", "cap_tier"]])
    bull_top.insert(0, "rank", range(1, len(bull_top) + 1))
    bull_top["mom_wl"] = bull_top["mom_in_bull_watchlist"].map({True: "✓", False: "—"})
    bull_top["rev_wl"] = bull_top["rev_in_bull_watchlist"].map({True: "✓", False: "—"})
    bull_top["conv_bull_pct"]  = (bull_top["conv_bull"] * 100).round(1).astype(str) + "%"
    bull_top["mom_score_pct"]  = (bull_top["mom_bull_model_score"] * 100).round(1).astype(str) + "%"
    bull_top["rev_score_pct"]  = (bull_top["rev_bull_model_score"] * 100).round(1).astype(str) + "%"
    print(f"\n  {'#':<4} {'Ticker':<8} {'Combined':>9} {'Mom-Bull':>9} {'Rev-Bull':>9}"
          f"  {'MomWL':>5} {'RevWL':>5}  {'Sector':<25} {'Cap'}")
    print("  " + "-" * 90)
    for i, (ticker, row) in enumerate(bull_top.iterrows(), 1):
        sec = str(row.get("sector", ""))[:24]
        cap = str(row.get("cap_tier", ""))
        print(f"  {i:<4} {ticker:<8} {row['conv_bull_pct']:>9} {row['mom_score_pct']:>9}"
              f" {row['rev_score_pct']:>9}  {row['mom_wl']:>5} {row['rev_wl']:>5}"
              f"  {sec:<25} {cap}")

    # ── 2. High-conviction BEAR ───────────────────────────────────────────────
    _section(f"HIGH-CONVICTION BEAR — top {top} (both models agree: short/avoid)")
    bear_top = (df.sort_values("conv_bear", ascending=False)
                  .head(top)
                  [["conv_bear", "mom_bear_model_score", "rev_bear_model_score",
                    "mom_in_bear_watchlist", "rev_in_bear_watchlist",
                    "sector", "cap_tier"]])
    print(f"\n  {'#':<4} {'Ticker':<8} {'Combined':>9} {'Mom-Bear':>9} {'Rev-Bear':>9}"
          f"  {'MomWL':>5} {'RevWL':>5}  {'Sector':<25} {'Cap'}")
    print("  " + "-" * 90)
    for i, (ticker, row) in enumerate(bear_top.iterrows(), 1):
        sec = str(row.get("sector", ""))[:24]
        cap = str(row.get("cap_tier", ""))
        mom_wl = "✓" if row["mom_in_bear_watchlist"] else "—"
        rev_wl = "✓" if row["rev_in_bear_watchlist"] else "—"
        cb  = f"{row['conv_bear']*100:.1f}%"
        mb  = f"{row['mom_bear_model_score']*100:.1f}%"
        rb  = f"{row['rev_bear_model_score']*100:.1f}%"
        print(f"  {i:<4} {ticker:<8} {cb:>9} {mb:>9} {rb:>9}  {mom_wl:>5} {rev_wl:>5}"
              f"  {sec:<25} {cap}")

    # ── 3. Signal divergence ─────────────────────────────────────────────────
    _section("SIGNAL DIVERGENCE — momentum says bull, reversal says bear (or vice versa)")

    # Momentum strongly bullish, reversal strongly bearish
    print(f"\n  >> MOMENTUM BULL / REVERSAL BEAR (top {top//2})")
    div_mb = df[df["divergence"] > 0].sort_values("divergence", ascending=False).head(top // 2)
    print(f"  {'#':<4} {'Ticker':<8} {'Diverge':>9} {'Mom-Bull':>9} {'Rev-Bull':>9}  {'Sector':<25} {'Cap'}")
    print("  " + "-" * 80)
    for i, (ticker, row) in enumerate(div_mb.iterrows(), 1):
        sec = str(row.get("sector", ""))[:24]
        cap = str(row.get("cap_tier", ""))
        dv  = f"{row['divergence']*100:+.1f}%"
        mb  = f"{row['mom_bull_model_score']*100:.1f}%"
        rb  = f"{row['rev_bull_model_score']*100:.1f}%"
        print(f"  {i:<4} {ticker:<8} {dv:>9} {mb:>9} {rb:>9}  {sec:<25} {cap}")

    # Reversal strongly bullish, momentum strongly bearish
    print(f"\n  >> REVERSAL BULL / MOMENTUM BEAR (top {top//2})")
    div_rb = df[df["divergence"] < 0].sort_values("divergence").head(top // 2)
    print(f"  {'#':<4} {'Ticker':<8} {'Diverge':>9} {'Mom-Bull':>9} {'Rev-Bull':>9}  {'Sector':<25} {'Cap'}")
    print("  " + "-" * 80)
    for i, (ticker, row) in enumerate(div_rb.iterrows(), 1):
        sec = str(row.get("sector", ""))[:24]
        cap = str(row.get("cap_tier", ""))
        dv  = f"{row['divergence']*100:+.1f}%"
        mb  = f"{row['mom_bull_model_score']*100:.1f}%"
        rb  = f"{row['rev_bull_model_score']*100:.1f}%"
        print(f"  {i:<4} {ticker:<8} {dv:>9} {mb:>9} {rb:>9}  {sec:<25} {cap}")

    # ── 4. Watchlist enrichment ───────────────────────────────────────────────
    _section("MOMENTUM WATCHLIST BULL — enriched with reversal model score")
    mom_wl = df[df["mom_in_bull_watchlist"]].sort_values("mom_bull_rank")
    print(f"\n  {len(mom_wl)} stocks in momentum bull watchlist. Reversal model view:")
    print(f"  {'#':<4} {'Ticker':<8} {'MomScore':>9} {'RevScore':>9} {'Rev-View':>10}  {'Sector'}")
    print("  " + "-" * 72)
    for i, (ticker, row) in enumerate(mom_wl.iterrows(), 1):
        rev_s = row["rev_bull_model_score"]
        view = "AGREES ✓" if rev_s > 0.55 else ("NEUTRAL" if rev_s > 0.45 else "DISAGREES ✗")
        ms = f"{row['mom_bull_model_score']*100:.1f}%"
        rs = f"{rev_s*100:.1f}%"
        sec = str(row.get("sector", ""))[:24]
        print(f"  {i:<4} {ticker:<8} {ms:>9} {rs:>9} {view:>10}  {sec}")

    _section("REVERSAL WATCHLIST BULL — enriched with momentum model score")
    rev_wl = df[df["rev_in_bull_watchlist"]].sort_values("rev_bull_rank")
    print(f"\n  {len(rev_wl)} stocks in reversal bull watchlist. Momentum model view:")
    print(f"  {'#':<4} {'Ticker':<8} {'RevScore':>9} {'MomScore':>9} {'Mom-View':>10}  {'Sector'}")
    print("  " + "-" * 72)
    for i, (ticker, row) in enumerate(rev_wl.iterrows(), 1):
        mom_s = row["mom_bull_model_score"]
        view = "AGREES ✓" if mom_s > 0.55 else ("NEUTRAL" if mom_s > 0.45 else "DISAGREES ✗")
        rs = f"{row['rev_bull_model_score']*100:.1f}%"
        ms = f"{mom_s*100:.1f}%"
        sec = str(row.get("sector", ""))[:24]
        print(f"  {i:<4} {ticker:<8} {rs:>9} {ms:>9} {view:>10}  {sec}")

    # ── 5. Sector distribution of high-conviction bulls ──────────────────────
    if "sector" in df.columns:
        _section("SECTOR DISTRIBUTION — high-conviction bulls (conv_bull > 0.6)")
        hc = df[df["conv_bull"] > 0.60]
        if len(hc) > 0:
            sec_counts = hc["sector"].value_counts()
            print(f"\n  {len(hc)} stocks with combined bull score > 60%")
            for sec, cnt in sec_counts.items():
                bar = "█" * cnt
                print(f"  {str(sec):<30} {cnt:3d}  {bar}")
        else:
            print("\n  No stocks with combined bull score > 60%")

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        out_path = output_dir / f"cross_model_analysis_{date}.csv"
        save_cols = ["conv_bull", "conv_bear", "divergence",
                     "mom_bull_model_score", "rev_bull_model_score",
                     "mom_bear_model_score", "rev_bear_model_score",
                     "mom_bull_rank", "rev_bull_rank",
                     "mom_in_bull_watchlist", "rev_in_bull_watchlist",
                     "mom_in_bear_watchlist", "rev_in_bear_watchlist",
                     "sector", "cap_tier", "universe"]
        save_cols = [c for c in save_cols if c in df.columns]
        df[save_cols].sort_values("conv_bull", ascending=False).to_csv(out_path)
        print(f"\n  Saved: {out_path}")

    print(f"\n{'='*72}")
    print(f"  Analysis complete — date: {date}  |  universe: {n} tickers")
    print(f"{'='*72}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default="/mnt/data/artefacts/us_local/output",
                    help="Directory containing scores_detail and watchlist files")
    ap.add_argument("--date", default=None,
                    help="Scoring date YYYY-MM-DD (default: auto-detect latest)")
    ap.add_argument("--top", type=int, default=20,
                    help="Rows to show per section (default: 20)")
    ap.add_argument("--save", action="store_true",
                    help="Save full cross-model DataFrame to CSV")
    args = ap.parse_args()

    analyse(Path(args.output_dir), args.date, args.top, args.save)


if __name__ == "__main__":
    main()
