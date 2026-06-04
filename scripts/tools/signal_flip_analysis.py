"""
signal_flip_analysis.py  —  Independent Watchlist Signal-Flip Evaluation

Answers:
  1. What does the watchlist look like over time? (ticker persistence, churn)
  2. Did any ticker flip BULL→BEAR (or BEAR→BULL) across dates?
  3. If you "bought" the bull signal date and "sold" at the bear signal date,
     what was the actual P&L?
  4. Summary: does the flip signal have predictive value?

Completely standalone — no dependency on backtest_segments or any other
internal tooling. Just reads watchlist CSVs and yfinance prices.

Usage:
    python scripts/tools/signal_flip_analysis.py
    python scripts/tools/signal_flip_analysis.py --output_dir output/us_local
    python scripts/tools/signal_flip_analysis.py --model reversal --scoring composite
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD ALL WATCHLISTS
# ─────────────────────────────────────────────────────────────────────────────

def load_watchlists(output_dir: Path, model: str, scoring: str) -> pd.DataFrame:
    """
    Scans output_dir/<date>/watchlist_{model}_{scoring}_{side}_{date}.csv
    for all dates and sides. Returns a single DataFrame with columns:
        date | ticker | side | rank | score | ...feature cols...
    """
    records = []
    for date_dir in sorted(output_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name
        for side in ("bull", "bear"):
            fname = f"watchlist_{model}_{scoring}_{side}_{date_str}.csv"
            fpath = date_dir / fname
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            df["signal_date"] = pd.Timestamp(date_str)
            df["side"] = df["side"].str.lower()
            records.append(df)

    if not records:
        print(f"ERROR: No watchlist files found in {output_dir} "
              f"for model={model} scoring={scoring}")
        sys.exit(1)

    all_df = pd.concat(records, ignore_index=True)
    # Normalise column names
    all_df.columns = [c.lower() for c in all_df.columns]
    if "ticker" not in all_df.columns and "symbol" in all_df.columns:
        all_df.rename(columns={"symbol": "ticker"}, inplace=True)
    all_df["ticker"] = all_df["ticker"].astype(str).str.strip().str.upper()
    all_df["signal_date"] = pd.to_datetime(all_df["signal_date"])
    return all_df.sort_values(["ticker", "signal_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TICKER TIMELINE  (what signal did each ticker carry on each date?)
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(wl: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a wide pivot: index=ticker, columns=dates, values=side (bull/bear/None)
    """
    dates = sorted(wl["signal_date"].unique())
    tickers = sorted(wl["ticker"].unique())

    lookup = {}
    for _, row in wl.iterrows():
        lookup[(row["ticker"], row["signal_date"])] = row["side"]

    rows = []
    for t in tickers:
        row = {"ticker": t}
        for d in dates:
            row[d.strftime("%Y-%m-%d")] = lookup.get((t, d), None)
        rows.append(row)

    return pd.DataFrame(rows).set_index("ticker")


# ─────────────────────────────────────────────────────────────────────────────
# 3. DETECT FLIPS
# ─────────────────────────────────────────────────────────────────────────────

def find_flips(wl: pd.DataFrame) -> list[dict]:
    """
    For each ticker, walk its chronological signal history.
    A FLIP is when the side changes (bull→bear or bear→bull) in consecutive
    appearances (ignoring dates where the ticker was absent).

    Returns list of dicts:
        ticker | buy_date | buy_side | sell_date | sell_side | flip_type
        (buy_date = first signal, sell_date = first opposite signal)
    """
    flips = []
    for ticker, grp in wl.groupby("ticker"):
        grp = grp.sort_values("signal_date")
        appearances = grp[["signal_date", "side", "rank", "score"]].to_dict("records")

        i = 0
        while i < len(appearances):
            entry = appearances[i]
            # look forward for a flip
            for j in range(i + 1, len(appearances)):
                later = appearances[j]
                if later["side"] != entry["side"]:
                    flips.append({
                        "ticker":       ticker,
                        "entry_date":   entry["signal_date"],
                        "entry_side":   entry["side"],
                        "entry_rank":   entry["rank"],
                        "entry_score":  round(entry["score"], 4),
                        "exit_date":    later["signal_date"],
                        "exit_side":    later["side"],
                        "exit_rank":    later["rank"],
                        "exit_score":   round(later["score"], 4),
                        "flip_type":    f"{entry['side']}→{later['side']}",
                        "days_between": (later["signal_date"] - entry["signal_date"]).days,
                    })
                    break   # only take the FIRST flip from each entry signal
            i += 1

    return sorted(flips, key=lambda x: (x["entry_date"], x["ticker"]))


# ─────────────────────────────────────────────────────────────────────────────
# 4. FETCH PRICES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Robustly extract Close series from yfinance download result.
    Handles both flat columns and multi-level (ticker, field) columns.
    """
    if raw is None or raw.empty:
        return None
    # Flatten MultiIndex columns — yfinance returns (field, ticker) structure
    if isinstance(raw.columns, pd.MultiIndex):
        # Level 0 = field name (Close, Open...), Level 1 = ticker
        if "Close" in raw.columns.get_level_values(0):
            series = raw["Close"]
            # If still a DataFrame (multiple tickers), pick the right one
            if isinstance(series, pd.DataFrame):
                if ticker in series.columns:
                    series = series[ticker]
                else:
                    series = series.iloc[:, 0]
            return series.dropna()
        return None
    # Flat columns
    if "Close" in raw.columns:
        return raw["Close"].dropna()
    return None


def _nearest_price(close: pd.Series, target_date: pd.Timestamp) -> float | None:
    """Get close price on or just after target_date."""
    if close is None or close.empty:
        return None
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    candidates = close[close.index >= target_date]
    if candidates.empty:
        # target beyond data — use last available
        candidates = close[close.index <= target_date]
    if candidates.empty:
        return None
    return float(candidates.iloc[0])


def fetch_prices(tickers: list[str],
                 dates: list[pd.Timestamp],
                 delay: float = 0.5) -> dict[tuple, float]:
    """
    Returns {(ticker, date): price} for all (ticker, date) combos.
    Downloads full range per ticker, reads closest trading day price.
    """
    if not tickers:
        return {}

    min_date = min(dates) - pd.Timedelta(days=5)
    max_date = max(dates) + pd.Timedelta(days=5)

    price_map: dict[tuple, float] = {}
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"  Fetching {ticker} ({i}/{total})...", end=" ", flush=True)
        try:
            raw = yf.download(
                ticker,
                start=min_date.strftime("%Y-%m-%d"),
                end=max_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )
            close = _extract_close(raw, ticker)
            if close is None:
                print("no data")
                time.sleep(delay)
                continue
            hits = 0
            for d in dates:
                p = _nearest_price(close, d)
                if p is not None:
                    price_map[(ticker, d)] = p
                    hits += 1
            print(f"got {hits}/{len(dates)} prices")
        except Exception as exc:
            print(f"ERROR: {exc}")
        time.sleep(delay)

    return price_map


# ─────────────────────────────────────────────────────────────────────────────
# 5. ENRICHMENT — add actual P&L to each flip
# ─────────────────────────────────────────────────────────────────────────────

def enrich_flips(flips: list[dict], price_map: dict) -> pd.DataFrame:
    rows = []
    for f in flips:
        entry_px = price_map.get((f["ticker"], f["entry_date"]))
        exit_px  = price_map.get((f["ticker"], f["exit_date"]))
        if entry_px and exit_px:
            raw_ret = (exit_px - entry_px) / entry_px * 100
            # For a bull→bear flip we BOUGHT at entry, SOLD at exit  → positive is profit
            # For a bear→bull flip we SHORTED at entry, covered at exit → negative ret is profit
            if f["entry_side"] == "bull":
                pnl_pct = raw_ret          # long: profit if price went up
            else:
                pnl_pct = -raw_ret         # short: profit if price went down
        else:
            raw_ret = pnl_pct = None

        rows.append({
            **f,
            "entry_price": round(entry_px, 2) if entry_px else None,
            "exit_price":  round(exit_px, 2)  if exit_px  else None,
            "raw_return_pct": round(raw_ret, 1) if raw_ret is not None else None,
            "trade_pnl_pct":  round(pnl_pct, 1) if pnl_pct is not None else None,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def report_overview(wl: pd.DataFrame) -> None:
    dates = sorted(wl["signal_date"].unique())
    print("\n" + "═" * 72)
    print("  WATCHLIST OVERVIEW")
    print("═" * 72)
    print(f"  Dates spanned : {dates[0].date()}  →  {dates[-1].date()}  ({len(dates)} snapshots)")
    print(f"  Unique tickers: {wl['ticker'].nunique()}")
    print(f"  Total records : {len(wl)}")
    print()
    print(f"  {'Date':<14}  {'Bull':>5}  {'Bear':>5}  {'Tickers':>8}")
    print(f"  {'────':<14}  {'────':>5}  {'────':>5}  {'───────':>8}")
    for d in dates:
        sub = wl[wl["signal_date"] == d]
        bull_n = (sub["side"] == "bull").sum()
        bear_n = (sub["side"] == "bear").sum()
        tickers_n = sub["ticker"].nunique()
        print(f"  {d.date()!s:<14}  {bull_n:>5}  {bear_n:>5}  {tickers_n:>8}")


def report_persistence(wl: pd.DataFrame) -> None:
    dates = sorted(wl["signal_date"].unique())
    n_dates = len(dates)
    print("\n" + "─" * 72)
    print("  TICKER PERSISTENCE  (how many dates each ticker appeared)")
    print("─" * 72)

    counts = wl.groupby(["ticker", "side"]).size().reset_index(name="appearances")
    for side in ("bull", "bear"):
        sub = counts[counts["side"] == side].sort_values("appearances", ascending=False)
        print(f"\n  {'━' * 40}")
        print(f"  BULL side  —  persistent across multiple dates:" if side == "bull"
              else f"  BEAR side  —  persistent across multiple dates:")
        persistent = sub[sub["appearances"] >= 3]
        if persistent.empty:
            print("    None appeared 3+ times")
        else:
            for _, row in persistent.head(15).iterrows():
                bar = "█" * row["appearances"]
                print(f"    {row['ticker']:<7} {bar:<12} {row['appearances']}/{n_dates} snapshots")


def report_flips(df: pd.DataFrame) -> None:
    print("\n" + "═" * 72)
    print("  SIGNAL FLIPS  (ticker appeared BULL then BEAR, or vice versa)")
    print("═" * 72)

    if df.empty:
        print("  No flips found.")
        return

    for flip_type in ("bull→bear", "bear→bull"):
        sub = df[df["flip_type"] == flip_type].sort_values(
            ["entry_date", "entry_rank"])
        print(f"\n  ── {flip_type.upper()}  ({len(sub)} flips) ──")
        if sub.empty:
            print("    None.")
            continue

        hdr = (f"  {'Ticker':<7}  {'Entry':>10}  {'Rk':>3}  {'Exit':>10}  "
               f"{'Rk':>3}  {'Days':>4}  {'EntryPx':>8}  {'ExitPx':>8}  "
               f"{'RawRet':>7}  {'TradePnL':>9}")
        print(hdr)
        print("  " + "─" * 68)
        for _, r in sub.iterrows():
            raw  = f"{r['raw_return_pct']:+.1f}%" if r["raw_return_pct"] is not None else "   N/A"
            pnl  = f"{r['trade_pnl_pct']:+.1f}%"  if r["trade_pnl_pct"]  is not None else "   N/A"
            epx  = f"${r['entry_price']:.2f}"      if r["entry_price"]    is not None else "   N/A"
            xpx  = f"${r['exit_price']:.2f}"       if r["exit_price"]     is not None else "   N/A"
            icon = "✅" if (r["trade_pnl_pct"] or 0) > 0 else "❌"
            print(f"  {r['ticker']:<7}  {str(r['entry_date'].date()):>10}  "
                  f"{int(r['entry_rank']):>3}  {str(r['exit_date'].date()):>10}  "
                  f"{int(r['exit_rank']):>3}  {int(r['days_between']):>4}  "
                  f"{epx:>8}  {xpx:>8}  {raw:>7}  {pnl:>9}  {icon}")


def report_pnl_summary(df: pd.DataFrame) -> None:
    print("\n" + "═" * 72)
    print("  P&L SUMMARY  —  Signal-Flip Trades")
    print("═" * 72)

    valid = df[df["trade_pnl_pct"].notna()].copy()
    if valid.empty:
        print("  No trades with price data.")
        return

    for flip_type in ("bull→bear", "bear→bull"):
        sub = valid[valid["flip_type"] == flip_type]
        if sub.empty:
            continue
        wins    = (sub["trade_pnl_pct"] > 0).sum()
        losses  = (sub["trade_pnl_pct"] <= 0).sum()
        avg_pnl = sub["trade_pnl_pct"].mean()
        med_pnl = sub["trade_pnl_pct"].median()
        best    = sub.loc[sub["trade_pnl_pct"].idxmax()]
        worst   = sub.loc[sub["trade_pnl_pct"].idxmin()]

        print(f"\n  {flip_type.upper()}  ({len(sub)} trades):")
        print(f"    Win rate     : {wins}/{len(sub)} = {wins/len(sub)*100:.0f}%")
        print(f"    Avg P&L      : {avg_pnl:+.1f}%")
        print(f"    Median P&L   : {med_pnl:+.1f}%")
        print(f"    Best trade   : {best['ticker']} {best['entry_date'].date()} → "
              f"{best['exit_date'].date()}  P&L={best['trade_pnl_pct']:+.1f}%")
        print(f"    Worst trade  : {worst['ticker']} {worst['entry_date'].date()} → "
              f"{worst['exit_date'].date()}  P&L={worst['trade_pnl_pct']:+.1f}%")

    print()
    all_valid = valid["trade_pnl_pct"]
    print(f"  COMBINED  ({len(valid)} trades):")
    print(f"    Win rate  : {(all_valid > 0).sum()}/{len(valid)} = "
          f"{(all_valid > 0).mean()*100:.0f}%")
    print(f"    Avg P&L   : {all_valid.mean():+.1f}%")
    print(f"    Median    : {all_valid.median():+.1f}%")
    print(f"    Total PnL : {all_valid.sum():+.1f}%  (sum of trade returns, not compounded)")


def report_signal_integrity(wl: pd.DataFrame) -> None:
    """How often does a signal repeat on the next date vs get dropped?"""
    print("\n" + "─" * 72)
    print("  SIGNAL CONTINUITY  (did the signal hold on the next snapshot?)")
    print("─" * 72)

    dates = sorted(wl["signal_date"].unique())
    lookup = {(r["ticker"], r["signal_date"]): r["side"]
              for _, r in wl.iterrows()}

    for side in ("bull", "bear"):
        held = 0
        flipped = 0
        dropped = 0
        total = 0

        for i in range(len(dates) - 1):
            d_now  = dates[i]
            d_next = dates[i + 1]
            tickers_now = wl[(wl["signal_date"] == d_now) & (wl["side"] == side)]["ticker"]
            for t in tickers_now:
                total += 1
                next_signal = lookup.get((t, d_next))
                if next_signal == side:
                    held += 1
                elif next_signal is not None:
                    flipped += 1
                else:
                    dropped += 1

        if total:
            print(f"\n  {side.upper()} signal (n={total} consecutive-pair observations):")
            print(f"    Held same signal next date : {held:>4}  ({held/total*100:.0f}%)")
            print(f"    Flipped to opposite        : {flipped:>4}  ({flipped/total*100:.0f}%)")
            print(f"    Dropped off watchlist      : {dropped:>4}  ({dropped/total*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="output/us_local",
                    help="Root dir containing date subdirs with watchlist CSVs")
    ap.add_argument("--model",   default="momentum",
                    choices=["momentum", "reversal"],
                    help="Model to analyse (default: momentum)")
    ap.add_argument("--scoring", default="composite",
                    choices=["composite", "pureml"],
                    help="Scoring variant (default: composite)")
    ap.add_argument("--delay",   default=0.4, type=float,
                    help="Seconds between yfinance calls (default: 0.4)")
    ap.add_argument("--no_prices", action="store_true",
                    help="Skip price fetching (just show watchlist structure)")
    ap.add_argument("--save", default="output/evaluation",
                    help="Directory to save CSV outputs (default: output/evaluation)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    save_dir   = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  SIGNAL FLIP ANALYSIS  |  {args.model.upper()} / {args.scoring.upper()}")
    print(f"  Source: {output_dir}")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading watchlists...")
    wl = load_watchlists(output_dir, args.model, args.scoring)
    print(f"  Loaded {len(wl)} records  |  {wl['ticker'].nunique()} unique tickers  "
          f"|  {wl['signal_date'].nunique()} dates")

    # ── Overview ──────────────────────────────────────────────────────────────
    report_overview(wl)
    report_persistence(wl)
    report_signal_integrity(wl)

    # ── Flips ─────────────────────────────────────────────────────────────────
    print("\n[2] Finding signal flips...")
    flips = find_flips(wl)
    print(f"  Found {len(flips)} flips  "
          f"({sum(1 for f in flips if f['flip_type']=='bull→bear')} bull→bear, "
          f"{sum(1 for f in flips if f['flip_type']=='bear→bull')} bear→bull)")

    if not flips:
        print("  No flips detected — all tickers maintained consistent signal.")
        sys.exit(0)

    if args.no_prices:
        flip_df = pd.DataFrame(flips)
        report_flips(flip_df)
    else:
        # ── Prices ────────────────────────────────────────────────────────────
        print("\n[3] Fetching actual prices for flip trades...")
        all_dates  = sorted({pd.Timestamp(f["entry_date"]) for f in flips} |
                            {pd.Timestamp(f["exit_date"])  for f in flips})
        all_tickers = sorted({f["ticker"] for f in flips})
        price_map  = fetch_prices(all_tickers, all_dates, delay=args.delay)
        print(f"  Got {len(price_map)} price points for {len(all_tickers)} tickers")

        # ── Enrich + Report ───────────────────────────────────────────────────
        flip_df = enrich_flips(flips, price_map)
        report_flips(flip_df)
        report_pnl_summary(flip_df)

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = save_dir / f"signal_flip_{args.model}_{args.scoring}.csv"
        flip_df.to_csv(out_path, index=False)
        print(f"\n[4] Saved flip detail → {out_path}")

    # ── Timeline table ────────────────────────────────────────────────────────
    print("\n[5] Building ticker signal timeline...")
    timeline = build_timeline(wl)
    tl_path = save_dir / f"signal_timeline_{args.model}_{args.scoring}.csv"
    timeline.to_csv(tl_path)
    print(f"  Saved timeline → {tl_path}")

    # Print a compact version for tickers that appeared on 3+ dates
    dates_cols = [c for c in timeline.columns]
    appear_counts = timeline.notna().sum(axis=1)
    frequent = timeline[appear_counts >= 3].copy()
    if not frequent.empty:
        print(f"\n  Tickers on 3+ dates ({len(frequent)} tickers):")
        print(f"\n  {'Ticker':<8}", end="")
        for c in dates_cols:
            print(f"  {c[5:]:>10}", end="")   # show MM-DD
        print()
        print("  " + "─" * (8 + 12 * len(dates_cols)))
        for ticker, row in frequent.iterrows():
            print(f"  {ticker:<8}", end="")
            for c in dates_cols:
                val = row[c]
                if val == "bull":
                    cell = "  ▲ BULL"
                elif val == "bear":
                    cell = "  ▼ BEAR"
                else:
                    cell = "       —"
                print(f"{cell:>12}", end="")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
