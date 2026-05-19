"""
rescore_watchlist.py
---------------------
Re-ranks existing scores_detail JSON using two alternate weighting schemes
and produces a comparison Excel file. No retraining required.

  List A — Pure ML    : ranked by model_score only     (composite ignored)
  List B — Pure Comp  : ranked by composite_score only (ML ignored)
  List C — Blended    : current 70/30 (original run)
  List D — 85/15      : model-heavy blend

Usage:
    python rescore_watchlist.py --market sp500
    python rescore_watchlist.py --market nse
    python rescore_watchlist.py --market sp500 --date 2024-03-01 --top 50
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

MARKET_CONFIG = {
    "sp500": {
        "output_dir": Path(r"C:\Victor\Project\ml-stock-predictor\output\us_local"),
        "label":      "SP500 + NASDAQ",
    },
    "nse": {
        "output_dir": Path(r"C:\Victor\Project\ml-stock-predictor\output\nse_local"),
        "label":      "NSE",
    },
}

EVAL_DIR = Path(r"C:\Victor\Project\ml-stock-predictor\output\evaluation")

SCHEMES = {
    "Pure ML (model only)":        (1.00, 0.00),
    "Pure Composite (signal only)":(0.00, 1.00),
    "Blended 85/15":               (0.85, 0.15),
    "Original 70/30":              (0.70, 0.30),
}


def load_scores(output_dir: Path, date_str: str) -> dict[str, dict]:
    """Load both momentum + reversal scores_detail JSONs."""
    merged = {}
    for model in ["momentum", "reversal"]:
        f = output_dir / f"scores_detail_{model}_{date_str}.json"
        if not f.exists():
            f = output_dir / f"scores_detail_{date_str}.json"
        if not f.exists():
            print(f"  WARNING: {f.name} not found — skipping")
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        merged.setdefault(model, data)
        print(f"  Loaded {len(data):,} tickers from {f.name}")
    return merged


def rank_by_scheme(
    model_data: dict,
    side: str,
    model_wt: float,
    comp_wt: float,
    top_n: int,
) -> pd.DataFrame:
    """Re-rank a model's side using the given weights."""
    rows = []
    for ticker, info in model_data.items():
        side_data = info.get(side, {})
        if not side_data:
            continue
        m_sc = float(side_data.get("model_score",     0))
        c_sc = float(side_data.get("composite_score", 0))
        final = m_sc * model_wt + c_sc * comp_wt
        rows.append({
            "ticker":          ticker,
            "final_score":     round(final, 4),
            "model_score":     round(m_sc,  4),
            "composite_score": round(c_sc,  4),
            "orig_rank":       info.get(f"{side}_rank",
                                        side_data.get("rank_in_universe", None)),
            "in_orig_wl":      info.get(f"in_{side}_watchlist", False),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.insert(0, "new_rank", df.index + 1)
    return df.head(top_n)


def run(market: str, date_str: str, top_n: int):
    cfg        = MARKET_CONFIG[market]
    output_dir = cfg["output_dir"]

    print(f"\n{'='*60}")
    print(f"  Rescore Watchlist — {cfg['label']}")
    print(f"  Date : {date_str}  |  Top N : {top_n}")
    print(f"{'='*60}")

    all_scores = load_scores(output_dir, date_str)
    if not all_scores:
        print("ERROR: No scores_detail files found.")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EVAL_DIR / f"rescore_{market}_{date_str}.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:

        # ── Per model × side × scheme ────────────────────────────────────────
        for model_name, model_data in all_scores.items():
            for side in ["bull", "bear"]:
                # Build comparison sheet: side-by-side columns per scheme
                combo_frames = {}
                for scheme_name, (mw, cw) in SCHEMES.items():
                    df = rank_by_scheme(model_data, side, mw, cw, top_n)
                    df = df.rename(columns={
                        "new_rank":        f"rank",
                        "ticker":          f"ticker",
                        "final_score":     f"score",
                        "model_score":     f"model_sc",
                        "composite_score": f"comp_sc",
                        "orig_rank":       f"orig_rank",
                        "in_orig_wl":      f"was_selected",
                    })
                    combo_frames[scheme_name] = df

                # Individual scheme sheets
                for scheme_name, df in combo_frames.items():
                    short = (scheme_name.replace("Pure ", "")
                                        .replace(" (model only)", "")
                                        .replace(" (signal only)", "")
                                        .replace("Blended ", "Blend_")
                                        .replace("Original ", "Orig_")
                                        .replace("/", "-"))
                    sheet = f"{model_name[:3].title()}_{side[:2].title()}_{short[:10]}"
                    df.to_excel(writer, sheet_name=sheet, index=False)

                # ── Combined comparison sheet ────────────────────────────────
                # Show all 4 scheme tickers side by side
                max_rows = top_n
                comp_data = {}
                for scheme_name, df in combo_frames.items():
                    short = scheme_name.split("(")[0].strip()
                    comp_data[f"{short} ticker"] = df["ticker"].tolist()
                    comp_data[f"{short} score"]  = df["score"].tolist()
                    comp_data[f"{short} was_wl"] = df["was_selected"].tolist()

                df_comp = pd.DataFrame(comp_data)

                # Add overlap column: how many schemes agree on this ticker
                pure_ml_tickers   = set(combo_frames["Pure ML (model only)"]["ticker"])
                pure_comp_tickers = set(combo_frames["Pure Composite (signal only)"]["ticker"])
                blend_8515        = set(combo_frames["Blended 85/15"]["ticker"])
                orig_7030         = set(combo_frames["Original 70/30"]["ticker"])

                sheet = f"{model_name[:3].title()}_{side[:2].title()}_Compare"
                df_comp.to_excel(writer, sheet_name=sheet, index=False)

                # ── Overlap analysis ─────────────────────────────────────────
                overlap_rows = []
                all_tickers = pure_ml_tickers | pure_comp_tickers | blend_8515 | orig_7030
                for t in sorted(all_tickers):
                    in_ml   = t in pure_ml_tickers
                    in_comp = t in pure_comp_tickers
                    in_8515 = t in blend_8515
                    in_7030 = t in orig_7030
                    count   = sum([in_ml, in_comp, in_8515, in_7030])

                    # get model score
                    info = model_data.get(t, {})
                    sd   = info.get(side, {})
                    m_sc = round(float(sd.get("model_score",     0)), 4) if sd else 0
                    c_sc = round(float(sd.get("composite_score", 0)), 4) if sd else 0

                    overlap_rows.append({
                        "ticker":        t,
                        "model_score":   m_sc,
                        "comp_score":    c_sc,
                        "in_PureML":     "Y" if in_ml   else "",
                        "in_PureComp":   "Y" if in_comp else "",
                        "in_85-15":      "Y" if in_8515 else "",
                        "in_Orig70-30":  "Y" if in_7030 else "",
                        "schemes_agree": count,
                    })

                df_ov = (pd.DataFrame(overlap_rows)
                           .sort_values(["schemes_agree","model_score"], ascending=[False, False]))
                sheet_ov = f"{model_name[:3].title()}_{side[:2].title()}_Overlap"
                df_ov.to_excel(writer, sheet_name=sheet_ov, index=False)

                # Print summary to console
                print(f"\n  {model_name.upper()} {side.upper()} — top {top_n} per scheme")
                print(f"  {'Scheme':<30}  {'Unique tickers':>15}  {'Also in Orig 70/30':>20}")
                for sn, df in combo_frames.items():
                    tickers = set(df["ticker"])
                    overlap = len(tickers & orig_7030)
                    new     = len(tickers - orig_7030)
                    print(f"  {sn:<30}  {len(tickers):>6} total  "
                          f"{overlap:>6} same / {new:>4} new vs orig")

        # ── Summary sheet ────────────────────────────────────────────────────
        print(f"\n  Saved: {out_file}")

    print(f"\nDone. Open: {out_file}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="sp500", choices=["sp500","nse"])
    p.add_argument("--date",   default=None,
                   help="Scoring date (default: auto-detect from latest scores_detail file)")
    p.add_argument("--top",    type=int, default=50,
                   help="Top N stocks per list (default: 50)")
    return p.parse_args()


def detect_date(output_dir: Path) -> str | None:
    files = sorted(output_dir.glob("scores_detail_momentum_*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        return files[0].stem.replace("scores_detail_momentum_", "")
    return None


if __name__ == "__main__":
    args = parse_args()
    cfg  = MARKET_CONFIG[args.market]
    date = args.date or detect_date(cfg["output_dir"])
    if not date:
        print("ERROR: Could not detect date. Use --date YYYY-MM-DD")
        exit(1)
    print(f"Using date: {date}")
    run(args.market, date, args.top)
