"""
Generate per-cap-tier top-10 watchlists from an existing scores_detail JSON.
Works for any market that has an Indices column (SPX/NDX/MID/SML) in its
constituent CSV.

Usage:
    python generate_tier_watchlists.py                    # auto-detect latest date
    python generate_tier_watchlists.py --date 2024-04-30  # specific date

Output files (written to output/us_local/):
    watchlist_momentum_bull_large_<date>.csv
    watchlist_momentum_bull_mid_<date>.csv
    watchlist_momentum_bull_small_<date>.csv
    watchlist_momentum_bear_large_<date>.csv
    ... (6 files per model × 2 models = 12 files total)
"""

import argparse
import json
from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
STOCK_LIST_CSV = Path(r"C:\Victor\Learning_charts\stock_lists\constituents_us_combined.csv")
OUTPUT_DIR     = Path("output/us_local")
TOP_N_TIER     = 10   # top stocks per cap tier

TIER_MAP_INDEX = {
    # Indices values → tier key
    "SPX":     "large",
    "NDX":     "large",
    "NDX|SPX": "large",
    "MID":     "mid",
    "SML":     "small",
}

TIER_LABELS = {
    "large": "Large Cap",
    "mid":   "Mid Cap",
    "small": "Small Cap",
}


def build_cap_tier_map(stock_list_csv: Path) -> dict[str, str]:
    df = pd.read_csv(stock_list_csv)
    cap_tier: dict[str, str] = {}
    if "Indices" not in df.columns:
        print("WARNING: No 'Indices' column found — cannot build cap-tier map")
        return cap_tier
    for sym, idx in zip(df["Symbol"].str.strip(), df["Indices"].fillna("")):
        idx_s = str(idx).strip()
        tier = TIER_MAP_INDEX.get(idx_s)
        if tier:
            cap_tier[str(sym).strip()] = tier
    counts = {t: sum(1 for v in cap_tier.values() if v == t)
              for t in ("large", "mid", "small")}
    print(f"Cap tier map: {counts['large']} large, {counts['mid']} mid, {counts['small']} small")
    return cap_tier


def load_scores_detail(output_dir: Path, model: str, date_str: str) -> dict:
    path = output_dir / f"scores_detail_{model}_{date_str}.json"
    if not path.exists():
        print(f"  Not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def generate_tier_files(output_dir: Path, date_str: str,
                        cap_tier_map: dict[str, str]) -> None:
    MODELS = ["momentum", "reversal"]
    SIDES  = ["bull", "bear"]

    for model in MODELS:
        scores = load_scores_detail(output_dir, model, date_str)
        if not scores:
            continue
        print(f"\n  Model: {model}  ({len(scores)} tickers in scores_detail)")

        for side in SIDES:
            side_key = "bull_rank" if side == "bull" else "bear_rank"

            # Rank all tickers by their rank (ascending = rank 1 is best)
            ranked = sorted(
                [(t, d.get(side_key, 9999)) for t, d in scores.items()],
                key=lambda x: x[1]
            )

            for tier_key, tier_label in TIER_LABELS.items():
                tier_tickers = [t for t, _ in ranked
                                if cap_tier_map.get(t) == tier_key][:TOP_N_TIER]
                if not tier_tickers:
                    print(f"    {tier_label} {side.upper()}: no tickers found")
                    continue

                rows = []
                for rank_pos, ticker in enumerate(tier_tickers, 1):
                    d     = scores[ticker]
                    side_d = d.get(side, {})
                    rows.append({
                        "rank":             rank_pos,
                        "side":             side.upper(),
                        "ticker":           ticker,
                        "cap_tier":         tier_label,
                        "score":            round(side_d.get("model_score", 0.0), 4),
                        "composite_score":  round(side_d.get("composite_score", 0.0), 4),
                        "rank_in_universe": side_d.get("rank_in_universe", 0),
                        "universe_size":    side_d.get("universe_size", 0),
                        "date":             date_str,
                    })

                tier_df = pd.DataFrame(rows)
                tier_path = output_dir / f"watchlist_{model}_{side}_{tier_key}_{date_str}.csv"
                tier_df.to_csv(tier_path, index=False)
                print(f"    {tier_label} {side.upper()} top-{TOP_N_TIER}: {tier_path.name}")


def detect_latest_date(output_dir: Path) -> str | None:
    files = sorted(
        output_dir.glob("scores_detail_momentum_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None
    # Extract date from filename: scores_detail_momentum_YYYY-MM-DD.json
    stem = files[0].stem  # e.g. scores_detail_momentum_2024-04-30
    return stem.replace("scores_detail_momentum_", "")


def main():
    parser = argparse.ArgumentParser(description="Generate per-cap-tier watchlists from scores_detail JSON")
    parser.add_argument("--date",       default=None, help="Scoring date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    date_str = args.date or detect_latest_date(out)
    if not date_str:
        print("ERROR: No scores_detail files found. Run the scoring pipeline first.")
        return

    print(f"Generating cap-tier watchlists for date: {date_str}")
    print(f"Output dir: {out}")

    cap_tier_map = build_cap_tier_map(STOCK_LIST_CSV)
    if not cap_tier_map:
        print("ERROR: Empty cap_tier_map — cannot proceed")
        return

    generate_tier_files(out, date_str, cap_tier_map)
    print("\nDone.")


if __name__ == "__main__":
    main()
