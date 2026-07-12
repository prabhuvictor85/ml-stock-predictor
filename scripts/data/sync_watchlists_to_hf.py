"""
sync_watchlists_to_hf.py — Push ML watchlists into the ml-stock-dashboard HF Space.

Reads output/{market}/watchlist_*.csv for the latest date of each market,
converts CSVs to JSON, builds a manifest.json, and uploads everything to the
Space repo's public/data/ folder so the React frontend can fetch them.

Usage
─────
    # Default: push the latest date for all 3 markets
    python scripts/data/sync_watchlists_to_hf.py

    # Push for a specific date (use for backtest snapshots)
    python scripts/data/sync_watchlists_to_hf.py --date 2024-03-01

    # Subset markets
    python scripts/data/sync_watchlists_to_hf.py --markets nse_tradingv,sp500

    # Dry run — print what would be uploaded
    python scripts/data/sync_watchlists_to_hf.py --dry-run

    # Override Space repo
    python scripts/data/sync_watchlists_to_hf.py --space-id your_user/your_space

Prerequisites
─────────────
1. pip install huggingface_hub
2. huggingface-cli login   (paste a write-scope token)
3. The target HF Space must exist (Settings → Visibility = Private recommended)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration — edit if your output/ folder names differ
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from local output/ subfolder → (manifest key, human label, available tiers)
MARKETS: Dict[str, Dict] = {
    "nse_local": {
        "label": "NSE (India)",
        "tiers": ["all", "large", "mid", "small", "micro"],
    },
    "us_local": {
        "label": "S&P 500 + NASDAQ",
        "tiers": ["all", "large", "mid", "small"],
    },
    "nse_tradingv": {
        "label": "NSE TradingView",
        "tiers": ["all", "large", "mid", "small", "micro"],
    },
}

MODES = ["momentum", "reversal"]
SIDES = ["bull", "bear"]
DEFAULT_SPACE_ID = "prabhuvictor85/ml-stock-dashboard"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"

# Filename patterns:
#   watchlist_<mode>_<side>_<date>.csv                       (main)
#   watchlist_<mode>_<side>_<tier>_<date>.csv                (per-tier)
#   watchlist_<mode>_<type>_<side>_<date>.csv                (with type e.g. composite/pureml)
#   watchlist_<mode>_<type>_<side>_<tier>_<date>.csv         (with type + tier)
_WL_RE = re.compile(
    r"^watchlist_(?P<mode>momentum|reversal)"
    r"(?:_(?P<model_type>composite|pureml|combined))?"   # optional type segment
    r"_(?P<side>bull|bear)"
    r"(?:_(?P<tier>large|mid|small|micro))?_(?P<date>\d{4}-\d{2}-\d{2})\.csv$"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_dates(market_dir: Path) -> List[str]:
    """Return all unique dates found in watchlist filenames, sorted descending.

    Handles two layouts:
      - Flat:  output/<market>/watchlist_*_<date>.csv
      - Nested: output/<market>/<date>/watchlist_*_<date>.csv
    """
    if not market_dir.is_dir():
        return []
    dates = set()
    # Flat files directly in market_dir
    for f in market_dir.glob("watchlist_*.csv"):
        m = _WL_RE.match(f.name)
        if m:
            dates.add(m.group("date"))
    # Files nested inside date sub-folders
    for sub in market_dir.iterdir():
        if sub.is_dir():
            for f in sub.glob("watchlist_*.csv"):
                m = _WL_RE.match(f.name)
                if m:
                    dates.add(m.group("date"))
    return sorted(dates, reverse=True)


def list_files_for_date(market_dir: Path, date: str) -> List[Path]:
    """Return all watchlist CSVs for a given date.

    Searches both flat (market_dir/) and nested (market_dir/<date>/) layouts.
    """
    out: List[Path] = []
    # Flat layout
    for f in market_dir.glob(f"watchlist_*_{date}.csv"):
        m = _WL_RE.match(f.name)
        if m and m.group("date") == date:
            out.append(f)
    # Nested layout — look inside the matching date subfolder
    date_sub = market_dir / date
    if date_sub.is_dir():
        for f in date_sub.glob(f"watchlist_*_{date}.csv"):
            m = _WL_RE.match(f.name)
            if m and m.group("date") == date:
                out.append(f)
    return sorted(set(out))


# ─────────────────────────────────────────────────────────────────────────────
#  CSV → JSON conversion
# ─────────────────────────────────────────────────────────────────────────────

def csv_to_records(csv_path: Path) -> List[dict]:
    """Read CSV and return a list-of-dicts. NaN → None for JSON safety."""
    df = pd.read_csv(csv_path)
    # Replace NaN/Inf with None so json.dumps is happy
    df = df.where(pd.notna(df), None)
    return df.to_dict(orient="records")


def out_filename(side: str, tier: Optional[str]) -> str:
    """Frontend-expected filename. e.g. bull.json or bull_large.json
    All modes (momentum/reversal) and types (composite/pureml) are merged
    into one file per side+tier, tagged with 'mode' and 'model_type' columns.
    """
    suffix = f"_{tier}" if tier else ""
    return f"{side}{suffix}.json"


# ─────────────────────────────────────────────────────────────────────────────
#  Build staging tree
# ─────────────────────────────────────────────────────────────────────────────

def build_staging(target_markets: List[str], target_date: Optional[str],
                   stage_dir: Path, all_dates: bool = False) -> dict:
    """
    Populate stage_dir with the file tree the frontend expects:
      stage_dir/manifest.json
      stage_dir/<market>/<date>/<mode>_<side>[_<tier>].json

    When all_dates=True every available date is staged; otherwise only
    target_date (or latest) is staged, but all date labels are recorded
    in the manifest so the frontend can show the full history.

    Returns the manifest dict.
    """
    manifest_markets: List[dict] = []

    for market_key in target_markets:
        if market_key not in MARKETS:
            print(f"  [WARN] Unknown market '{market_key}' - skipping")
            continue

        market_dir = OUTPUT_DIR / market_key
        meta = MARKETS[market_key]

        dates_available = discover_dates(market_dir)
        if not dates_available:
            print(f"  [WARN] No watchlists found in {market_dir} - skipping")
            continue

        # Which dates to stage
        if all_dates:
            dates_to_stage = dates_available          # all dates, newest first
        elif target_date and target_date in dates_available:
            dates_to_stage = [target_date]
        elif target_date:
            print(f"  [WARN] {market_key}: requested date {target_date} not found. "
                  f"Available: {dates_available[:3]} ...")
            continue
        else:
            dates_to_stage = [dates_available[0]]     # latest only

        tiers_found: set  = {"all"}
        modes_found: set  = set()
        sides_found: set  = set()
        total_written = 0

        for date in dates_to_stage:
            files = list_files_for_date(market_dir, date)
            if not files:
                print(f"  [WARN] {market_key} @ {date}: no files matched - skipping")
                continue

            out_market_dir = stage_dir / market_key / date
            out_market_dir.mkdir(parents=True, exist_ok=True)
            n_written = 0

            # Group files by (side, tier) so we can merge all modes/types
            # into one JSON file per side+tier, tagged with mode & model_type.
            from collections import defaultdict
            groups: dict = defaultdict(list)   # (side, tier) -> list of (mode, model_type, Path)
            for f in files:
                m = _WL_RE.match(f.name)
                if not m:
                    continue
                mode       = m.group("mode")
                model_type = m.group("model_type") or "default"
                side       = m.group("side")
                tier       = m.group("tier")   # None for the overall file
                modes_found.add(mode)
                sides_found.add(side)
                if tier:
                    tiers_found.add(tier)
                groups[(side, tier)].append((mode, model_type, f))

            for (side, tier), entries in groups.items():
                # ── Dedup: one row per (ticker, mode) ─────────────────────
                # Within each mode, a ticker may appear in both composite
                # and pureml.  We collapse those into one row and set
                # model_type = "PureML", "Composite", or "PureML + Composite".
                from collections import defaultdict as _dd

                # mode -> ticker -> { model_type -> record }
                mode_ticker_map: dict = _dd(lambda: _dd(dict))

                for mode, model_type, f in entries:
                    records = csv_to_records(f)
                    label = (
                        "PureML"    if model_type == "pureml"    else
                        "Composite" if model_type == "composite" else
                        model_type.title()
                    )
                    for rec in records:
                        tkr = str(rec.get("ticker") or rec.get("Ticker") or "")
                        if not tkr:
                            continue
                        rec["mode"] = mode
                        # Keep the record with the higher score for this type
                        existing = mode_ticker_map[mode][tkr].get(label)
                        if existing is None or (rec.get("score") or 0) > (existing.get("score") or 0):
                            mode_ticker_map[mode][tkr][label] = rec

                merged: List[dict] = []
                for mode, ticker_map in mode_ticker_map.items():
                    for tkr, type_records in ticker_map.items():
                        types = sorted(type_records.keys())          # e.g. ["Composite","PureML"]
                        label = " + ".join(types)                    # "Composite", "PureML", or "Composite + PureML"
                        # Use the record with the highest score as the base row
                        best = max(type_records.values(),
                                   key=lambda r: r.get("score") or 0)
                        row = dict(best)
                        row["model_type"] = label
                        merged.append(row)

                # Sort: by mode (momentum first) then score descending,
                # then reassign rank sequentially within each mode so rank
                # numbers are clean (1, 2, 3 …) after dedup.
                merged.sort(key=lambda r: (r.get("mode", ""), -(r.get("score") or 0)))
                mode_counters: dict = {}
                for row in merged:
                    m = row.get("mode", "")
                    mode_counters[m] = mode_counters.get(m, 0) + 1
                    row["rank"] = mode_counters[m]

                out_name = out_filename(side, tier)
                (out_market_dir / out_name).write_text(
                    json.dumps(merged, indent=None), encoding="utf-8"
                )
                n_written += 1
                total_written += 1

            print(f"  OK   {market_key} @ {date}: {n_written} files staged")

        if total_written == 0:
            continue

        manifest_markets.append({
            "key":         market_key,
            "label":       meta["label"],
            "latest_date": dates_to_stage[0],         # newest date staged
            "dates":       dates_to_stage,            # full list for date picker
            "modes":       sorted(modes_found),
            "sides":       sorted(sides_found),
            "tiers":       [t for t in meta["tiers"] if t in tiers_found],
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "markets":      manifest_markets,
    }
    (stage_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
#  Upload
# ─────────────────────────────────────────────────────────────────────────────

def upload(stage_dir: Path, space_id: str, commit_msg: str) -> str:
    """Upload stage_dir contents to <space>/public/data/. Returns commit URL."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )

    api = HfApi()
    # Confirm the user is logged in
    try:
        whoami = api.whoami()
        print(f"  HF user: {whoami['name']}")
    except Exception as e:
        raise SystemExit(
            f"Not logged in to HF: {e}\n"
            "Run: huggingface-cli login  (paste write-scope token)"
        )

    print(f"  Uploading to space '{space_id}' under public/data/ ...")
    info = api.upload_folder(
        folder_path=str(stage_dir),
        path_in_repo="public/data",
        repo_id=space_id,
        repo_type="space",
        commit_message=commit_msg,
    )
    return getattr(info, "commit_url", "(no commit url returned)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync ML watchlists to HF Space")
    p.add_argument(
        "--space-id", default=DEFAULT_SPACE_ID,
        help=f"HF Space repo id (default: {DEFAULT_SPACE_ID})",
    )
    p.add_argument(
        "--markets", default=",".join(MARKETS.keys()),
        help="Comma-separated market keys (default: all configured markets)",
    )
    p.add_argument(
        "--date", default=None,
        help="Specific date YYYY-MM-DD (default: latest per market)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build the staging tree but don't upload",
    )
    p.add_argument(
        "--all-dates", action="store_true",
        help="Stage and upload every available date (not just latest). "
             "Recommended for the initial push or after backfill runs.",
    )
    p.add_argument(
        "--keep-staging", action="store_true",
        help="Keep the staging directory after the run (for inspection)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    target_markets = [m.strip() for m in args.markets.split(",") if m.strip()]

    print("=" * 66)
    print("  sync_watchlists_to_hf.py")
    print(f"  Source     : {OUTPUT_DIR}")
    print(f"  Markets    : {target_markets}")
    print(f"  Date       : {args.date or 'latest per market'}")
    print(f"  Space      : {args.space_id}")
    print(f"  All dates  : {args.all_dates}")
    print(f"  Dry run    : {args.dry_run}")
    print("=" * 66)

    if not OUTPUT_DIR.is_dir():
        print(f"✗ Output directory not found: {OUTPUT_DIR}")
        return 1

    stage_root = Path(tempfile.mkdtemp(prefix="ml_wl_sync_"))
    try:
        print(f"\n[1/2] Building staging tree at {stage_root}")
        manifest = build_staging(target_markets, args.date, stage_root,
                                 all_dates=args.all_dates)

        if not manifest["markets"]:
            print("\n✗ No markets with usable data — nothing to upload.")
            return 1

        print("\n  Manifest summary:")
        for m in manifest["markets"]:
            print(f"    - {m['label']:<22} key={m['key']:<14} date={m['latest_date']}"
                  f"  tiers={m['tiers']}")

        if args.dry_run:
            print("\n[2/2] DRY RUN — skipping upload.")
            print(f"  Staging dir kept at: {stage_root}")
            args.keep_staging = True  # force keep
            return 0

        print(f"\n[2/2] Uploading to HF Space...")
        commit_msg = (
            f"watchlists: {len(manifest['markets'])} markets, "
            f"dates={','.join(sorted({m['latest_date'] for m in manifest['markets']}))}"
        )
        commit_url = upload(stage_root, args.space_id, commit_msg)
        print(f"\nUploaded OK. {commit_url}")
        print(f"  Space rebuild will start automatically. Check:")
        print(f"  https://huggingface.co/spaces/{args.space_id}")
        return 0
    finally:
        if args.keep_staging:
            print(f"\n  (Staging dir kept at {stage_root})")
        else:
            shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
