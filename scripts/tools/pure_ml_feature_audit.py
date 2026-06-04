#!/usr/bin/env python
"""
pure_ml_feature_audit.py — audit which features actually drive the PURE ML model.

The explanations_<date>.json files store, per watchlist pick, the top SHAP
attributions taken from shap.TreeExplainer run on the LGBM ranker over X_inf.
Those SHAP values explain `model_score` (the pure ML ranking) and are completely
independent of the 0.15 composite overlay / hand-coded signal_weights.

So aggregating them answers: "what does the PURE ML model lean on, and is
per-timeframe ICT actually picked?"

Usage
-----
    python scripts/tools/pure_ml_feature_audit.py
    python scripts/tools/pure_ml_feature_audit.py --output_dir output/us_local
    python scripts/tools/pure_ml_feature_audit.py --top 25 --side bull
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

# UTF-8 console (Windows cp1252 safety)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_TF_RE = re.compile(r"_(1d|1wk|1mo|3mo|1y)$")


def classify(raw: str) -> str:
    """Bucket a feature name into a pure-ML driver category."""
    f = raw.replace("features_", "")
    if f.startswith("ict_") and "htf" in f:
        return "ICT_composite"      # ict_bull_htf_score / ict_bear_htf_score
    if f.startswith("ict_"):
        return "ICT_perTF"          # every other ICT feature (per-timeframe flags)
    if re.match(r"(sdz|ssz|dz|sz)_", f) or "zone" in f:
        return "ZONE"
    if "sector" in f or "etf" in f:
        return "SECTOR"
    return "TECH/PRICE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default="output/us_local",
                    help="dir containing <date>/explanations_<date>.json (default: output/us_local)")
    ap.add_argument("--side", choices=["bull", "bear", "both"], default="both",
                    help="restrict to one side (default: both)")
    ap.add_argument("--top", type=int, default=20, help="top-N drivers to list (default: 20)")
    args = ap.parse_args()

    pattern = os.path.join(args.output_dir, "*", "explanations_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # fall back to flat layout
        files = sorted(glob.glob(os.path.join(args.output_dir, "explanations_*.json")))
    if not files:
        print(f"No explanations_*.json found under {args.output_dir!r}")
        sys.exit(1)

    cat = Counter()
    feat = Counter()
    ict_pertf = Counter()
    ict_comp = Counter()
    n_picks = 0

    for path in files:
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            continue
        for pick in data:
            if args.side != "both" and pick.get("side") != args.side:
                continue
            n_picks += 1
            feats = ([f for f, _ in pick.get("top_positive_features", [])] +
                     [f for f, _ in pick.get("top_negative_features", [])])
            for f in feats:
                c = classify(f)
                cat[c] += 1
                feat[f] += 1
                if c == "ICT_perTF":
                    ict_pertf[f.replace("features_", "")] += 1
                elif c == "ICT_composite":
                    ict_comp[f.replace("features_", "")] += 1

    tot = sum(cat.values())
    if tot == 0:
        print("No feature mentions found (check --side filter).")
        sys.exit(1)

    print(f"Files: {len(files)}   Picks: {n_picks}   SHAP feature mentions: {tot}   "
          f"Side: {args.side}")
    print()
    print("=== Category share of PURE-ML SHAP drivers ===")
    for c, n in cat.most_common():
        print(f"  {c:15s} {n:6d}  {100*n/tot:5.1f}%")
    print()

    print("=== Per-timeframe ICT (pure ML) ===")
    if ict_pertf:
        for f, n in ict_pertf.most_common():
            print(f"  {f:35s} {n}")
    else:
        print("  NONE")
    s = sum(ict_pertf.values())
    print(f"  per-TF ICT total: {s} ({100*s/tot:.1f}%)")
    print()

    print("=== Composite ICT (pure ML) ===")
    for f, n in ict_comp.most_common():
        print(f"  {f:35s} {n}")
    s = sum(ict_comp.values())
    print(f"  composite ICT total: {s} ({100*s/tot:.1f}%)")
    print()

    print(f"=== Top {args.top} pure-ML drivers overall ===")
    for f, n in feat.most_common(args.top):
        print(f"  {f:38s} {n:5d}  [{classify(f)}]")


if __name__ == "__main__":
    main()
