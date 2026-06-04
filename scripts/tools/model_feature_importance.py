#!/usr/bin/env python
"""
model_feature_importance.py — read the ACTUAL pure-ML feature drivers straight
from the trained model artefact on the server (no inference of saved SHAP picks).

Two views:
  1. Native LightGBM importance (gain + split) — needs ONLY the .pkl, no data.
  2. True SHAP (mean |SHAP|) — needs a feature matrix; pass --features <parquet/csv>.

Both are bucketed into ZONE / TECH-PRICE / ICT_composite / ICT_perTF / SECTOR so
you can see exactly how much the pure ML model leans on per-timeframe ICT.

Artefact layout expected (matches pipeline/train.py + pipeline/infer.py):
    <artefacts_dir>/<market>/<model>/lgbm_ranker.pkl   (per-model LGBMRanker)
    <artefacts_dir>/<market>/<model>/ensemble.pkl      (EnsembleRanker, has .lgbm)
    <artefacts_dir>/<market>/ensemble.pkl              (flat fallback)
    <artefacts_dir>/<market>/lgbm_ranker.pkl           (flat fallback)
  where <model> is e.g. momentum / reversal.

Usage
-----
    # both momentum + reversal, native importance only (no data needed)
    python scripts/tools/model_feature_importance.py \
        --artefacts_dir /mnt/data/artefacts --market us_local

    # one model
    python scripts/tools/model_feature_importance.py \
        --artefacts_dir /mnt/data/artefacts --market us_local --models momentum

    # add true SHAP from a feature snapshot
    python scripts/tools/model_feature_importance.py \
        --artefacts_dir /mnt/data/artefacts --market us_local \
        --features data/panel/panel_features.parquet --sample 2000

    # point straight at specific pickles
    python scripts/tools/model_feature_importance.py \
        --model_pkl /mnt/data/artefacts/us_local/momentum/lgbm_ranker.pkl \
                    /mnt/data/artefacts/us_local/reversal/lgbm_ranker.pkl
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path so pickle can resolve pipeline.* classes
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

# UTF-8 console (Windows cp1252 safety)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── feature bucketing ────────────────────────────────────────────────────────
def classify(raw: str) -> str:
    f = raw.replace("features_", "")
    if f.startswith("ict_") and "htf" in f:
        return "ICT_composite"      # ict_bull_htf_score / ict_bear_htf_score
    if f.startswith("ict_"):
        return "ICT_perTF"          # per-timeframe ICT flags
    if re.match(r"(sdz|ssz|dz|sz)_", f) or "zone" in f:
        return "ZONE"
    if "sector" in f or "etf" in f:
        return "SECTOR"
    return "TECH/PRICE"


# ── model loading ────────────────────────────────────────────────────────────
def _unwrap_lgbm(obj):
    """Return an LGBMRanker-like object that exposes .model_ (lgb.Booster) and
    .feature_names_, from whatever was pickled."""
    # EnsembleRanker -> .lgbm
    if hasattr(obj, "lgbm"):
        obj = obj.lgbm
    if hasattr(obj, "model_") and obj.model_ is not None:
        return obj
    raise TypeError(f"Could not find a trained booster on {type(obj).__name__}. "
                    "Expected an LGBMRanker or EnsembleRanker.")


def _load_one(candidates):
    """Return (lgbm, path) for the first existing candidate, else (None, tried)."""
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return _unwrap_lgbm(obj), path
    return None, candidates


def resolve_models(args):
    """
    Yield (label, lgbm, path) for every model to report on.

    Priority:
      1. explicit --model_pkl paths (one or more)
      2. per-model subdirs: <artefacts_dir>/<market>/<model>/{lgbm_ranker,ensemble}.pkl
      3. flat fallback:       <artefacts_dir>/<market>/{ensemble,lgbm_ranker}.pkl
    """
    if args.model_pkl:
        for raw in args.model_pkl:
            p = Path(raw)
            lgbm, used = _load_one([p])
            if lgbm is None:
                print(f"  WARNING: not found, skipping: {p}")
                continue
            label = p.parent.name or p.stem
            yield label, lgbm, used
        return

    base = Path(args.artefacts_dir) / args.market
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    found_any = False
    for m in models:
        mdir = base / m
        lgbm, used = _load_one([mdir / "lgbm_ranker.pkl", mdir / "ensemble.pkl"])
        if lgbm is not None:
            found_any = True
            yield m, lgbm, used

    if not found_any:
        # flat layout (single un-split model)
        lgbm, used = _load_one([base / "ensemble.pkl", base / "lgbm_ranker.pkl"])
        if lgbm is not None:
            yield args.market, lgbm, used
        else:
            raise FileNotFoundError(
                f"No model artefact found under {base} "
                f"(tried {models} subdirs and flat ensemble/lgbm_ranker.pkl).")


# ── reporting ────────────────────────────────────────────────────────────────
def report_importance(title: str, series: pd.Series, top: int) -> None:
    total = float(series.sum()) or 1.0
    cats: dict[str, float] = {}
    for feat, val in series.items():
        cats[classify(feat)] = cats.get(classify(feat), 0.0) + float(val)
    print(f"\n=== {title} — category share ===")
    for c, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:15s} {v:14.2f}  {100*v/total:5.1f}%")

    # per-TF ICT breakout
    pertf = series[[f for f in series.index if classify(f) == "ICT_perTF"]]
    print(f"\n=== {title} — per-timeframe ICT ===")
    if len(pertf) and pertf.sum() > 0:
        for f, v in pertf.sort_values(ascending=False).items():
            if v > 0:
                print(f"  {f.replace('features_',''):35s} {float(v):12.2f}")
        print(f"  per-TF ICT share: {100*float(pertf.sum())/total:.1f}%")
    else:
        print("  (none with non-zero importance)")

    print(f"\n=== {title} — top {top} features ===")
    for f, v in series.sort_values(ascending=False).head(top).items():
        print(f"  {f:40s} {float(v):14.2f}  [{classify(f)}]")


def compute_shap(lgbm, features_path: str, sample: int) -> pd.Series | None:
    try:
        import shap
    except ImportError:
        print("\n[SHAP skipped] `pip install shap` to enable true SHAP view.")
        return None

    p = Path(features_path)
    if not p.exists():
        print(f"\n[SHAP skipped] features file not found: {p}")
        return None
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    feats = list(lgbm.feature_names_)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"\n[SHAP skipped] features file missing {len(missing)} model columns "
              f"(e.g. {missing[:3]}). Pass the panel that was used for inference.")
        return None

    X = df[feats]
    if sample and len(X) > sample:
        X = X.sample(sample, random_state=42)
    X = X.fillna(0)
    print(f"\nComputing SHAP on {len(X)} rows x {len(feats)} features ...")
    explainer = shap.TreeExplainer(lgbm.model_)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.Series(mean_abs, index=feats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefacts_dir", default="artefacts",
                    help="root artefacts dir (default: artefacts)")
    ap.add_argument("--market", default="us_local",
                    help="market subdir under artefacts_dir (default: us_local)")
    ap.add_argument("--models", default="momentum,reversal",
                    help="comma-separated model subdirs to report (default: momentum,reversal)")
    ap.add_argument("--model_pkl", nargs="+", default=None,
                    help="explicit path(s) to lgbm_ranker.pkl/ensemble.pkl "
                         "(overrides --artefacts_dir/--market/--models). Accepts multiple.")
    ap.add_argument("--features", default=None,
                    help="optional parquet/csv of features to compute TRUE SHAP")
    ap.add_argument("--sample", type=int, default=2000,
                    help="rows to sample for SHAP (default: 2000)")
    ap.add_argument("--top", type=int, default=25, help="top-N features to list (default: 25)")
    args = ap.parse_args()

    for label, lgbm, path in resolve_models(args):
        print("\n" + "#" * 72)
        print(f"# MODEL: {label}    ({path})")
        print("#" * 72)
        n_feats = len(lgbm.feature_names_)
        print(f"Booster best_iteration: {getattr(lgbm.model_, 'best_iteration', 'n/a')}   "
              f"features: {n_feats}")

        # 1. native gain
        gain = pd.Series(lgbm.model_.feature_importance(importance_type="gain"),
                         index=lgbm.feature_names_)
        report_importance(f"[{label}] NATIVE GAIN", gain, args.top)

        # 2. native split
        split = pd.Series(lgbm.model_.feature_importance(importance_type="split"),
                          index=lgbm.feature_names_)
        report_importance(f"[{label}] NATIVE SPLIT", split, args.top)

        # 3. true SHAP (optional)
        if args.features:
            shap_imp = compute_shap(lgbm, args.features, args.sample)
            if shap_imp is not None:
                report_importance(f"[{label}] TRUE SHAP (mean |value|)", shap_imp, args.top)


if __name__ == "__main__":
    main()
