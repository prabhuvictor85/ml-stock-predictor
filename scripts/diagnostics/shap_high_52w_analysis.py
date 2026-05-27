"""
shap_high_52w_analysis.py — Quantify the importance of features_high_52w_dist
(and friends) in the trained LightGBM models.

Two passes:
  PASS 1 — LightGBM's built-in gain/split importance (instant, no panel load)
  PASS 2 — Real TreeSHAP on a sample of the panel (slower, ~1-3 min)

Usage
─────
    python scripts/diagnostics/shap_high_52w_analysis.py
    python scripts/diagnostics/shap_high_52w_analysis.py --mode reversal
    python scripts/diagnostics/shap_high_52w_analysis.py --sample 10000   # bigger SHAP sample
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["momentum", "reversal", "both"],
                   default="both")
    p.add_argument("--sample", type=int, default=5000,
                   help="Number of rows to use for TreeSHAP (default 5000)")
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top features to print")
    p.add_argument("--no-shap", action="store_true",
                   help="Skip the SHAP pass (only show built-in importance)")
    return p.parse_args()


def banner(s: str):
    print()
    print("=" * 70)
    print("  " + s)
    print("=" * 70)


def analyse_one_mode(mode: str, sample_size: int, top_n: int, skip_shap: bool):
    banner(f"MODE: {mode.upper()}")
    art_dir = PROJECT_ROOT / "artefacts" / "nse_tradingv" / mode
    ranker_path = art_dir / "lgbm_ranker.pkl"
    panel_path  = art_dir / "panel.pkl"

    if not ranker_path.exists():
        print(f"  Skipping {mode}: {ranker_path} not found.")
        return

    # ── PASS 1: built-in importance ────────────────────────────────────────
    print(f"\n[Pass 1/2] Loading {ranker_path.name} ...")
    with open(ranker_path, "rb") as f:
        ranker = pickle.load(f)

    booster = getattr(ranker, "model_", None)
    if booster is None:
        # Fallback: ensemble wrapper
        ensemble_path = art_dir / "ensemble.pkl"
        with open(ensemble_path, "rb") as f:
            ensemble = pickle.load(f)
        booster = getattr(ensemble.lgbm, "model_", None)

    if booster is None:
        print(f"  Could not extract booster from {ranker_path.name} — skipping.")
        return

    feature_names = list(getattr(ranker, "feature_names_", booster.feature_name()))
    n_features    = len(feature_names)

    gain_imp  = booster.feature_importance(importance_type="gain")
    split_imp = booster.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "gain":    gain_imp,
        "split":   split_imp,
    })
    imp_df["gain_norm"]  = imp_df["gain"]  / imp_df["gain"].sum()
    imp_df["split_norm"] = imp_df["split"] / imp_df["split"].sum()
    imp_df = imp_df.sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1

    # Where do the 52w features rank?
    target_features = ["features_high_52w_dist", "features_low_52w_dist"]
    print(f"\n  Top {top_n} by GAIN importance (n_features={n_features}):")
    print(f"  {'Rank':>4}  {'Feature':<45}  {'Gain%':>7}  {'Split%':>7}")
    print(f"  {'-'*4}  {'-'*45}  {'-'*7}  {'-'*7}")
    for _, r in imp_df.head(top_n).iterrows():
        marker = "  <--" if r["feature"] in target_features else ""
        print(f"  {int(r['rank']):>4}  {r['feature']:<45}  "
              f"{r['gain_norm']*100:>6.2f}%  {r['split_norm']*100:>6.2f}%{marker}")

    print(f"\n  Position of 52-week features:")
    for feat in target_features:
        hit = imp_df[imp_df["feature"] == feat]
        if len(hit) == 0:
            print(f"    {feat:<45} NOT in selected features for {mode}")
        else:
            r = hit.iloc[0]
            print(f"    {feat:<45} rank #{int(r['rank']):>3}/{n_features}  "
                  f"gain={r['gain_norm']*100:.2f}%  split={r['split_norm']*100:.2f}%")

    # ── PASS 2: TreeSHAP on a sample ───────────────────────────────────────
    if skip_shap:
        return

    print(f"\n[Pass 2/2] TreeSHAP on {sample_size:,}-row sample ...")
    try:
        import shap
    except ImportError:
        print("  shap not installed. Skipping. (pip install shap)")
        return

    if not panel_path.exists():
        print(f"  Panel not found at {panel_path} — skipping SHAP pass.")
        return

    print(f"  Loading panel ({panel_path.stat().st_size / 1e9:.1f} GB)...")
    with open(panel_path, "rb") as f:
        panel = pickle.load(f)

    # Filter to universe rows (in_universe == True) and to the actual feature cols
    if "in_universe" in panel.columns:
        panel = panel[panel["in_universe"] == True]
    print(f"  Panel rows after universe filter: {len(panel):,}")

    avail = [f for f in feature_names if f in panel.columns]
    missing = set(feature_names) - set(avail)
    if missing:
        print(f"  WARN: {len(missing)} model features missing from panel: "
              f"{list(missing)[:5]}...")

    X = panel[avail].fillna(0).astype("float32")
    del panel  # release the big object

    # Random sample
    rng = np.random.default_rng(42)
    n_sample = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n_sample, replace=False)
    X_sample = X.iloc[idx]
    del X
    print(f"  Sampled {len(X_sample):,} rows for SHAP.")

    # Compute SHAP
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)
    # For LambdaRank, shap_values is a 2D array (samples × features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature":       X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    })
    shap_df["shap_pct"] = shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["rank"] = shap_df.index + 1

    print(f"\n  Top {top_n} by SHAP mean|value|:")
    print(f"  {'Rank':>4}  {'Feature':<45}  {'SHAP%':>7}  {'Mean|SHAP|':>11}")
    print(f"  {'-'*4}  {'-'*45}  {'-'*7}  {'-'*11}")
    for _, r in shap_df.head(top_n).iterrows():
        marker = "  <--" if r["feature"] in target_features else ""
        print(f"  {int(r['rank']):>4}  {r['feature']:<45}  "
              f"{r['shap_pct']:>6.2f}%  {r['mean_abs_shap']:>10.4f}{marker}")

    print(f"\n  Position of 52-week features (SHAP):")
    for feat in target_features:
        hit = shap_df[shap_df["feature"] == feat]
        if len(hit) == 0:
            print(f"    {feat:<45} NOT in feature set")
        else:
            r = hit.iloc[0]
            print(f"    {feat:<45} rank #{int(r['rank']):>3}/{len(shap_df)}  "
                  f"shap={r['shap_pct']:.2f}%  mean|SHAP|={r['mean_abs_shap']:.4f}")

    # Distribution of SHAP for high_52w_dist
    if "features_high_52w_dist" in X_sample.columns:
        col_idx = list(X_sample.columns).index("features_high_52w_dist")
        sv_52w  = shap_values[:, col_idx]
        val_52w = X_sample["features_high_52w_dist"].values

        print(f"\n  features_high_52w_dist — interaction profile:")
        print(f"    Feature value:  min={val_52w.min():.3f}  max={val_52w.max():.3f}  "
              f"mean={val_52w.mean():.3f}")
        print(f"    SHAP value:     min={sv_52w.min():.4f}  max={sv_52w.max():.4f}  "
              f"mean={sv_52w.mean():.4f}")

        # Bin by feature value, show average SHAP per bin
        bins = pd.cut(val_52w, bins=[-1, -0.5, -0.30, -0.20, -0.15, -0.10, -0.05, 0.01],
                      labels=["<-50%", "-50/-30", "-30/-20", "-20/-15",
                              "-15/-10", "-10/-5", "-5/0%"])
        prof = pd.DataFrame({"bin": bins, "shap": sv_52w}).groupby("bin", observed=True)["shap"].agg(
            ["count", "mean", "std"]).reset_index()
        print(f"\n    Average SHAP contribution by feature value bin:")
        print(f"    {'Bin (high_52w_dist)':<22}  {'Count':>7}  {'Mean SHAP':>10}  {'Std SHAP':>10}")
        for _, r in prof.iterrows():
            sign = "pushes UP" if r["mean"] > 0 else "pushes DOWN" if r["mean"] < 0 else "neutral"
            print(f"    {str(r['bin']):<22}  {int(r['count']):>7}  "
                  f"{r['mean']:>10.4f}  {r['std']:>10.4f}  {sign}")


def main():
    args = parse_args()
    modes = ["momentum", "reversal"] if args.mode == "both" else [args.mode]

    print(f"Project root: {PROJECT_ROOT}")
    for m in modes:
        analyse_one_mode(m, args.sample, args.top_n, args.no_shap)

    print("\nDone.")


if __name__ == "__main__":
    main()
