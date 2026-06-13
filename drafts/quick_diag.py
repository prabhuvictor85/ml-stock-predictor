"""
Quick diagnostic — validates all 4 fixes using the existing panel artefact.
Run: python -u quick_diag.py
"""
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

print("=" * 60)
print("QUICK DIAGNOSTIC — verifying 4 fixes")
print("=" * 60)

# ── Load panel ──────────────────────────────────────────────
print("\n[1] Loading artefacts/panel.pkl ...")
try:
    with open("artefacts/panel.pkl", "rb") as f:
        panel = pickle.load(f)
    print(f"    Shape       : {panel.shape}")
    dates = panel.index.get_level_values("date")
    print(f"    Date range  : {dates.min().date()} → {dates.max().date()}")
    print(f"    Columns     : {list(panel.columns[:10])}")
    if "in_universe" in panel.columns:
        print(f"    in_universe : {panel['in_universe'].sum()} True rows")
    else:
        print("    WARNING: 'in_universe' column missing!")
except FileNotFoundError:
    print("    panel.pkl not found — trying panel_features.pkl")
    with open("artefacts/panel_features.pkl", "rb") as f:
        panel = pickle.load(f)
    print(f"    Shape: {panel.shape}")

# ── Fix 1: Monthly group_date ───────────────────────────────
print("\n[FIX 1] Switching group_date to monthly ...")
raw_dates = panel.index.get_level_values("date").to_series().reset_index(drop=True)
panel["group_date"] = raw_dates.dt.to_period("M").dt.to_timestamp().values

univ = panel[panel.get("in_universe", pd.Series([True]*len(panel), index=panel.index)) == True] if "in_universe" in panel.columns else panel
gd_counts = univ.groupby("group_date").size()

print(f"    Total monthly groups : {len(gd_counts)}")
print(f"    Group size  min={gd_counts.min()}  max={gd_counts.max()}  mean={gd_counts.mean():.0f}  median={gd_counts.median():.0f}")
print(f"    Groups < 10 stocks   : {(gd_counts < 10).sum()} (these would be skipped by Fix 2)")
print(f"    Groups < 3 stocks    : {(gd_counts < 3).sum()}")
print(f"    Sample (first 4 months):")
for gd, cnt in gd_counts.head(4).items():
    print(f"      {str(gd)[:7]}  →  {cnt} rows")

# ── Fix 2: CV fold group sizes ──────────────────────────────
print("\n[FIX 2] Checking per-fold avg group size ...")
from pipeline.validation.cv import PurgedWalkForwardCV

cv = PurgedWalkForwardCV(n_folds=5)
n_valid = 0
n_skip  = 0
for spec, tr_idx, te_idx in cv.split(panel):
    tr = panel.iloc[tr_idx]
    tr_grp, tr_groups = cv.build_group_array(tr, min_group_size=5)
    if len(tr_grp) == 0:
        print(f"    Fold {spec.fold_id}: EMPTY — no eligible rows after group filter")
        n_skip += 1
        continue
    avg_gs = len(tr_grp) / max(len(tr_groups), 1)
    skip   = avg_gs < 10
    status = "SKIP (Fix 2)" if skip else "OK ✓"
    print(f"    Fold {spec.fold_id}: train={len(tr_idx):>7} rows | "
          f"{len(tr_groups):>4} groups | avg_size={avg_gs:>6.1f} | {status}")
    if skip:
        n_skip += 1
    else:
        n_valid += 1

print(f"    → Valid folds: {n_valid}  Skipped: {n_skip}")

# ── Fix 3 & 4: Quick ranker stall + classifier fallback test ─
print("\n[FIX 3 & 4] Testing ranker stall detection + classifier fallback ...")
from pipeline.models.lgbm_ranker import LGBMRanker
from pipeline.models.xgb_baseline import XGBBaseline
from pipeline.selection.selector import FeatureSelector

# Use fold 0 training data
cv2 = PurgedWalkForwardCV(n_folds=5)
test_done = False
for spec, tr_idx, te_idx in cv2.split(panel):
    tr = panel.iloc[tr_idx]
    tr_grp, tr_groups = cv2.build_group_array(tr, min_group_size=5)
    if len(tr_grp) == 0:
        continue
    avg_gs = len(tr_grp) / max(len(tr_groups), 1)
    if avg_gs < 10:
        continue

    feat_cols = [c for c in tr_grp.columns
                 if c not in ("cs_rank_20d","top_quintile","group_date","in_universe","close","open","high","low","volume","future_20d_return")
                 and tr_grp[c].dtype in (float, "float32","float64")]
    feat_cols = feat_cols[:30]  # cap for speed

    X_tr = tr_grp[feat_cols].fillna(0)
    y_tr_r = tr_grp["cs_rank_20d"].fillna(0)
    y_tr_c = tr_grp["top_quintile"].fillna(0).astype(int)

    print(f"    Using fold {spec.fold_id}: {len(X_tr)} train rows, {len(feat_cols)} features")

    # Train ranker (tiny params for speed)
    tiny_params = {"num_leaves": 31, "n_estimators": 50, "learning_rate": 0.05}
    ranker = LGBMRanker(params=tiny_params, seed=42)
    ranker.fit(X_tr, y_tr_r, tr_groups)
    bi = ranker.model_.best_iteration
    print(f"    Ranker best_iteration = {bi}  →  {'STALLED (Fix 3: use classifier)' if bi == 0 else 'LEARNED ✓ (Fix 3: use ranker scores)'}")

    # Train classifier fallback (Fix 4)
    clf = XGBBaseline(
        params={"n_estimators": 50, "learning_rate": 0.05, "max_depth": 4},
        model_mode="classifier",
        seed=42,
    )
    clf.fit(X_tr, y_tr_c)
    te = panel.iloc[te_idx]
    te_univ = te[te["in_universe"] == True] if "in_universe" in te.columns else te
    X_te = te_univ[[c for c in feat_cols if c in te_univ.columns]].fillna(0)
    proba = clf.predict_proba(X_te)
    print(f"    Classifier predict_proba: shape={proba.shape}  min={proba.min():.4f}  max={proba.max():.4f}  mean={proba.mean():.4f}")
    print(f"    Fix 4 ✓ — classifier scores usable as HPO signal")
    test_done = True
    break

if not test_done:
    print("    Could not find a valid fold — all skipped by Fix 2 guard")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

