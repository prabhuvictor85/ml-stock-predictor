"""
Quick diagnostic: runs feature engineering + targets + a single CV fold
on 5 stocks to verify cs_rank_20d, NDCG, and benchmark alignment are correct.
Run: python debug_pipeline.py
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path

STOCK_DATA_DIR = Path(r"C:\Victor\Learning_charts\stock_data")
STOCK_LIST_CSV = Path(r"C:\Victor\Learning_charts\stock_lists\constituentsi.csv")
N_STOCKS = 10   # small sample — fast

print("=" * 60)
print("STEP 1: Load sample stocks")
print("=" * 60)
from run_nse_local import load_local_ohlcv, load_htf_zones, merge_htf_zones_to_daily, load_benchmark

tickers_all = pd.read_csv(STOCK_LIST_CSV)
# Use "Symbol" col (has .NS suffix) not "Symbol1"
sym_col = "Symbol" if "Symbol" in tickers_all.columns else tickers_all.columns[0]
tickers = tickers_all[sym_col].dropna().astype(str).str.strip().tolist()[:N_STOCKS]
print(f"Using tickers: {tickers}")

benchmark_close = load_benchmark(STOCK_DATA_DIR)
print(f"Benchmark: {len(benchmark_close)} bars | {benchmark_close.index.min().date()} -> {benchmark_close.index.max().date()}")
print(f"Benchmark sample:\n{benchmark_close.tail(5)}")

frames = []
for t in tickers:
    df = load_local_ohlcv(t, STOCK_DATA_DIR)
    if df.empty or len(df) < 252:
        print(f"  {t}: skipped ({len(df)} bars)")
        continue
    htf = load_htf_zones(t, STOCK_DATA_DIR)
    if htf:
        zc = merge_htf_zones_to_daily(df.index, htf)
        df = df.join(zc, how="left")
    df["ticker"] = t
    df["adv_20d_usd"] = (df["volume"] * df["close"]).rolling(20).mean()
    df["market_cap_usd"] = float("nan")
    df["sector"] = "NSE"
    df["in_universe"] = True
    frames.append(df)
    print(f"  {t}: {len(df)} bars OK")

panel = pd.concat(frames).reset_index().set_index(["date","ticker"]).sort_index()
dates = panel.index.get_level_values("date").to_series().reset_index(drop=True)
panel["group_date"] = dates.dt.to_period("M").dt.to_timestamp().values
print(f"\nPanel shape: {panel.shape}")

print("\n" + "=" * 60)
print("STEP 2: Feature engineering")
print("=" * 60)
from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
from pipeline.config.nse import NSE_CONFIG
fe = FeatureEngineer(NSE_CONFIG, benchmark_close)
panel = fe.build(panel)
feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
print(f"Features computed: {len(feat_cols)}")

print("\n" + "=" * 60)
print("STEP 3: Target building — check benchmark alignment")
print("=" * 60)
from pipeline.targets.builder import TargetBuilder
tb = TargetBuilder(NSE_CONFIG)
panel = tb.build(panel, benchmark_close)

print("\n--- Key target columns ---")
for col in ["future_20d_return", "benchmark_20d_return", "future_20d_excess_return",
            "cs_rank_20d", "cs_rank_composite"]:
    s = panel[col] if col in panel.columns else None
    if s is None:
        print(f"  {col}: MISSING!")
    else:
        nn = s.notna().sum()
        total = len(s)
        print(f"  {col}: {nn}/{total} non-null | mean={s.mean():.4f} | range=[{s.min():.4f}, {s.max():.4f}]")

# Spot check: pick one date and show the cross-section
sample_date = panel.index.get_level_values("date").unique()[-30]
cs = panel.xs(sample_date, level="date")[["future_20d_return","benchmark_20d_return","future_20d_excess_return","cs_rank_20d"]]
print(f"\n--- Cross-section on {sample_date.date()} ---")
print(cs.to_string())

print("\n" + "=" * 60)
print("STEP 4: CV fold — check label and NDCG")
print("=" * 60)
from pipeline.validation.cv import PurgedWalkForwardCV
from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
from pipeline.validation.metrics import compute_fold_metrics, ndcg_at_k

cv = PurgedWalkForwardCV(n_folds=4, min_train_window=252)
fold_specs = list(cv.split(panel))

for spec, tr_idx, te_idx in fold_specs[-2:]:   # last 2 folds
    tr = panel.iloc[tr_idx]
    te = panel.iloc[te_idx]
    te_univ = te[te["in_universe"] == True]

    if len(tr) == 0 or len(te_univ) < 5:
        print(f"Fold {spec.fold_id}: skipped (too small)")
        continue

    tr_grp, tr_groups = cv.build_group_array(tr, min_group_size=3)
    if len(tr_grp) == 0:
        print(f"Fold {spec.fold_id}: no groups")
        continue

    print(f"\nFold {spec.fold_id}: train={len(tr_grp)} rows | test={len(te_univ)} rows")

    # Check cs_rank_composite in training data
    cr = tr_grp["cs_rank_composite"]
    print(f"  train cs_rank_composite: {cr.notna().sum()} non-null | mean={cr.mean():.4f}")

    # Check cs_rank_20d in test data
    cr20 = te_univ["cs_rank_20d"]
    print(f"  test  cs_rank_20d: {cr20.notna().sum()} non-null | mean={cr20.mean():.4f}")

    # Train a quick model
    params = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 100,
              "min_child_samples": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}
    sf = [f for f in feat_cols if f in tr_grp.columns][:20]
    y_tr = tr_grp["cs_rank_composite"].fillna(0)

    ranker = LGBMRanker(params, seed=42)
    ranker.fit(tr_grp[sf].fillna(0), y_tr, tr_groups)
    print(f"  Model: {ranker.model_.num_trees()} trees")

    # Score and compute NDCG manually for first group_date
    scores = pd.Series(ranker.predict(te_univ[sf].fillna(0)), index=te_univ.index)
    te_univ_copy = te_univ.copy()
    te_univ_copy["_score"] = scores

    # Manual NDCG for first group_date
    first_gd = te_univ_copy["group_date"].dropna().unique()[0]
    grp = te_univ_copy[te_univ_copy["group_date"] == first_gd]
    rel = cs_rank_to_label(grp["cs_rank_20d"].fillna(0)).values
    sc  = grp["_score"].values
    print(f"\n  group_date={first_gd.date()} | n_rows={len(grp)}")
    print(f"  rel (labels): min={rel.min()} max={rel.max()} mean={rel.mean():.1f} zeros={( rel==0).sum()}")
    print(f"  scores:       min={sc.min():.4f} max={sc.max():.4f} unique={len(np.unique(sc))}")
    ndcg = ndcg_at_k(rel, sc, k=10)
    print(f"  NDCG@10 = {ndcg:.4f}  (expected > 0 if labels are valid)")

    # Deep trace: show top-10 labels and DCG/IDCG breakdown
    k = min(10, len(sc))
    order = np.argsort(sc)[::-1][:k]
    top10_labels = rel[order]
    ideal_order = np.argsort(rel)[::-1][:k]
    ideal_labels = rel[ideal_order]
    print(f"  top-10 labels by score:  {top10_labels}")
    print(f"  ideal top-10 labels:     {ideal_labels}")
    dcg  = sum((2 ** int(top10_labels[i]) - 1) / np.log2(i + 2) for i in range(k))
    idcg = sum((2 ** int(ideal_labels[i])  - 1) / np.log2(i + 2) for i in range(k))
    print(f"  DCG={dcg:.4f}  IDCG={idcg:.4f}  ratio={dcg/idcg if idcg>0 else 'idcg=0'}")
    print(f"  rel dtype={rel.dtype}  sc dtype={sc.dtype}")

    # Full fold metrics
    bm_ret = benchmark_close.pct_change().fillna(0)
    m = compute_fold_metrics(te_univ, scores, sf, bm_ret, 10, 10)
    print(f"\n  Fold metrics: NDCG@10={m['mean_ndcg_at_10']:.4f} | TopDec_exc={m['top_decile_excess_return']:+.4f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
