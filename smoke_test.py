"""
smoke_test.py — Lightweight end-to-end smoke test for the NSE pipeline.

Uses:
  - First 40 tickers only  (fast CSV load)
  - min_history_days = 252  (1 year minimum)
  - n_folds  = 5            (minimum)
  - n_trials = 0            (skip Optuna HPO, use defaults)
  - Separate artefact dir   (artefacts/smoke_test/) — won't overwrite real run

Run:
    python smoke_test.py
    python smoke_test.py --gpu
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
sys.path.insert(0, ".")

# ── Config ─────────────────────────────────────────────────────────────────
STOCK_LIST_CSV = Path(r"C:\Victor\Learning_charts\stock_lists\constituentsi.csv")
STOCK_DATA_DIR = Path(r"C:\Victor\Learning_charts\stock_data")
SMOKE_ARTEFACTS = Path("artefacts/smoke_test")
N_TICKERS   = 40
N_FOLDS     = 5
N_TRIALS    = 0          # skip HPO
MIN_HISTORY = 252
TOP_N       = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    use_gpu = False
    if args.gpu:
        try:
            import subprocess
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
            use_gpu = True
            print("GPU detected — GPU acceleration ENABLED")
        except Exception:
            print("WARNING: --gpu specified but no CUDA GPU detected — running on CPU")

    # ── Import pipeline pieces ──────────────────────────────────────────────
    from run_nse_local import (
        build_panel_from_local,
        load_benchmark,
        score_and_rank,
        save_outputs,
    )
    from pipeline.config.nse import NSE_CONFIG as cfg
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
    from pipeline.targets.builder import TargetBuilder
    from pipeline.validation.cv import PurgedWalkForwardCV
    from pipeline.validation.leakage_tests import LeakageTestSuite
    from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
    from pipeline.models.catboost_model import CatBoostModel
    from pipeline.models.xgb_baseline import XGBBaseline
    from pipeline.models.calibrator import ProbabilityCalibrator
    from pipeline.models.ensemble import EnsembleRanker
    from pipeline.selection.selector import FeatureSelector
    from pipeline.validation.metrics import compute_fold_metrics
    from pipeline.monitoring.drift_monitor import FeatureDriftMonitor

    SMOKE_ARTEFACTS.mkdir(parents=True, exist_ok=True)

    # ── 1. Load small ticker subset ─────────────────────────────────────────
    ticker_df = pd.read_csv(STOCK_LIST_CSV)
    tickers   = ticker_df["Symbol"].str.strip().dropna().tolist()[:N_TICKERS]
    print(f"\n{'='*60}")
    print(f"SMOKE TEST  —  {N_TICKERS} tickers / {N_FOLDS} folds / {N_TRIALS} HPO trials")
    print(f"{'='*60}\n")
    print(f"[0] Tickers: {tickers[:5]} ... ({len(tickers)} total)")

    # ── 2. Load benchmark ───────────────────────────────────────────────────
    print("[1] Loading benchmark (^NSEI)...")
    benchmark_close = load_benchmark(STOCK_DATA_DIR)
    if benchmark_close.empty:
        print("  WARNING: Benchmark unavailable — using equal-weight proxy")

    # ── 3. Build panel ──────────────────────────────────────────────────────
    print("[2] Building panel from local CSVs...")
    panel = build_panel_from_local(tickers, STOCK_DATA_DIR,
                                   min_history_days=MIN_HISTORY)
    if benchmark_close.empty:
        benchmark_close = panel.groupby(level="date")["close"].mean().rename("benchmark_close")

    dates = panel.index.get_level_values("date")
    print(f"    Panel: {panel.shape}  |  {dates.min().date()} -> {dates.max().date()}")
    print(f"    in_universe True: {(panel['in_universe']==True).sum()}")

    # ── 4. Feature engineering ──────────────────────────────────────────────
    print("[3] Feature engineering...")
    fe = FeatureEngineer(cfg, benchmark_close)
    panel = fe.build(panel)
    feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
    print(f"    {len(feat_cols)} feature columns")

    # Sanity: check date index is real timestamps
    sample_date = panel.index.get_level_values("date")[0]
    if not isinstance(sample_date, (pd.Timestamp,)) and not hasattr(sample_date, 'year'):
        print(f"  FAIL: date index dtype={panel.index.get_level_values('date').dtype} "
              f"— expected datetime, got {type(sample_date)}")
        sys.exit(1)
    print(f"    Date index OK: dtype={panel.index.get_level_values('date').dtype}")

    # ── 5. Build targets ────────────────────────────────────────────────────
    print("[4] Building targets...")
    tb = TargetBuilder(cfg)
    panel = tb.build(panel, benchmark_close)
    n_pos = int((panel["top_quintile"] == 1).sum())
    n_cs  = int(panel["cs_rank_20d"].notna().sum())
    print(f"    cs_rank_20d non-null: {n_cs}  |  top_quintile positives: {n_pos}")
    if n_pos == 0:
        print("  FAIL: top_quintile has 0 positives — targets broken, aborting.")
        sys.exit(1)

    # ── 6. Leakage tests ────────────────────────────────────────────────────
    print("[5] Leakage tests...")
    suite = LeakageTestSuite(panel, feat_cols)
    suite.run_all()

    # ── 7. Walk-forward CV ──────────────────────────────────────────────────
    print(f"[6] Walk-forward CV ({N_FOLDS} folds)...")
    cv = PurgedWalkForwardCV(n_folds=N_FOLDS, min_train_window=504)
    fold_specs = cv.get_fold_specs(panel)
    print(f"    {len(fold_specs)} fold specs generated")
    if len(fold_specs) == 0:
        print("  FAIL: 0 folds generated — not enough history.")
        sys.exit(1)

    # ── 8. Train final models (no HPO) ──────────────────────────────────────
    print("[7] Training final models (no HPO)...")
    best_params = {
        "num_leaves": 31, "learning_rate": 0.05,
        "n_estimators": 100, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    }
    best_k = 20

    full_grp, full_groups = cv.build_group_array(panel, min_group_size=5)
    avail    = [f for f in feat_cols if f in full_grp.columns]
    X_full   = full_grp[avail]
    y_full_r = full_grp["cs_rank_20d"].fillna(0)

    cls_mask     = full_grp["top_quintile"].notna()
    X_full_cls   = full_grp.loc[cls_mask, avail]
    y_full_c_cls = full_grp.loc[cls_mask, "top_quintile"].astype(int)
    n_pos_cls    = int(y_full_c_cls.sum())
    print(f"    Classifier rows: {cls_mask.sum()}  ({n_pos_cls} pos / {cls_mask.sum()-n_pos_cls} neg)")
    if n_pos_cls == 0:
        print("  FAIL: 0 positive labels for classifier.")
        sys.exit(1)

    # Feature selection
    sel = FeatureSelector(seed=cfg.random_seed)
    final_features = sel.select(X_full_cls, y_full_c_cls, top_k=best_k)
    print(f"    Selected {len(final_features)} features")

    X_fin     = X_full[final_features].fillna(0)
    X_fin_cls = X_full_cls[final_features].fillna(0)

    # LGBM Ranker
    final_ranker = LGBMRanker(best_params, seed=cfg.random_seed)
    final_ranker.fit(X_fin, y_full_r, full_groups)
    print("    LGBMRanker  OK")

    # CatBoost
    cb_extra = {"task_type": "GPU", "devices": "0"} if use_gpu else {}
    final_cb = CatBoostModel({
        "iterations": best_params.get("n_estimators", 100),
        "learning_rate": best_params.get("learning_rate", 0.05),
        "depth": 6, **cb_extra,
    }, seed=cfg.random_seed)
    final_cb.fit(X_fin_cls, y_full_c_cls)
    print("    CatBoostModel OK")

    # XGB (classifier mode -- explicit!)
    xgb_extra = {"device": "cuda"} if use_gpu else {}
    xgb = XGBBaseline({
        "n_estimators":  best_params.get("n_estimators", 100),
        "learning_rate": best_params.get("learning_rate", 0.05),
        "max_depth": 6, **xgb_extra,
    }, model_mode="classifier", seed=cfg.random_seed)
    xgb.fit(X_fin_cls, y_full_c_cls)
    print("    XGBBaseline   OK")

    # Calibration
    cb_probs  = final_cb.predict_proba(X_fin_cls)
    cal       = ProbabilityCalibrator()
    cal.fit(cb_probs, y_full_c_cls.values)
    print("    ProbabilityCalibrator OK")

    ensemble = EnsembleRanker(final_ranker, final_cb, cal)

    # ── 9. Drift monitor ───────────────────────────────────────────────────
    drift_monitor = FeatureDriftMonitor(cfg, final_features)
    drift_monitor.fit_baseline(full_grp[final_features])

    # ── 10. Score & rank ───────────────────────────────────────────────────
    print("[8] Scoring latest cross-section...")
    result = score_and_rank(
        panel=panel,
        ensemble=ensemble,
        final_features=final_features,
        benchmark_close=benchmark_close,
        cfg=cfg,
        top_n=TOP_N,
        weighting="equal",
        as_of_date=None,
    )
    watchlist = result.get("watchlist_combined") or result.get("watchlist") or result.get("bull")
    if watchlist is not None and not watchlist.empty:
        print(f"    Watchlist ({len(watchlist)} stocks):")
        print(watchlist.head(TOP_N).to_string())
    else:
        print("    (no watchlist rows produced — check score_and_rank output keys)")

    # ── 11. Save smoke artefacts ───────────────────────────────────────────
    print(f"\n[9] Saving smoke artefacts -> {SMOKE_ARTEFACTS}/")
    for name, obj in [
        ("ensemble.pkl",       ensemble),
        ("lgbm_ranker.pkl",    final_ranker),
        ("catboost_model.pkl", final_cb),
        ("calibrator.pkl",     cal),
        ("drift_monitor.pkl",  drift_monitor),
    ]:
        with open(SMOKE_ARTEFACTS / name, "wb") as f:
            pickle.dump(obj, f)
    (SMOKE_ARTEFACTS / "selected_features.txt").write_text("\n".join(final_features))

    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED -- safe to launch the full run.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()




