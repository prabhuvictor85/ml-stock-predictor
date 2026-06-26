"""
train.py — Full training pipeline entry point.

Usage:
    python train.py --market nse
    python train.py --market sp500 --n_folds 8 --n_trials 150

Pipeline:
  1. Load config
  2. Fetch / load panel
  3. Feature engineering
  4. Target building
  5. Purged walk-forward CV + Optuna HPO
  6. Final model training on full data
  7. Probability calibration
  8. Ensemble assembly
  9. SHAP global explanations
 10. Feature drift baseline
 11. Save artefacts
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def set_seeds(seed: int) -> None:
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Stock Predictor — Training Pipeline")
    p.add_argument("--market", required=True, choices=["nse", "sp500", "nasdaq"])
    p.add_argument("--start", default="2015-01-01", help="Data start date")
    p.add_argument("--end", default=None, help="Data end date (default: today)")
    p.add_argument("--n_folds", type=int, default=8)
    p.add_argument("--n_trials", type=int, default=150)
    p.add_argument("--timeout", type=int, default=None,  # FIX 14: was 3600 — kills HPO after 2 trials
                   help="Optuna timeout in seconds (default: None = no timeout). "
                        "Set to e.g. 7200 for a 2-hour cap.")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Parallel Optuna workers (-1 = all CPU cores). "
                        "Uses SQLite storage for thread-safety.")
    p.add_argument("--use_gpu", action="store_true",
                   help="Enable GPU acceleration for LightGBM.")
    p.add_argument("--panel_dir", default="panel", help="Panel parquet directory")
    p.add_argument("--output_dir", default="artefacts", help="Output directory for models")
    p.add_argument("--polygon_key", default="", help="Polygon.io API key")
    p.add_argument("--tiingo_key", default="", help="Tiingo API key")
    p.add_argument("--skip_fetch", action="store_true", help="Skip data fetch, load existing panel")
    p.add_argument("--tickers_file", default=None, help="Path to text file with one ticker per line")
    return p.parse_args()


# ── Optuna objective ────────────────────────────────────────────────────────

def make_optuna_objective(
    panel: pd.DataFrame,
    feature_cols: List[str],
    cfg,
    n_folds: int,
    benchmark_close: pd.Series,
    pre_selected_features: List[str],   # FIX 2: accept pre-selected features, skip per-fold selection
    use_gpu: bool = False,
) -> Any:
    """Return a closure that is the Optuna objective function."""
    import logging
    import lightgbm as lgb  # FIX 10: needed for early_stopping callback
    _obj_log = logging.getLogger("pipeline.optuna.objective")  # FIX 1: module-level logger avoids NameError

    from pipeline.validation.cv import PurgedWalkForwardCV
    from pipeline.validation.metrics import compute_fold_metrics, ndcg_at_k
    from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
    from pipeline.selection.selector import ALWAYS_INCLUDE
    import optuna

    # FIX 2b: split the pre-selected list into forced vs data-ranked ONCE.
    # select() returns [forced..., ordinary...] with forced at the FRONT, so a
    # naive pre_selected[:top_k] slice scoops only forced features and never
    # reaches the importance-ranked ones — HPO would tune on the hand-picked set,
    # not the data. Mirror select()'s construction (forced + top_k ordinary) so
    # each trial uses the SAME feature set the final model is built from.
    _forced_pre   = [f for f in pre_selected_features if f in ALWAYS_INCLUDE]
    _ordinary_pre = [f for f in pre_selected_features if f not in ALWAYS_INCLUDE]

    def objective(trial: optuna.Trial) -> float:
        # FIX 9: tightened search space — avoids ultra-slow trials and degenerate configs
        params = {
            "num_leaves":        trial.suggest_int("num_leaves", 31, 127),
            "learning_rate":     trial.suggest_float("lr", 0.01, 0.1, log=True),
            "n_estimators":      trial.suggest_int("n_estimators", 200, 600),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
        }
        # FIX 2: forced features + top_k DATA-RANKED ordinary features — mirrors
        # FeatureSelector.select() so HPO tunes on the same feature set the final
        # model ships with (not a front-slice of only forced features).
        top_k = trial.suggest_categorical("feature_top_K", [20, 30])
        selected_feats = _forced_pre + _ordinary_pre[:top_k]

        cv = PurgedWalkForwardCV(n_folds=n_folds)
        ndcg_list: List[float] = []
        top_dec_list: List[float] = []

        for fold_spec, train_idx, test_idx in cv.split(panel):
            train_panel = panel.iloc[train_idx]
            test_panel  = panel.iloc[test_idx]

            # Build group array (in-universe only, min 10 per group)
            train_grp, train_groups = cv.build_group_array(train_panel)
            if len(train_grp) == 0:
                continue

            # Fix 2 — skip fold if average group size is too small for ranker
            avg_group_size = len(train_grp) / max(len(train_groups), 1)
            if avg_group_size < 10:
                _obj_log.warning(  # FIX 1: was log.warning — log not in scope here
                    f"Fold {fold_spec.fold_id}: avg group size {avg_group_size:.1f} < 10 "
                    f"— skipping fold (ranker won't learn)"
                )
                continue

            avail_feats  = [f for f in selected_feats if f in train_grp.columns]
            X_tr         = train_grp[avail_feats].fillna(0)
            y_tr_rank    = train_grp["cs_rank_20d"].fillna(0)

            # Score test set
            test_univ = test_panel[test_panel["in_universe"] == True].copy()
            if len(test_univ) < 5:
                continue

            # Use reindex to guarantee the same column set as training — fills
            # any genuinely missing columns with 0 rather than silently dropping
            # them (which causes a 0-feature matrix and LightGBM crash).
            X_te = test_univ.reindex(columns=avail_feats).fillna(0)

            # FIX 3: safe attribute probe — best_iteration=0 is normal without early stopping;
            # use num_trees() to detect genuine stalls; multiple attribute paths for robustness.
            ranker = LGBMRanker(params=params, seed=cfg.random_seed, use_gpu=use_gpu)
            # FIX 10: pass early_stopping callback so ranker stops at best round, not always n_estimators
            ranker.fit(
                X_tr, y_tr_rank, train_groups,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            try:
                n_trees = (
                    getattr(getattr(ranker, "model_",    None), "num_trees",      lambda: 0)()
                    or getattr(getattr(ranker, "booster_", None), "num_trees",    lambda: 0)()
                    or getattr(getattr(ranker, "_model",  None), "num_trees",     lambda: 0)()
                )
            except Exception:
                n_trees = 0

            if n_trees == 0:
                _obj_log.warning(  # FIX 1: was log.warning
                    f"Fold {fold_spec.fold_id}: ranker has 0 trees — skipping fold"
                )
                continue
            ranker_scores = ranker.predict(X_te)
            score_series  = pd.Series(ranker_scores, index=test_univ.index)

            # FIX 12: positional call, no top_n kwarg (matches compute_fold_metrics signature)
            metrics = compute_fold_metrics(
                test_univ,
                score_series,
                avail_feats,
                benchmark_close.pct_change().fillna(0),
                cfg.commission_bps,
                cfg.get_slippage_bps(cfg.min_adv_usd),
            )
            ndcg_list.append(metrics["mean_ndcg_at_10"])
            top_dec_list.append(metrics["top_decile_excess_return"])

            trial.report(metrics["mean_ndcg_at_10"], step=fold_spec.fold_id)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # FIX 5: guard against empty lists — np.mean([]) returns NaN, nan<=0 is False (silent bug)
        if len(ndcg_list) < 3:
            raise optuna.exceptions.TrialPruned()

        mean_ndcg = float(np.mean(ndcg_list))
        std_ndcg  = float(np.std(ndcg_list))

        if np.mean(top_dec_list) <= 0:
            raise optuna.exceptions.TrialPruned()

        return mean_ndcg - 0.5 * std_ndcg

    return objective


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    from pipeline.config import get_config
    from pipeline.data.fetcher import DataFetcher
    from pipeline.data.universe import UniverseBuilder
    from pipeline.data.panel import PanelConstructor
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
    from pipeline.targets.builder import TargetBuilder
    from pipeline.validation.cv import PurgedWalkForwardCV
    from pipeline.models.lgbm_ranker import LGBMRanker, cs_rank_to_label
    from pipeline.models.ensemble import EnsembleRanker
    from pipeline.selection.selector import FeatureSelector
    from pipeline.monitoring.drift_monitor import FeatureDriftMonitor
    from pipeline.explainability.shap_explainer import SHAPExplainer
    from pipeline.utils.logging import get_logger
    import optuna

    log = get_logger("train")
    cfg = get_config(args.market)
    set_seeds(cfg.random_seed)

    end_date   = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir) / args.market
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting training pipeline: market={cfg.market_id}, period={args.start}->{end_date}")

    # ── 1. Load or fetch panel ─────────────────────────────────────────────
    panel_dir = Path(args.panel_dir) / args.market

    if args.skip_fetch and panel_dir.exists():
        log.info("Loading existing panel from parquet...")
        panel = PanelConstructor.load(panel_dir)
    else:
        fetcher          = DataFetcher(cfg, polygon_api_key=args.polygon_key, tiingo_api_key=args.tiingo_key)
        universe_builder = UniverseBuilder(cfg)

        if args.tickers_file:
            with open(args.tickers_file) as f:
                tickers = [l.strip() for l in f if l.strip()]
        else:
            log.warning("No tickers_file provided. Using benchmark only as demo.")
            tickers = [cfg.benchmark_ticker]

        pc    = PanelConstructor(cfg, fetcher, universe_builder)
        panel = pc.build(tickers, args.start, end_date)
        pc.save(panel, panel_dir)

    # FIX 15: validate panel immediately after load — catches corruption before confusing KeyErrors
    REQUIRED_COLS = {"open", "high", "low", "close", "volume", "in_universe"}
    missing = REQUIRED_COLS - set(panel.columns)
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")
    if len(panel) == 0:
        raise ValueError("Panel is empty — check data source and date range.")
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Panel must have a MultiIndex of (date, ticker).")
    log.info(
        f"Panel validation passed: {panel.shape}, "
        f"dates {panel.index.get_level_values('date').min().date()} -> "
        f"{panel.index.get_level_values('date').max().date()}"
    )

    # ── 2. Benchmark ──────────────────────────────────────────────────────
    # FIX 11: avoid fetching benchmark twice — extract from panel if already present
    if "benchmark_close" in panel.columns:
        benchmark_close = panel["benchmark_close"].groupby(level="date").first()
        log.info("Benchmark close extracted from panel.")
    else:
        fetcher_bm      = DataFetcher(cfg, polygon_api_key=args.polygon_key, tiingo_api_key=args.tiingo_key)
        bm_raw          = fetcher_bm.fetch_benchmark(args.start, end_date)
        benchmark_close = bm_raw["close"].rename("benchmark_close")
        log.info("Benchmark close fetched from DataFetcher.")

    # ── 3a. Time-honest zone labeling ─────────────────────────────────────
    try:
        from pipeline.utils.zone_analyzer import ZoneAnalyzer
        zone_analyzer = ZoneAnalyzer()
        from pipeline.data.zone_labeler import build_time_honest_zones
        log.info("Computing time-honest zone labels (expanding window)...")
        panel = build_time_honest_zones(
            panel=panel,
            analyze_zones_fn=zone_analyzer.analyze_zones,
            checkpoint_freq="YE",
            timeframes=["1d", "1wk", "1mo", "3mo", "1y"],
        )
        log.info("Zone labeling complete.")
    except Exception as e:
        # FIX 6: fail hard if no zone columns exist — silent training without zones is unacceptable
        zone_cols_present = [c for c in panel.columns if "zone" in c.lower()]
        if zone_cols_present:
            log.warning(
                f"Zone labeling failed ({e}) — using existing zone columns: {zone_cols_present[:5]}"
            )
        else:
            raise RuntimeError(
                f"Zone labeling failed and NO zone columns exist in panel. "
                f"Cannot proceed without zone data. Error: {e}"
            ) from e

    # ── 3b. Feature engineering ───────────────────────────────────────────
    log.info("Running feature engineering...")
    feat_eng = FeatureEngineer(cfg, benchmark_close)
    panel    = feat_eng.build(panel)

    # ── 4. Target building ────────────────────────────────────────────────
    log.info("Building targets...")
    target_builder = TargetBuilder(cfg)
    panel          = target_builder.build(panel, benchmark_close)

    feature_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
    log.info(f"Total feature columns: {len(feature_cols)}")

    # ── 5. Optuna HPO ─────────────────────────────────────────────────────
    log.info(f"Starting Optuna HPO: {args.n_trials} trials, timeout={args.timeout}s, "
             f"n_jobs={args.n_jobs}, use_gpu={args.use_gpu}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # FIX 8: SQLite storage is MANDATORY when n_jobs != 1 — in-memory + parallel = race conditions
    storage = None
    db_path = output_dir / "optuna_study.db"
    if args.n_jobs != 1:
        try:
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{db_path}",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )
            log.info(f"Optuna SQLite storage: {db_path}")
        except Exception as e:
            raise RuntimeError(
                f"n_jobs={args.n_jobs} requires SQLite storage but creation failed: {e}\n"
                f"Check disk space and write permissions for {output_dir}"
            ) from e

    # FIX 2: run FeatureSelector ONCE before HPO on full labelled data
    cv_pre = PurgedWalkForwardCV(n_folds=args.n_folds)
    full_train_pre, _ = cv_pre.build_group_array(panel)
    avail_pre    = [f for f in feature_cols if f in full_train_pre.columns]
    X_pre        = full_train_pre[avail_pre]
    y_pre_cls    = full_train_pre["top_quintile"].fillna(0).astype(int)
    selector_pre = FeatureSelector(seed=cfg.random_seed)
    pre_selected = selector_pre.select(X_pre, y_pre_cls, top_k=50)
    log.info(f"Pre-selected {len(pre_selected)} features for HPO")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2),
        storage=storage,
        load_if_exists=True,
    )
    # FIX 2: pass pre_selected_features so objective() slices instead of re-running selector
    objective = make_optuna_objective(
        panel, feature_cols, cfg, args.n_folds,
        benchmark_close,
        pre_selected_features=pre_selected,
        use_gpu=args.use_gpu,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   n_jobs=args.n_jobs, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_k      = best_params.pop("feature_top_K", 30)
    # HPO stores lr under key "lr" — remap to learning_rate for model constructors
    if "lr" in best_params:
        best_params["learning_rate"] = best_params.pop("lr")
    log.info(f"Best params: {best_params}, feature_top_K={best_k}")
    log.info(f"Best Optuna value: {study.best_value:.4f}")

    # FIX 13: save study metadata (name + db path) instead of stale in-memory pickle
    study_meta = {
        "study_name":   study.study_name,
        "storage_url":  str(db_path) if storage else "in-memory",
        "best_value":   study.best_value,
        "best_params":  study.best_params,
        "n_trials":     len(study.trials),
    }
    with open(output_dir / "optuna_study_meta.json", "w") as f:
        json.dump(study_meta, f, indent=2)
    log.info(
        f"Study metadata saved. Reload with: "
        f"optuna.load_study(study_name='{study.study_name}', storage='sqlite:///{db_path}')"
    )

    # ── 6. Final training on ALL data ─────────────────────────────────────
    log.info("Training final models on full data...")
    cv            = PurgedWalkForwardCV(n_folds=args.n_folds)
    full_train, full_groups = cv.build_group_array(panel)

    avail_feats  = [f for f in feature_cols if f in full_train.columns]
    X_full       = full_train[avail_feats]
    y_full_rank  = full_train["cs_rank_20d"].fillna(0)
    y_full_cls   = full_train["top_quintile"].fillna(0).astype(int)

    selector       = FeatureSelector(seed=cfg.random_seed)
    final_features = selector.select(X_full, y_full_cls, top_k=best_k)
    log.info(f"Final feature count: {len(final_features)}")

    feat_path = output_dir / "selected_features.txt"
    feat_path.write_text("\n".join(final_features))

    X_final = X_full[final_features].fillna(0)

    # LightGBM Ranker
    final_ranker = LGBMRanker(params=best_params, seed=cfg.random_seed, use_gpu=args.use_gpu)
    final_ranker.fit(X_final, y_full_rank, full_groups)

    # ── 7. Ensemble ───────────────────────────────────────────────────────
    # LGBM-only + inverse-vol tilt — matches the production EnsembleRanker used
    # by the run_*_local.py scripts. CatBoost/XGBoost/calibrator were removed:
    # their artefacts were pickled but never loaded anywhere, and a second GBM
    # on the same features correlates 0.85+ with LGBM (see ensemble.py).
    ensemble = EnsembleRanker(final_ranker)

    from pipeline.validation.metrics import compute_fold_metrics
    fold_specs = cv.get_fold_specs(panel)
    if fold_specs:
        last_fold = fold_specs[-1]
        dates     = panel.index.get_level_values("date")
        test_mask = (dates >= last_fold.test_start) & (dates <= last_fold.test_end)
        test_panel = panel.iloc[np.where(test_mask)[0]]
        test_univ  = test_panel[test_panel["in_universe"] == True]
        if len(test_univ) > 0:
            X_te    = test_univ.reindex(columns=final_features).fillna(0)
            bm_rets = benchmark_close.pct_change().fillna(0)
            slippage = cfg.get_slippage_bps(cfg.min_adv_usd)

            lgbm_scores      = pd.Series(final_ranker.predict(X_te), index=test_univ.index)
            # FIX 12: positional call — no top_n kwarg, matches compute_fold_metrics signature
            lgbm_metrics     = compute_fold_metrics(test_univ, lgbm_scores, final_features, bm_rets, cfg.commission_bps, slippage)

            log.info(f"LGBM net_sharpe={lgbm_metrics['net_sharpe']:.3f} (last-fold OOS)")

    # ── 9. SHAP global explanations ───────────────────────────────────────
    log.info("Computing SHAP global feature importances...")
    try:
        shap_exp        = SHAPExplainer(final_ranker)
        X_shap_sample   = X_final.sample(min(2000, len(X_final)), random_state=cfg.random_seed)
        shap_exp.compute(X_shap_sample)
        shap_importance = shap_exp.global_importance(top_k=20)
        log.info(f"Top 5 SHAP features:\n{shap_importance.head(5).to_string()}")
        shap_exp.plot_global(X_shap_sample)
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")

    # ── 10. Feature drift baseline ────────────────────────────────────────
    log.info("Fitting feature drift monitor baseline...")
    drift_monitor = FeatureDriftMonitor(cfg, final_features)
    drift_monitor.fit_baseline(full_train[final_features])

    # ── 11. Save artefacts ─────────────────────────────────────────────────
    log.info("Saving artefacts...")
    with open(output_dir / "lgbm_ranker.pkl", "wb") as f:
        pickle.dump(final_ranker, f)
    with open(output_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(output_dir / "drift_monitor.pkl", "wb") as f:
        pickle.dump(drift_monitor, f)
    # FIX 13: study metadata JSON replaces stale pickle (study object lags SQLite state)
    # (optuna_study_meta.json already written above)

    log.info(f"Training complete. Artefacts saved to {output_dir}")


if __name__ == "__main__":
    main()

