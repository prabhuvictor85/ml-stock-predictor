"""
backtest_run.py — Walk-forward backtest entry point.

Usage:
    python backtest_run.py --market nse --start 2018-01-01 --weighting inverse_vol

Outputs:
  - reports/report_{market}.html
  - reports/equity_curve_{market}.parquet
  - reports/performance_tables_{market}.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Stock Predictor — Backtest Runner")
    p.add_argument("--market", required=True, choices=["nse", "sp500", "nasdaq"])
    p.add_argument("--start", default=None, help="Backtest start date (format: YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Backtest end date (format: YYYY-MM-DD)")
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--weighting", choices=["equal", "inverse_vol"], default="equal")
    p.add_argument("--initial_nav", type=float, default=1_000_000)
    p.add_argument("--artefacts_dir", default="artefacts")
    p.add_argument("--panel_dir", default="panel")
    p.add_argument("--output_dir", default="reports")
    p.add_argument("--polygon_key", default="")
    p.add_argument("--tiingo_key", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline.config import get_config
    from pipeline.data.fetcher import DataFetcher
    from pipeline.data.panel import PanelConstructor
    from pipeline.features.engineer import FeatureEngineer
    from pipeline.targets.builder import TargetBuilder
    from pipeline.backtest.engine import BacktestEngine
    from pipeline.backtest.execution import ExecutionModel
    from pipeline.portfolio.constructor import PortfolioConstructor
    from pipeline.reports.generator import ReportGenerator
    from pipeline.explainability.shap_explainer import SHAPExplainer
    from pipeline.utils.logging import get_logger

    log = get_logger("backtest_run")
    cfg = get_config(args.market)
    artefacts_dir = Path(args.artefacts_dir) / args.market
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. LOAD ARTEFACTS ──────────────────────────────────────────────────
    def _load(name: str) -> Any:
        path = artefacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}. Run train.py first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    ensemble = _load("ensemble.pkl")
    selected_features = (artefacts_dir / "selected_features.txt").read_text().strip().split("\n")
    expected_features = set(selected_features)

    # ── 2. VALIDATE MODEL BINARY FEATURES VS selected_features.txt ────────
    base_model = ensemble.lgbm if hasattr(ensemble, "lgbm") else ensemble
    if hasattr(base_model, "feature_name"):
        model_expected_features = base_model.feature_name()
        if model_expected_features != selected_features:
            mismatch_selected = set(selected_features) - set(model_expected_features)
            mismatch_model = set(model_expected_features) - set(selected_features)
            raise ValueError(
                f"CRITICAL: Alignment Mismatch! The feature mapping array does not match the model requirements.\n"
                f"In selected_features.txt but missing from model layout: {mismatch_selected}\n"
                f"In model layout but missing from selected_features.txt: {mismatch_model}"
            )
        log.info("✅ Model feature vector structural validation passed.")

    # ── 3. LOAD COMPREHENSIVE PANEL & CHECK METADATA VERSIONS ─────────────
    panel_dir = Path(args.panel_dir) / args.market
    log.info(f"Loading panel matrix from {panel_dir}")
    panel = PanelConstructor.load(panel_dir)

    expected_version = getattr(ensemble, "feature_version", "v1.0.0")
    current_panel_version = panel.attrs.get("version", "v1.0.0")
    if expected_version != current_panel_version:
        log.warning(
            f"⚠️ VERSION MISMATCH: Ensemble built for feature version [{expected_version}], "
            f"but loaded data panel is version [{current_panel_version}]."
        )

    # Cache boundaries
    initial_dates = panel.index.get_level_values("date")
    start_str = initial_dates.min().strftime("%Y-%m-%d")
    end_str = initial_dates.max().strftime("%Y-%m-%d")

    # Fetch baseline benchmark series
    fetcher = DataFetcher(cfg, polygon_api_key=args.polygon_key, tiingo_api_key=args.tiingo_key)
    bm_raw = fetcher.fetch_benchmark(start_str, end_str)
    benchmark_close = bm_raw["close"].rename("benchmark_close")

    # ── 4. FEATURE & TARGET VALIDATION ────────────────────────────────────
    missing_features = expected_features - set(panel.columns)
    if missing_features:
        log.warning(f"Missing {len(missing_features)} expected features in panel schema. Re-building...")
        feat_eng = FeatureEngineer(cfg, benchmark_close)
        panel = feat_eng.build(panel)
        if expected_features - set(panel.columns):
            raise ValueError("Critical Error: Core features missing after explicit calculation loop.")
    else:
        log.info("✅ Feature validation passed. All selected features present in panel matrix.")

    required_targets = {"cs_rank_20d", "cs_rank_40d", "cs_rank_60d"}
    missing_targets = required_targets - set(panel.columns)
    if missing_targets:
        log.info(f"Target columns {missing_targets} missing. Generating target matrices...")
        target_builder = TargetBuilder(cfg)
        panel = target_builder.build(panel, benchmark_close)
        if required_targets - set(panel.columns):
            raise ValueError("Critical Error: Target building failed to materialize expected columns.")
    else:
        log.info("✅ Target validation passed. All forward horizons present.")

    # ── 5. TEMPORAL SLICING ────────────────────────────────────────────────
    panel_dates = panel.index.get_level_values("date")
    if args.start:
        panel = panel[panel_dates >= pd.Timestamp(args.start)]
        panel_dates = panel.index.get_level_values("date")
    if args.end:
        panel = panel[panel_dates <= pd.Timestamp(args.end)]
        panel_dates = panel.index.get_level_values("date")

    if panel.empty:
        raise ValueError(
            f"CRITICAL: Panel is empty after date filters. "
            f"Requested: {args.start} to {args.end}. "
            f"Available: {initial_dates.min().strftime('%Y-%m-%d')} to {initial_dates.max().strftime('%Y-%m-%d')}"
        )
    log.info(f"Panel after date filter: {len(panel)} rows.")

    # ── 6. BENCHMARK ALIGNMENT ─────────────────────────────────────────────
    unique_backtest_dates = panel_dates.unique().sort_values()
    sliced_bm = benchmark_close.reindex(unique_backtest_dates)
    missing_proportion = sliced_bm.isna().sum() / len(sliced_bm)
    if missing_proportion > 0.05:
        raise ValueError(
            f"CRITICAL: Benchmark series contains {missing_proportion:.2%} missing rows."
        )
    benchmark_close = sliced_bm.ffill()
    log.info(f"Aligned benchmark to {len(benchmark_close)} dates.")

    # ── 7. RUN BACKTEST ENGINE ─────────────────────────────────────────────
    execution = ExecutionModel(cfg)
    port_ctor = PortfolioConstructor(cfg, top_n=args.top_n, weighting=args.weighting)

    engine = BacktestEngine(
        cfg=cfg,
        ensemble=ensemble,
        port_ctor=port_ctor,
        execution=execution,
        feature_cols=selected_features,
        top_n=args.top_n,
        initial_nav=args.initial_nav,
    )

    log.info("Running backtest simulation engine...")
    report = engine.run(panel, benchmark_close)

    # ── 8. SHAP GLOBAL SNAPSHOT ────────────────────────────────────────────
    shap_img = output_dir / f"shap_global_{args.market}.png"
    try:
        latest_date = panel_dates.max()
        recent_univ = panel[(panel_dates == latest_date) & (panel["in_universe"] == True)]
        avail = [f for f in selected_features if f in recent_univ.columns]
        if avail and len(recent_univ) > 5:
            X_shap = recent_univ[avail]
            shap_exp = SHAPExplainer(base_model)
            shap_exp.plot_global(X_shap, output_path=shap_img)
            log.info(f"SHAP chart generated: {shap_img}")
    except Exception as e:
        log.warning(f"SHAP plot skipped: {e}")
        shap_img = None

    # ── 9. SAVE OUTPUTS ────────────────────────────────────────────────────
    rg = ReportGenerator(
        report=report,
        market_id=cfg.market_id,
        shap_img=shap_img if shap_img and Path(shap_img).exists() else None,
    )
    html_path = rg.generate(output_dir / f"report_{args.market}.html")

    eq_df = pd.DataFrame({
        "equity_gross": report.equity_curve_gross,
        "equity_net":   report.equity_curve_net,
        "benchmark":    report.benchmark_curve,
    })
    eq_df.to_parquet(output_dir / f"equity_curve_{args.market}.parquet")

    perf_dict = {
        "market":                  cfg.market_id,
        "gross_annual_return":     report.gross_annual_return,
        "net_annual_return":       report.net_annual_return,
        "gross_sharpe":            report.gross_sharpe,
        "net_sharpe":              report.net_sharpe,
        "max_drawdown":            report.max_drawdown,
        "calmar_ratio":            report.calmar_ratio,
        "hit_ratio":               report.hit_ratio,
        "top_decile_excess_return":report.top_decile_excess_return,
        "mean_weekly_turnover":    report.mean_weekly_turnover,
        "annualized_turnover":     report.annualized_turnover,
        "sector_attribution":      report.sector_attribution,
    }
    with open(output_dir / f"performance_tables_{args.market}.json", "w") as f:
        json.dump(perf_dict, f, indent=2)

    log.info(f"✅ Backtest complete. Report: {html_path}")


if __name__ == "__main__":
    main()
