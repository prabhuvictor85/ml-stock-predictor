"""
backtest_run.py — Walk-forward backtest entry point.

Usage:
    python backtest_run.py --market sp500
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
    p.add_argument("--start", default=None, help="Backtest start date (default: full panel)")
    p.add_argument("--end", default=None, help="Backtest end date (default: latest in panel)")
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
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
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

    # ── Load artefacts ─────────────────────────────────────────────────────
    def _load(name: str) -> Any:
        path = artefacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}. Run train.py first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    ensemble = _load("ensemble.pkl")
    selected_features = (artefacts_dir / "selected_features.txt").read_text().strip().split("\n")

    # ── Load panel ─────────────────────────────────────────────────────────
    panel_dir = Path(args.panel_dir) / args.market
    log.info(f"Loading panel from {panel_dir}")
    panel = PanelConstructor.load(panel_dir)

    # Filter date range if specified
    dates = panel.index.get_level_values("date")
    if args.start:
        panel = panel[dates >= pd.Timestamp(args.start)]
    if args.end:
        panel = panel[dates <= pd.Timestamp(args.end)]
    log.info(f"Panel after date filter: {len(panel)} rows")

    # ── Fetch benchmark ────────────────────────────────────────────────────
    all_dates = panel.index.get_level_values("date")
    start_str = all_dates.min().strftime("%Y-%m-%d")
    end_str = all_dates.max().strftime("%Y-%m-%d")

    fetcher = DataFetcher(cfg, polygon_api_key=args.polygon_key, tiingo_api_key=args.tiingo_key)
    bm_raw = fetcher.fetch_benchmark(start_str, end_str)
    benchmark_close = bm_raw["close"].rename("benchmark_close")

    # ── Feature engineering (if not already done) ─────────────────────────
    feat_cols_existing = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
    if not feat_cols_existing:
        log.info("Running feature engineering on panel...")
        feat_eng = FeatureEngineer(cfg, benchmark_close)
        panel = feat_eng.build(panel)

    # ── Target building (if not already done) ─────────────────────────────
    if "cs_rank_20d" not in panel.columns:
        log.info("Building targets on panel...")
        target_builder = TargetBuilder(cfg)
        panel = target_builder.build(panel, benchmark_close)

    # ── Run backtest ───────────────────────────────────────────────────────
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

    log.info("Running backtest...")
    report = engine.run(panel, benchmark_close)

    # ── SHAP global for latest test fold ──────────────────────────────────
    shap_img = output_dir / f"shap_global_{args.market}.png"
    try:
        latest_date = panel.index.get_level_values("date").max()
        lookback_start = latest_date - pd.Timedelta(days=252)
        recent = panel[panel.index.get_level_values("date") >= lookback_start]
        recent_univ = recent[recent["in_universe"] == True]
        avail = [f for f in selected_features if f in recent_univ.columns]
        if avail and len(recent_univ) > 50:
            X_shap = recent_univ[avail].fillna(0).sample(min(2000, len(recent_univ)), random_state=42)
            shap_exp = SHAPExplainer(ensemble.lgbm)
            shap_exp.plot_global(X_shap, output_path=shap_img)
    except Exception as e:
        log.warning(f"SHAP plot failed: {e}")
        shap_img = None

    # ── Generate HTML report ───────────────────────────────────────────────
    rg = ReportGenerator(
        report=report,
        market_id=cfg.market_id,
        shap_img=shap_img if shap_img and Path(shap_img).exists() else None,
    )
    html_path = rg.generate(output_dir / f"report_{args.market}.html")

    # ── Save equity curves to parquet ──────────────────────────────────────
    eq_df = pd.DataFrame({
        "equity_gross": report.equity_curve_gross,
        "equity_net": report.equity_curve_net,
        "benchmark": report.benchmark_curve,
    })
    eq_path = output_dir / f"equity_curve_{args.market}.parquet"
    eq_df.to_parquet(eq_path)
    log.info(f"Equity curve saved: {eq_path}")

    # ── Save performance tables to JSON ───────────────────────────────────
    perf_dict = {
        "market": cfg.market_id,
        "gross_annual_return": report.gross_annual_return,
        "net_annual_return": report.net_annual_return,
        "gross_sharpe": report.gross_sharpe,
        "net_sharpe": report.net_sharpe,
        "max_drawdown": report.max_drawdown,
        "calmar_ratio": report.calmar_ratio,
        "hit_ratio": report.hit_ratio,
        "top_decile_excess_return": report.top_decile_excess_return,
        "mean_weekly_turnover": report.mean_weekly_turnover,
        "annualized_turnover": report.annualized_turnover,
        "sector_attribution": report.sector_attribution,
    }
    perf_path = output_dir / f"performance_tables_{args.market}.json"
    with open(perf_path, "w") as f:
        json.dump(perf_dict, f, indent=2)
    log.info(f"Performance tables saved: {perf_path}")

    log.info(f"✅ Backtest complete. HTML report: {html_path}")


if __name__ == "__main__":
    main()

