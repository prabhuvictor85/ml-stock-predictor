"""
infer.py — Weekly inference pipeline.

Usage:
    python infer.py --market nse
    python infer.py --market sp500 --top_n 10 --weighting inverse_vol

Steps:
  1. Load config + trained artefacts
  2. Build inference snapshot (NEVER a tail of the training panel — built fresh)
  3. Feature engineering on snapshot
  4. Ensemble scoring
  5. Portfolio construction
  6. SHAP per-stock explanations + similar-setup matching
  7. Output: watchlist.csv + explanations.json
  8. Weekly drift monitoring
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Stock Predictor — Inference Pipeline")
    p.add_argument("--market", required=True, choices=["nse", "sp500", "nasdaq"])
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--weighting", choices=["equal", "inverse_vol"], default="equal")
    p.add_argument("--artefacts_dir", default="artefacts")
    p.add_argument("--output_dir", default="output")
    p.add_argument("--polygon_key", default="")
    p.add_argument("--tiingo_key", default="")
    p.add_argument("--tickers_file", default=None)
    p.add_argument("--lookback_days", type=int, default=504,
                   help="Days of history needed for feature computation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from pipeline.config import get_config
    from pipeline.data.fetcher import DataFetcher
    from pipeline.data.universe import UniverseBuilder
    from pipeline.data.panel import PanelConstructor
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
    from pipeline.portfolio.constructor import PortfolioConstructor
    from pipeline.explainability.shap_explainer import SHAPExplainer
    from pipeline.explainability.setup_matcher import SetupMatcher
    from pipeline.monitoring.drift_monitor import FeatureDriftMonitor
    from pipeline.utils.logging import get_logger
    from pipeline.utils.calendar import get_trading_days

    log = get_logger("infer")
    cfg = get_config(args.market)
    artefacts_dir = Path(args.artefacts_dir) / args.market
    output_dir = Path(args.output_dir) / args.market
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load artefacts ─────────────────────────────────────────────────────
    log.info(f"Loading artefacts from {artefacts_dir}")

    def _load(name: str) -> Any:
        path = artefacts_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Artefact not found: {path}. Run train.py first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    ensemble = _load("ensemble.pkl")
    drift_monitor: FeatureDriftMonitor = _load("drift_monitor.pkl")
    selected_features = (artefacts_dir / "selected_features.txt").read_text().strip().split("\n")

    log.info(f"Loaded ensemble. Using {len(selected_features)} features.")

    # ── Build inference snapshot (fresh fetch — NOT a tail of training panel) ─
    # RULE 2: Inference snapshot is built separately from training panel.
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=args.lookback_days + 50)).strftime("%Y-%m-%d")

    fetcher = DataFetcher(cfg, polygon_api_key=args.polygon_key, tiingo_api_key=args.tiingo_key)
    universe_builder = UniverseBuilder(cfg)

    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [l.strip() for l in f if l.strip()]
    else:
        log.warning("No tickers_file — using benchmark ticker only (demo mode).")
        tickers = [cfg.benchmark_ticker]

    pc = PanelConstructor(cfg, fetcher, universe_builder)
    log.info(f"Building inference snapshot for {len(tickers)} tickers...")
    snapshot_panel = pc.build(tickers, start_date, end_date)

    # Fetch benchmark for feature engineering
    bm_raw = fetcher.fetch_benchmark(start_date, end_date)
    benchmark_close = bm_raw["close"].rename("benchmark_close")

    # ── Feature engineering ────────────────────────────────────────────────
    feat_eng = FeatureEngineer(cfg, benchmark_close)
    snapshot_panel = feat_eng.build(snapshot_panel)

    # ── Get the LATEST date's cross-section ───────────────────────────────
    latest_date = snapshot_panel.index.get_level_values("date").max()
    log.info(f"Inference date: {latest_date.date()}")

    current_cross = snapshot_panel.xs(latest_date, level="date")
    # Only in-universe tickers
    current_cross = current_cross[current_cross["in_universe"] == True]

    if len(current_cross) == 0:
        log.warning("No in-universe tickers at latest date. Exiting.")
        return

    # Re-attach date level for ensemble
    current_cross.index = pd.MultiIndex.from_arrays(
        [[latest_date] * len(current_cross), current_cross.index],
        names=["date", "ticker"]
    )

    # ── Ensemble scoring ───────────────────────────────────────────────────
    avail = [f for f in selected_features if f in current_cross.columns]
    X_infer = current_cross[avail].fillna(0)

    vol_col = "future_vol_20d" if "future_vol_20d" in current_cross.columns else None
    vol_series = current_cross[vol_col] if vol_col else None

    scores = ensemble.score(X_infer, vol_series)
    score_series = pd.Series(scores, index=current_cross.index)

    # ── Portfolio construction ─────────────────────────────────────────────
    port_ctor = PortfolioConstructor(cfg, top_n=args.top_n, weighting=args.weighting)
    # For portfolio construction, cs needs group_date; add it
    current_cross["group_date"] = latest_date
    ticker_scores, weights = port_ctor.construct(current_cross, score_series)

    if not weights:
        log.warning("Portfolio construction returned no holdings.")
        return

    log.info(f"Selected {len(weights)} stocks")

    # ── SHAP per-stock explanations ────────────────────────────────────────
    lgbm_model = ensemble.lgbm
    explanations: List[Dict] = []
    try:
        shap_exp = SHAPExplainer(lgbm_model)
        shap_values = shap_exp.compute(X_infer)

        # Determine regime for this date
        regime_col_bull = f"{FEATURE_PREFIX}regime_bull"
        regime_col_bear = f"{FEATURE_PREFIX}regime_bear"
        if regime_col_bull in current_cross.columns:
            bull_val = current_cross[regime_col_bull].mean()
            bear_val = current_cross[regime_col_bear].mean() if regime_col_bear in current_cross.columns else 0
            regime = "bull" if bull_val > 0.5 else ("bear" if bear_val > 0.5 else "choppy")
        else:
            regime = "unknown"

        ranked_tickers = sorted(weights.keys(), key=lambda t: -weights[t])
        for rank_pos, ticker in enumerate(ranked_tickers, 1):
            try:
                row_loc = current_cross.xs(ticker, level="ticker", drop_level=False)
                if row_loc.empty:
                    continue
                feat_idx = [list(X_infer.columns).index(f) for f in avail if f in avail]
                ticker_idx = list(X_infer.index.get_level_values("ticker")).index(ticker)
                shap_row = shap_values[ticker_idx]
                feat_names = list(X_infer.columns)
                feat_shap = list(zip(feat_names, shap_row))
                top5 = sorted(feat_shap, key=lambda x: -abs(x[1]))[:5]

                exp_dict = shap_exp.explain_stock(
                    ticker=ticker,
                    rank=rank_pos,
                    rank_score=float(weights[ticker]),
                    X_row=X_infer.loc[X_infer.index.get_level_values("ticker") == ticker].iloc[0],
                    shap_row=shap_row,
                    regime=regime,
                )
                # Attempt similar-setup matching (requires historical panel — use snapshot as proxy)
                try:
                    matcher = SetupMatcher(
                        historical_panel=snapshot_panel,
                        shap_values=shap_exp._shap_values if shap_exp._shap_values is not None else shap_values,
                        feature_names=feat_names,
                    )
                    sim = matcher.match(regime, top5, ticker)
                    exp_dict["similar_setups"] = sim
                except Exception:
                    exp_dict["similar_setups"] = {"note": "insufficient_history"}

                explanations.append(exp_dict)
            except Exception as e:
                log.warning(f"Explanation failed for {ticker}: {e}")
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")

    # ── Drift monitoring ───────────────────────────────────────────────────
    try:
        drift_df = drift_monitor.compute_weekly_drift(snapshot_panel, latest_date)
        # Save to a live-inference-specific path so these records never mix with
        # walk-forward monitoring (which reads monitoring/{mode}/feature_drift.parquet).
        # Mixing caused walk-forward to latch onto live 2026 dates and retrain every step.
        drift_monitor.save(Path(f"monitoring/live/{args.mode}"))
        log.info(f"Drift check: {len(drift_df)} features checked, "
                 f"{drift_df['alert'].sum()} alerts, "
                 f"{drift_df['retrain_flag'].sum()} retrain flags")
    except Exception as e:
        log.warning(f"Drift monitoring failed: {e}")

    # ── Output watchlist ───────────────────────────────────────────────────
    rows = []
    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        row = {
            "ticker": ticker,
            "weight": round(weight, 4),
            "score": round(float(ticker_scores.get(ticker, 0)), 4),
            "inference_date": str(latest_date.date()),
        }
        # Add key features
        if ticker in current_cross.index.get_level_values("ticker"):
            t_row = current_cross.xs(ticker, level="ticker", drop_level=False).iloc[0]
            for feat in ["features_sector_rs_20d", "features_rolling_beta_60d",
                         "features_adx_14", "features_vol_contraction"]:
                if feat in t_row.index:
                    row[feat] = round(float(t_row[feat]), 4)
        rows.append(row)

    watchlist_df = pd.DataFrame(rows)
    watchlist_path = output_dir / f"watchlist_{latest_date.strftime('%Y%m%d')}.csv"
    watchlist_df.to_csv(watchlist_path, index=False)
    log.info(f"Watchlist saved: {watchlist_path}")

    # Output explanations
    expl_path = output_dir / f"explanations_{latest_date.strftime('%Y%m%d')}.json"
    with open(expl_path, "w") as f:
        json.dump(explanations, f, indent=2)
    log.info(f"Explanations saved: {expl_path}")

    print(f"\n{'='*55}")
    print(f"  WEEKLY WATCHLIST — {args.market.upper()} — {latest_date.date()}")
    print(f"{'='*55}")
    print(watchlist_df.to_string(index=False))
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

