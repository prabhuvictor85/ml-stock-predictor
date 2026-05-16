"""
PortfolioConstructor — selection, weighting, and risk limits (§8).

Rules:
- Top N by ensemble score.
- Sector cap: replace lowest-ranked over-sector-weight stock.
- Liquidity filter: exclude adv < min_adv_usd.
- Max single stock weight: cfg.max_single_stock_weight.
- Max portfolio beta: cfg.max_portfolio_beta.
- Weighting: equal or inverse-volatility.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


def _normalize_weights(weights: Dict[str, float], max_weight: float) -> Dict[str, float]:
    """Cap weights at max_weight and renormalize to sum=1."""
    for _ in range(10):  # iterate until convergence
        total = sum(weights.values())
        if total <= 0:
            break
        weights = {t: w / total for t, w in weights.items()}
        # Apply cap
        capped = False
        for t in list(weights.keys()):
            if weights[t] > max_weight:
                weights[t] = max_weight
                capped = True
        if not capped:
            break
    # Final renorm
    total = sum(weights.values())
    if total > 0:
        weights = {t: w / total for t, w in weights.items()}
    return weights


class PortfolioConstructor:
    """
    Constructs the weekly portfolio from a ranked cross-section.

    Parameters
    ----------
    cfg       : MarketConfig
    top_n     : number of stocks to hold
    weighting : 'equal' or 'inverse_vol'
    """

    def __init__(
        self,
        cfg: MarketConfig,
        top_n: int = 10,
        weighting: str = "equal",
    ) -> None:
        self.cfg = cfg
        self.top_n = top_n
        self.weighting = weighting

    def construct(
        self,
        cross_section: pd.DataFrame,
        scores: pd.Series,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Parameters
        ----------
        cross_section : DataFrame for one group_date, MultiIndex (date, ticker),
                        in_universe rows only.
        scores        : ensemble score Series (same index as cross_section)

        Returns
        -------
        (selected_tickers, weights_dict)
        selected_tickers : dict ticker → rank_score
        weights_dict     : dict ticker → portfolio weight (sum=1)
        """
        cfg = self.cfg
        cs = cross_section.copy()
        cs["_score"] = scores.reindex(cs.index).fillna(-999)

        # ── Liquidity filter ───────────────────────────────────────────────
        if "adv_20d_usd" in cs.columns:
            before = len(cs)
            cs = cs[cs["adv_20d_usd"] >= cfg.min_adv_usd]
            removed = before - len(cs)
            if removed > 0:
                log.warning(f"Portfolio: removed {removed} tickers below min_adv_usd={cfg.min_adv_usd}")

        if cs.empty:
            return {}, {}

        # ── Rank descending by score ───────────────────────────────────────
        cs = cs.sort_values("_score", ascending=False)

        # ── Greedy sector-capped top-N selection ──────────────────────────
        sector_col = "sector"
        max_sec_w = cfg.max_sector_weight  # 40%
        max_per_sector = max(1, int(np.ceil(self.top_n * max_sec_w)))

        selected_tickers: List[str] = []
        sector_counts: Dict[str, int] = {}

        for idx_row in cs.itertuples():
            if len(selected_tickers) >= self.top_n:
                break
            ticker = idx_row.Index[1] if isinstance(idx_row.Index, tuple) else idx_row.Index
            sec = getattr(idx_row, sector_col, "Unknown")
            if sector_counts.get(sec, 0) >= max_per_sector:
                continue
            selected_tickers.append(ticker)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        if not selected_tickers:
            return {}, {}

        selected_rows = cs.loc[cs.index.get_level_values("ticker").isin(selected_tickers)]
        selected_scores = selected_rows["_score"].to_dict()

        # ── Weighting ──────────────────────────────────────────────────────
        weights: Dict[str, float] = {}

        if self.weighting == "inverse_vol":
            vol_col = "future_vol_20d" if "future_vol_20d" in cs.columns else None
            if vol_col:
                vols = {}
                for t in selected_tickers:
                    t_rows = selected_rows.xs(t, level="ticker", drop_level=False) if t in selected_rows.index.get_level_values("ticker") else pd.DataFrame()
                    v = t_rows.iloc[0][vol_col] if not t_rows.empty else np.nan
                    vols[t] = float(v) if not np.isnan(float(v) if v is not None else np.nan) else 0.20
                inv_vol = {t: 1.0 / max(v, 1e-4) for t, v in vols.items()}
                total_inv = sum(inv_vol.values())
                weights = {t: inv_vol[t] / total_inv for t in selected_tickers}
            else:
                # Fallback to equal weight
                weights = {t: 1.0 / len(selected_tickers) for t in selected_tickers}
        else:
            # Equal weight
            weights = {t: 1.0 / len(selected_tickers) for t in selected_tickers}

        # ── Apply single-stock cap + renormalize ──────────────────────────
        weights = _normalize_weights(weights, cfg.max_single_stock_weight)

        # ── Portfolio beta check ───────────────────────────────────────────
        if "rolling_beta_60d" in cs.columns or "features_rolling_beta_60d" in cs.columns:
            beta_col = "features_rolling_beta_60d" if "features_rolling_beta_60d" in cs.columns else "rolling_beta_60d"
            port_beta = sum(
                weights.get(t, 0) * float(
                    selected_rows.xs(t, level="ticker", drop_level=False).iloc[0].get(beta_col, 1.0)
                    if t in selected_rows.index.get_level_values("ticker") else 1.0
                )
                for t in selected_tickers
            )
            if port_beta > cfg.max_portfolio_beta:
                log.warning(
                    f"Portfolio beta={port_beta:.2f} exceeds max {cfg.max_portfolio_beta}. "
                    "Scaling down high-beta positions."
                )
                # Scale down high-beta positions proportionally
                betas = {}
                for t in selected_tickers:
                    t_rows = selected_rows.xs(t, level="ticker", drop_level=False) if t in selected_rows.index.get_level_values("ticker") else pd.DataFrame()
                    betas[t] = float(t_rows.iloc[0].get(beta_col, 1.0)) if not t_rows.empty else 1.0

                scale_factor = cfg.max_portfolio_beta / port_beta
                adj_weights = {}
                for t in selected_tickers:
                    if betas[t] > 1.0:
                        adj_weights[t] = weights[t] * scale_factor
                    else:
                        adj_weights[t] = weights[t]
                weights = _normalize_weights(adj_weights, cfg.max_single_stock_weight)

        # Map ticker → score
        ticker_scores = {t: selected_scores.get(
            next((k for k in selected_scores.keys() if (k[1] if isinstance(k, tuple) else k) == t), None), 0.0
        ) for t in selected_tickers}

        return ticker_scores, weights

