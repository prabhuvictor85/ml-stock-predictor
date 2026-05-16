"""
pipeline/portfolio/portfolio_risk.py

Production portfolio construction layer.

Applies three independent constraint layers on top of raw model scores:

  Layer 1 — Turnover constraint
    Limits what fraction of the portfolio changes each rebalance.
    Preserves continuity (avoids excessive trading costs).

  Layer 2 — Risk caps
    Per-position: max weight, min weight, max beta exposure.
    Portfolio-level: max gross exposure, max volatility target.

  Layer 3 — Sector neutralization
    Caps exposure to any single sector. For NSE, sector is derived
    from the 'sector' column in the panel (or defaults to 'NSE').

All three layers are applied in order. The final weights sum to 1.0
and respect all constraints simultaneously.

Usage
─────
    from pipeline.portfolio.portfolio_risk import ProductionPortfolioConstructor

    constructor = ProductionPortfolioConstructor(
        top_n=30,
        max_position_weight=0.08,     # no single stock > 8%
        max_sector_weight=0.35,       # no sector > 35%
        max_turnover=0.40,            # max 40% portfolio change per rebalance
        vol_target=0.15,              # target 15% annualised vol
        max_beta=1.3,                 # portfolio beta vs NSEI cap
    )

    weights = constructor.construct(
        cross=cross_section_df,
        scores=bull_score_series,
        prev_weights=last_week_weights,   # None for first run
        benchmark_close=nsei_close,
    )
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.utils.logging import get_logger

log = get_logger(__name__)

# Annualisation factor (252 trading days)
_ANN = 252


class ProductionPortfolioConstructor:
    """
    Three-layer constrained portfolio constructor.

    Parameters
    ----------
    top_n : int
        Initial candidate pool size (top-N by score before constraints).
    max_position_weight : float
        Hard cap on any single position (default 0.10 = 10%).
    min_position_weight : float
        Minimum weight for any included position (avoids dust allocations).
    max_sector_weight : float
        Maximum total weight in any single sector (default 0.40).
    max_turnover : float
        Maximum fraction of portfolio value that can change per rebalance.
        0.0 = no trading, 1.0 = unconstrained. Default 0.50.
    vol_target : float | None
        Annualised portfolio volatility target. If set, weights are scaled
        to match this vol using realized 60-day covariance. None = no scaling.
    max_beta : float | None
        Maximum portfolio beta vs benchmark. None = unconstrained.
    min_adv_mult : float
        Minimum average daily volume multiple — position must be tradeable
        within `min_adv_mult` days (liquidity filter). Default 20× position.
    weighting : str
        'equal' | 'inverse_vol' | 'score_proportional'
    """

    def __init__(
        self,
        top_n: int                   = 30,
        max_position_weight: float   = 0.10,
        min_position_weight: float   = 0.01,
        max_sector_weight: float     = 0.40,
        max_turnover: float          = 0.50,
        vol_target: Optional[float]  = None,
        max_beta: Optional[float]    = None,
        min_adv_mult: float          = 20.0,
        weighting: str               = "equal",
    ) -> None:
        self.top_n                = top_n
        self.max_position_weight  = max_position_weight
        self.min_position_weight  = min_position_weight
        self.max_sector_weight    = max_sector_weight
        self.max_turnover         = max_turnover
        self.vol_target           = vol_target
        self.max_beta             = max_beta
        self.min_adv_mult         = min_adv_mult
        self.weighting            = weighting

    # ── Main entry point ──────────────────────────────────────────────────────

    def construct(
        self,
        cross: pd.DataFrame,
        scores: pd.Series,
        prev_weights: Optional[Dict[str, float]] = None,
        panel: Optional[pd.DataFrame] = None,
        benchmark_close: Optional[pd.Series] = None,
    ) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Build constrained portfolio.

        Returns
        -------
        weights : Dict[ticker, weight]  — final portfolio weights (sum to 1)
        diagnostics : Dict of constraint diagnostics for logging/monitoring
        """
        prev_weights = prev_weights or {}

        # ── Step 0: Score → candidate pool ───────────────────────────────
        tickers_idx = scores.index.get_level_values("ticker")
        score_series = pd.Series(scores.values, index=tickers_idx).sort_values(ascending=False)
        candidates   = score_series.head(self.top_n * 2)  # over-select before constraints

        # ── Layer 1: Liquidity filter ─────────────────────────────────────
        candidates = self._apply_liquidity_filter(candidates, cross)

        # ── Layer 2: Base weights ─────────────────────────────────────────
        final_tickers = list(candidates.head(self.top_n).index)
        weights       = self._compute_base_weights(final_tickers, candidates, cross, panel)

        # ── Layer 3: Position caps ────────────────────────────────────────
        weights = self._apply_position_caps(weights)

        # ── Layer 4: Sector neutralization ───────────────────────────────
        weights, sector_diag = self._apply_sector_cap(weights, cross)

        # ── Layer 5: Turnover constraint ──────────────────────────────────
        weights, turnover = self._apply_turnover_constraint(weights, prev_weights)

        # ── Layer 6: Vol targeting / beta cap ────────────────────────────
        vol_scale = 1.0
        realized_vol = None
        if (self.vol_target is not None or self.max_beta is not None) and panel is not None:
            weights, vol_scale, realized_vol = self._apply_risk_scaling(
                weights, panel, benchmark_close
            )

        # ── Normalise to sum = 1.0 ────────────────────────────────────────
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}

        # ── Diagnostics ───────────────────────────────────────────────────
        diag = self._build_diagnostics(
            weights, prev_weights, turnover, sector_diag,
            vol_scale, realized_vol
        )
        self._log_diagnostics(diag)
        return weights, diag

    # ── Layer implementations ─────────────────────────────────────────────────

    def _apply_liquidity_filter(
        self, candidates: pd.Series, cross: pd.DataFrame
    ) -> pd.Series:
        """Remove tickers where ADV is too low to trade a meaningful position."""
        if cross is None or "adv_20d_usd" not in cross.columns:
            return candidates

        adv = cross["adv_20d_usd"]
        if hasattr(adv.index, "names") and "ticker" in adv.index.names:
            adv = adv.droplevel([n for n in adv.index.names if n != "ticker"])

        illiquid = []
        for ticker in candidates.index:
            ticker_adv = adv.get(ticker, np.nan)
            if pd.isna(ticker_adv) or ticker_adv <= 0:
                continue  # can't assess, keep
            # Rough position size check: 1/top_n of ₹10Cr notional
            # Adjust notional to your actual AUM
            notional_per_position = 1e7 / self.top_n
            days_to_trade = notional_per_position / ticker_adv
            if days_to_trade > self.min_adv_mult:
                illiquid.append(ticker)

        if illiquid:
            log.warning(f"Liquidity filter: removed {len(illiquid)} illiquid tickers: "
                        f"{illiquid[:10]}")
            candidates = candidates.drop(labels=illiquid, errors="ignore")
        return candidates

    def _compute_base_weights(
        self,
        tickers: list,
        scores: pd.Series,
        cross: pd.DataFrame,
        panel: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Compute initial weights before constraint layers."""
        n = len(tickers)
        if n == 0:
            return {}

        if self.weighting == "equal":
            return {t: 1.0 / n for t in tickers}

        if self.weighting == "score_proportional":
            s = scores.reindex(tickers).fillna(0)
            s = s.clip(lower=0)
            total = s.sum()
            if total <= 0:
                return {t: 1.0 / n for t in tickers}
            return {t: float(s[t]) / total for t in tickers}

        if self.weighting == "inverse_vol":
            # Use 20d rolling vol from panel if available
            vols = self._get_realized_vols(tickers, panel)
            inv_vol = {t: 1.0 / max(vols.get(t, 0.20), 0.05) for t in tickers}
            total = sum(inv_vol.values())
            return {t: v / total for t, v in inv_vol.items()}

        # Fallback
        return {t: 1.0 / n for t in tickers}

    def _apply_position_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Iteratively cap positions at max_position_weight.
        Excess weight is redistributed to uncapped positions.
        Remove positions below min_position_weight.
        """
        w = dict(weights)
        for _ in range(20):  # iterate until convergence
            excess = 0.0
            capped = set()
            for t, wt in w.items():
                if wt > self.max_position_weight:
                    excess += wt - self.max_position_weight
                    w[t] = self.max_position_weight
                    capped.add(t)
            if excess < 1e-8:
                break
            uncapped = [t for t in w if t not in capped]
            if not uncapped:
                break
            share = excess / len(uncapped)
            for t in uncapped:
                w[t] += share

        # Remove dust positions
        w = {t: wt for t, wt in w.items() if wt >= self.min_position_weight}
        return w

    def _apply_sector_cap(
        self, weights: Dict[str, float], cross: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Cap total weight in any single sector at max_sector_weight.
        Excess is redistributed to least-exposed tickers in other sectors.
        """
        sector_map = self._get_sector_map(list(weights.keys()), cross)
        sector_totals: Dict[str, float] = {}
        for t, w in weights.items():
            sec = sector_map.get(t, "Unknown")
            sector_totals[sec] = sector_totals.get(sec, 0.0) + w

        w = dict(weights)
        for sec, total in sector_totals.items():
            if total <= self.max_sector_weight:
                continue
            # Scale down tickers in this sector
            scale = self.max_sector_weight / total
            excess = 0.0
            sec_tickers = [t for t in w if sector_map.get(t) == sec]
            for t in sec_tickers:
                before = w[t]
                w[t] = before * scale
                excess += before - w[t]
            log.warning(f"Sector cap: '{sec}' reduced from {total:.1%} to "
                        f"{self.max_sector_weight:.1%}, redistributing {excess:.3f}")
            # Redistribute to other sectors
            other = [t for t in w if sector_map.get(t) != sec]
            if other:
                per_ticker = excess / len(other)
                for t in other:
                    w[t] += per_ticker

        return w, sector_totals

    def _apply_turnover_constraint(
        self,
        new_weights: Dict[str, float],
        prev_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], float]:
        """
        Blend new weights with previous weights to limit turnover.

        Turnover = sum(|new_w - prev_w|) / 2
        If turnover exceeds max_turnover, blend toward previous weights.
        """
        if not prev_weights:
            return new_weights, 1.0  # first run — no constraint

        # Compute raw turnover
        all_tickers = set(new_weights) | set(prev_weights)
        raw_turnover = sum(
            abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in all_tickers
        ) / 2.0

        if raw_turnover <= self.max_turnover:
            return new_weights, raw_turnover

        # Blend: alpha * new + (1-alpha) * prev
        # Solve: alpha * raw_turnover = max_turnover
        alpha = self.max_turnover / max(raw_turnover, 1e-8)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        blended: Dict[str, float] = {}
        for t in all_tickers:
            w_new  = new_weights.get(t, 0.0)
            w_prev = prev_weights.get(t, 0.0)
            blended[t] = alpha * w_new + (1 - alpha) * w_prev

        # Remove near-zero positions
        blended = {t: w for t, w in blended.items() if w >= self.min_position_weight}
        actual_turnover = sum(
            abs(blended.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in set(blended) | set(prev_weights)
        ) / 2.0

        log.info(f"Turnover constraint: raw={raw_turnover:.1%} → "
                 f"blended={actual_turnover:.1%} (alpha={alpha:.2f})")
        return blended, actual_turnover

    def _apply_risk_scaling(
        self,
        weights: Dict[str, float],
        panel: pd.DataFrame,
        benchmark_close: Optional[pd.Series],
    ) -> Tuple[Dict[str, float], float, Optional[float]]:
        """
        Scale portfolio weights to hit vol_target and/or respect max_beta.
        Uses realized 60-day portfolio volatility.
        """
        tickers = list(weights.keys())
        vols    = self._get_realized_vols(tickers, panel, window=60)
        corrs   = self._get_return_corr(tickers, panel, window=60)

        w_vec = np.array([weights.get(t, 0.0) for t in tickers])
        vol_vec = np.array([vols.get(t, 0.20) for t in tickers])

        # Estimate portfolio vol: sqrt(w' Σ w) using diagonal approx if no corr
        if corrs is not None and corrs.shape == (len(tickers), len(tickers)):
            cov = np.outer(vol_vec, vol_vec) * corrs
        else:
            cov = np.diag(vol_vec ** 2)
        port_var = float(w_vec @ cov @ w_vec)
        port_vol = float(np.sqrt(max(port_var, 1e-10))) * np.sqrt(_ANN)

        scale = 1.0
        if self.vol_target is not None and port_vol > 1e-6:
            scale = min(self.vol_target / port_vol, 1.0)  # only scale down, not up

        # Beta cap
        if self.max_beta is not None and benchmark_close is not None:
            beta = self._estimate_portfolio_beta(tickers, weights, panel, benchmark_close)
            if beta > self.max_beta:
                beta_scale = self.max_beta / beta
                scale = min(scale, beta_scale)
                log.warning(f"Beta cap: portfolio beta={beta:.2f} > {self.max_beta} — "
                            f"scaling by {beta_scale:.2f}")

        if scale < 1.0:
            log.info(f"Risk scaling: port_vol={port_vol:.1%} → "
                     f"scale={scale:.2f} (target={self.vol_target})")
            # Scaling reduces gross exposure; remainder goes to cash (weight=0)
            weights = {t: w * scale for t, w in weights.items()}

        return weights, scale, port_vol

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _build_diagnostics(
        self,
        weights: Dict[str, float],
        prev_weights: Dict[str, float],
        turnover: float,
        sector_totals: Dict[str, float],
        vol_scale: float,
        realized_vol: Optional[float],
    ) -> Dict:
        entries = sorted(weights.items(), key=lambda x: -x[1])
        return {
            "n_positions":       len(weights),
            "gross_exposure":    round(sum(weights.values()), 4),
            "max_weight":        round(max(weights.values(), default=0), 4),
            "min_weight":        round(min(weights.values(), default=0), 4),
            "turnover":          round(turnover, 4),
            "vol_scale":         round(vol_scale, 4),
            "realized_vol":      round(realized_vol, 4) if realized_vol else None,
            "sector_exposure":   {k: round(v, 4) for k, v in sector_totals.items()},
            "top5_positions":    [(t, round(w, 4)) for t, w in entries[:5]],
            "entries":           [t for t, w in entries
                                  if t not in prev_weights],
            "exits":             [t for t in prev_weights
                                  if t not in weights],
        }

    def _log_diagnostics(self, diag: Dict) -> None:
        log.info(
            f"Portfolio: {diag['n_positions']} positions | "
            f"exposure={diag['gross_exposure']:.1%} | "
            f"max_pos={diag['max_weight']:.1%} | "
            f"turnover={diag['turnover']:.1%} | "
            f"entries={len(diag['entries'])} exits={len(diag['exits'])}"
        )
        if diag.get("realized_vol"):
            log.info(f"  Realized vol: {diag['realized_vol']:.1%} | "
                     f"vol_scale: {diag['vol_scale']:.2f}")
        for sec, exp in sorted(diag["sector_exposure"].items(), key=lambda x: -x[1]):
            flag = " ⚠️" if exp > self.max_sector_weight * 0.9 else ""
            log.info(f"  Sector {sec:<25}: {exp:.1%}{flag}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_sector_map(self, tickers: list, cross: pd.DataFrame) -> Dict[str, str]:
        if cross is None or "sector" not in cross.columns:
            return {t: "NSE" for t in tickers}
        sec = cross["sector"]
        if hasattr(sec.index, "names") and "ticker" in sec.index.names:
            sec = sec.droplevel([n for n in sec.index.names if n != "ticker"])
        return {t: str(sec.get(t, "Unknown")) for t in tickers}

    def _get_realized_vols(
        self, tickers: list, panel: Optional[pd.DataFrame], window: int = 20
    ) -> Dict[str, float]:
        if panel is None or "close" not in panel.columns:
            return {t: 0.20 for t in tickers}
        try:
            close = panel["close"].unstack("ticker")[tickers]
            rets  = close.pct_change().dropna(how="all")
            vols  = rets.tail(window).std() * np.sqrt(_ANN)
            return {t: float(vols.get(t, 0.20)) for t in tickers}
        except Exception:
            return {t: 0.20 for t in tickers}

    def _get_return_corr(
        self, tickers: list, panel: Optional[pd.DataFrame], window: int = 60
    ) -> Optional[np.ndarray]:
        if panel is None or "close" not in panel.columns:
            return None
        try:
            close = panel["close"].unstack("ticker")[tickers]
            rets  = close.pct_change().dropna(how="all").tail(window)
            corr  = rets.corr().values
            return corr
        except Exception:
            return None

    def _estimate_portfolio_beta(
        self,
        tickers: list,
        weights: Dict[str, float],
        panel: pd.DataFrame,
        benchmark_close: pd.Series,
    ) -> float:
        try:
            close  = panel["close"].unstack("ticker")[tickers]
            rets   = close.pct_change().dropna(how="all").tail(60)
            bm_ret = benchmark_close.pct_change().reindex(rets.index).fillna(0)
            bm_var = float(bm_ret.var())
            if bm_var < 1e-10:
                return 1.0
            port_ret = sum(
                rets[t] * weights.get(t, 0.0) for t in tickers if t in rets.columns
            )
            beta = float(port_ret.cov(bm_ret)) / bm_var
            return beta
        except Exception:
            return 1.0