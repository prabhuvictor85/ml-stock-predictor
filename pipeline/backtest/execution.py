"""
ExecutionModel — tiered slippage, commission, market impact, liquidity cap (§7.1).
All market-specific values read from cfg: MarketConfig.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pipeline.config.base import MarketConfig
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TradeResult:
    ticker: str
    direction: int          # +1 entry, -1 exit
    target_weight: float
    executed_weight: float  # after liquidity cap
    fill_price: float       # T+1 open
    slippage_bps: float
    commission_bps: float
    market_impact_bps: float
    total_cost_bps: float
    nav: float
    trade_size_usd: float
    adv_20d_usd: float


class ExecutionModel:
    """
    Models execution costs and liquidity constraints.

    Signal generated at close T → execution at open T+1.
    """

    def __init__(self, cfg: MarketConfig) -> None:
        self.cfg = cfg

    def compute_trade(
        self,
        ticker: str,
        target_weight: float,
        current_weight: float,
        nav: float,
        open_price_t1: float,
        adv_20d_usd: float,
    ) -> TradeResult:
        """
        Compute trade for one position change.

        Parameters
        ----------
        ticker           : ticker symbol
        target_weight    : desired portfolio weight
        current_weight   : current portfolio weight
        nav              : portfolio NAV in base currency
        open_price_t1    : T+1 open price (fill price)
        adv_20d_usd      : 20-day ADV in USD equivalent at signal date T
        """
        cfg = self.cfg

        # ── Liquidity cap ─────────────────────────────────────────────────
        max_position_usd = cfg.adv_participation_cap * adv_20d_usd
        max_weight = max_position_usd / nav if nav > 0 else target_weight
        if target_weight > max_weight:
            log.warning(
                f"{ticker}: target_weight={target_weight:.3f} capped to {max_weight:.3f} "
                f"(ADV cap: ${max_position_usd:,.0f})"
            )
        executed_weight = min(target_weight, max_weight)

        trade_size_usd = abs(executed_weight - current_weight) * nav
        direction = 1 if target_weight >= current_weight else -1

        # ── Slippage (tiered by ADV) ───────────────────────────────────────
        slippage_bps = cfg.get_slippage_bps(adv_20d_usd)

        # ── Commission ────────────────────────────────────────────────────
        commission_bps = cfg.commission_bps  # per leg

        # ── Market impact (linear model, §7.1) ────────────────────────────
        # impact_bps = 10 × sqrt(trade_size_usd / adv_20d_usd)
        # Applied only when trade > 1% ADV
        impact_bps = 0.0
        if adv_20d_usd > 0 and (trade_size_usd / adv_20d_usd) > 0.01:
            impact_bps = 10.0 * np.sqrt(trade_size_usd / adv_20d_usd)

        total_cost_bps = slippage_bps + commission_bps + impact_bps

        return TradeResult(
            ticker=ticker,
            direction=direction,
            target_weight=target_weight,
            executed_weight=executed_weight,
            fill_price=open_price_t1,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            market_impact_bps=impact_bps,
            total_cost_bps=total_cost_bps,
            nav=nav,
            trade_size_usd=trade_size_usd,
            adv_20d_usd=adv_20d_usd,
        )

    def apply_costs(
        self,
        gross_return: float,
        trades: list[TradeResult],
        portfolio_nav: float,
    ) -> float:
        """
        Deduct execution costs from gross return.
        Returns net return.
        """
        total_cost = 0.0
        for t in trades:
            cost_fraction = t.total_cost_bps / 10000
            total_cost += cost_fraction * (t.trade_size_usd / portfolio_nav)
        return gross_return - total_cost

