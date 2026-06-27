"""
MarketConfig — single source of truth for all market-specific parameters.
No module may import a market constant from anywhere else.
Adding a new market requires only a new config instance — zero code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class SlippageTier:
    """ADV bucket (in USD) → slippage in basis points."""
    adv_min_usd: float          # inclusive lower bound (0 = no floor)
    adv_max_usd: float          # exclusive upper bound (inf = no cap)
    slippage_bps: float


@dataclass
class MarketConfig:
    """
    All market-specific constants.  Every pipeline module receives an instance
    of this class and reads exclusively from it.

    RULE: MarketConfig is the single source of truth for all market-specific
    values.  Adding a new market requires only a new config instance.
    """

    # ── Identity ────────────────────────────────────────────────────────────
    market_id: str                        # e.g. "NSE", "SP500", "NASDAQ"
    exchange_calendar: str                # pandas_market_calendars key
    benchmark_ticker: str                 # e.g. "^NSEI", "SPY", "QQQ"
    currency: str                         # "INR" | "USD"

    # ── Data sources ────────────────────────────────────────────────────────
    data_source_primary: str              # primary fetch adapter identifier
    data_source_fallback: str             # fallback fetch adapter identifier

    # ── Universe & liquidity ────────────────────────────────────────────────
    sector_classification: str            # "NSE_SECTOR" | "GICS_L1"
    min_adv_usd: float                    # minimum 20d ADV in USD equivalent
    slippage_tiers_bps: List[SlippageTier] = field(default_factory=list)
    commission_bps: float = 1.0           # bps per leg
    lot_size: int = 1
    adv_participation_cap: float = 0.10   # max % of 20d ADV per order
    allow_short: bool = False

    # ── Rebalance schedule ───────────────────────────────────────────────────
    universe_reconstitution_freq: str = "monthly_first"   # first trading day of month
    signal_rebalance_freq: str = "weekly_last"            # last trading day of ISO week

    # ── Risk limits ──────────────────────────────────────────────────────────
    max_sector_weight: float = 0.40
    max_single_stock_weight: float = 0.15
    max_portfolio_beta: float = 1.3

    # ── Trade management ─────────────────────────────────────────────────────
    profit_target_pct: float = 0.08       # 8%
    stop_loss_pct: float = 0.04           # 4%
    position_stop_loss_pct: Optional[float] = None  # optional position-level stop

    # ── Drift monitoring ─────────────────────────────────────────────────────
    psi_alert_threshold: float = 0.20
    psi_retrain_threshold: float = 0.25

    # ── Reproducibility ──────────────────────────────────────────────────────
    random_seed: int = 42

    # ── Experimental features (toggle on/off) ────────────────────────────────
    # BOS/CHoCH/liquidity-sweep market-structure features (pipeline.features.
    # structure_features). Default OFF → baseline behaviour unchanged. When ON,
    # engineer.py emits features_{major,internal}_* + features_structure_alignment,
    # computed causally per CV fold (cutoff_date guard, mirrors zone features).
    use_structure_features: bool = False
    structure_major_swing: int = 25
    structure_minor_swing: int = 5

    def get_slippage_bps(self, adv_usd: float) -> float:
        """Return slippage bps for the given 20d ADV in USD."""
        for tier in sorted(self.slippage_tiers_bps, key=lambda t: t.adv_min_usd, reverse=True):
            if adv_usd >= tier.adv_min_usd:
                return tier.slippage_bps
        # fallback: highest-cost tier
        return max(t.slippage_bps for t in self.slippage_tiers_bps) if self.slippage_tiers_bps else 50.0

