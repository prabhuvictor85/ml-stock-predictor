"""NSE (India) MarketConfig preset."""
from pipeline.config.base import MarketConfig, SlippageTier

NSE_CONFIG = MarketConfig(
    market_id="NSE",
    exchange_calendar="XBOM",
    benchmark_ticker="^NSEI",
    currency="INR",
    data_source_primary="nsepy",          # nsepy / Kite Connect
    data_source_fallback="yfinance",      # Yahoo Finance fallback
    sector_classification="NSE_SECTOR",
    min_adv_usd=500_000,                  # USD equivalent
    slippage_tiers_bps=[
        SlippageTier(adv_min_usd=50_000_000, adv_max_usd=float("inf"), slippage_bps=5),
        SlippageTier(adv_min_usd=10_000_000, adv_max_usd=50_000_000,  slippage_bps=15),
        SlippageTier(adv_min_usd=1_000_000,  adv_max_usd=10_000_000,  slippage_bps=40),
        SlippageTier(adv_min_usd=0,          adv_max_usd=1_000_000,   slippage_bps=80),
    ],
    commission_bps=3.0,                   # SEBI discount broker
    lot_size=1,
    adv_participation_cap=0.10,
    allow_short=False,                    # NSE regulatory constraint
    universe_reconstitution_freq="monthly_first",
    signal_rebalance_freq="weekly_last",
    max_sector_weight=0.40,
    max_single_stock_weight=0.15,
    max_portfolio_beta=1.3,
    profit_target_pct=0.08,
    stop_loss_pct=0.04,
    psi_alert_threshold=0.20,
    psi_retrain_threshold=0.25,
    random_seed=42,
)

