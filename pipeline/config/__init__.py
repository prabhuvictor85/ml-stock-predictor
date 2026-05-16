"""config package."""
from pipeline.config.base import MarketConfig, SlippageTier
from pipeline.config.nse import NSE_CONFIG
from pipeline.config.sp500 import SP500_CONFIG
from pipeline.config.nasdaq import NASDAQ_CONFIG

MARKET_CONFIGS = {
    "nse": NSE_CONFIG,
    "sp500": SP500_CONFIG,
    "nasdaq": NASDAQ_CONFIG,
}

def get_config(market: str) -> MarketConfig:
    key = market.lower()
    if key not in MARKET_CONFIGS:
        raise ValueError(f"Unknown market '{market}'. Choose from: {list(MARKET_CONFIGS.keys())}")
    return MARKET_CONFIGS[key]

__all__ = ["MarketConfig", "SlippageTier", "NSE_CONFIG", "SP500_CONFIG", "NASDAQ_CONFIG", "get_config"]

