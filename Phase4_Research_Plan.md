# Phase 4: Institutional-Grade Feature Engineering Research Plan

## Critique of Current Architecture
The current feature engineering pipeline (`pipeline/features/engineer.py`, `ict_features.py`, `zone_features.py`) relies heavily on specialized Smart Money Concepts (ICT/Zones) and standard moving averages (SMA distance, slope). 
While it attempts to capture regime and trend, it lacks rigorous statistical properties typical of an institutional quant platform:
- **Missing Microstructure & Volume Signals:** Features like VWAP distance, rigorous volume imbalances (OBV, CMF), and high-fidelity volatility estimators (Parkinson, Yang-Zhang) are absent. This leaves the model blind to institutional accumulation footprints.
- **Naïve Volatility:** Uses standard Wilder ATR and basic return standard deviation. These do not distinguish between overnight gap volatility and intraday trend volatility.
- **Limited Factor Cross-Sectionalization:** Raw distances and basic ranks are used, but institutional alpha relies on beta-neutralization, sector-neutralization, and residual momentum to isolate idiosyncratic stock sentiment.
- **Distributional Blindspots:** No tracking of higher-order return moments (Skewness, Kurtosis), which are highly predictive of tail risks and mean-reversion characteristics.
- **Over-reliance on Binary Logic:** ICT/Zone features often rely on binary triggers, leading to sparse or saturated arrays. Continuous representations defined by factor distributions offer smoother gradients for LightGBM.

## Prioritized Experiment Backlog

### Exp-401: Volume Microstructure & Price-Volume Agreement [DONE]
- **Description:** Implement Volume-Weighted Average Price (VWAP) distance, Relative Volume (RVOL), On-Balance Volume (OBV), and Chaikin Money Flow (CMF).
- **Hypothesis:** Tracking whether price closes above/below VWAP alongside institutional relative volume identifies genuine accumulation vs. retail noise.
- **Expected Impact:** High. Volume confirms price action.
- **Estimation:** 1 day of implementation/testing.

### Exp-402: Extreme Volatility & Distribution Estimation [DONE]
- **Description:** Implement Parkinson, Garman-Klass, and Yang-Zhang volatility estimators. Track rolling skew and kurtosis over 20d/60d horizons.
- **Hypothesis:** Yang-Zhang volatility captures both overnight gaps and continuous intraday variance, providing a superior risk adjuster than basic ATR. Positive skew signals lottery-ticket stocks which tend to underperform long-term.
- **Expected Impact:** Medium-High. Improves risk-adjusted IC by filtering out high-volatility traps.
- **Estimation:** 1 day.

### Exp-403: Cross-Sectional & Residual Momentum [DONE]
- **Description:** Implement Beta-neutralized and Sector-neutralized momentum (Residual Momentum) using rolling linear regression against benchmark and sector means. Convert base features to Cross-Sectional Z-Scores (Standardized).
- **Hypothesis:** Raw momentum is heavily exposed to market beta. Isolating idiosyncratic return (residual momentum) provides purer alpha and creates a market-neutral signal edge.
- **Expected Impact:** High. Crucial for stability in varying market regimes.
- **Estimation:** 2 days (requires regression across groups).

### Exp-404: Trend Persistence & Market Breadth [DONE]
- **Description:** Implement Hurst Exponent, Fractal Dimension, and explicit Advance/Decline indicators (Breadth thrusts).
- **Hypothesis:** Hurst exponent efficiently discriminates between mean-reverting (H < 0.5), random walk (H = 0.5), and trending (H > 0.5) assets, directly conditioning momentum effectiveness. 
- **Expected Impact:** Medium. Excellent interaction feature for tree models.
- **Estimation:** 1-2 days.

## Success Criteria for Phase 4
1. **Rank IC Improvement:** A consistent increase in out-of-sample Rank IC by at least +0.01 to +0.015 absolute points across the 20d horizon.
2. **ICIR Boost:** Evaluated Information Ratio stays robust (> 1.0), penalizing volatile alpha delivery.
3. **Orthogonality:** Newly introduced features must demonstrate low absolute pairwise correlation (|rho| < 0.6) with the current ICT and Zone feature blocks, proving they add orthogonal information. 
4. **Execution Speed:** Feature engineer script must maintain parallel processing efficiency without excessive memory explosions (use Numba or vectorized Pandas strictly).

## Next Steps
Awaiting approval of this research plan. Once approved, we will begin sequential implementation starting with **Exp-401: Volume Microstructure**.
