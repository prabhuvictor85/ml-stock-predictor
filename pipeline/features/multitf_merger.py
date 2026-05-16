from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from pipeline.utils.logging import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class MultiTFConfig:
    lookbacks: dict = None
    atr_window: int = 14
    normalize_atr: bool = True
    min_periods: int = 10

    def __post_init__(self):
        if self.lookbacks is None:
            self.lookbacks = {
                "weekly": 20,
                "monthly": 60,
                "quarterly": 120,
                "yearly": 240,
            }


# ─────────────────────────────────────────────────────────────
# Core Engine
# ─────────────────────────────────────────────────────────────

class MultiTFMerger:
    """
    Leakage-safe multi-timeframe feature generator.

    KEY DESIGN:
    ❌ NO resample-based calendar TFs
    ❌ NO merge_asof shifting hacks
    ✅ PURE rolling-window features aligned per row
    ✅ Index order matches panel convention: ["date", "ticker"]
    """

    def __init__(self, cfg: MultiTFConfig = None):
        self.cfg = cfg or MultiTFConfig()

    # ─────────────────────────────────────────────
    # ATR (normalized)
    # ─────────────────────────────────────────────
    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = np.maximum(high - low,
             np.maximum((high - close.shift(1)).abs(),
                        (low - close.shift(1)).abs()))

        atr = tr.rolling(self.cfg.atr_window,
                         min_periods=self.cfg.min_periods).mean()

        if self.cfg.normalize_atr:
            return atr / (close + 1e-8)

        return atr

    # ─────────────────────────────────────────────
    # Rolling trend engine
    # ─────────────────────────────────────────────
    def _rolling_trend(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Trend = close above rolling SMA. Fully leakage-safe."""
        sma = df["close"].rolling(window, min_periods=self.cfg.min_periods).mean()
        return (df["close"] > sma).astype(np.float32)

    # ─────────────────────────────────────────────
    # Volatility feature
    # ─────────────────────────────────────────────
    def _rolling_vol(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df["close"].pct_change().rolling(window).std()

    # ─────────────────────────────────────────────
    # transform: returns standalone features DataFrame
    # ─────────────────────────────────────────────
    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Input:  MultiIndex [date, ticker], columns: open/high/low/close/volume
        Output: MultiIndex [date, ticker] features-only DataFrame
        """
        panel = panel.sort_index().copy()
        out_parts = []

        for ticker, df in panel.groupby(level="ticker"):
            df = df.droplevel("ticker").sort_index()

            features = pd.DataFrame(index=df.index)

            features["atr_pct"] = self._atr(df)

            for tf, window in self.cfg.lookbacks.items():
                features[f"{tf}_trend"] = self._rolling_trend(df, window)
                features[f"{tf}_vol"] = self._rolling_vol(df, window)

            features["return_20d"] = df["close"].pct_change(20)
            features["return_60d"] = df["close"].pct_change(60)

            features = features.replace([np.inf, -np.inf], np.nan)

            # ── Index: [date, ticker] to match panel convention ──────��─────
            features.index = pd.MultiIndex.from_arrays(
                [features.index, [ticker] * len(features)],
                names=["date", "ticker"],
            )

            out_parts.append(features)

        result = pd.concat(out_parts).sort_index()

        log.info(
            f"MultiTFMerger generated | rows={len(result)} | "
            f"features={result.shape[1]}"
        )

        return result

    # ─────────────────────────────────────────────
    # merge: join features back into the full panel
    # ─────────────────────────────────────────────
    def merge(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Compute multi-TF features and left-join them back into panel.
        Returns panel with new columns added; original columns preserved.
        """
        features = self.transform(panel)
        return panel.join(features, how="left")


# Backward-compat alias
MultiTFMergerV2 = MultiTFMerger
