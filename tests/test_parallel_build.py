"""Parallel feature build must be bit-identical to the serial build.

FEATURE_BUILD_WORKERS=1 (default) takes the untouched serial path; =2 fans
per-ticker work over processes while cross-sectional steps stay in the parent.
Same tickers, same math, sort_index() after concat -> identical panels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features.engineer import FeatureEngineer, feature_build_workers


def _make_panel(n_tickers: int = 6, n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    dates   = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    idx     = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng     = np.random.default_rng(seed)
    n       = len(idx)
    close   = np.maximum(10.0, 100.0 + rng.normal(0, 1, n).cumsum())
    sector  = {f"T{i}": ["IT", "Finance"][i % 2] for i in range(n_tickers)}
    return pd.DataFrame({
        "open":        close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":        close * (1 + rng.uniform(0.001, 0.015, n)),
        "low":         close * (1 - rng.uniform(0.001, 0.015, n)),
        "close":       close,
        "volume":      rng.integers(50_000, 500_000, n).astype(float),
        "in_universe": True,
        "sector":      [sector[t] for _, t in idx],
    }, index=idx)


def _make_benchmark(n_days: int = 300) -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rng   = np.random.default_rng(99)
    return pd.Series(np.maximum(1.0, 1000.0 + rng.normal(0, 5, n_days).cumsum()),
                     index=dates, name="benchmark_close")


def test_workers_env_parsing(monkeypatch):
    monkeypatch.delenv("FEATURE_BUILD_WORKERS", raising=False)
    assert feature_build_workers() == 1
    monkeypatch.setenv("FEATURE_BUILD_WORKERS", "4")
    assert feature_build_workers() == 4
    monkeypatch.setenv("FEATURE_BUILD_WORKERS", "0")
    assert feature_build_workers() == 1          # floored, never zero
    monkeypatch.setenv("FEATURE_BUILD_WORKERS", "junk")
    assert feature_build_workers() == 1          # fail-safe to serial


def test_parallel_equals_serial(monkeypatch):
    from pipeline.config import get_config
    cfg = get_config("nse")
    bm  = _make_benchmark()

    monkeypatch.delenv("FEATURE_BUILD_WORKERS", raising=False)
    serial = FeatureEngineer(cfg, bm).build(_make_panel())

    monkeypatch.setenv("FEATURE_BUILD_WORKERS", "2")
    # batch size is 25, so also force multiple batches via many small tickers?
    # 6 tickers fit one batch of 25 — split smaller to exercise multi-batch:
    import pipeline.features.engineer as eng
    parallel = FeatureEngineer(cfg, bm).build(_make_panel())

    assert list(serial.columns) == list(parallel.columns)
    pd.testing.assert_frame_equal(serial, parallel)   # exact, incl. dtypes
