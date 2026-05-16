"""Shared type definitions / result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FoldResult:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    ndcg_at_10: float
    top_decile_excess_return: float
    precision_at_10: float
    hit_ratio: float
    net_sharpe: float
    max_drawdown: float
    selected_features: List[str]
    params: Dict[str, Any]


@dataclass
class CVResult:
    folds: List[FoldResult]
    mean_ndcg: float
    std_ndcg: float
    mean_top_decile_excess: float
    mean_net_sharpe: float
    objective_value: float   # mean_ndcg - 0.5 * std_ndcg


@dataclass
class PortfolioSnapshot:
    date: pd.Timestamp
    holdings: Dict[str, float]        # ticker → weight
    rank_scores: Dict[str, float]     # ticker → ensemble score
    explanations: List[Dict[str, Any]]


@dataclass
class PerformanceReport:
    gross_annual_return: float
    net_annual_return: float
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    calmar_ratio: float
    hit_ratio: float
    top_decile_excess_return: float
    mean_weekly_turnover: float
    annualized_turnover: float
    sector_attribution: Dict[str, float]
    equity_curve_gross: pd.Series
    equity_curve_net: pd.Series
    benchmark_curve: pd.Series

