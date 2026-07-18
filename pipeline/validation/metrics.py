"""
CV metrics computation for the validation layer.
Computes mean_rank_ic, icir, NDCG@10, precision@10, hit_ratio, net_sharpe,
top_decile_excess_return.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.utils.logging import get_logger

log = get_logger(__name__)


def ndcg_at_k(relevance: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k for a single query group using LINEAR gain (gain = label).

    Linear gain matches LightGBM's label_gain=[0,1,...,n_bins-1] setting so
    that our evaluation metric is consistent with the lambdarank training
    objective. Exponential gain (2^label - 1) with labels up to 99 causes
    numerical collapse because IDCG ∝ 2^99 dwarfs any realistic DCG.

    Parameters
    ----------
    relevance : ground-truth relevance labels (integer, 0-based)
    scores    : predicted ranking scores (higher = better ranked)
    k         : cutoff
    """
    n = min(k, len(scores))
    if n == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:n]
    ideal_order = np.argsort(relevance)[::-1][:n]
    dcg  = sum(float(relevance[order[i]])       / np.log2(i + 2) for i in range(n))
    idcg = sum(float(relevance[ideal_order[i]]) / np.log2(i + 2) for i in range(n))
    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(y_true_top_quintile: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """Fraction of top-k ranked that are actually top quintile."""
    n = min(k, len(scores))
    if n == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:n]
    return float(y_true_top_quintile[order].mean())


def compute_fold_metrics(
    panel_test: pd.DataFrame,
    scores: pd.Series,
    feature_cols: List[str],
    benchmark_returns: pd.Series,
    commission_bps: float,
    slippage_bps: float,
    top_n: int = 10,
    invert_relevance: bool = False,
) -> dict:
    """
    Compute all CV metrics for one test fold.

    Parameters
    ----------
    panel_test        : test fold panel (MultiIndex date, ticker)
    scores            : ensemble score Series (same index)
    feature_cols      : used to extract cs_rank and quintile labels
    benchmark_returns : date-indexed benchmark return Series on the SAME horizon
                        as the portfolio returns it is subtracted from (~20d).
                        Pass e.g. benchmark_20d_return grouped per date — a
                        daily pct_change series here understates the benchmark
                        leg ~20x and inflates excess/net-sharpe.
    commission_bps    : from MarketConfig
    slippage_bps      : average slippage in bps
    top_n             : number of top stocks per week

    Returns
    -------
    dict with all metrics
    """
    from pipeline.models.lgbm_ranker import cs_rank_to_label

    # Slice only the columns this function reads before copying — the incoming
    # test fold carries ~300 feature columns that are never touched here, and
    # this runs once per fold per Optuna trial (~30x smaller copy).
    _METRIC_COLS = ["group_date", "in_universe", "cs_rank_20d",
                    "top_quintile", "bot_quintile",
                    "future_20d_return", "future_20d_excess_return"]
    panel_test = panel_test[[c for c in _METRIC_COLS if c in panel_test.columns]].copy()
    panel_test["_score"] = scores.reindex(panel_test.index).fillna(-999)

    ndcg_values: List[float] = []
    prec_values: List[float] = []
    hit_values: List[float] = []
    weekly_gross_rets: List[float] = []
    weekly_net_rets: List[float] = []
    weekly_bm_rets: List[float] = []

    # --- Per-date Rank IC computation ---
    # To get a statistically sound ICIR, we compute IC across ALL test dates
    # with sufficient observations, not just group_dates. Universe-filtered to
    # match the labels (cs_rank is built on in_universe rows only). Single
    # boolean filter + groupby instead of a per-date .loc loop — this runs
    # once per fold per Optuna trial, so the loop shape matters.
    # Interpretation note: adjacent daily ICs share 19/20 of the 20d label
    # window (autocorrelated series) — confirm any "ICIR > 1.0" claim on
    # non-overlapping dates before treating it as met.
    rank_ic_values: List[float] = []
    _ic_rows = panel_test[
        panel_test["future_20d_excess_return"].notna()
        & (panel_test["in_universe"] == True)
        & (panel_test["_score"] != -999)
    ]
    for _d, _gk_d in _ic_rows.groupby(level="date", sort=True):
        if len(_gk_d) >= 5:
            _ic_sc = _gk_d["_score"].values
            if _ic_sc.std() > 1e-9:
                ic, _ = spearmanr(_ic_sc, _gk_d["future_20d_excess_return"].values)
                if invert_relevance:
                    ic = -ic
                if not np.isnan(ic):
                    rank_ic_values.append(ic)

    group_dates = panel_test["group_date"].dropna().unique()

    for gd in sorted(group_dates):
        grp = panel_test[
            (panel_test["group_date"] == gd) & (panel_test["in_universe"] == True)
        ]
        if len(grp) < 5:
            continue

        sc = grp["_score"].values   # full-group scores (used for portfolio selection below)

        # Rank-quality metrics (NDCG / precision) are graded ONLY on stocks whose
        # forward outcome is known. Zero-filling a missing cs_rank label would grade
        # an unknown-outcome pick (delisted/halted/tail) as the WORST stock, biasing
        # the HPO objective. The return metrics below already dropna, so this keeps
        # the ranking and return sides consistent.
        _gk = grp[grp["cs_rank_20d"].notna()]
        if len(_gk) >= 5:
            _cs = _gk["cs_rank_20d"]
            if invert_relevance:
                _cs = 1.0 - _cs   # rank worst performers highest for bear model
            rel  = cs_rank_to_label(_cs).values
            sc_k = _gk["_score"].values
            _q_col = "bot_quintile" if invert_relevance else "top_quintile"
            top_q = _gk[_q_col].fillna(0).astype(int).values
            
            # NDCG and Precision
            ndcg_values.append(ndcg_at_k(rel, sc_k, k=10))
            prec_values.append(precision_at_k(top_q, sc_k, k=10))

        # Select top-N for this week
        top_idx = np.argsort(sc)[::-1][:top_n]
        top_rows = grp.iloc[top_idx]

        # Hit ratio: bear mode wants negative excess returns (stocks declined)
        exc_rets = top_rows["future_20d_excess_return"].dropna()
        if len(exc_rets) > 0:
            hit_values.append((exc_rets < 0).mean() if invert_relevance else (exc_rets > 0).mean())

        # 20-day portfolio return (equal weight)
        if "future_20d_return" in top_rows.columns:
            port_ret = top_rows["future_20d_return"].dropna().mean()
            cost_bps = commission_bps + slippage_bps
            net_ret = port_ret - 2 * cost_bps / 10000  # 2-way cost estimate
            weekly_gross_rets.append(port_ret)   # named legacy; actually 20d returns
            weekly_net_rets.append(net_ret)

        # Benchmark return for this week
        gd_ts = pd.Timestamp(gd)
        if gd_ts in benchmark_returns.index:
            weekly_bm_rets.append(benchmark_returns[gd_ts])

    mean_ndcg = float(np.mean(ndcg_values)) if ndcg_values else 0.0
    std_ndcg = float(np.std(ndcg_values)) if ndcg_values else 0.0
    mean_prec = float(np.mean(prec_values)) if prec_values else 0.0
    mean_hit = float(np.mean(hit_values)) if hit_values else 0.0

    mean_rank_ic = float(np.mean(rank_ic_values)) if rank_ic_values else 0.0
    # Adjacent daily ICs share 19/20 of their 20d label window, so the series
    # is smooth/autocorrelated and its short-sample std underestimates the
    # marginal IC dispersion — flattering the ICIR. Estimate the std from
    # non-overlapping subseries instead, POOLED across all 20 phase offsets:
    # a single offset (e.g. [::20]) keeps ~5% of the points and makes the
    # ICIR depend on which weekday the fold happens to start on.
    _n_ic = len(rank_ic_values)
    if _n_ic >= 40:  # ≥ ~2 points in every offset subseries
        _ic_arr = np.asarray(rank_ic_values, dtype=float)
        _ss, _dof = 0.0, 0
        for _k in range(20):
            _sub = _ic_arr[_k::20]
            if len(_sub) > 1:
                _ss  += float(np.var(_sub, ddof=1)) * (len(_sub) - 1)
                _dof += len(_sub) - 1
        std_rank_ic = float(np.sqrt(_ss / _dof)) if _dof > 0 else 0.0
    else:
        std_rank_ic = float(np.std(rank_ic_values, ddof=1)) if _n_ic > 1 else 0.0

    icir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0.0

    # Net Sharpe — align lengths before array subtraction to avoid shape mismatch
    # (weekly_net_rets and weekly_bm_rets can differ if some group_dates are
    #  missing from the benchmark index)
    # Returns are 20-day (monthly) periods — annualisation factor = 12 (months/year)
    # group_date fires ~monthly (every 20 trading days), so ~12 periods per year.
    _PERIODS_PER_YEAR = 12
    if len(weekly_net_rets) > 1:
        n_aligned   = min(len(weekly_net_rets), len(weekly_bm_rets))
        net_arr     = np.array(weekly_net_rets[:n_aligned])
        bm_arr      = np.array(weekly_bm_rets[:n_aligned]) if n_aligned > 0 else np.zeros(n_aligned)
        excess_net  = net_arr - bm_arr
        net_sharpe  = float(np.mean(excess_net) / (np.std(excess_net) + 1e-10) * np.sqrt(_PERIODS_PER_YEAR))
        max_dd      = _max_drawdown(np.cumprod(1 + np.array(weekly_net_rets)))
    else:
        net_sharpe = 0.0
        max_dd = 0.0

    # Top decile excess return.
    # For bear mode (invert_relevance=True) we negate so the metric is positive
    # when the model correctly selects stocks that decline vs benchmark.
    # This lets HPO use the same "prune if <= 0" guard for both modes.
    top_decile_exc = 0.0
    if len(weekly_gross_rets) > 0 and len(weekly_bm_rets) > 0:
        n = min(len(weekly_gross_rets), len(weekly_bm_rets))
        exc = np.array(weekly_gross_rets[:n]) - np.array(weekly_bm_rets[:n])
        top_decile_exc = float(np.mean(exc) * _PERIODS_PER_YEAR)  # annualized (12 × 20d periods)
        if invert_relevance:
            top_decile_exc = -top_decile_exc  # positive = stocks declined more than benchmark

    return {
        "mean_rank_ic": mean_rank_ic,
        "icir": icir,
        "mean_ndcg_at_10": mean_ndcg,
        "std_ndcg_at_10": std_ndcg,
        "precision_at_10": mean_prec,
        "hit_ratio": mean_hit,
        "net_sharpe": net_sharpe,
        "max_drawdown": max_dd,
        "top_decile_excess_return": top_decile_exc,
    }


def _max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute peak-to-trough max drawdown."""
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.where(peak > 0, peak, 1e-10)
    return float(dd.min())

