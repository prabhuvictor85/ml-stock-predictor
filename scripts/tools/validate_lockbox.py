"""
validate_lockbox.py -- independent statistical verdict on watchlist scores.

Answers the question the pipeline metrics never do: *is the edge distinguishable
from luck?* It reads the scores_detail_{mode}_{date}.json files a run already
emits, recomputes forward returns DIRECTLY from the price CSVs (never via
pipeline code, so a pipeline bug can't grade its own homework), and reports:

  - per-cross-section rank-IC (Spearman of score vs realized forward return)
  - mean IC, its t-stat (naive AND non-overlapping), and bootstrap 95% CI
  - IC decay curve across horizons (5/10/20/40/60d)
  - top-decile excess return (mean fwd return of top-10% scored minus universe)

This is READ-ONLY: it never trains, never writes pickles, never touches model
state. Run it on a clean lockbox walk for the real verdict, or on existing
walk-forward output as a "pulse check" (in-sample -- an upper bound on the edge).

Usage (Hetzner):
    python scripts/tools/validate_lockbox.py \
        --scores_dir /mnt/data/artefacts/us_local \
        --mode momentum --score_field model_score \
        --start 2024-01-01 --end 2026-05-13 \
        --out /mnt/data/artefacts/us_local/lockbox_verdict.json

Interpretation cheat-sheet:
    mean IC > 0 and |t-stat| > ~2  -> edge unlikely to be luck
    IC decay: should fade smoothly with horizon, not be all noise
    top-decile excess > 0 with CI excluding 0 -> tradeable picks beat the field
    Naive t-stat with overlapping windows is INFLATED -- trust the
    non-overlapping one, which subsamples dates >= horizon apart.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.json$")
HORIZONS = [5, 10, 20, 40, 60]


# -- data loading --------------------------------------------------------------

def load_score_files(scores_dir: Path, mode: str,
                     start: Optional[str], end: Optional[str]) -> Dict[pd.Timestamp, Path]:
    """Find scores_detail_{mode}_{date}.json under scores_dir (recursive)."""
    pattern = f"scores_detail_{mode}_*.json"
    hits = glob.glob(str(scores_dir / "**" / pattern), recursive=True)
    out: Dict[pd.Timestamp, Path] = {}
    for h in hits:
        m = DATE_RE.search(h)
        if not m:
            continue
        dt = pd.Timestamp(m.group(1))
        if start and dt < pd.Timestamp(start):
            continue
        if end and dt > pd.Timestamp(end):
            continue
        # If duplicate dates exist across folders, keep the lexically last path
        if dt not in out or h > str(out[dt]):
            out[dt] = Path(h)
    return dict(sorted(out.items()))


def extract_scores(path: Path, side: str, score_field: str) -> Dict[str, float]:
    """Return {ticker: score} for one date's scores_detail file."""
    d = json.load(open(path))
    scores: Dict[str, float] = {}
    for ticker, rec in d.items():
        node = rec.get(side, {})
        v = node.get(score_field)
        if v is not None and np.isfinite(v):
            scores[ticker] = float(v)
    return scores


def load_close_series(data_dir: Path, tickers: set) -> Dict[str, pd.Series]:
    """Load close price Series (indexed by date) for each ticker that has a CSV."""
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        p = data_dir / f"{t}-1d.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        dcol = next((c for c in df.columns if c in ("date", "datetime", "timestamp")), None)
        if dcol is None or "close" not in df.columns:
            continue
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
        s = (df.dropna(subset=[dcol])
               .set_index(dcol)["close"]
               .sort_index())
        s = s[~s.index.duplicated(keep="last")]
        out[t] = s
    return out


def forward_return(close: pd.Series, as_of: pd.Timestamp, horizon: int) -> float:
    """close[t+h]/close[t]-1 using TRADING-day offset. NaN if insufficient data."""
    pos = close.index.searchsorted(as_of, side="right") - 1
    if pos < 0 or pos >= len(close):
        return np.nan
    fwd = pos + horizon
    if fwd >= len(close):
        return np.nan
    c0, c1 = close.iloc[pos], close.iloc[fwd]
    if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0:
        return np.nan
    return c1 / c0 - 1.0


# -- statistics ----------------------------------------------------------------

def spearman_ic(scores: Dict[str, float], fwd: Dict[str, float]) -> Optional[float]:
    """Cross-sectional rank-IC = Spearman(score, forward return)."""
    common = [t for t in scores if t in fwd and np.isfinite(fwd[t])]
    if len(common) < 20:
        return None
    s = pd.Series({t: scores[t] for t in common}).rank()
    r = pd.Series({t: fwd[t] for t in common}).rank()
    if s.std() == 0 or r.std() == 0:
        return None
    return float(np.corrcoef(s.values, r.values)[0, 1])


def top_decile_excess(scores: Dict[str, float], fwd: Dict[str, float]) -> Optional[float]:
    """Mean fwd return of top-10% by score, minus the cross-section mean."""
    common = [t for t in scores if t in fwd and np.isfinite(fwd[t])]
    if len(common) < 20:
        return None
    df = pd.DataFrame({"s": [scores[t] for t in common],
                       "r": [fwd[t] for t in common]})
    cut = df["s"].quantile(0.90)
    top = df[df["s"] >= cut]
    if len(top) == 0:
        return None
    return float(top["r"].mean() - df["r"].mean())


def t_stat(series: np.ndarray) -> float:
    series = series[np.isfinite(series)]
    if len(series) < 2 or series.std(ddof=1) == 0:
        return 0.0
    return float(series.mean() / series.std(ddof=1) * np.sqrt(len(series)))


def bootstrap_ci(series: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple:
    """IID bootstrap CI of the mean (ignores autocorrelation — see block version)."""
    series = series[np.isfinite(series)]
    if len(series) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(series, size=len(series), replace=True).mean()
             for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# -- autocorrelation-aware statistics ------------------------------------------

def newey_west_tstat(series: np.ndarray, lags: int) -> float:
    """
    HAC (Newey-West) t-stat of the mean, robust to autocorrelation up to `lags`.
    Overlapping forward-return windows make consecutive ICs serially correlated;
    the naive t-stat ignores that and is inflated. This corrects the standard
    error with a Bartlett kernel.
    """
    x = series[np.isfinite(series)]
    n = len(x)
    if n < 3:
        return 0.0
    mu = float(x.mean())
    e = x - mu
    var = float(np.dot(e, e) / n)                      # gamma_0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)                     # Bartlett weight
        cov = float(np.dot(e[k:], e[:-k]) / n)
        var += 2.0 * w * cov
    if var <= 0:
        return 0.0
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else 0.0


def block_bootstrap_ci(series: np.ndarray, block_len: int,
                       n_boot: int = 5000, seed: int = 42) -> tuple:
    """
    Moving-block bootstrap CI of the mean. Resamples contiguous blocks so serial
    dependence (from overlapping windows) is preserved — an IID bootstrap on
    autocorrelated ICs understates the CI width.
    """
    x = series[np.isfinite(series)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len + 1
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        means.append(sample.mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def forward_return_lagged(close: pd.Series, as_of: pd.Timestamp,
                          horizon: int, lag: int = 1) -> float:
    """
    Realistic fill: rank on close[t], ENTER at close[t+lag], exit close[t+lag+h].
    Removes the same-bar look-ahead in forward_return() (which enters at the very
    close used to rank — unbuyable live).
    """
    pos = close.index.searchsorted(as_of, side="right") - 1
    if pos < 0:
        return np.nan
    entry = pos + lag
    exit_ = entry + horizon
    if entry >= len(close) or exit_ >= len(close):
        return np.nan
    c0, c1 = close.iloc[entry], close.iloc[exit_]
    if not (np.isfinite(c0) and np.isfinite(c1)) or c0 <= 0:
        return np.nan
    return c1 / c0 - 1.0


def top_decile_members(scores: Dict[str, float], fwd: Dict[str, float]) -> set:
    """Tickers in the top 10% by score among names with a realized forward return."""
    common = [t for t in scores if t in fwd and np.isfinite(fwd[t])]
    if len(common) < 20:
        return set()
    s = pd.Series({t: scores[t] for t in common})
    return set(s[s >= s.quantile(0.90)].index)


# -- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores_dir", required=True,
                    help="Root holding dated folders with scores_detail_*.json")
    ap.add_argument("--data_dir", default=None,
                    help="Price CSV dir (default: PATHS.stock_data.us)")
    ap.add_argument("--mode", default="momentum", choices=["momentum", "reversal"])
    ap.add_argument("--side", default="bull", choices=["bull", "bear"])
    ap.add_argument("--score_field", default="model_score",
                    choices=["model_score", "composite_score"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--primary_horizon", type=int, default=20)
    ap.add_argument("--fill_lag", type=int, default=1,
                    help="Trading-day lag between ranking and entry for the realistic "
                         "fill check (1 = rank close[t], buy close[t+1]).")
    ap.add_argument("--cost_bps", type=float, default=10.0,
                    help="Round-trip transaction cost (bps) for the net-of-cost "
                         "top-decile estimate.")
    ap.add_argument("--out", default=None, help="Write verdict JSON here")
    args = ap.parse_args()

    scores_dir = Path(args.scores_dir)
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from pipeline.config.paths import PATHS
        data_dir = Path(PATHS.stock_data.us)

    print(f"\n{'='*64}\n  LOCKBOX VALIDATOR -- {args.mode}/{args.side}/{args.score_field}\n{'='*64}")
    print(f"  scores_dir: {scores_dir}")
    print(f"  data_dir:   {data_dir}")

    files = load_score_files(scores_dir, args.mode, args.start, args.end)
    if not files:
        print("  No scores_detail files found in range -- nothing to validate.")
        sys.exit(1)
    print(f"  score dates: {len(files)}  "
          f"({min(files).date()} -> {max(files).date()})")

    # Universe of tickers referenced anywhere -> load their close series once.
    all_scores: Dict[pd.Timestamp, Dict[str, float]] = {}
    universe: set = set()
    for dt, p in files.items():
        sc = extract_scores(p, args.side, args.score_field)
        all_scores[dt] = sc
        universe |= set(sc.keys())
    print(f"  tickers referenced: {len(universe)}  -- loading price CSVs ...")
    closes = load_close_series(data_dir, universe)
    print(f"  price CSVs loaded:  {len(closes)}/{len(universe)} "
          f"({100*len(closes)/max(len(universe),1):.0f}% coverage)")

    # -- per-date IC at each horizon + top-decile excess at primary horizon --
    ph = args.primary_horizon
    ic_by_h: Dict[int, List[float]] = {h: [] for h in HORIZONS}
    ic_dates: Dict[int, List[pd.Timestamp]] = {h: [] for h in HORIZONS}
    tde_vals: List[float] = []
    tde_dates: List[pd.Timestamp] = []

    # Post-run diagnostics collected at the primary horizon:
    surv_scored: List[int] = []      # scored names that have a price CSV
    surv_valid:  List[int] = []      # of those, how many had a realized forward return
    topdec_seq:  List[set] = []      # top-decile membership per date (for turnover)
    ic_lag_vals: List[float] = []    # IC under realistic (lagged) fill
    tde_lag_vals: List[float] = []   # top-decile excess under realistic fill

    for dt, sc in all_scores.items():
        for h in HORIZONS:
            fwd = {t: forward_return(closes[t], dt, h) for t in sc if t in closes}
            ic = spearman_ic(sc, fwd)
            if ic is not None:
                ic_by_h[h].append(ic)
                ic_dates[h].append(dt)
            if h == ph:
                tde = top_decile_excess(sc, fwd)
                if tde is not None:
                    tde_vals.append(tde)
                    tde_dates.append(dt)
                # survivorship audit: scored-with-CSV vs realized-return available
                _scored = [t for t in sc if t in closes]
                surv_scored.append(len(_scored))
                surv_valid.append(sum(1 for t in _scored
                                      if np.isfinite(fwd.get(t, np.nan))))
                # turnover: top-decile membership this date
                topdec_seq.append(top_decile_members(sc, fwd))
                # realistic fill (rank close[t], enter close[t+lag])
                fwd_lag = {t: forward_return_lagged(closes[t], dt, h, args.fill_lag)
                           for t in sc if t in closes}
                _icl = spearman_ic(sc, fwd_lag)
                _tdl = top_decile_excess(sc, fwd_lag)
                if _icl is not None:
                    ic_lag_vals.append(_icl)
                if _tdl is not None:
                    tde_lag_vals.append(_tdl)

    # -- verdict assembly ----------------------------------------------------
    ic_arr = np.array(ic_by_h[ph], dtype=float)
    ph_dates = ic_dates[ph]

    # Median spacing between score dates -> #overlapping periods per window.
    # Drives the HAC lag count and the bootstrap block length.
    if len(ph_dates) > 1:
        _sp = np.diff([d.value for d in ph_dates]) / 8.64e13   # ns -> days
        med_spacing = float(np.median(_sp))
    else:
        med_spacing = float(ph)
    overlap_lags = max(1, int(np.ceil(ph / max(med_spacing, 1.0))))

    # Non-overlapping subsample: keep dates >= horizon trading days apart.
    nonoverlap_idx = []
    last_kept: Optional[pd.Timestamp] = None
    for i, d in enumerate(ph_dates):
        if last_kept is None or (d - last_kept).days >= ph * 1.4:  # ~h trading days
            nonoverlap_idx.append(i)
            last_kept = d
    ic_nonoverlap = ic_arr[nonoverlap_idx] if nonoverlap_idx else ic_arr

    mean_ic = float(np.mean(ic_arr)) if len(ic_arr) else 0.0
    ci_lo, ci_hi = bootstrap_ci(ic_arr)
    block_lo, block_hi = block_bootstrap_ci(ic_arr, block_len=overlap_lags + 1)
    nw_t = newey_west_tstat(ic_arr, overlap_lags)
    tde_arr = np.array(tde_vals, dtype=float)
    tde_ci = bootstrap_ci(tde_arr)

    # -- (1) autocorrelation-aware significance is folded into ic_primary below

    # -- (2) regime robustness: IC by calendar year ---------------------------
    yr_series: Dict[int, List[float]] = {}
    for d, v in zip(ph_dates, ic_arr):
        yr_series.setdefault(d.year, []).append(v)
    by_year = {}
    for yr, vals in sorted(yr_series.items()):
        a = np.array(vals, dtype=float)
        by_year[str(yr)] = {
            "mean_ic":       float(a.mean()),
            "n":             int(len(a)),
            "positive_rate": float((a > 0).mean()),
        }

    # -- (3) turnover + net-of-cost top-decile --------------------------------
    turnovers = [1.0 - len(a & b) / len(a)
                 for a, b in zip(topdec_seq[:-1], topdec_seq[1:]) if a and b]
    mean_turnover = float(np.mean(turnovers)) if turnovers else float("nan")
    rt_cost = args.cost_bps / 1e4
    gross_tde = float(np.mean(tde_arr)) if len(tde_arr) else 0.0
    # cost per rebalance = turnover * round-trip cost; subtract from per-period excess
    net_tde = (gross_tde - mean_turnover * rt_cost) if turnovers else gross_tde

    # -- (4) survivorship audit -----------------------------------------------
    drop_frac = [1.0 - v / s for s, v in zip(surv_scored, surv_valid) if s > 0]
    mean_drop = float(np.mean(drop_frac)) if drop_frac else float("nan")

    # -- (5) realistic-fill (lagged entry) IC + excess ------------------------
    ic_lag_arr  = np.array(ic_lag_vals, dtype=float)
    tde_lag_arr = np.array(tde_lag_vals, dtype=float)
    mean_ic_lag = float(ic_lag_arr.mean()) if len(ic_lag_arr) else 0.0

    verdict = {
        "generated_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "config": {
            "mode": args.mode, "side": args.side,
            "score_field": args.score_field,
            "primary_horizon": ph,
            "fill_lag": args.fill_lag,
            "cost_bps": args.cost_bps,
            "median_date_spacing_days": med_spacing,
            "overlap_lags": overlap_lags,
            "date_range": [str(min(files).date()), str(max(files).date())],
            "n_score_dates": len(files),
        },
        "ic_primary": {
            "mean": mean_ic,
            "std": float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
            "n_periods": int(len(ic_arr)),
            "t_stat_naive": t_stat(ic_arr),
            "t_stat_nonoverlap": t_stat(ic_nonoverlap),
            "n_nonoverlap": int(len(ic_nonoverlap)),
            "t_stat_newey_west": nw_t,
            "positive_rate": float((ic_arr > 0).mean()) if len(ic_arr) else 0.0,
            "boot_ci95_iid": [ci_lo, ci_hi],
            "boot_ci95_block": [block_lo, block_hi],
        },
        "ic_decay": {str(h): float(np.mean(ic_by_h[h])) if ic_by_h[h] else None
                     for h in HORIZONS},
        "ic_by_year": by_year,
        "top_decile_excess": {
            "mean": gross_tde,
            "t_stat": t_stat(tde_arr),
            "boot_ci95": list(tde_ci),
            "n_periods": int(len(tde_arr)),
        },
        "turnover": {
            "mean_per_rebalance": mean_turnover,
            "round_trip_cost_bps": args.cost_bps,
            "gross_excess_per_period": gross_tde,
            "net_excess_per_period": net_tde,
        },
        "realistic_fill": {
            "fill_lag_days": args.fill_lag,
            "mean_ic": mean_ic_lag,
            "t_stat_newey_west": newey_west_tstat(ic_lag_arr, overlap_lags),
            "mean_top_decile_excess": float(tde_lag_arr.mean()) if len(tde_lag_arr) else 0.0,
        },
        "survivorship": {
            "mean_frac_scored_without_realized_return": mean_drop,
            "note": ("Names scored but missing a realized fwd return (mostly "
                     "delisted) are dropped from IC -> survivor-biased UPWARD. "
                     "Cure needs --pit_universe + dead-ticker prices, not a "
                     "post-run fix."),
        },
    }

    # -- print human-readable verdict ----------------------------------------
    ip = verdict["ic_primary"]
    print(f"\n{'-'*64}\n  RANK-IC @ {ph}d (score vs realized forward return)\n{'-'*64}")
    print(f"  mean IC          : {ip['mean']:+.4f}   (std {ip['std']:.4f}, n={ip['n_periods']})")
    print(f"  t-stat (naive)   : {ip['t_stat_naive']:+.2f}   <- inflated by overlapping windows")
    print(f"  t-stat (non-ovl) : {ip['t_stat_nonoverlap']:+.2f}   (n={ip['n_nonoverlap']})")
    print(f"  t-stat (Newey-W) : {ip['t_stat_newey_west']:+.2f}   (HAC, lags={verdict['config']['overlap_lags']}) <- trust this")
    print(f"  positive rate    : {ip['positive_rate']:.0%}")
    print(f"  95% CI  (IID)    : [{ip['boot_ci95_iid'][0]:+.4f}, {ip['boot_ci95_iid'][1]:+.4f}]")
    print(f"  95% CI  (block)  : [{ip['boot_ci95_block'][0]:+.4f}, {ip['boot_ci95_block'][1]:+.4f}]  <- autocorr-aware")

    print(f"\n  IC decay curve:")
    for h in HORIZONS:
        v = verdict["ic_decay"][str(h)]
        bar = "#" * int(abs(v) * 200) if v is not None else ""
        print(f"    {h:>3}d : {v:+.4f} {bar}" if v is not None else f"    {h:>3}d :   n/a")

    print(f"\n  IC by year (regime robustness):")
    for yr, d in verdict["ic_by_year"].items():
        print(f"    {yr} : IC {d['mean_ic']:+.4f}  pos {d['positive_rate']:.0%}  (n={d['n']})")

    td = verdict["top_decile_excess"]
    print(f"\n  Top-decile excess @ {ph}d (close[t] fill):")
    print(f"    mean {td['mean']*100:+.2f}%  t={td['t_stat']:+.2f}  "
          f"CI95 [{td['boot_ci95'][0]*100:+.2f}%, {td['boot_ci95'][1]*100:+.2f}%]")

    rf = verdict["realistic_fill"]
    print(f"\n  Realistic fill (rank close[t], enter close[t+{rf['fill_lag_days']}]):")
    print(f"    IC {rf['mean_ic']:+.4f}  t(NW)={rf['t_stat_newey_west']:+.2f}  "
          f"top-decile excess {rf['mean_top_decile_excess']*100:+.2f}%")

    tv = verdict["turnover"]
    print(f"\n  Turnover & cost:")
    print(f"    top-decile turnover/rebalance : {tv['mean_per_rebalance']:.0%}")
    print(f"    gross excess/period {tv['gross_excess_per_period']*100:+.2f}%  ->  "
          f"net (after {tv['round_trip_cost_bps']:.0f}bps) {tv['net_excess_per_period']*100:+.2f}%")

    sv = verdict["survivorship"]
    print(f"\n  Survivorship audit:")
    print(f"    {sv['mean_frac_scored_without_realized_return']:.1%} of scored names/date had "
          f"NO realized return (dropped from IC) -> IC is survivor-biased UPWARD.")

    # -- one-line automated read (uses the autocorrelation-aware stats) -------
    t_use = ip["t_stat_newey_west"]                       # HAC — the honest t
    ci = ip["boot_ci95_block"]                            # autocorr-aware CI
    ci_excl_0 = (ci[0] > 0) or (ci[1] < 0)
    if ip["mean"] > 0 and abs(t_use) > 2 and ci_excl_0:
        read = ("PULSE: positive IC, Newey-West t>2, block-CI excludes 0 -- edge "
                "unlikely to be luck (still survivor-biased; see audit).")
    elif ip["mean"] > 0 and abs(t_use) > 1:
        read = ("WEAK: positive IC but HAC t<2 -- suggestive, not significant once "
                "autocorrelation is accounted for.")
    else:
        read = "FLATLINE: IC indistinguishable from zero under autocorrelation-aware stats."
    print(f"\n  >>> {read}\n")

    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2))
        print(f"  verdict written: {args.out}\n")


if __name__ == "__main__":
    main()
