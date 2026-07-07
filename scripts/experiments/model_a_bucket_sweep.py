#!/usr/bin/env python3
"""
MODEL_A — Bucket Sweep
======================
Tests each non-ICT, non-pivot feature bucket independently against the
zone-only baseline. Each bucket is added ON TOP of the 30 zone features
and the 4 trend features (the best configuration tested so far).

Buckets tested:
  B1  — Trend (4 features)              [already confirmed: IC +0.1958]
  B2  — Market Regime (3)
  B3  — Volatility & ADX full (11)
  B4  — Price vs SMA & Breakouts (7)
  B5  — Volume (2)
  B6  — Market Context (2)
  B7  — Returns / Momentum (4)
  B8  — SMA Slopes (3)
  ALL — All buckets B1-B8 combined

Run on Hetzner:
    cd /root/ml-stock-predictor
    python3 -u scripts/experiments/model_a_bucket_sweep.py \
        2>&1 | tee /tmp/bucket_sweep.log
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
OUT_PATH      = "/mnt/data/artefacts/experiments/bucket_sweep_results.json"

# ── Zone baseline (always included) ──────────────────────────────────────────
ZONE_COLS = (
    "features_sdz_", "features_ssz_", "features_dz_",
    "features_sz_",  "features_zone_",
)

# ── Buckets (exact column names — used as startswith prefixes) ────────────────
BUCKETS = {
    "B1_trend": [
        "features_weekly_trend", "features_monthly_trend",
        "features_quarterly_trend", "features_yearly_trend",
    ],
    "B2_regime": [
        "features_regime_bull", "features_regime_bear", "features_regime_choppy",
    ],
    "B3_vol_adx": [
        "features_atr_pct_rank_252", "features_vol_contraction",
        "features_compression_score", "features_adx_14",
        "features_plus_di", "features_minus_di", "features_adx_dir",
        "features_adx_bull", "features_adx_bear",
        "features_hist_vol_20d", "features_atr_expansion",
    ],
    "B4_sma_breakout": [
        "features_price_vs_sma20", "features_price_vs_sma50",
        "features_price_vs_sma200", "features_high_52w_dist",
        "features_low_52w_dist", "features_20d_breakout", "features_50d_breakout",
    ],
    "B5_volume": [
        "features_vol_ratio_5d", "features_vol_ratio_20d",
    ],
    "B6_context": [
        "features_sector_rs_20d", "features_market_breadth",
    ],
    "B7_returns": [
        "features_return_1d", "features_return_5d",
        "features_return_20d", "features_return_60d",
    ],
    "B8_sma_slopes": [
        "features_sma20_slope_5", "features_sma50_slope_5",
        "features_sma200_slope_10",
    ],
}

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def run_cv(panel: pd.DataFrame, features: list, date_level: str) -> dict:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    ics, top_dec = [], []
    for test_year in FOLD_YEARS:
        dates = panel.index.get_level_values(date_level)
        tr = panel[dates.year < test_year].dropna(subset=features + ["cs_rank_composite"])
        te = panel[dates.year == test_year].dropna(subset=features + ["future_20d_excess_return"])
        if len(tr) < 5000 or len(te) < 500:
            continue

        model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(
                tr[features], label=cs_rank_to_label(tr["cs_rank_composite"]),
                group=tr.groupby(level=date_level).size().sort_index().values,
                free_raw_data=False,
            ),
            num_boost_round=LGBM_PARAMS["n_estimators"],
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        scores = pd.Series(model.predict(te[features]), index=te.index)

        fold_ics, fold_top = [], []
        for dt in te.index.get_level_values(date_level).unique():
            grp = te.xs(dt, level=date_level)
            sc  = scores.xs(dt, level=date_level)
            m   = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
            if len(m) < 20 or m["score"].std() < 1e-9:
                continue
            ic, _ = spearmanr(m["score"], m["ret"])
            fold_ics.append(ic)
            fold_top.append(m.nlargest(max(1, len(m)//10), "score")["ret"].mean())

        mean_ic = float(np.mean(fold_ics)) if fold_ics else float("nan")
        ics.append(mean_ic)
        top_dec.append(float(np.mean(fold_top)) if fold_top else float("nan"))
        print(f"    {test_year}: IC={mean_ic:+.4f}")
        gc.collect()

    valid_ics = [x for x in ics if not np.isnan(x)]
    n = len(valid_ics)
    mean_ic = float(np.mean(valid_ics)) if n else float("nan")
    std_ic  = float(np.std(valid_ics))  if n else float("nan")
    t_stat  = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    return {
        "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat,
        "n_folds_positive": sum(1 for x in valid_ics if x > 0),
        "mean_top_decile_exc": float(np.mean([x for x in top_dec if not np.isnan(x)])),
        "n_features": len(features),
    }


def get_zone_cols(panel_cols):
    return [c for c in panel_cols if c.startswith(ZONE_COLS)]


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("=" * 64)
    print("  MODEL_A — Bucket Sweep (zone baseline + each bucket)")
    print("=" * 64)

    print(f"\nLoading panel: {DEFAULT_PANEL}")
    panel = pd.read_pickle(DEFAULT_PANEL)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    print(f"Fenced panel: {panel.shape[0]:,} rows, "
          f"{panel.index.get_level_values(date_level).nunique()} dates\n")

    zone_cols = get_zone_cols(panel.columns)
    print(f"Zone baseline: {len(zone_cols)} features\n")

    # Build ALL bucket cols for the combined test
    all_bucket_cols = []
    for cols in BUCKETS.values():
        all_bucket_cols.extend(cols)

    results = {}

    # ── Zone-only baseline ────────────────────────────────────────────────────
    print("─" * 48)
    print(f"BASELINE — Zone only ({len(zone_cols)} features)")
    keep = zone_cols + ["cs_rank_composite", "future_20d_excess_return"]
    p = panel[[c for c in keep if c in panel.columns]].copy()
    results["BASELINE_zone"] = run_cv(p, zone_cols, date_level)
    results["BASELINE_zone"]["label"] = "Zone only (30)"
    print(f"  → IC={results['BASELINE_zone']['mean_ic']:+.4f}  "
          f"t={results['BASELINE_zone']['t_stat']:+.2f}\n")
    del p; gc.collect()

    # ── Per-bucket tests (zone + each bucket) ────────────────────────────────
    for bucket_name, bucket_cols in BUCKETS.items():
        avail = [c for c in bucket_cols if c in panel.columns]
        missing = [c for c in bucket_cols if c not in panel.columns]
        if not avail:
            print(f"{bucket_name}: no columns found in panel — skipping\n")
            continue
        if missing:
            print(f"  WARNING {bucket_name}: {len(missing)}/{len(bucket_cols)} columns absent "
                  f"from panel — results NOT comparable to full-bucket run: {missing}")
            print(f"  SKIPPING {bucket_name} to keep experiments comparable\n")
            continue

        features = zone_cols + avail
        print("─" * 48)
        print(f"{bucket_name} — zone + {len(avail)} features = {len(features)} total")
        keep = features + ["cs_rank_composite", "future_20d_excess_return"]
        p = panel[[c for c in keep if c in panel.columns]].copy()
        r = run_cv(p, features, date_level)
        r["label"] = f"Zone + {bucket_name} ({len(features)})"
        r["bucket_cols"] = avail
        baseline_ic = results["BASELINE_zone"]["mean_ic"]
        r["delta_ic"] = round(r["mean_ic"] - baseline_ic, 4)
        results[bucket_name] = r
        verdict = "KEEP ✓" if r["delta_ic"] >= 0.005 else ("MARGINAL" if r["delta_ic"] > 0 else "DROP ✗")
        print(f"  → IC={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  "
              f"delta={r['delta_ic']:+.4f}  {verdict}\n")
        del p; gc.collect()

    # ── All buckets combined ──────────────────────────────────────────────────
    avail_all = [c for c in all_bucket_cols if c in panel.columns]
    features_all = zone_cols + avail_all
    print("─" * 48)
    print(f"ALL buckets combined — {len(features_all)} features total")
    keep = features_all + ["cs_rank_composite", "future_20d_excess_return"]
    p = panel[[c for c in keep if c in panel.columns]].copy()
    r = run_cv(p, features_all, date_level)
    r["label"] = f"Zone + ALL buckets ({len(features_all)})"
    r["delta_ic"] = round(r["mean_ic"] - results["BASELINE_zone"]["mean_ic"], 4)
    results["ALL_combined"] = r
    print(f"  → IC={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  "
          f"delta={r['delta_ic']:+.4f}\n")
    del p; gc.collect()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  {'Config':<28} {'n_zone':>6} {'n_bkt':>6} {'n_tot':>6} {'IC':>7} {'t':>6} {'delta':>7} {'top-dec':>8}")
    print("  " + "-" * 74)
    n_zone = len(zone_cols)
    for key, r in results.items():
        verdict = ""
        if key != "BASELINE_zone" and key != "ALL_combined":
            verdict = " KEEP" if r.get("delta_ic", 0) >= 0.005 else (" MARG" if r.get("delta_ic", 0) > 0 else " DROP")
        n_bkt = len(r.get("bucket_cols", [])) if key != "BASELINE_zone" else 0
        n_tot = r["n_features"]
        print(f"  {key:<28} {n_zone:>6} {n_bkt:>6} {n_tot:>6} {r['mean_ic']:>+7.4f} "
              f"{r['t_stat']:>+6.2f} {r.get('delta_ic', 0.0):>+7.4f} "
              f"{r['mean_top_decile_exc']:>+8.4f}{verdict}")

    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
