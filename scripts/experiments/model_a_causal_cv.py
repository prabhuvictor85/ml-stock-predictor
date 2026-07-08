#!/usr/bin/env python3
"""
MODEL_A — CAUSAL zone CV (decision-grade re-test after look-ahead finding)
===========================================================================
The panel's zone columns were built with no cutoff (engineer.py build →
compute_zone_features(ohlcv)), so ZoneAnalyzer rewrote history using future
data (formation shift(-1), SDZ/SSZ breach scans, base_eliminator). All prior
zone CV numbers are inflated upper bounds.

This harness recomputes zone columns PER FOLD with a hard cutoff:
  fold test_year=Y →
    slice  = panel[dates <= Y-12-31]
    recompute_fold_features(slice, cutoff_date=(Y-1)-12-31)  [skip_ict]
    train  = rows with year <  Y   (zone state as known at cutoff — exactly
                                    what a live retrain on that date sees)
    test   = rows with year == Y   (zone state FROZEN at cutoff, carried
                                    forward via merge_asof)

Semantics of the result:
  * Train side matches live retrains / lockbox walk-forward steps exactly.
  * Test side is STALER than live (live refreshes zones daily; the lockbox
    walk refreshes every 14 days). So this CV is a conservative LOWER bound,
    the old leaked CV is the upper bound, and the lockbox should land in
    between. If this gate fails badly, do not burn the lockbox.

Leak-safety: raw zone columns are BLANKED before recompute, so a ticker whose
zone recompute throws contributes empty zones — never silently-leaked panel
values (recompute_fold_features keeps existing values on per-ticker failure).

Configs (decided by the 2026-07 bucket sweep, screening-grade):
  Z        30 zone features            (baseline)
  Z+B1     + 4 trend flags             (sweep: +0.0049, subsumed by B7)
  Z+B7     + 4 return/momentum         (sweep winner: +0.0062)
  Z+B1+B7  + 8                         (overlap check on honest baseline)

Run on Hetzner (~1.5-2.5 hrs; do NOT run anything else panel-sized):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_a_causal_cv.py \
        2>&1 | tee /tmp/causal_zone_cv.log &
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
OUT_PATH      = "/mnt/data/artefacts/experiments/causal_zone_cv_results.json"

ZONE_PREFIXES = ("features_sdz_", "features_ssz_", "features_dz_",
                 "features_sz_",  "features_zone_")

B1_TREND   = ["features_weekly_trend", "features_monthly_trend",
              "features_quarterly_trend", "features_yearly_trend"]
B7_RETURNS = ["features_return_1d", "features_return_5d",
              "features_return_20d", "features_return_60d"]

# Raw columns recompute_fold_features needs: OHLCV to re-run the analyzer,
# raw trend flags to rebuild sdz/ssz_htf_score at panel level.
OHLCV_COLS     = ["open", "high", "low", "close", "volume"]
RAW_TREND_COLS = ["weekly_trend", "monthly_trend", "quarterly_trend", "yearly_trend"]
LABEL_COLS     = ["cs_rank_composite", "future_20d_excess_return"]

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]

# Leaked-panel references for the comparison table (2026-07 bucket sweep).
LEAKED_REF = {"Z": 0.1920, "Z+B1": 0.1969, "Z+B7": 0.1983, "Z+B1+B7": None}


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def eval_fold(tr: pd.DataFrame, te: pd.DataFrame, features: list, date_level: str):
    import lightgbm as lgb
    from scipy.stats import spearmanr

    tr = tr.dropna(subset=features + ["cs_rank_composite"])
    te = te.dropna(subset=features + ["future_20d_excess_return"])
    if len(tr) < 5000 or len(te) < 500:
        return None

    # Group-contiguity invariant: rows must be date-major or the lambdarank
    # group array silently misaligns. Fail loudly, never train on garbage.
    if not tr.index.get_level_values(date_level).is_monotonic_increasing:
        raise RuntimeError("train slice is not date-major contiguous — "
                           "lambdarank groups would misalign; sort before training")

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

    ics, tops = [], []
    for dt in te.index.get_level_values(date_level).unique():
        grp = te.xs(dt, level=date_level)
        sc  = scores.xs(dt, level=date_level)
        m   = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
        if len(m) < 20 or m["score"].std() < 1e-9:
            continue
        ic, _ = spearmanr(m["score"], m["ret"])
        ics.append(ic)
        tops.append(m.nlargest(max(1, len(m)//10), "score")["ret"].mean())
    if not ics:
        return None
    return {"ic": float(np.mean(ics)), "top_dec": float(np.mean(tops)), "n_dates": len(ics)}


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    from pipeline.features.engineer import FeatureEngineer
    from pipeline.features.zone_features import _ZONE_COLS

    print("=" * 68)
    print("  MODEL_A — CAUSAL zone CV (per-fold cutoff recompute, skip_ict)")
    print("=" * 68)

    print(f"\nLoading panel: {DEFAULT_PANEL}")
    panel = pd.read_pickle(DEFAULT_PANEL)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    if "ticker" not in panel.index.names:
        sys.exit("FATAL: panel index has no 'ticker' level — recompute needs it.")

    zone_cols = [c for c in panel.columns if c.startswith(ZONE_PREFIXES)]

    # ── Fail loudly on anything missing ──────────────────────────────────────
    required = (OHLCV_COLS + RAW_TREND_COLS + LABEL_COLS + B1_TREND + B7_RETURNS
                + [c for c in _ZONE_COLS])
    missing = [c for c in required if c not in panel.columns]
    if missing:
        sys.exit(f"FATAL: panel missing required columns: {missing}")
    if len(zone_cols) < 25:
        sys.exit(f"FATAL: only {len(zone_cols)} zone feature columns found — expected ~30.")

    keep = list(dict.fromkeys(
        OHLCV_COLS + list(_ZONE_COLS) + zone_cols + RAW_TREND_COLS
        + B1_TREND + B7_RETURNS + LABEL_COLS
    ))
    panel = panel[keep].copy()
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    gc.collect()
    print(f"Trimmed panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols  "
          f"({panel.memory_usage(deep=True).sum()/1e9:.2f} GB)")
    print(f"Zone features: {len(zone_cols)}")

    # cfg only needs use_structure_features (absent → False); benchmark unused
    # in recompute_fold_features. skip_ict=True skips ICT recompute entirely.
    fe = FeatureEngineer(cfg=SimpleNamespace(), benchmark_close=pd.Series(dtype=float),
                         skip_ict=True)

    configs = {
        "Z":        zone_cols,
        "Z+B1":     zone_cols + B1_TREND,
        "Z+B7":     zone_cols + B7_RETURNS,
        "Z+B1+B7":  zone_cols + B1_TREND + B7_RETURNS,
    }
    fold_results: dict = {name: {} for name in configs}

    for test_year in FOLD_YEARS:
        cutoff   = pd.Timestamp(f"{test_year - 1}-12-31")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        dates    = panel.index.get_level_values(date_level)
        fold     = panel[dates <= test_end].copy()

        # Leak-safety: blank raw zone cols so a per-ticker recompute failure
        # yields EMPTY zones, never the panel's leaked values.
        for col in _ZONE_COLS:
            fold[col] = "" if "type" in col else np.nan

        t0 = time.time()
        print(f"\n─── fold {test_year}: recomputing zones with cutoff={cutoff.date()} "
              f"on {fold.shape[0]:,} rows ...", flush=True)
        fold = fe.recompute_fold_features(fold, cutoff_date=cutoff)
        # recompute_fold_features returns TICKER-major order; LightGBM group
        # arrays require DATE-major contiguous rows (cv.build_group_array
        # re-sorts for exactly this reason). Without this reorder the
        # lambdarank groups are misaligned — bug found 2026-07-08, which
        # invalidated the first causal run's verdict.
        fold = fold.reorder_levels([date_level, "ticker"]).sort_index()
        print(f"    recompute done in {(time.time()-t0)/60:.1f} min", flush=True)

        fdates = fold.index.get_level_values(date_level)
        tr = fold[fdates.year < test_year]
        te = fold[fdates.year == test_year]

        for name, feats in configs.items():
            r = eval_fold(tr, te, feats, date_level)
            if r is None:
                print(f"    {name:<8} skipped (insufficient data)")
                continue
            fold_results[name][test_year] = r
            print(f"    {name:<8} IC={r['ic']:+.4f}  top-dec={r['top_dec']:+.4f}  "
                  f"n_dates={r['n_dates']}")
        del fold, tr, te
        gc.collect()

    # ── Aggregate + report ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  CAUSAL RESULTS (vs leaked-panel references)")
    print("=" * 68)
    print(f"  {'Config':<9} {'IC':>8} {'t':>6} {'minIC':>8} {'f+':>3} "
          f"{'top-dec':>8} {'leakedIC':>9} {'inflation':>9}")
    print("  " + "-" * 66)

    out = {}
    for name in configs:
        ics = [r["ic"] for r in fold_results[name].values()]
        n = len(ics)
        if n == 0:
            continue
        mean_ic = float(np.mean(ics))
        std_ic  = float(np.std(ics))
        t_stat  = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
        n_pos   = sum(1 for x in ics if x > 0)
        top_d   = float(np.mean([r["top_dec"] for r in fold_results[name].values()]))
        gate    = (mean_ic >= 0.03) and (t_stat >= 2.0) and (n_pos >= 4)
        leaked  = LEAKED_REF.get(name)
        infl    = f"{leaked - mean_ic:+.4f}" if leaked is not None else "      —"
        lk      = f"{leaked:+.4f}" if leaked is not None else "        —"
        print(f"  {name:<9} {mean_ic:>+8.4f} {t_stat:>+6.2f} {min(ics):>+8.4f} "
              f"{n_pos:>3} {top_d:>+8.4f} {lk:>9} {infl:>9}"
              f"  {'GATE PASS' if gate else 'GATE FAIL'}")
        out[name] = {
            "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat,
            "min_fold_ic": float(min(ics)), "n_folds": n, "n_folds_positive": n_pos,
            "mean_top_decile_exc": top_d, "gate_pass": gate,
            "leaked_reference_ic": leaked,
            "fold_ics": {str(y): round(r["ic"], 4) for y, r in fold_results[name].items()},
            "n_features": len(configs[name]),
        }

    out["_meta"] = {
        "semantics": "train=zone state as of fold cutoff (matches live retrain); "
                     "test=state frozen at cutoff carried forward (staler than live "
                     "-> conservative lower bound). Leaked CV is the upper bound.",
        "gate": "mean_ic>=0.03 AND t_stat>=2.0 AND n_folds_positive>=4",
        "panel": DEFAULT_PANEL,
    }
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n  Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
