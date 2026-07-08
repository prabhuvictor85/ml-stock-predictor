#!/usr/bin/env python3
"""
MODEL_A — Bucket Sweep (3-phase screening)
==========================================
Screens each non-ICT, non-pivot feature bucket against the zone-only
baseline, then greedily builds the best bucket combination.

Phases:
  1. INDIVIDUAL — zone + each bucket alone (main effects).
  2. GREEDY     — forward selection over buckets: each round, test every
                  remaining bucket on top of the currently-selected set;
                  add the best one if it improves IC by >= KEEP_THRESHOLD;
                  stop when nothing clears the bar. Captures interactions
                  involving the selected set.
  3. ALL        — zone + every bucket (ceiling check). If ALL beats the
                  greedy final by >= KEEP_THRESHOLD, positive interactions
                  exist among buckets greedy dropped — investigate pairs.

Known limitations (this is a SCREENING tool, not final selection):
  * Feature-count bias: an 11-feature bucket has more split candidates than
    a 2-feature one, so raw delta favors big buckets. delta_per_feature is
    reported alongside to expose this; neither number is capacity-fair.
  * Greedy misses pairs where NEITHER bucket helps alone but BOTH together
    do. The ALL-vs-greedy gap is the (weak) detector for that case.
  * Buckets are internally correlated (e.g. the four trend flags move
    together); the production FeatureSelector prunes within-bucket
    redundancy later. This sweep decides buckets, not features.
  * Fold ICs are equal-weighted across years regardless of date count.
  * Final arbiter is the 2024+ lockbox, which is never iterated on.

Run on Hetzner (phase 2 makes runtime data-dependent; typical 2-3.5 hrs,
--skip_greedy cuts it to ~100 min):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_a_bucket_sweep.py \
        2>&1 | tee /tmp/bucket_sweep.log &
"""
from __future__ import annotations

import argparse
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

# KEEP threshold: seeded re-runs of the zone baseline on this panel move mean
# IC by ~0.0015 (fold-sampling / LGBM nondeterminism noise floor). 0.005 is
# ~3x that floor. Engineering heuristic, not a formal test — the t-stat and
# min_fold_ic columns are the statistical guards; a bucket that raises mean
# IC while dropping t below ~8 or going negative in folds is still rejected.
KEEP_THRESHOLD = 0.005

# ── Zone baseline (always included) ──────────────────────────────────────────
ZONE_COLS = (
    "features_sdz_", "features_ssz_", "features_dz_",
    "features_sz_",  "features_zone_",
)

# ── Buckets (exact column names) ──────────────────────────────────────────────
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

    fold_ics: dict = {}          # year -> mean IC across that year's dates
    top_dec = []
    for test_year in FOLD_YEARS:
        dates = panel.index.get_level_values(date_level)
        tr = panel[dates.year < test_year].dropna(subset=features + ["cs_rank_composite"])
        te = panel[dates.year == test_year].dropna(subset=features + ["future_20d_excess_return"])
        if len(tr) < 5000 or len(te) < 500:
            continue

        # Group-contiguity invariant (bug class found 2026-07-08): lambdarank
        # group arrays require date-major contiguous rows. Fail loudly.
        if not tr.index.get_level_values(date_level).is_monotonic_increasing:
            raise RuntimeError("train slice not date-major — sort before training")

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

        date_ics, date_top = [], []
        for dt in te.index.get_level_values(date_level).unique():
            grp = te.xs(dt, level=date_level)
            sc  = scores.xs(dt, level=date_level)
            m   = pd.DataFrame({"score": sc, "ret": grp["future_20d_excess_return"]}).dropna()
            if len(m) < 20 or m["score"].std() < 1e-9:
                continue
            ic, _ = spearmanr(m["score"], m["ret"])
            date_ics.append(ic)
            date_top.append(m.nlargest(max(1, len(m)//10), "score")["ret"].mean())

        if date_ics:
            fold_ics[test_year] = float(np.mean(date_ics))
            top_dec.append(float(np.mean(date_top)))
            print(f"    {test_year}: IC={fold_ics[test_year]:+.4f}")
        gc.collect()

    valid = list(fold_ics.values())
    n = len(valid)
    mean_ic = float(np.mean(valid)) if n else float("nan")
    std_ic  = float(np.std(valid))  if n else float("nan")
    t_stat  = float(mean_ic / (std_ic / np.sqrt(n))) if n > 1 and std_ic > 0 else 0.0
    return {
        "mean_ic": mean_ic, "std_ic": std_ic, "t_stat": t_stat,
        "fold_ics": {str(y): round(v, 4) for y, v in fold_ics.items()},
        "min_fold_ic": float(min(valid)) if n else float("nan"),
        "n_folds_positive": sum(1 for x in valid if x > 0),
        "mean_top_decile_exc": float(np.mean(top_dec)) if top_dec else float("nan"),
        "n_features": len(features),
    }


def test_config(panel: pd.DataFrame, features: list, date_level: str) -> dict:
    keep = features + ["cs_rank_composite", "future_20d_excess_return"]
    p = panel[[c for c in keep if c in panel.columns]].copy()
    r = run_cv(p, features, date_level)
    del p
    gc.collect()
    return r


def main():
    ap = argparse.ArgumentParser(description="MODEL_A 3-phase bucket sweep")
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--skip_greedy", action="store_true",
                    help="Run phases 1 and 3 only (individual + ALL), ~100 min.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 64)
    print("  MODEL_A — Bucket Sweep (individual → greedy → ALL)")
    print("=" * 64)

    print(f"\nLoading panel: {args.panel}")
    panel = pd.read_pickle(args.panel)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    print(f"Fenced panel: {panel.shape[0]:,} rows, "
          f"{panel.index.get_level_values(date_level).nunique()} dates\n")

    zone_cols = [c for c in panel.columns if c.startswith(ZONE_COLS)]
    print(f"Zone baseline: {len(zone_cols)} features\n")

    # Fail loudly: only buckets with ALL columns present participate anywhere
    # (individual, greedy, ALL) — a shrunken bucket is a different experiment.
    usable: dict = {}
    for name, cols in BUCKETS.items():
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            print(f"  WARNING {name}: {len(missing)}/{len(cols)} columns absent — "
                  f"EXCLUDED from all phases: {missing}")
        else:
            usable[name] = cols
    print()

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("─" * 48)
    print(f"BASELINE — Zone only ({len(zone_cols)} features)")
    base = test_config(panel, zone_cols, date_level)
    base["label"] = "Zone only"
    results["BASELINE_zone"] = base
    print(f"  → IC={base['mean_ic']:+.4f}  t={base['t_stat']:+.2f}  "
          f"min={base['min_fold_ic']:+.4f}\n")

    # ── Phase 1: individual buckets ───────────────────────────────────────────
    print("=" * 64)
    print("  PHASE 1 — individual buckets (main effects)")
    print("=" * 64)
    for name, cols in usable.items():
        features = zone_cols + cols
        print("─" * 48)
        print(f"{name} — zone + {len(cols)} = {len(features)} features")
        r = test_config(panel, features, date_level)
        r["label"] = f"Zone + {name}"
        r["bucket_cols"] = cols
        r["delta_ic"] = round(r["mean_ic"] - base["mean_ic"], 4)
        r["delta_per_feature"] = round(r["delta_ic"] / len(cols), 5)
        results[name] = r
        verdict = ("KEEP" if r["delta_ic"] >= KEEP_THRESHOLD
                   else ("MARGINAL" if r["delta_ic"] > 0 else "DROP"))
        print(f"  → IC={r['mean_ic']:+.4f}  t={r['t_stat']:+.2f}  "
              f"min={r['min_fold_ic']:+.4f}  delta={r['delta_ic']:+.4f}  "
              f"d/feat={r['delta_per_feature']:+.5f}  {verdict}\n")

    # ── Phase 2: greedy forward selection ─────────────────────────────────────
    selected: list = []
    current_ic = base["mean_ic"]
    if not args.skip_greedy:
        print("=" * 64)
        print("  PHASE 2 — greedy forward selection")
        print("=" * 64)
        remaining = dict(usable)
        greedy_path = []
        round_no = 1
        while remaining:
            # Round 1 reuses phase-1 results (identical configs) — no re-run.
            if round_no == 1:
                cand = {n: results[n]["mean_ic"] for n in remaining}
            else:
                cand = {}
                for name, cols in remaining.items():
                    feats = zone_cols + [c for b in selected for c in usable[b]] + cols
                    print(f"  round {round_no}: trying +{name} ({len(feats)} features)")
                    cand[name] = test_config(panel, feats, date_level)["mean_ic"]
            best = max(cand, key=cand.get)
            delta = cand[best] - current_ic
            if delta >= KEEP_THRESHOLD:
                selected.append(best)
                current_ic = cand[best]
                del remaining[best]
                greedy_path.append({"round": round_no, "added": best,
                                    "ic": round(current_ic, 4), "delta": round(delta, 4)})
                print(f"  round {round_no}: ADD {best}  IC={current_ic:+.4f}  "
                      f"(delta={delta:+.4f})\n")
                round_no += 1
            else:
                print(f"  round {round_no}: best candidate {best} adds only "
                      f"{delta:+.4f} < {KEEP_THRESHOLD} — STOP\n")
                break
        greedy_feats = zone_cols + [c for b in selected for c in usable[b]]
        results["GREEDY_final"] = {
            "label": "Zone + " + (" + ".join(selected) if selected else "nothing"),
            "path": greedy_path, "selected_buckets": selected,
            "mean_ic": current_ic,
            "delta_ic": round(current_ic - base["mean_ic"], 4),
            "n_features": len(greedy_feats),
        }
        print(f"GREEDY result: {results['GREEDY_final']['label']}  "
              f"IC={current_ic:+.4f}\n")

    # ── Phase 3: ALL buckets combined ─────────────────────────────────────────
    print("=" * 64)
    print("  PHASE 3 — ALL usable buckets combined (ceiling check)")
    print("=" * 64)
    all_cols = [c for cols in usable.values() for c in cols]
    features_all = zone_cols + all_cols
    print(f"ALL — zone + {len(all_cols)} = {len(features_all)} features")
    r_all = test_config(panel, features_all, date_level)
    r_all["label"] = "Zone + ALL buckets"
    r_all["delta_ic"] = round(r_all["mean_ic"] - base["mean_ic"], 4)
    results["ALL_combined"] = r_all
    print(f"  → IC={r_all['mean_ic']:+.4f}  t={r_all['t_stat']:+.2f}  "
          f"min={r_all['min_fold_ic']:+.4f}  delta={r_all['delta_ic']:+.4f}\n")

    if not args.skip_greedy:
        gap = r_all["mean_ic"] - current_ic
        results["all_vs_greedy_gap"] = round(gap, 4)
        if gap >= KEEP_THRESHOLD:
            print(f"  ⚠ ALL beats greedy by {gap:+.4f} — hidden positive interactions "
                  f"among non-selected buckets. Investigate pairs before lockbox.\n")
        else:
            print(f"  ALL-vs-greedy gap {gap:+.4f} < {KEEP_THRESHOLD} — "
                  f"no evidence of hidden interactions.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"  {'Config':<20} {'n_tot':>5} {'IC':>8} {'t':>6} {'minIC':>8} "
          f"{'f+':>3} {'delta':>8} {'d/feat':>8}  verdict")
    print("  " + "-" * 82)
    for key, r in results.items():
        if not isinstance(r, dict) or "mean_ic" not in r:
            continue
        verdict = ""
        if key in usable:
            verdict = ("KEEP" if r["delta_ic"] >= KEEP_THRESHOLD
                       else ("MARG" if r["delta_ic"] > 0 else "DROP"))
        minic = f"{r['min_fold_ic']:>+8.4f}" if "min_fold_ic" in r else " " * 8
        fpos  = f"{r['n_folds_positive']:>3}"  if "n_folds_positive" in r else "  -"
        dpf   = f"{r['delta_per_feature']:>+8.5f}" if "delta_per_feature" in r else " " * 8
        print(f"  {key:<20} {r['n_features']:>5} {r['mean_ic']:>+8.4f} "
              f"{r.get('t_stat', 0.0):>+6.2f} {minic} {fpos} "
              f"{r.get('delta_ic', 0.0):>+8.4f} {dpf}  {verdict}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  Full results → {args.out}")


if __name__ == "__main__":
    main()
