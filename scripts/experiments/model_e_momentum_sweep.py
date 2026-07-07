#!/usr/bin/env python3
"""
MODEL_E — Momentum + Base-Feature Sweep (pre-registered)
=========================================================
Implements docs/MODEL_E_PREREGISTRATION.md EXACTLY. Read that first; do not
change configs/thresholds here without a new pre-registration document.

Baseline: B7 trailing returns (4 features, audited causal).
Phases:   1) individual buckets on B7   2) greedy forward selection
          3) ALL buckets (ceiling / hidden-interaction check)
Gate:     20d IC >= 0.03 AND t >= 2.0 AND >= 4/6 folds positive
          (applied to baseline and greedy-final).
KEEP:     delta >= +0.005 AND t >= baseline_t - 1.0.
Horizons: 40d/60d ICs reported for information only — the gate is 20d.

All features here passed the 2026-07-07 truncation audit — no per-fold
recompute needed, so each config is just a training run (~1.5 min/fold).

Run on Hetzner (NOT concurrently with any other panel-loading job):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_e_momentum_sweep.py \
        2>&1 | tee /tmp/model_e_sweep.log &
"""
from __future__ import annotations

import gc
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
OUT_PATH      = "/mnt/data/artefacts/experiments/model_e_results.json"

# Pre-registered thresholds — see docs/MODEL_E_PREREGISTRATION.md §4.
KEEP_DELTA   = 0.005     # ~3x the ~0.0015 seeded-rerun IC noise floor
T_GUARD_GAP  = 1.0       # bucket t-stat may not fall below baseline_t - 1.0

BASELINE_B7 = ["features_return_1d", "features_return_5d",
               "features_return_20d", "features_return_60d"]

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
    "B8_sma_slopes": [
        "features_sma20_slope_5", "features_sma50_slope_5",
        "features_sma200_slope_10",
    ],
    # B9_misc is discovered at runtime: every features_* column that is not
    # zone/ICT/pivot/structure and not already in B7 or B1-B8. Printed loudly.
}

EXCLUDE_PREFIXES = (
    "features_sdz_", "features_ssz_", "features_dz_", "features_sz_",
    "features_zone_", "features_any_valid", "features_ict_",
    "features_pivot_", "features_structure_",
)

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
HORIZONS   = [20, 40, 60]   # 20d = gate; 40/60d informational (pre-registered)


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def run_cv(panel: pd.DataFrame, features: list, date_level: str) -> dict:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    fold_ics = {h: {} for h in HORIZONS}   # horizon -> {year: mean IC}
    top_dec  = []
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

        date_ics = {h: [] for h in HORIZONS}
        date_top = []
        for dt in te.index.get_level_values(date_level).unique():
            grp = te.xs(dt, level=date_level)
            sc  = scores.xs(dt, level=date_level)
            for h in HORIZONS:
                col = f"future_{h}d_excess_return"
                if col not in grp.columns:
                    continue
                m = pd.DataFrame({"score": sc, "ret": grp[col]}).dropna()
                if len(m) < 20 or m["score"].std() < 1e-9:
                    continue
                ic, _ = spearmanr(m["score"], m["ret"])
                date_ics[h].append(ic)
                if h == 20:
                    date_top.append(m.nlargest(max(1, len(m)//10), "score")["ret"].mean())

        for h in HORIZONS:
            if date_ics[h]:
                fold_ics[h][test_year] = float(np.mean(date_ics[h]))
        if date_top:
            top_dec.append(float(np.mean(date_top)))
        ic20 = fold_ics[20].get(test_year, float("nan"))
        print(f"    {test_year}: IC20={ic20:+.4f}"
              + "".join(f"  IC{h}={fold_ics[h].get(test_year, float('nan')):+.4f}"
                        for h in (40, 60)))
        gc.collect()

    def agg(h):
        vals = list(fold_ics[h].values())
        n = len(vals)
        if n == 0:
            return {"mean_ic": float("nan"), "t_stat": 0.0, "n_folds": 0,
                    "n_folds_positive": 0, "min_fold_ic": float("nan")}
        m, s = float(np.mean(vals)), float(np.std(vals))
        return {
            "mean_ic": m, "std_ic": s,
            "t_stat": float(m / (s / np.sqrt(n))) if n > 1 and s > 0 else 0.0,
            "n_folds": n, "n_folds_positive": sum(1 for x in vals if x > 0),
            "min_fold_ic": float(min(vals)),
            "fold_ics": {str(y): round(v, 4) for y, v in fold_ics[h].items()},
        }

    out = {f"h{h}": agg(h) for h in HORIZONS}
    out["mean_top_decile_exc"] = float(np.mean(top_dec)) if top_dec else float("nan")
    out["n_features"] = len(features)
    return out


def gate_pass(r) -> bool:
    a = r["h20"]
    return (a["mean_ic"] >= 0.03) and (a["t_stat"] >= 2.0) and (a["n_folds_positive"] >= 4)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print("=" * 66)
    print("  MODEL_E — Momentum sweep (pre-registered; gate = 20d IC)")
    print("=" * 66)

    print(f"\nLoading panel: {DEFAULT_PANEL}")
    panel = pd.read_pickle(DEFAULT_PANEL)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]

    # ── B9 discovery + fail-loudly checks ─────────────────────────────────────
    named = set(BASELINE_B7) | {c for cols in BUCKETS.values() for c in cols}
    b9 = sorted(
        c for c in panel.columns
        if c.startswith("features_") and not c.startswith(EXCLUDE_PREFIXES)
        and c not in named
    )
    if b9:
        print(f"\nB9_misc discovered {len(b9)} unclaimed base columns: {b9}")
        BUCKETS["B9_misc"] = b9

    label_cols = ["cs_rank_composite"] + [f"future_{h}d_excess_return" for h in HORIZONS]
    missing = [c for c in BASELINE_B7 + label_cols if c not in panel.columns]
    if missing:
        sys.exit(f"FATAL: panel missing required columns: {missing}")

    usable: dict = {}
    for name, cols in BUCKETS.items():
        absent = [c for c in cols if c not in panel.columns]
        if absent:
            print(f"  WARNING {name}: {len(absent)}/{len(cols)} columns absent — "
                  f"EXCLUDED from all phases: {absent}")
        else:
            usable[name] = cols

    keep = list(dict.fromkeys(
        BASELINE_B7 + [c for cols in usable.values() for c in cols] + label_cols
    ))
    panel = panel[keep].copy()
    panel = panel[panel.index.get_level_values(date_level) <= pd.Timestamp("2023-12-31")]
    gc.collect()
    print(f"\nFenced panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols  "
          f"({panel.memory_usage(deep=True).sum()/1e9:.2f} GB)\n")

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("─" * 48)
    print(f"BASELINE — B7 returns ({len(BASELINE_B7)} features)")
    base = run_cv(panel, BASELINE_B7, date_level)
    base["label"] = "B7 returns only"
    results["BASELINE_B7"] = base
    base_t = base["h20"]["t_stat"]
    print(f"  → IC20={base['h20']['mean_ic']:+.4f}  t={base_t:+.2f}  "
          f"IC40={base['h40']['mean_ic']:+.4f}  IC60={base['h60']['mean_ic']:+.4f}  "
          f"gate={'PASS' if gate_pass(base) else 'FAIL'}\n")

    # ── Phase 1: individual buckets ───────────────────────────────────────────
    print("=" * 66)
    print("  PHASE 1 — individual buckets on top of B7")
    print("=" * 66)
    for name, cols in usable.items():
        feats = BASELINE_B7 + cols
        print("─" * 48)
        print(f"{name} — B7 + {len(cols)} = {len(feats)} features")
        r = run_cv(panel, feats, date_level)
        r["label"] = f"B7 + {name}"
        r["bucket_cols"] = cols
        r["delta_ic"] = round(r["h20"]["mean_ic"] - base["h20"]["mean_ic"], 4)
        r["delta_per_feature"] = round(r["delta_ic"] / len(cols), 5)
        results[name] = r
        keepit = (r["delta_ic"] >= KEEP_DELTA
                  and r["h20"]["t_stat"] >= base_t - T_GUARD_GAP)
        verdict = "KEEP" if keepit else ("MARGINAL" if r["delta_ic"] > 0 else "DROP")
        print(f"  → IC20={r['h20']['mean_ic']:+.4f}  t={r['h20']['t_stat']:+.2f}  "
              f"delta={r['delta_ic']:+.4f}  d/feat={r['delta_per_feature']:+.5f}  {verdict}\n")

    # ── Phase 2: greedy forward selection ─────────────────────────────────────
    print("=" * 66)
    print("  PHASE 2 — greedy forward selection")
    print("=" * 66)
    selected: list = []
    current_ic = base["h20"]["mean_ic"]
    remaining = dict(usable)
    greedy_path = []
    round_no = 1
    while remaining:
        if round_no == 1:
            cand = {n: results[n]["h20"]["mean_ic"] for n in remaining}
        else:
            cand = {}
            for name, cols in remaining.items():
                feats = BASELINE_B7 + [c for b in selected for c in usable[b]] + cols
                print(f"  round {round_no}: trying +{name} ({len(feats)} features)")
                cand[name] = run_cv(panel, feats, date_level)["h20"]["mean_ic"]
        best = max(cand, key=cand.get)
        delta = cand[best] - current_ic
        if delta >= KEEP_DELTA:
            selected.append(best)
            current_ic = cand[best]
            del remaining[best]
            greedy_path.append({"round": round_no, "added": best,
                                "ic": round(current_ic, 4), "delta": round(delta, 4)})
            print(f"  round {round_no}: ADD {best}  IC20={current_ic:+.4f} "
                  f"(delta={delta:+.4f})\n")
            round_no += 1
        else:
            print(f"  round {round_no}: best candidate {best} adds only "
                  f"{delta:+.4f} < {KEEP_DELTA} — STOP\n")
            break

    greedy_feats = BASELINE_B7 + [c for b in selected for c in usable[b]]
    if selected:
        print(f"Re-scoring greedy final (full metrics) ...")
        gfin = run_cv(panel, greedy_feats, date_level)
    else:
        gfin = base
    gfin_out = dict(gfin)
    gfin_out["label"] = "B7 + " + (" + ".join(selected) if selected else "nothing")
    gfin_out["path"] = greedy_path
    gfin_out["selected_buckets"] = selected
    results["GREEDY_final"] = gfin_out
    print(f"GREEDY result: {gfin_out['label']}  IC20={gfin['h20']['mean_ic']:+.4f}  "
          f"gate={'PASS' if gate_pass(gfin) else 'FAIL'}\n")

    # ── Phase 3: ALL buckets ──────────────────────────────────────────────────
    print("=" * 66)
    print("  PHASE 3 — ALL usable buckets (ceiling check)")
    print("=" * 66)
    all_feats = BASELINE_B7 + [c for cols in usable.values() for c in cols]
    print(f"ALL — {len(all_feats)} features")
    r_all = run_cv(panel, all_feats, date_level)
    r_all["label"] = "B7 + ALL buckets"
    r_all["delta_ic"] = round(r_all["h20"]["mean_ic"] - base["h20"]["mean_ic"], 4)
    results["ALL_combined"] = r_all
    gap = r_all["h20"]["mean_ic"] - gfin["h20"]["mean_ic"]
    results["all_vs_greedy_gap"] = round(gap, 4)
    print(f"  → IC20={r_all['h20']['mean_ic']:+.4f}  delta={r_all['delta_ic']:+.4f}  "
          f"ALL-vs-greedy gap={gap:+.4f} "
          f"({'hidden interactions — investigate' if gap >= KEEP_DELTA else 'no hidden interactions'})\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 66)
    print("  SUMMARY  (gate horizon = 20d; 40/60d informational)")
    print("=" * 66)
    print(f"  {'Config':<16} {'n':>3} {'IC20':>8} {'t':>6} {'minIC':>8} {'f+':>3} "
          f"{'IC40':>8} {'IC60':>8} {'delta':>8}  verdict")
    print("  " + "-" * 88)
    for key, r in results.items():
        if not isinstance(r, dict) or "h20" not in r:
            continue
        a = r["h20"]
        verdict = ""
        if key in usable:
            keepit = (r.get("delta_ic", 0) >= KEEP_DELTA
                      and a["t_stat"] >= base_t - T_GUARD_GAP)
            verdict = "KEEP" if keepit else ("MARG" if r.get("delta_ic", 0) > 0 else "DROP")
        elif key in ("BASELINE_B7", "GREEDY_final"):
            verdict = "GATE PASS" if gate_pass(r) else "GATE FAIL"
        print(f"  {key:<16} {r['n_features']:>3} {a['mean_ic']:>+8.4f} {a['t_stat']:>+6.2f} "
              f"{a['min_fold_ic']:>+8.4f} {a['n_folds_positive']:>3} "
              f"{r['h40']['mean_ic']:>+8.4f} {r['h60']['mean_ic']:>+8.4f} "
              f"{r.get('delta_ic', 0.0):>+8.4f}  {verdict}")

    print("\n  Pre-registered gate: mean 20d IC >= 0.03 AND t >= 2.0 AND >= 4/6 folds positive")
    print("  References: MODEL_C ICT −0.00002 | MODEL_D pivot +0.0092 | "
          "MODEL_A zone (causal) — see causal_zone_cv_results.json")

    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  Full results → {OUT_PATH}")


if __name__ == "__main__":
    main()
