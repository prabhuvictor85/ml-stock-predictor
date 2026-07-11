#!/usr/bin/env python3
"""
MODEL_F — All-Causal-Features CV (pre-registered)
==================================================
Implements docs/MODEL_F_PREREGISTRATION.md EXACTLY, including the 2026-07-11
pre-run amendment (§8: fold-boundary purge, NaN handling, selection
determinism). One run, as specified.

Feature set: every `features_*` column EXCEPT the zone family (prefixes
sdz/ssz/dz/sz/zone/any_valid — non-causal in the panel AND proven dead),
PLUS the E2 momentum kernel (e2_mom_3m/6m/12m, computed in-script from close —
pure trailing arithmetic, truncation-safe by construction).

Configs (§4):
  MOM        — E2 KERNEL+S momentum reference (7 features, known ~+0.0168)
  ALL        — every causal feature + kernel, no selection
  BASE+MOM   — base technical state + kernel (ICT and pivots dropped)
  ALL+SELECT — ALL, then fold-local LGBM gain-importance top-40, refit

Gate (§5, applied to ALL and ALL+SELECT only):
  mean IC20 >= 0.03 AND t >= 2.0 (ddof=1) AND >= 4/6 folds positive.

Purge (§8.1): the train slice for test year Y ends MAX_FORWARD_HORIZON (60)
trading days before Y's first trading day, so no train label window overlaps
test-period returns.

Run on Hetzner (only when nothing else panel-sized is running):
    cd /root/ml-stock-predictor
    nohup python3 -u scripts/experiments/model_f_all_causal.py \
        > /tmp/model_f.log 2>&1 &

Local smoke test (synthetic panel, small trees — infra check only, NOT a run):
    python scripts/experiments/model_f_all_causal.py \
        --panel <synthetic.pkl> --out <tmp.json> --smoke
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

import numpy as np
import pandas as pd

from pipeline.targets.builder import MAX_FORWARD_HORIZON  # purge gap, stays synced

DEFAULT_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"
DEFAULT_OUT   = "/mnt/data/artefacts/experiments/model_f_results.json"

FENCE      = pd.Timestamp("2023-12-31")
FOLD_YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
HORIZONS   = [20, 40, 60]
PURGE_TD   = MAX_FORWARD_HORIZON          # 60 trading days (§8.1)
SELECT_K   = 40                           # fold-local gain top-K (§4.4)

# E2 momentum kernel — identical construction to model_e2_formation_momentum.py
SKIP_D   = 21
WINDOWS  = {"e2_mom_3m": 63, "e2_mom_6m": 126, "e2_mom_12m": 252}
KERNEL   = list(WINDOWS.keys())
SHORT_B7 = ["features_return_1d", "features_return_5d",
            "features_return_20d", "features_return_60d"]

# Zone family exclusion (§3): prefixes on the part after "features_".
ZONE_PREFIXES = ("sdz", "ssz", "dz", "sz", "zone", "any_valid")
ICT_PREFIX    = "ict"
PIVOT_PREFIX  = "pivot"

LGBM_PARAMS = dict(
    objective="lambdarank", metric="ndcg", ndcg_eval_at=[10],
    label_gain=list(range(100)), num_leaves=31, min_child_samples=50,
    learning_rate=0.05, n_estimators=400, colsample_bytree=0.9,
    subsample=0.8, reg_alpha=0.05, reg_lambda=0.1,
    verbosity=-1, n_jobs=4,
    seed=42, feature_fraction_seed=42, bagging_seed=42, data_random_seed=42,
)

# Overridden by --smoke (infra check only; a smoke result is NOT a verdict).
MIN_TR_ROWS = 5000
MIN_TE_ROWS = 500


def cs_rank_to_label(cs_rank, n_bins=100):
    return (cs_rank * (n_bins - 1)).round().clip(0, n_bins - 1).fillna(0).astype(int)


def classify_features(cols: list[str]) -> dict[str, list[str]]:
    """Split panel `features_*` columns into causal families; zones excluded."""
    feats = [c for c in cols if c.startswith("features_")]
    zone, ict, pivot, base = [], [], [], []
    for c in feats:
        rest = c[len("features_"):]
        if rest.startswith(ZONE_PREFIXES):
            zone.append(c)
        elif rest.startswith(ICT_PREFIX):
            ict.append(c)
        elif rest.startswith(PIVOT_PREFIX):
            pivot.append(c)
        else:
            base.append(c)
    return {"zone_excluded": zone, "ict": ict, "pivot": pivot, "base": base}


def build_kernel(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel["close"].astype(float)
    grp = close.groupby(level="ticker")
    near = grp.shift(SKIP_D)
    for name, far_d in WINDOWS.items():
        far = grp.shift(far_d)
        panel[name] = (near / far - 1.0).astype(np.float32)
    return panel


def run_cv(panel: pd.DataFrame, features: list, date_level: str,
           udates: np.ndarray, select_top_k: int | None = None,
           num_boost_round: int | None = None) -> dict:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    nbr = num_boost_round or LGBM_PARAMS["n_estimators"]
    fold_ics = {h: {} for h in HORIZONS}
    fold_pit, fold_meta, fold_selected = {}, {}, {}
    dates = panel.index.get_level_values(date_level)

    for test_year in FOLD_YEARS:
        test_dates = udates[pd.DatetimeIndex(udates).year == test_year]
        if len(test_dates) == 0:
            continue
        boundary = pd.Timestamp(test_dates[0])
        pos = int(np.searchsorted(udates, boundary))
        if pos <= PURGE_TD:
            continue
        cutoff = pd.Timestamp(udates[pos - PURGE_TD])  # §8.1 purge gap

        # NaN policy (§8.3): drop only on kernel + required label; all other
        # features stay NaN-native for LGBM.
        tr = panel[dates <= cutoff].dropna(subset=KERNEL + ["cs_rank_composite"])
        te = panel[dates.year == test_year].dropna(
            subset=KERNEL + ["future_20d_excess_return"])
        if len(tr) < MIN_TR_ROWS or len(te) < MIN_TE_ROWS:
            continue

        # Purge invariant: no train label window may reach past the boundary.
        tr_max = tr.index.get_level_values(date_level).max()
        assert tr_max <= cutoff < boundary, "purge gap violated"

        # Group-contiguity invariant (bug class found 2026-07-08): lambdarank
        # group arrays require date-major contiguous rows. Fail loudly.
        if not tr.index.get_level_values(date_level).is_monotonic_increasing:
            raise RuntimeError("train slice not date-major — sort before training")

        def _fit(feats: list) -> "lgb.Booster":
            return lgb.train(
                LGBM_PARAMS,
                lgb.Dataset(
                    tr[feats], label=cs_rank_to_label(tr["cs_rank_composite"]),
                    group=tr.groupby(level=date_level).size().sort_index().values,
                    free_raw_data=False,
                ),
                num_boost_round=nbr,
                callbacks=[lgb.log_evaluation(period=-1)],
            )

        feats_used = features
        if select_top_k:
            # §8.4: seeded preliminary fit on the train fold only, rank by
            # gain, keep top K, refit on those K with the same seeds.
            prelim = _fit(features)
            gain = prelim.feature_importance(importance_type="gain")
            order = np.argsort(gain)[::-1][:select_top_k]
            feats_used = [features[i] for i in sorted(order)]
            fold_selected[test_year] = [features[i] for i in order]
            del prelim
            gc.collect()

        model = _fit(feats_used)
        scores = pd.Series(model.predict(te[feats_used]), index=te.index)

        date_ics = {h: [] for h in HORIZONS}
        pit_ics  = []
        for dt in te.index.get_level_values(date_level).unique():
            grp = te.xs(dt, level=date_level)
            sc  = scores.xs(dt, level=date_level)
            for h in HORIZONS:
                col = f"future_{h}d_excess_return"
                if col not in grp.columns:
                    continue
                m = pd.DataFrame({"s": sc, "r": grp[col]}).dropna()
                if len(m) < 20 or m["s"].std() < 1e-9:
                    continue
                ic, _ = spearmanr(m["s"], m["r"])
                date_ics[h].append(ic)
            if "in_universe" in grp.columns:
                pit = grp["in_universe"] == True
                m = pd.DataFrame({"s": sc[pit],
                                  "r": grp.loc[pit, "future_20d_excess_return"]}).dropna()
                if len(m) >= 20 and m["s"].std() > 1e-9:
                    ic, _ = spearmanr(m["s"], m["r"])
                    pit_ics.append(ic)

        for h in HORIZONS:
            if date_ics[h]:
                fold_ics[h][test_year] = float(np.mean(date_ics[h]))
        if pit_ics:
            fold_pit[test_year] = float(np.mean(pit_ics))
        fold_meta[test_year] = {"train_max": str(tr_max.date()),
                                "purge_cutoff": str(cutoff.date()),
                                "test_start": str(boundary.date()),
                                "gap_td": int(pos - np.searchsorted(udates, tr_max)),
                                "n_train": int(len(tr)), "n_test": int(len(te))}
        print(f"    {test_year}: IC20={fold_ics[20].get(test_year, float('nan')):+.4f}  "
              f"IC40={fold_ics[40].get(test_year, float('nan')):+.4f}  "
              f"IC60={fold_ics[60].get(test_year, float('nan')):+.4f}  "
              f"PIT20={fold_pit.get(test_year, float('nan')):+.4f}  "
              f"(train<={fold_meta[test_year]['train_max']}, gap {fold_meta[test_year]['gap_td']} td)")
        del model, scores, tr, te
        gc.collect()

    def agg(vals: dict):
        v = list(vals.values())
        n = len(v)
        if n == 0:
            return {"mean_ic": float("nan"), "t_stat": 0.0, "n_folds": 0,
                    "n_folds_positive": 0, "min_fold_ic": float("nan"), "fold_ics": {}}
        m = float(np.mean(v))
        s = float(np.std(v, ddof=1)) if n > 1 else float("nan")   # sample std for 1-sample t
        return {"mean_ic": m, "std_ic": s,
                "t_stat": float(m / (s / np.sqrt(n))) if n > 1 and s > 0 else 0.0,
                "n_folds": n, "n_folds_positive": sum(1 for x in v if x > 0),
                "min_fold_ic": float(min(v)),
                "fold_ics": {str(y): round(x, 4) for y, x in vals.items()}}

    out = {f"h{h}": agg(fold_ics[h]) for h in HORIZONS}
    out["pit20"] = agg(fold_pit)
    out["n_features"] = len(features)
    out["purge"] = fold_meta
    if fold_selected:
        out["selected_per_fold"] = {str(y): f for y, f in fold_selected.items()}
    return out


def gate_pass(r) -> bool:
    a = r["h20"]
    return (a["mean_ic"] >= 0.03) and (a["t_stat"] >= 2.0) and (a["n_folds_positive"] >= 4)


def main():
    global MIN_TR_ROWS, MIN_TE_ROWS
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--out",   default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true",
                    help="infra check on a tiny/synthetic panel: small trees, "
                         "low row minimums, feature-count band check off. "
                         "A smoke result is NOT a verdict.")
    args = ap.parse_args()
    nbr = 25 if args.smoke else None
    if args.smoke:
        MIN_TR_ROWS, MIN_TE_ROWS = 200, 50

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print("=" * 66)
    print("  MODEL_F - All-causal-features CV (gate = 20d IC, full universe)")
    if args.smoke:
        print("  *** SMOKE MODE - infrastructure check only, NOT a verdict ***")
    print("=" * 66)

    print(f"\nLoading panel: {args.panel}")
    panel = pd.read_pickle(args.panel)
    date_level = "date" if "date" in panel.index.names else panel.index.names[0]
    if "ticker" not in panel.index.names:
        sys.exit("FATAL: panel index has no 'ticker' level.")

    fam = classify_features(list(panel.columns))
    print(f"Feature families: base={len(fam['base'])}  ict={len(fam['ict'])}  "
          f"pivot={len(fam['pivot'])}  zone(EXCLUDED)={len(fam['zone_excluded'])}")

    label_cols = ["cs_rank_composite"] + [f"future_{h}d_excess_return" for h in HORIZONS]
    required = ["close"] + label_cols + SHORT_B7
    missing = [c for c in required if c not in panel.columns]
    if missing:
        sys.exit(f"FATAL: panel missing required columns: {missing}")

    causal = fam["base"] + fam["ict"] + fam["pivot"]
    if not args.smoke and not (200 <= len(causal) + len(KERNEL) <= 360):
        sys.exit(f"FATAL: causal feature count {len(causal) + len(KERNEL)} outside "
                 "the ~290 band the pre-registration expects (§3) — wrong panel? "
                 "PIVOT_FEATURES off? Aborting rather than testing a different set.")

    keep = sorted(set(required + causal + (["in_universe"] if "in_universe" in panel.columns else [])))
    panel = panel[keep].copy()
    gc.collect()

    print("Building E2 momentum kernel (3m/6m/12m skip-1m) ...")
    panel = panel.sort_index()
    panel = build_kernel(panel)
    panel = panel[panel.index.get_level_values(date_level) <= FENCE]
    gc.collect()
    for k in KERNEL:
        nn = panel[k].notna().mean()
        print(f"  {k:<12} non-null {nn:.1%}")
        if not args.smoke and nn < 0.5:
            sys.exit(f"FATAL: {k} mostly NaN — feature build wrong; aborting.")
    print(f"Fenced panel: {panel.shape[0]:,} rows x {panel.shape[1]} cols  "
          f"({panel.memory_usage(deep=True).sum()/1e9:.2f} GB)\n")

    configs = {
        "MOM":        (KERNEL + SHORT_B7, None),
        "ALL":        (causal + KERNEL, None),
        "BASE+MOM":   (fam["base"] + KERNEL, None),
        "ALL+SELECT": (causal + KERNEL, SELECT_K),
    }
    # Zone exclusion is load-bearing — fail loudly if any slipped through.
    for name, (feats, _) in configs.items():
        leaked = [c for c in feats
                  if c.startswith("features_") and c[len("features_"):].startswith(ZONE_PREFIXES)]
        assert not leaked, f"{name}: zone columns leaked into feature set: {leaked[:5]}"

    udates = np.array(sorted(panel.index.get_level_values(date_level).unique()))
    results = {}
    base = None
    for name, (feats, sel) in configs.items():
        print("-" * 48)
        print(f"{name} - {len(feats)} features" + (f", fold-local top-{sel}" if sel else ""))
        r = run_cv(panel, feats, date_level, udates, select_top_k=sel,
                   num_boost_round=nbr)
        r["label"] = name
        if base is None:
            base = r
        r["delta_ic_vs_mom"] = round(r["h20"]["mean_ic"] - base["h20"]["mean_ic"], 4)
        results[name] = r
        a = r["h20"]
        print(f"  -> IC20={a['mean_ic']:+.4f}  t={a['t_stat']:+.2f}  "
              f"IC40={r['h40']['mean_ic']:+.4f}  IC60={r['h60']['mean_ic']:+.4f}  "
              f"PIT20={r['pit20']['mean_ic']:+.4f}  "
              f"gate={'PASS' if gate_pass(r) else 'FAIL'}\n")

    # ── Summary (§5 gate on ALL / ALL+SELECT; MOM & BASE+MOM diagnostic) ────
    print("=" * 66)
    print("  SUMMARY  (gate applies to ALL and ALL+SELECT; ddof=1 t-stats)")
    print("=" * 66)
    print(f"  {'Config':<12} {'n':>4} {'IC20':>8} {'t':>6} {'minIC':>8} {'f+':>3} "
          f"{'IC40':>8} {'IC60':>8} {'PIT20':>8} {'dMOM':>7}  verdict")
    print("  " + "-" * 92)
    for name, r in results.items():
        a = r["h20"]
        gated = name in ("ALL", "ALL+SELECT")
        verdict = ("GATE PASS" if gate_pass(r) else "GATE FAIL") if gated else "diagnostic"
        print(f"  {name:<12} {r['n_features']:>4} {a['mean_ic']:>+8.4f} {a['t_stat']:>+6.2f} "
              f"{a['min_fold_ic']:>+8.4f} {a['n_folds_positive']:>3} "
              f"{r['h40']['mean_ic']:>+8.4f} {r['h60']['mean_ic']:>+8.4f} "
              f"{r['pit20']['mean_ic']:>+8.4f} {r['delta_ic_vs_mom']:>+7.4f}  {verdict}")

    dsel = results["ALL+SELECT"]["delta_ic_vs_mom"]
    print(f"\n  Decision tree (§6): ALL+SELECT - MOM = {dsel:+.4f} "
          f"(focused-follow-up trigger at >= +0.005)")
    print("  Purge: train ends 60 td before each test year (§8.1 amendment); "
          "§7 references were computed WITHOUT the gap (marginally optimistic).")
    print("  Survivorship note: dead tickers absent from panel - measured IC "
          "is survivorship-tinted (187 ex-members lack prices, 14.5% of "
          "membership-days).")
    print("  References: ICT -0.000 | pivots +0.0092 | zones(causal) pending | "
          "momentum KERNEL+S +0.0168")

    results["_meta"] = {"smoke": bool(args.smoke), "purge_td": PURGE_TD,
                        "select_k": SELECT_K, "fence": str(FENCE.date()),
                        "panel": args.panel}
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n  Full results -> {args.out}")


if __name__ == "__main__":
    main()
