"""
A/B: do GRADED ICT validators improve the ranker over the current SHAPE-only ICT?

  BASELINE = panel's existing 118 features_*  (already includes shape-based ICT:
             features_ict_bob_active / *_atr_dist / sweeps / htf_score ...)
  GRADED   = BASELINE + 8 features_ict_{bull,bear}_ob_{disp_atr,made_fvg,
             broke_struct,quality}  (from ICTGradedEngine, section 9b)

Design mirrors drafts/ab_watchlist.py (daily cross-section ranking groups, purged
expanding folds, day-correct purge) but runs MULTIPLE SEEDS so the NDCG delta is
reported as a distribution, not a single point — the rigorous version of the test.

Three questions answered:
  1. NDCG@10 delta (graded - baseline), mean +/- std over seeds x folds.
  2. Where the 8 graded validators rank by gain importance — and CRUCIALLY,
     whether they OUTRANK the existing shape-based features_ict_* columns they
     are meant to improve on.
  3. Watchlist top-15 diff (seed 42) baseline vs graded.

Causality: graded features computed ONCE per ticker over full history; the engine
is causal per-row (verified in validate_ict_graded.py), so test rows see no future.
"""
from __future__ import annotations
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.features.ict_features import _wilder_atr
from drafts.ict_features_graded import ICTGradedEngine, GRADED_OB_COLS
from pipeline.models.lgbm_ranker import LGBMRanker

PANEL = "artefacts/nse_local/panel.pkl"
TRAIN_END = pd.Timestamp("2024-03-22")
PURGE = 63
TEST_WIN = 63
TARGET = "cs_rank_composite"
UNIV = "in_universe"
TOPN = 15
SEEDS = [42, 7, 123, 2024, 99]
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=250,
              min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1)
PREFIX = "features_"

LOG = open("drafts/ab_ict_results.txt", "w", encoding="utf-8")
def out(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); LOG.write(msg + "\n"); LOG.flush()


def ndcg_at_k(df_test: pd.DataFrame, score_col: str, k: int = 10) -> float:
    from sklearn.metrics import ndcg_score
    vals = []
    for _, g in df_test.groupby("date"):
        if len(g) < 2:
            continue
        vals.append(ndcg_score(g[TARGET].values.reshape(1, -1),
                               g[score_col].values.reshape(1, -1), k=min(k, len(g))))
    return float(np.mean(vals)) if vals else float("nan")


def fit_score(train, score_on, feats, seed):
    tr = train.sort_values("date")
    sizes = tr.groupby("date", sort=False).size().values
    r = LGBMRanker(params=PARAMS, seed=seed, use_monotone_constraints=False, num_threads=4)
    r.fit(tr[feats], tr[TARGET], sizes)
    return r.predict(score_on[feats]), r


def main():
    t0 = time.time()
    out("=== A/B GRADED ICT validators vs shape-only ICT (multi-seed) ===")
    out(f"seeds={SEEDS}")
    import joblib
    panel = joblib.load(PANEL)
    panel = panel[panel.index.get_level_values("date") <= TRAIN_END].copy()
    base_feats = [c for c in panel.columns if c.startswith(PREFIX)]
    out(f"panel<= {TRAIN_END.date()}: {panel.shape} | baseline features={len(base_feats)} "
        f"({time.time()-t0:.0f}s)")

    # ── Graded ICT validators once per ticker (causal per-row) ───────────────
    t1 = time.time()
    eng = ICTGradedEngine()
    pieces = []
    tickers = panel.index.get_level_values("ticker").unique()
    for i, tk in enumerate(tickers):
        g = panel.xs(tk, level="ticker")[["open", "high", "low", "close", "volume"]].sort_index().copy()
        if len(g) < 60:
            continue
        try:
            g["atr_14"] = _wilder_atr(g["high"].values, g["low"].values, g["close"].values, 14)
            res = eng.compute(g, disp_mult=0.0)
            sf = res[GRADED_OB_COLS].astype(np.float32)
            sf.columns = [f"{PREFIX}{c}" for c in GRADED_OB_COLS]
            sf.index = pd.MultiIndex.from_product([[tk], sf.index], names=["ticker", "date"])
            pieces.append(sf)
        except Exception as e:
            out(f"  WARN {tk}: {e}")
        if (i + 1) % 100 == 0:
            out(f"  ...graded {i+1}/{len(tickers)} ({time.time()-t1:.0f}s)")
    graded = pd.concat(pieces).swaplevel().sort_index()
    graded_cols = list(graded.columns)
    panel = panel.join(graded, how="left")
    for c in graded_cols:
        panel[c] = panel[c].fillna(0.0).astype(np.float32)
    all_feats = base_feats + graded_cols
    out(f"graded: +{len(graded_cols)} cols {graded_cols} ({time.time()-t1:.0f}s)")

    # ── CV frame ─────────────────────────────────────────────────────────────
    df = panel.reset_index()
    if UNIV in df.columns:
        df = df[df[UNIV].astype(bool)]
    df["date"] = pd.to_datetime(df["date"])
    cv = df.dropna(subset=[TARGET]).sort_values("date").copy()
    dates = pd.DatetimeIndex(sorted(cv["date"].unique()))
    n = len(dates)
    out(f"CV rows={len(cv)} daily-dates={n} range {dates[0].date()}->{dates[-1].date()}")

    # ── Purged expanding folds x multi-seed ──────────────────────────────────
    out("\n=== NDCG@10 per fold x seed (relevance = cs_rank_composite, group = day) ===")
    fold_defs = []
    for fi, frac in enumerate((0.45, 0.60, 0.75), 1):
        tr_end_i = int(n * frac)
        te_start_i = tr_end_i + PURGE
        te_end_i = min(te_start_i + TEST_WIN, n - 1)
        if te_start_i >= n - 1:
            out(f"fold {fi}: not enough data after purge — skipped"); continue
        fold_defs.append((fi, dates[tr_end_i], dates[te_start_i], dates[te_end_i]))

    per_fold = {fi: [] for fi, *_ in fold_defs}    # fi -> list of (nb, ns)
    all_deltas = []
    for fi, tr_end, te_start, te_end in fold_defs:
        tr = cv[cv["date"] <= tr_end]
        te_full = cv[(cv["date"] >= te_start) & (cv["date"] <= te_end)].copy()
        if len(te_full) < 50:
            out(f"fold {fi}: test too small ({len(te_full)}) — skipped"); continue
        for seed in SEEDS:
            te = te_full.copy()
            sc_b, _ = fit_score(tr, te, base_feats, seed)
            sc_s, _ = fit_score(tr, te, all_feats, seed)
            te["__b"], te["__s"] = sc_b, sc_s
            nb, ns = ndcg_at_k(te, "__b"), ndcg_at_k(te, "__s")
            per_fold[fi].append((nb, ns))
            all_deltas.append(ns - nb)
            out(f"fold {fi} seed {seed:>4}: train<= {tr_end.date()} test "
                f"{te_start.date()}->{te_end.date()} | base={nb:.4f} graded={ns:.4f} "
                f"delta={ns-nb:+.4f}")

    out("\n=== Per-fold summary (mean +/- std over seeds) ===")
    for fi, *_ in fold_defs:
        rs = per_fold[fi]
        if not rs:
            continue
        b = np.array([r[0] for r in rs]); s = np.array([r[1] for r in rs])
        d = s - b
        wins = int((d > 0).sum())
        out(f"fold {fi}: base={b.mean():.4f}+/-{b.std():.4f} "
            f"graded={s.mean():.4f}+/-{s.std():.4f} "
            f"delta={d.mean():+.4f}+/-{d.std():.4f} | graded wins {wins}/{len(rs)} seeds")
    if all_deltas:
        ad = np.array(all_deltas)
        out(f"\nOVERALL delta: mean={ad.mean():+.5f} std={ad.std():.5f} "
            f"min={ad.min():+.4f} max={ad.max():+.4f} | "
            f"graded>baseline in {int((ad>0).sum())}/{len(ad)} (seed x fold) runs")

    # ── Watchlist (seed 42) ──────────────────────────────────────────────────
    out("\n=== Watchlist (top-15) comparison, seed 42 ===")
    sizes_by_date = df.groupby("date").size()
    asof = sizes_by_date[sizes_by_date >= 100].index.max()
    asof_i = int(dates.searchsorted(pd.Timestamp(asof)))
    tr_cut = dates[max(0, asof_i - PURGE)]
    train = cv[cv["date"] <= tr_cut]
    score = df[df["date"] == asof].copy()
    out(f"as-of: {pd.Timestamp(asof).date()} (universe={len(score)}) | train<= {tr_cut.date()}")
    sc_b, rb = fit_score(train, score, base_feats, 42)
    sc_s, rs = fit_score(train, score, all_feats, 42)
    score["__b"], score["__s"] = sc_b, sc_s
    wl_b = score.sort_values("__b", ascending=False)["ticker"].head(TOPN).tolist()
    wl_s = score.sort_values("__s", ascending=False)["ticker"].head(TOPN).tolist()
    out(f"BASELINE top{TOPN}: {wl_b}")
    out(f"GRADED   top{TOPN}: {wl_s}")
    out(f"overlap: {len(set(wl_b)&set(wl_s))}/{TOPN} | "
        f"new: {sorted(set(wl_s)-set(wl_b))} | dropped: {sorted(set(wl_b)-set(wl_s))}")

    # ── Importance: graded validators vs existing shape ICT ──────────────────
    out(f"\n=== Importance (gain) — graded model, {len(all_feats)} features ===")
    imp = rs.feature_importance().sort_values(ascending=False)
    rank = {f: i + 1 for i, f in enumerate(imp.index)}
    out("  [graded validators]")
    for f in graded_cols:
        out(f"    rank {rank.get(f, 9999):>4}/{len(all_feats)}  gain={float(imp.get(f,0)):>10.1f}  {f}")
    shape_ict = [c for c in base_feats if c.startswith(f"{PREFIX}ict_")]
    out(f"  [existing shape ICT — {len(shape_ict)} cols, for comparison]")
    for f in sorted(shape_ict, key=lambda x: rank.get(x, 9999)):
        out(f"    rank {rank.get(f, 9999):>4}/{len(all_feats)}  gain={float(imp.get(f,0)):>10.1f}  {f}")
    best_graded = min(rank.get(f, 9999) for f in graded_cols)
    best_shape = min(rank.get(f, 9999) for f in shape_ict) if shape_ict else 9999
    out(f"\nbest graded rank={best_graded}/{len(all_feats)} | best shape-ICT rank={best_shape}/{len(all_feats)}")
    out(f"\nDONE in {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
