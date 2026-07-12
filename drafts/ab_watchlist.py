"""
A/B: does adding BOS/CHoCH structure features change/improve the ranker?

Runs entirely off the cached raw panel (artefacts/nse_local/panel.pkl). Compares
two arms on IDENTICAL daily cross-sections / seed / params:
  BASELINE  = existing 118 features_* columns
  STRUCTURE = baseline + 21 features_{major,internal}_* + features_structure_alignment

Ranking group = the actual trading DATE (one cross-section of ~457 names per day),
which gives proper fold granularity and a day-correct purge. (group_date in the
panel is a coarse monthly reconstitution key — unusable as a CV fold axis.)

Outputs:
  1. NDCG@10 over N purged expanding folds, baseline vs structure
  2. Top-15 watchlist as-of the last daily cross-section <= 2024-03-22, both arms
  3. Where structure features rank by LightGBM gain importance

Causality: structure features computed ONCE per ticker over full history
(cutoff=None). The engine is causal per-row by construction (a swing activates only
at confirmed_at = idx + swing_length), so test rows see no future info; only rows
within swing_length of each series end carry the unavoidable end-of-series clamp
(= exactly what live "today" inference faces).
"""
from __future__ import annotations
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.features.structure_features import structure_feature_frame
from pipeline.models.lgbm_ranker import LGBMRanker

PANEL = "artefacts/nse_local/panel.pkl"
TRAIN_END = pd.Timestamp("2024-03-22")
PURGE = 63               # trading-day purge between train and test (>= max 60d label horizon)
TEST_WIN = 63            # trading-day test window per fold
TARGET = "cs_rank_composite"
UNIV = "in_universe"
TOPN = 15
SEED = 42
PARAMS = dict(num_leaves=31, learning_rate=0.05, n_estimators=250,
              min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1)

LOG = open("drafts/ab_results.txt", "w", encoding="utf-8")
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


def fit_score(train: pd.DataFrame, score_on: pd.DataFrame, feats: list[str]):
    tr = train.sort_values("date")
    sizes = tr.groupby("date", sort=False).size().values
    r = LGBMRanker(params=PARAMS, seed=SEED, use_monotone_constraints=False, num_threads=4)
    r.fit(tr[feats], tr[TARGET], sizes)
    return r.predict(score_on[feats]), r


def main():
    t0 = time.time()
    out("=== A/B Structure Features — NSE cached panel (daily cross-sections) ===")
    import joblib
    panel = joblib.load(PANEL)
    panel = panel[panel.index.get_level_values("date") <= TRAIN_END].copy()
    base_feats = [c for c in panel.columns if c.startswith("features_")]
    out(f"panel<= {TRAIN_END.date()}: {panel.shape} | baseline features={len(base_feats)} ({time.time()-t0:.0f}s)")

    # ── Structure features once per ticker (causal per-row) ───────────────────
    t1 = time.time()
    pieces = []
    tickers = panel.index.get_level_values("ticker").unique()
    for i, tk in enumerate(tickers):
        g = panel.xs(tk, level="ticker")[["open", "high", "low", "close", "volume"]].sort_index()
        try:
            sf = structure_feature_frame(g, cutoff_date=None, prefix="features_")
            sf.index = pd.MultiIndex.from_product([[tk], sf.index], names=["ticker", "date"])
            pieces.append(sf)
        except Exception as e:
            out(f"  WARN {tk}: {e}")
    struct = pd.concat(pieces).swaplevel().sort_index()
    struct_feats = list(struct.columns)
    panel = panel.join(struct, how="left")
    for c in struct_feats:
        panel[c] = panel[c].fillna(0.0).astype(np.float32)
    all_feats = base_feats + struct_feats
    out(f"structure: +{len(struct_feats)} cols ({time.time()-t1:.0f}s)")

    # ── Frame for CV: in-universe, valid target, daily date column ────────────
    df = panel.reset_index()
    if UNIV in df.columns:
        df = df[df[UNIV].astype(bool)]
    df["date"] = pd.to_datetime(df["date"])
    cv = df.dropna(subset=[TARGET]).sort_values("date").copy()
    dates = pd.DatetimeIndex(sorted(cv["date"].unique()))
    n = len(dates)
    out(f"CV rows={len(cv)} daily-dates={n} range {dates[0].date()}->{dates[-1].date()}")

    # ── Purged expanding folds ───────────────────────────────────────────────
    out("\n=== NDCG@10 per fold (relevance = cs_rank_composite, group = trading day) ===")
    rows = []
    for fi, frac in enumerate((0.45, 0.60, 0.75), 1):
        tr_end_i = int(n * frac)
        te_start_i = tr_end_i + PURGE
        te_end_i = min(te_start_i + TEST_WIN, n - 1)
        if te_start_i >= n - 1:
            out(f"fold {fi}: not enough data after purge — skipped"); continue
        tr_end, te_start, te_end = dates[tr_end_i], dates[te_start_i], dates[te_end_i]
        tr = cv[cv["date"] <= tr_end]
        te = cv[(cv["date"] >= te_start) & (cv["date"] <= te_end)].copy()
        if len(te) < 50:
            out(f"fold {fi}: test too small ({len(te)}) — skipped"); continue
        sc_b, _ = fit_score(tr, te, base_feats)
        sc_s, _ = fit_score(tr, te, all_feats)
        te["__b"], te["__s"] = sc_b, sc_s
        nb, ns = ndcg_at_k(te, "__b"), ndcg_at_k(te, "__s")
        out(f"fold {fi}: train<= {tr_end.date()} | test {te_start.date()}->{te_end.date()} "
            f"(rows={len(te)}, days={te['date'].nunique()}) | "
            f"NDCG@10 base={nb:.4f} struct={ns:.4f} delta={ns-nb:+.4f}")
        rows.append((nb, ns))
    if rows:
        b = np.mean([r[0] for r in rows]); s = np.mean([r[1] for r in rows])
        wins = sum(1 for r in rows if r[1] > r[0])
        out(f"MEAN: base={b:.4f} struct={s:.4f} delta={s-b:+.4f} | structure wins {wins}/{len(rows)} folds")

    # ── Watchlist (top-15) as-of last daily cross-section ─────────────────────
    out("\n=== Watchlist (top-15) comparison ===")
    sizes_by_date = df.groupby("date").size()
    asof = sizes_by_date[sizes_by_date >= 100].index.max()
    asof_i = int(dates.searchsorted(pd.Timestamp(asof)))
    tr_cut = dates[max(0, asof_i - PURGE)]
    train = cv[cv["date"] <= tr_cut]
    score = df[df["date"] == asof].copy()
    out(f"as-of: {pd.Timestamp(asof).date()} (universe={len(score)}) | "
        f"train<= {tr_cut.date()} rows={len(train)}")
    sc_b, rb = fit_score(train, score, base_feats)
    sc_s, rs = fit_score(train, score, all_feats)
    score["__b"], score["__s"] = sc_b, sc_s
    wl_b = score.sort_values("__b", ascending=False)["ticker"].head(TOPN).tolist()
    wl_s = score.sort_values("__s", ascending=False)["ticker"].head(TOPN).tolist()
    out(f"BASELINE  top{TOPN}: {wl_b}")
    out(f"STRUCTURE top{TOPN}: {wl_s}")
    out(f"overlap: {len(set(wl_b)&set(wl_s))}/{TOPN} | "
        f"new: {sorted(set(wl_s)-set(wl_b))} | dropped: {sorted(set(wl_b)-set(wl_s))}")

    # ── Importance: where do structure features rank? ────────────────────────
    out(f"\n=== Structure feature gain-importance rank (of {len(all_feats)} total) ===")
    imp = rs.feature_importance().sort_values(ascending=False)
    rank = {f: i + 1 for i, f in enumerate(imp.index)}
    sr = sorted(((f, rank.get(f, 9999), float(imp.get(f, 0))) for f in struct_feats), key=lambda x: x[1])
    for f, rk, gain in sr:
        out(f"  rank {rk:>4}/{len(all_feats)}  gain={gain:>10.1f}  {f}")
    out(f"best structure rank: {min(r for _, r, _ in sr)}/{len(all_feats)}")
    out(f"\nDONE in {time.time()-t0:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
