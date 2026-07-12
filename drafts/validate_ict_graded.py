"""
Validate the graded ICT engine (drafts/ict_features_graded.py) before any
ablation. Three things must hold:

  1. SMOKE / RANGE   — runs, emits the 8 graded cols, all finite, in-range,
                       and zero wherever the OB zone is inactive.
  2. BASE UNCHANGED  — the production base columns (ict_bob_active, distances,
                       sweeps, ...) are byte-identical to the upstream
                       ICTFeatureEngine. i.e. we only ADDED, never altered.
  3. CAUSAL / NO LEAK— perturbing FUTURE bars (idx > T) by x1.5 leaves every
                       row idx <= T unchanged (to 1e-9). This is the same proof
                       used for structure_features.

Run:  .venv/Scripts/python.exe drafts/validate_ict_graded.py
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.features.ict_features import ICTFeatureEngine, _wilder_atr
from drafts.ict_features_graded import ICTGradedEngine, GRADED_OB_COLS

PASS = 0
FAIL = 0
def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    tag = "PASS" if ok else "FAIL"
    if ok: PASS += 1
    else:  FAIL += 1
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))


def synth_ohlcv(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Deterministic random-walk OHLCV with realistic intrabar ranges."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(ret))
    rng2 = rng.uniform(0.005, 0.03, n)            # intrabar range fraction
    high = close * (1 + rng2)
    low = close * (1 - rng2 * rng.uniform(0.3, 1.0, n))
    openp = low + (high - low) * rng.uniform(0, 1, n)
    vol = rng.uniform(1e5, 5e6, n)
    idx = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"open": openp, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    df["atr_14"] = _wilder_atr(df["high"].values, df["low"].values,
                               df["close"].values, 14)
    return df


def run_graded(df: pd.DataFrame) -> pd.DataFrame:
    return ICTGradedEngine().compute(df.copy(), disp_mult=0.0)


def main():
    print("=== Validate graded ICT engine ===")

    # ── 1. SMOKE / RANGE ─────────────────────────────────────────────────────
    print("\n[1] Smoke / range / active-masking")
    df = synth_ohlcv(1500, seed=1)
    g = run_graded(df)
    have_all = all(c in g.columns for c in GRADED_OB_COLS)
    check("all 8 graded columns present", have_all,
          f"{[c for c in GRADED_OB_COLS if c not in g.columns]}")
    sub = g[GRADED_OB_COLS].values
    check("all graded values finite", bool(np.isfinite(sub).all()))
    check("disp_atr within [0,20]",
          bool((g["ict_bull_ob_disp_atr"].between(0, 20).all()) and
               (g["ict_bear_ob_disp_atr"].between(0, 20).all())))
    check("made_fvg/broke_struct are 0/1 (when active)",
          bool(set(np.unique(g[["ict_bull_ob_made_fvg", "ict_bull_ob_broke_struct",
                                "ict_bear_ob_made_fvg", "ict_bear_ob_broke_struct"]].values
                             .round(6))).issubset({0.0, 1.0})))
    check("quality >= 0", bool((g["ict_bull_ob_quality"] >= 0).all() and
                               (g["ict_bear_ob_quality"] >= 0).all()))
    # active-masking: graded == 0 wherever the OB zone is inactive
    bull_off = g["ict_bob_active"].values == 0
    sob_off = g["ict_sob_active"].values == 0
    bull_cols = [c for c in GRADED_OB_COLS if c.startswith("ict_bull")]
    bear_cols = [c for c in GRADED_OB_COLS if c.startswith("ict_bear")]
    check("bull graded == 0 where bob inactive",
          bool((np.abs(g.loc[bull_off, bull_cols].values) < 1e-12).all()))
    check("bear graded == 0 where sob inactive",
          bool((np.abs(g.loc[sob_off, bear_cols].values) < 1e-12).all()))
    # coverage: features actually fire somewhere (not all-zero / dead)
    nz = {c: int((np.abs(g[c].values) > 0).sum()) for c in GRADED_OB_COLS}
    check("graded features have non-zero coverage",
          all(v > 0 for v in nz.values()), f"nonzero bars per col: {nz}")

    # ── 2. BASE COLUMNS UNCHANGED vs production ──────────────────────────────
    print("\n[2] Production base columns unchanged (added-only)")
    prod = ICTFeatureEngine().compute(df.copy(), disp_mult=0.0)
    base_cols = [c for c in prod.columns if c.startswith("ict_")
                 and c not in GRADED_OB_COLS]
    max_diff = 0.0
    worst = ""
    for c in base_cols:
        if c in g.columns:
            d = float(np.nanmax(np.abs(prod[c].values - g[c].values)))
            if d > max_diff:
                max_diff, worst = d, c
    check(f"all {len(base_cols)} base ict_* cols identical to production",
          max_diff < 1e-9, f"max|diff|={max_diff:.2e} (worst={worst})")

    # ── 3. CAUSALITY / NO LEAK ───────────────────────────────────────────────
    print("\n[3] Causality: perturb future, past must not move")
    base = run_graded(df)
    n = len(df)
    for frac in (0.4, 0.6, 0.8):
        T = int(n * frac)
        dfp = df.copy()
        # blow up EVERYTHING strictly after T (prices + recompute atr on full)
        for col in ("open", "high", "low", "close"):
            dfp.iloc[T + 1:, dfp.columns.get_loc(col)] *= 1.5
        dfp["atr_14"] = _wilder_atr(dfp["high"].values, dfp["low"].values,
                                    dfp["close"].values, 14)
        gp = run_graded(dfp)
        past_base = base.iloc[:T + 1][GRADED_OB_COLS].values
        past_pert = gp.iloc[:T + 1][GRADED_OB_COLS].values
        d = float(np.nanmax(np.abs(past_base - past_pert)))
        check(f"rows<=T unchanged when future x1.5 (T={T}, {int(frac*100)}%)",
              d < 1e-9, f"max|diff|={d:.2e}")

    # ── 4. Multi-seed robustness smoke ───────────────────────────────────────
    print("\n[4] Multi-seed smoke (no crash, all finite)")
    for s in (2, 3, 4, 5, 6):
        gg = run_graded(synth_ohlcv(1200, seed=s))
        ok = bool(np.isfinite(gg[GRADED_OB_COLS].values).all())
        check(f"seed={s} finite", ok)

    # ── Optional: real tickers from the cached panel ─────────────────────────
    panel_path = "artefacts/nse_local/panel.pkl"
    if os.path.exists(panel_path):
        print("\n[5] Real-ticker smoke (cached panel)")
        try:
            import joblib
            panel = joblib.load(panel_path)
            tks = list(panel.index.get_level_values("ticker").unique())[:3]
            for tk in tks:
                sub = panel.xs(tk, level="ticker")[
                    ["open", "high", "low", "close", "volume"]].sort_index().copy()
                sub["atr_14"] = _wilder_atr(sub["high"].values, sub["low"].values,
                                            sub["close"].values, 14)
                gg = run_graded(sub)
                fin = bool(np.isfinite(gg[GRADED_OB_COLS].values).all())
                cov = int((np.abs(gg["ict_bull_ob_quality"].values) > 0).sum())
                check(f"{tk}: finite & bull_ob_quality fires",
                      fin and cov > 0, f"rows={len(sub)} bull_q nonzero={cov}")
        except Exception as e:
            check("panel smoke", False, f"error: {e}")
    else:
        print("\n[5] (panel not found — skipping real-ticker smoke)")

    print(f"\n=== {PASS} passed / {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
