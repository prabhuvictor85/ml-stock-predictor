"""
Validation harness for drafts/structure_features.py — NOT integrated.
Runs: (1) smoke test, (2) leakage/cutoff-invariance test, (3) edge-case units.
Run:  python drafts/validate_structure.py
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drafts.structure_features import (
    compute_structure_features,
    compute_multiscale_structure_features,
    _detect_confirmed_swings,
)

CSV = r"C:\Victor\Learning_charts\stock_data\AAPL-1d.csv"
PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]]

# ───────────────────────── 1. SMOKE ─────────────────────────
def smoke(df):
    print("\n=== 1. SMOKE TEST (single-scale, AAPL daily) ===")
    out = compute_structure_features(df, swing_length=10)
    new_cols = [c for c in out.columns if c.startswith("structure_")]
    print(f"  rows={len(out)}  new_cols={len(new_cols)}")

    finite_cols = ["structure_trend_state", "structure_bos_flag", "structure_choch_flag",
                   "structure_bsl_swept", "structure_ssl_swept", "structure_level_dist_atr"]
    for c in finite_cols:
        v = out[c].values
        check(f"{c} all-finite", np.all(np.isfinite(v)),
              f"nan={np.isnan(v).sum()} inf={np.isinf(v).sum()}")

    dist = out["structure_level_dist_atr"].values
    check("level_dist within +/-20 clip", np.all(np.abs(dist) <= 20.0 + 1e-6),
          f"min={dist.min():.2f} max={dist.max():.2f}")
    for c in ["structure_bos_flag", "structure_choch_flag"]:
        check(f"{c} in {{-1,0,1}}", set(np.unique(out[c].values)).issubset({-1.0, 0.0, 1.0}))
    check("trend_state in {-1,0,1}", set(np.unique(out["structure_trend_state"].values)).issubset({-1.0, 0.0, 1.0}))

    n_bos = int((out["structure_bos_flag"] != 0).sum())
    n_choch = int((out["structure_choch_flag"] != 0).sum())
    n_bsl = int((out["structure_bsl_swept"] != 0).sum())
    n_ssl = int((out["structure_ssl_swept"] != 0).sum())
    print(f"  events: BOS={n_bos}  CHoCH={n_choch}  BSL_sweep={n_bsl}  SSL_sweep={n_ssl}")
    check("at least some BOS fired", n_bos > 0)
    check("at least some CHoCH fired", n_choch > 0)
    check("events are sparse (<25% of bars)", (n_bos + n_choch) < 0.25 * len(out),
          f"{n_bos+n_choch}/{len(out)}")
    # Sanity: every CHoCH must coincide with a trend flip vs prior state
    ts = out["structure_trend_state"].values
    choch = out["structure_choch_flag"].values
    bad_flip = 0
    for i in np.flatnonzero(choch != 0):
        if i > 0 and ts[i] == ts[i-1]:
            bad_flip += 1
    check("every CHoCH flips trend_state", bad_flip == 0, f"bad={bad_flip}")

    print("\n=== 1b. SMOKE TEST (multi-scale) ===")
    mout = compute_multiscale_structure_features(df, major_swing_length=25, minor_swing_length=5)
    for c in ["structure_alignment", "major_trend_state", "internal_trend_state"]:
        check(f"{c} present & finite", c in mout.columns and np.all(np.isfinite(mout[c].values)))
    check("alignment in {-1,0,1}", set(np.unique(mout["structure_alignment"].values)).issubset({-1.0, 0.0, 1.0}))
    check("judas_setup removed", "structure_judas_setup" not in mout.columns)
    print(f"  alignment dist: +1={int((mout['structure_alignment']==1).sum())} "
          f"-1={int((mout['structure_alignment']==-1).sum())} 0={int((mout['structure_alignment']==0).sum())}")

# ───────────────────────── 2. LEAKAGE ─────────────────────────
def leakage(df):
    print("\n=== 2. LEAKAGE / CUTOFF-INVARIANCE TEST ===")
    # Invariant: values for dates <= T must be identical whether the engine
    # sees future data (full df + cutoff=T) or not (df truncated at T + cutoff=T).
    # Any difference = future information leaking into the past.
    cutoff = df.index[2000]
    full = compute_structure_features(df, swing_length=10, cutoff_date=cutoff)
    trunc_df = df[df.index <= cutoff]
    trunc = compute_structure_features(trunc_df, swing_length=10, cutoff_date=cutoff)

    cols = [c for c in full.columns if c.startswith("structure_")]
    full_head = full[full.index <= cutoff][cols].replace([np.inf], 1e18)
    trunc_al = trunc[cols].replace([np.inf], 1e18)
    check("aligned row counts", len(full_head) == len(trunc_al), f"{len(full_head)} vs {len(trunc_al)}")
    for c in cols:
        a, b = full_head[c].values.astype(float), trunc_al[c].values.astype(float)
        identical = np.allclose(a, b, atol=1e-6, equal_nan=True)
        if not identical:
            diff_idx = np.flatnonzero(~np.isclose(a, b, atol=1e-6, equal_nan=True))
            check(f"{c} cutoff-invariant", False,
                  f"{len(diff_idx)} differing rows, first@{diff_idx[0]}: {a[diff_idx[0]]} vs {b[diff_idx[0]]}")
        else:
            check(f"{c} cutoff-invariant", True)

    # Stronger: vary FUTURE data only; past must not move.
    df2 = df.copy()
    fut = df2.index > cutoff
    df2.loc[fut, ["open","high","low","close"]] *= 1.5   # perturb only the future
    full2 = compute_structure_features(df2, swing_length=10, cutoff_date=cutoff)
    a = full[full.index <= cutoff][cols].replace([np.inf],1e18).values.astype(float)
    b = full2[full2.index <= cutoff][cols].replace([np.inf],1e18).values.astype(float)
    check("past invariant to future perturbation", np.allclose(a, b, atol=1e-6, equal_nan=True),
          "future *1.5 changed the past" if not np.allclose(a,b,atol=1e-6,equal_nan=True) else "")

    # Multi-scale leakage
    mfull = compute_multiscale_structure_features(df, 25, 5, cutoff_date=cutoff)
    mtrunc = compute_multiscale_structure_features(trunc_df, 25, 5, cutoff_date=cutoff)
    mcols = ["structure_alignment", "major_trend_state", "internal_trend_state"]
    for c in mcols:
        a = mfull[mfull.index <= cutoff][c].replace([np.inf],1e18).values.astype(float)
        b = mtrunc[c].replace([np.inf],1e18).values.astype(float)
        check(f"multiscale {c} cutoff-invariant", np.allclose(a, b, atol=1e-6, equal_nan=True))

# ───────────────────────── 3. EDGE CASES ─────────────────────────
def edges(df):
    print("\n=== 3. EDGE CASES ===")
    # Too-short series -> neutral columns, no crash
    short = df.head(5)
    try:
        o = compute_structure_features(short, swing_length=10)
        ok = (o["structure_trend_state"]==0).all() and np.isinf(o["structure_bars_since_bos"]).all()
        check("short series returns neutral, no crash", bool(ok))
    except Exception as e:
        check("short series returns neutral, no crash", False, repr(e))

    # All-flat prices -> no swings/breaks, all finite
    flat = df.head(200).copy()
    for c in ["open","high","low","close"]:
        flat[c] = 100.0
    try:
        o = compute_structure_features(flat, swing_length=10)
        ok = ((o["structure_bos_flag"]==0).all() and (o["structure_choch_flag"]==0).all()
              and np.isfinite(o["structure_level_dist_atr"]).all())
        check("flat prices: no events, finite", bool(ok))
    except Exception as e:
        check("flat prices: no events, finite", False, repr(e))

    # minor >= major must raise
    try:
        compute_multiscale_structure_features(df.head(300), major_swing_length=5, minor_swing_length=5)
        check("minor>=major raises ValueError", False, "did not raise")
    except ValueError:
        check("minor>=major raises ValueError", True)
    except Exception as e:
        check("minor>=major raises ValueError", False, f"wrong exc {repr(e)}")

    # Swing confirmation lag is causal: confirmed_at == idx + swing_length
    h = df["high"].values.astype(float); l = df["low"].values.astype(float)
    sw = _detect_confirmed_swings(h, l, 10)
    if sw:
        ok = all(c == min(i+10, len(h)-1) for (i,_,_,c) in sw)
        check("swing confirmed_at == idx+swing_length", ok, f"n_swings={len(sw)}")
    else:
        check("swing confirmed_at == idx+swing_length", False, "no swings detected")

# ───────────────────────── RUN ─────────────────────────
if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows  {df.index[0].date()} -> {df.index[-1].date()}")
    smoke(df); leakage(df); edges(df)
    print("\n" + "="*50)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILURES:")
        for f in FAIL: print("  -", f)
        sys.exit(1)
    print("ALL VALIDATION PASSED [OK]")
