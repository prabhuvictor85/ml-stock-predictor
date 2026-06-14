"""
test_feature_causality.py -- prove (or disprove) inference-time look-ahead.

The question: does fe.build() produce DIFFERENT features for a given date
depending on whether the panel contains bars AFTER that date? If yes, scoring a
historical date against CSVs that already extend past it (any backtest /
walk-forward re-run on a fully-downloaded machine) leaks the future into the
feature row -- because zone/ICT features carry state that later price action
can change, and fe.build runs with cutoff_date=None.

Method (uses the REAL build path, not a proxy):
  1. Load a sample of tickers into a panel via build_panel_from_local.
  2. Build features on the FULL panel  -> features_full
  3. Build features on the panel TRUNCATED at as_of -> features_trunc
  4. Compare the feature rows AT as_of between the two.

If the rows are identical -> fe.build is causal (no leak; live runs were fine).
If they differ          -> fe.build looks ahead; historical re-scores leak,
                           and the magnitude per feature is reported.

This needs no model artefacts -- it isolates the feature engineering only.

Run (Hetzner):
    python3 scripts/tools/validate_feature_causality.py \
        --as_of 2024-06-14 --n_tickers 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as_of", required=True, help="Date to test (YYYY-MM-DD).")
    ap.add_argument("--n_tickers", type=int, default=60,
                    help="How many tickers to sample for the test.")
    ap.add_argument("--data_dir", default=None,
                    help="Price CSV dir (default: PATHS.stock_data.us).")
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="Abs tolerance for declaring a feature value 'changed'.")
    args = ap.parse_args()

    from run_sp500_local import build_panel_from_local, load_benchmark, STOCK_DATA_DIR
    from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
    from pipeline.config.nse import NSE_CONFIG as cfg

    data_dir = Path(args.data_dir) if args.data_dir else STOCK_DATA_DIR
    as_of = pd.Timestamp(args.as_of)

    # -- pick tickers that have a CSV --------------------------------------------
    all_csv = sorted(p.stem.replace("-1d", "") for p in data_dir.glob("*-1d.csv"))
    tickers = all_csv[: args.n_tickers]
    print(f"\n{'='*64}\n  FEATURE CAUSALITY TEST @ {as_of.date()}\n{'='*64}")
    print(f"  data_dir : {data_dir}")
    print(f"  tickers  : {len(tickers)} (sample)")

    bench = load_benchmark(data_dir)

    # -- build features two ways -------------------------------------------------
    def build(trunc: bool) -> pd.DataFrame:
        panel = build_panel_from_local(tickers, data_dir)
        if trunc:
            d = panel.index.get_level_values("date")
            panel = panel[d <= as_of].copy()
        fe = FeatureEngineer(cfg, bench)
        return fe.build(panel)

    print("\n  Building features on FULL panel (cutoff_date=None, future present) ...")
    feat_full = build(trunc=False)
    print("  Building features on panel TRUNCATED at as_of (future removed) ...")
    feat_trunc = build(trunc=True)

    # -- compare the as_of cross-section -----------------------------------------
    def cross(df: pd.DataFrame) -> pd.DataFrame:
        dates = df.index.get_level_values("date").unique()
        avail = dates[dates <= as_of]
        if len(avail) == 0:
            return pd.DataFrame()
        d = avail.max()
        return df.xs(d, level="date")

    cf, ct = cross(feat_full), cross(feat_trunc)
    if cf.empty or ct.empty:
        print("\n  Could not isolate the as_of cross-section in one of the builds.")
        sys.exit(1)

    feat_cols = [c for c in cf.columns
                 if c.startswith(FEATURE_PREFIX) and c in ct.columns]
    common = cf.index.intersection(ct.index)
    cf, ct = cf.loc[common, feat_cols], ct.loc[common, feat_cols]

    a = cf.to_numpy(dtype=float)
    b = ct.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    diff = np.nan_to_num(diff, nan=np.inf)            # NaN-vs-number counts as changed
    changed_cells = int((diff > args.atol).sum())
    total_cells = a.size
    per_feat_changed = (diff > args.atol).sum(axis=0)
    changed_feats = [(feat_cols[i], int(per_feat_changed[i]))
                     for i in np.argsort(per_feat_changed)[::-1]
                     if per_feat_changed[i] > 0]

    print(f"\n{'-'*64}")
    print(f"  cross-section rows compared : {len(common)} tickers")
    print(f"  feature columns compared    : {len(feat_cols)}")
    print(f"  changed cells               : {changed_cells:,} / {total_cells:,} "
          f"({100*changed_cells/max(total_cells,1):.1f}%)")
    print(f"  feature columns that changed: {len(changed_feats)} / {len(feat_cols)}")
    if changed_feats:
        print(f"\n  Top changed features (col : #tickers affected):")
        for name, n in changed_feats[:15]:
            print(f"    {name:<40} {n}")

    print(f"\n{'-'*64}")
    if changed_cells == 0:
        print("  >>> CAUSAL: features at as_of are identical with/without future "
              "bars.\n      fe.build does not look ahead -- historical re-scores "
              "are safe.")
    else:
        print("  >>> LEAK CONFIRMED: features at as_of CHANGE when future bars are "
              "present.\n      Any backtest/walk-forward scored against CSVs that "
              "extend past\n      as_of is contaminated. Live runs (as_of == latest "
              "bar) are safe.\n      Fix: truncate panel to <= as_of before fe.build "
              "(now done in the\n      --skip_train path of run_sp500_local.py).")
    print()


if __name__ == "__main__":
    main()
