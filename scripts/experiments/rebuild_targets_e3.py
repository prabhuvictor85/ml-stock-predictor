#!/usr/bin/env python3
"""
MODEL_E3 target rebuild — extended horizons + optional TWAP terminal.
=====================================================================
Produces a new panel_targets pickle with labels at horizons {20,40,60,80,100,
120} (or whatever TARGET_HORIZONS is set to) and a chosen terminal-smoothing
window, WITHOUT recomputing features and WITHOUT touching the reference panel.

Method: load the reference panel_targets.pkl, strip every target-derived
column, re-run the validated TargetBuilder (which already implements the TWAP
terminal, see builder.py:113) with the env-selected horizons + window, save to
a new path. Reusing TargetBuilder means E3 grades on the exact same ruler as
validate_lockbox.py — no reimplementation.

Run on Hetzner (two builds — one per terminal window):
    cd /root/ml-stock-predictor
    TARGET_HORIZONS=20,40,60,80,100,120 TARGET_TWAP_WINDOW=1 \
        python3 -u scripts/experiments/rebuild_targets_e3.py \
        --out /mnt/data/artefacts/experiments/e3_targets_w1.pkl
    TARGET_HORIZONS=20,40,60,80,100,120 TARGET_TWAP_WINDOW=5 \
        python3 -u scripts/experiments/rebuild_targets_e3.py \
        --out /mnt/data/artefacts/experiments/e3_targets_w5.pkl

TARGET_HORIZONS / TARGET_TWAP_WINDOW are read by pipeline/targets/builder.py at
import time, so they MUST be set in the environment before this runs (as above).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

REF_PANEL = "/mnt/data/artefacts/us_pivot_v1/us_local/checkpoints/panel_targets.pkl"

# Column families produced by TargetBuilder — stripped before rebuild so the
# new horizon set fully replaces the old one (no stale 20/40/60-only leftovers).
TARGET_PREFIXES = ("future_", "benchmark_", "cs_rank_", "top_quintile",
                   "bot_quintile", "hit_target", "max_drawdown", "future_vol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_PANEL, help="reference panel_targets.pkl")
    ap.add_argument("--out", required=True, help="output pickle path")
    args = ap.parse_args()

    # Import AFTER env is set (builder reads TARGET_HORIZONS at import time)
    from pipeline.targets.builder import TargetBuilder, HORIZONS, MAX_FORWARD_HORIZON
    from pipeline.config import get_config
    from run_sp500_local import load_benchmark, STOCK_DATA_DIR

    twap = int(os.environ.get("TARGET_TWAP_WINDOW", "1"))
    print("=" * 60)
    print("  MODEL_E3 target rebuild")
    print(f"  HORIZONS            = {HORIZONS}")
    print(f"  MAX_FORWARD_HORIZON = {MAX_FORWARD_HORIZON}")
    print(f"  TARGET_TWAP_WINDOW  = {twap}")
    print(f"  out                 = {args.out}")
    print("=" * 60)
    if set(HORIZONS) == {20, 40, 60}:
        sys.exit("REFUSING: TARGET_HORIZONS not extended (still 20,40,60). "
                 "Set it in the environment before running (see header).")

    print(f"\nLoading reference panel: {args.ref}")
    panel = pd.read_pickle(args.ref)
    before_cols = panel.shape[1]

    # Strip old target columns
    drop = [c for c in panel.columns if c.startswith(TARGET_PREFIXES)]
    panel = panel.drop(columns=drop)
    print(f"  stripped {len(drop)} old target columns "
          f"({before_cols} -> {panel.shape[1]} cols)")

    # Sanity: features + ohlcv must survive
    for need in ("open", "high", "low", "close", "volume", "in_universe"):
        if need not in panel.columns:
            sys.exit(f"FATAL: reference panel missing base column '{need}'.")

    bm = load_benchmark(STOCK_DATA_DIR)
    print(f"  benchmark loaded: {len(bm):,} dates "
          f"[{bm.index.min().date()}..{bm.index.max().date()}]")

    cfg = get_config("sp500")
    tb = TargetBuilder(cfg)
    print("\nRebuilding targets ...")
    panel = tb.build(panel, bm)   # terminal_window read from env inside build()

    # Report label coverage per horizon
    print("\nLabel coverage (non-null future_{h}d_excess_return):")
    for h in HORIZONS:
        col = f"future_{h}d_excess_return"
        if col in panel.columns:
            print(f"  {col:<32} {panel[col].notna().mean():.1%}")
    print(f"  cs_rank_composite non-null: {panel['cs_rank_composite'].notna().mean():.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel.to_pickle(args.out)
    print(f"\nSaved -> {args.out}  ({panel.shape[0]:,} rows × {panel.shape[1]} cols)")


if __name__ == "__main__":
    main()
