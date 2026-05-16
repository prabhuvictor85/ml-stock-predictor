"""
Quick fold layout preview — no panel needed, uses the actual trading day count
from the smoke test (4031 unique trading days: 2010-01-04 to 2023-12-08).
"""
import sys; sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from datetime import timedelta

# ── Reproduce the actual trading calendar from the panel dates ───────────────
# Smoke test confirmed: 4031 unique dates, 2010-01-04 → 2023-12-08
# We'll use business days as a proxy for the fold index calculation.
bdays = pd.bdate_range("2010-01-04", "2026-05-04")  # actual data range
all_dates = np.array(bdays)
d0 = pd.Timestamp(all_dates[0]).date()
d1 = pd.Timestamp(all_dates[-1]).date()
print(f"Simulated trading days: {len(all_dates)}  ({d0} -> {d1})\n")

MIN_TRAIN = 504
TEST_WIN  = 252
N_FOLDS   = 14

possible_folds = (len(all_dates) - MIN_TRAIN) // TEST_WIN
print(f"n_folds requested : {N_FOLDS}")
print(f"n_folds possible  : {possible_folds}  (with {MIN_TRAIN}d min-train + {TEST_WIN}d test windows)")
print()

first_test_start_idx = MIN_TRAIN

print(f"{'Fold':<5} {'Train start':<14} {'Train end':<14} {'Test start':<14} {'Test end':<14} {'Train days'}")
print("-" * 80)
for fold_id in range(N_FOLDS):
    ts_idx = first_test_start_idx + fold_id * TEST_WIN
    te_idx = ts_idx + TEST_WIN - 1
    if te_idx >= len(all_dates):
        print(f"  Fold {fold_id} would exceed panel — stopping.")
        break

    train_start = all_dates[0]
    # purge+embargo = 45 days before test_start
    purge_cutoff = pd.Timestamp(all_dates[ts_idx]) - timedelta(days=45)
    train_end    = purge_cutoff
    test_start   = pd.Timestamp(all_dates[ts_idx])
    test_end     = pd.Timestamp(all_dates[te_idx])
    train_days   = ts_idx  # number of trading days in training window

    print(f"  {fold_id:<4} {str(pd.Timestamp(train_start).date()):<14} {str(train_end.date()):<14} "
          f"{str(test_start.date()):<14} {str(test_end.date()):<14} {train_days}")

print()
# Show how much data is left unused in test windows
last_test_end_idx = first_test_start_idx + (N_FOLDS - 1) * TEST_WIN + TEST_WIN - 1
last_test_end = pd.Timestamp(all_dates[min(last_test_end_idx, len(all_dates)-1)])
unused_days   = len(all_dates) - last_test_end_idx - 1
print(f"Last test window ends : {last_test_end.date()}")
print(f"Panel ends            : {pd.Timestamp(all_dates[-1]).date()}")
print(f"Days only in training : {max(0, unused_days)}  "
      f"(~{max(0, unused_days)/252:.1f} years after last test fold)")
print()
print(f"To cover the full 2010-2023 range in test windows, use --n_folds {possible_folds}")

