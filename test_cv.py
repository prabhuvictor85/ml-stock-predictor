import pandas as pd
import numpy as np
from pipeline.validation.cv import PurgedWalkForwardCV

np.random.seed(42)
n_days = 900
tickers = ['AAPL']
dates = pd.bdate_range('2022-01-03', periods=n_days)
rows = []
for t in tickers:
    for d in dates:
        rows.append({'date': d, 'ticker': t, 'close': 100.0, 'in_universe': True, 'group_date': d})
panel = pd.DataFrame(rows).set_index(['date', 'ticker']).sort_index()
panel['group_date'] = panel.index.get_level_values('date').to_series().dt.to_period('W').apply(lambda p: p.end_time.normalize()).values

cv = PurgedWalkForwardCV(n_folds=3, min_train_window=300)
all_dates = sorted(panel.index.get_level_values('date').unique())
group_dates = sorted(panel['group_date'].dropna().unique())

print("len all_dates:", len(all_dates))
print("n_folds:", cv.n_folds)

for fold_id in range(cv.n_folds):
    test_start_idx = cv.min_train_window + fold_id * cv.test_window
    test_end_idx = test_start_idx + cv.test_window - 1
    if test_end_idx >= len(all_dates):
        print(f"fold {fold_id}: break at test_end_idx ({test_end_idx}) >= len(all_dates) ({len(all_dates)})")
        break
    test_start_date = pd.Timestamp(all_dates[test_start_idx])
    test_end_date   = pd.Timestamp(all_dates[test_end_idx])
    gd_pd = [pd.Timestamp(g) for g in group_dates]
    gd_on_or_after = [g for g in gd_pd if g >= test_start_date]
    if not gd_on_or_after:
        print(f"fold {fold_id}: break at not gd_on_or_after")
        break
    aligned_test_start = gd_on_or_after[0]
    gd_on_or_before_end = [g for g in gd_pd if g <= test_end_date]
    if not gd_on_or_before_end:
        print(f"fold {fold_id}: break at not gd_on_or_before_end")
        break
    aligned_test_end = gd_on_or_before_end[-1]
    
    print(f'fold {fold_id} train end: test start: {aligned_test_start}, test end: {aligned_test_end}')


