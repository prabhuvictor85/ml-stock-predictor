import pickle, sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from pipeline.validation.cv import PurgedWalkForwardCV, _stock_first_dates

with open("artefacts/nse_local/checkpoints/panel_targets.pkl", "rb") as f:
    panel = pickle.load(f)

dates = panel.index.get_level_values("date")
tickers = panel.index.get_level_values("ticker")
print(f"Date range : {dates.min().date()} -> {dates.max().date()}")
print(f"Unique dates: {dates.nunique()}")
print(f"Unique tickers: {tickers.nunique()}")

cv = PurgedWalkForwardCV(n_folds=8, min_train_window=504)
specs = cv.get_fold_specs(panel)
print(f"\nFold specs generated: {len(specs)}")

stock_first = _stock_first_dates(panel)
all_trading_days = np.array(sorted(dates.unique()))
print(f"\nstock_first dtype: {stock_first.dtype}")
print(f"stock_first sample:\n{stock_first.head()}")
print(f"\ncal_days_needed (old approx): {int(504 * 365 / 252)}")

for s in specs:
    elig = cv._eligible_tickers(stock_first, s.train_end, all_trading_days)
    tr_mask = (dates >= s.train_start) & (dates <= s.train_end)
    tr_elig = tickers.isin(elig)
    tr_rows = (tr_mask & tr_elig).sum()
    print(f"Fold {s.fold_id}: train_end={s.train_end.date()}  eligible={len(elig)}/{len(stock_first)}  train_rows={tr_rows}")
