import pickle, sys, pathlib
sys.path.insert(0, ".")
import pandas as pd

for p in ["artefacts/panel.pkl", "artefacts/nse_local/panel.pkl", "artefacts/panel_targets.pkl"]:
    f = pathlib.Path(p)
    if not f.exists():
        continue
    with open(f, "rb") as fh:
        panel = pickle.load(fh)
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker")
    print(f"{p}:")
    print(f"  Start : {dates.min().date()}")
    print(f"  End   : {dates.max().date()}")
    print(f"  Days  : {dates.nunique()}")
    print(f"  Tickers: {tickers.nunique()}")

