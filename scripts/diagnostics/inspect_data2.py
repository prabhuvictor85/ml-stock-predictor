import pandas as pd, os

base = r'C:\Victor\Learning_charts\stock_data'

# Read daily OHLCV
f1d = os.path.join(base, '360ONE.NS-1d.csv')
df1d = pd.read_csv(f1d, nrows=5)
print("=== 1d OHLCV ===")
print("COLUMNS:", df1d.columns.tolist())
print(df1d.to_string())

# Read daily Drv (zone)
f1d_drv = os.path.join(base, '360ONE.NS-1d-Drv.csv')
ddrv = pd.read_csv(f1d_drv, nrows=8)
print("\n=== 1d-Drv (zones) ===")
print("COLUMNS:", ddrv.columns.tolist())
print(ddrv.to_string())

# Read weekly Drv
f1wk = os.path.join(base, '360ONE.NS-1wk-Drv.csv')
dwk = pd.read_csv(f1wk, nrows=8)
print("\n=== 1wk-Drv ===")
print("COLUMNS:", dwk.columns.tolist())
print(dwk.to_string())

# Check 1mo, 3mo, 1y
for tf in ['1mo','3mo','1y']:
    fp = os.path.join(base, f'360ONE.NS-{tf}-Drv.csv')
    d = pd.read_csv(fp, nrows=4)
    print(f"\n=== {tf}-Drv ===")
    print("COLUMNS:", d.columns.tolist())
    print(d.head(3).to_string())

# Count how many tickers have full data
files = os.listdir(base)
tickers_1d = set(f.replace('-1d.csv','') for f in files if f.endswith('-1d.csv'))
tickers_drv = set(f.replace('-1d-Drv.csv','') for f in files if f.endswith('-1d-Drv.csv'))
print(f"\n1d files: {len(tickers_1d)}, 1d-Drv files: {len(tickers_drv)}")
print("Sample tickers:", sorted(tickers_1d)[:5])

# Check date range
df_full = pd.read_csv(f1d)
print("\nFull 1d shape:", df_full.shape)
print("Date col:", df_full.columns[0], "Min:", df_full.iloc[:,0].min(), "Max:", df_full.iloc[:,0].max())

