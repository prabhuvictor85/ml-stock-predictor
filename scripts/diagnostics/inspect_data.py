import os, sys, pandas as pd

base = r'C:\Victor\Learning_charts'
print("BASE EXISTS:", os.path.exists(base))
print("BASE CONTENTS:")
for x in sorted(os.listdir(base)):
    full = os.path.join(base, x)
    if os.path.isdir(full):
        sub = os.listdir(full)
        print(f"  DIR  {x}/  ({len(sub)} items)")
        for s in sorted(sub)[:10]:
            sfull = os.path.join(full, s)
            print(f"       {os.path.getsize(sfull):>12,}  {s}")
    else:
        print(f"  FILE {os.path.getsize(full):>12,}  {x}")

# Probe stock_data
sd = os.path.join(base, 'stock_data')
for candidate in [sd, sd+'.csv', sd+'.parquet']:
    if os.path.isfile(candidate):
        print(f"\nPROBING: {candidate}")
        if candidate.endswith('.parquet'):
            df = pd.read_parquet(candidate)
        else:
            df = pd.read_csv(candidate, nrows=5)
        print("COLUMNS:", df.columns.tolist())
        print("SHAPE:", df.shape)
        print(df.head(3).to_string())
    elif os.path.isdir(candidate):
        print(f"\nDIR: {candidate}")
        files = os.listdir(candidate)
        print("FILES:", sorted(files)[:20])
        # read first file
        first = sorted(files)[0]
        fp = os.path.join(candidate, first)
        try:
            if first.endswith('.parquet'):
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp, nrows=5)
            print("COLUMNS:", df.columns.tolist())
            print("SHAPE:", df.shape)
            print(df.head(3).to_string())
        except Exception as e:
            print("READ ERROR:", e)

# Probe constituents
ci = os.path.join(base, 'stock_lists', 'constituentsi.csv')
if os.path.isfile(ci):
    df = pd.read_csv(ci)
    print(f"\nCONSTITUENTS: {ci}")
    print("SHAPE:", df.shape)
    print("COLUMNS:", df.columns.tolist())
    print(df.head(10).to_string())

