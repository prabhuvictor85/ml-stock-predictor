"""Find the exact working parameter combination for index_constituent_timeseries."""
import norgatedata as ng
import pandas as pd
import numpy as np
from datetime import date

TODAY = date.today().isoformat()

print("=== Testing index_constituent_timeseries variations ===\n")

# Test 1: bare minimum, default numpy-recarray
print("Test 1: minimal args, numpy-recarray (default)")
try:
    ts = ng.index_constituent_timeseries('AAPL', 'S&P 500')
    print(f"  OK  type={type(ts).__name__}  len={len(ts)}")
    if len(ts) > 0:
        print(f"  dtype={ts.dtype}  sample={ts[:3]}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: with start/end date, numpy-recarray
print("\nTest 2: start/end date, numpy-recarray")
try:
    ts = ng.index_constituent_timeseries('AAPL', 'S&P 500',
        start_date='2020-01-01', end_date='2021-01-01')
    print(f"  OK  type={type(ts).__name__}  len={len(ts)}")
    if len(ts) > 0:
        print(f"  dtype fields={ts.dtype.names}  sample={ts[:2]}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: pandas-dataframe format
print("\nTest 3: pandas-dataframe format")
try:
    ts = ng.index_constituent_timeseries('AAPL', 'S&P 500',
        start_date='2020-01-01', end_date='2021-01-01',
        timeseriesformat='pandas-dataframe')
    print(f"  OK  type={type(ts).__name__}  len={len(ts)}")
    if len(ts) > 0:
        print(f"  columns={list(ts.columns)}  index type={type(ts.index[0])}")
        print(ts.head(3))
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: with 'Current & Past' watchlist name
print("\nTest 4: 'S&P 500 Current & Past' watchlist")
try:
    ts = ng.index_constituent_timeseries('AAPL', 'S&P 500 Current & Past',
        start_date='2020-01-01', end_date='2021-01-01',
        timeseriesformat='pandas-dataframe')
    print(f"  OK  type={type(ts).__name__}  len={len(ts)}")
    if len(ts) > 0:
        print(ts.head(3))
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: dead ticker with Current & Past
print("\nTest 5: dead ticker CELG with 'S&P 500 Current & Past'")
try:
    ts = ng.index_constituent_timeseries('CELG', 'S&P 500 Current & Past',
        start_date='2015-01-01', end_date='2020-01-01',
        timeseriesformat='pandas-dataframe')
    print(f"  OK  type={type(ts).__name__}  len={len(ts)}")
    if len(ts) > 0:
        print(ts.head(3))
        print(ts.tail(3))
except Exception as e:
    print(f"  FAIL: {e}")

# Test 6: TSLA with SP500 (added 2020-12-21)
print("\nTest 6: TSLA membership check (added 2020-12-21)")
try:
    ts = ng.index_constituent_timeseries('TSLA', 'S&P 500 Current & Past',
        start_date='2020-01-01', end_date='2021-06-01',
        timeseriesformat='pandas-dataframe')
    print(f"  OK  len={len(ts)}")
    if len(ts) > 0:
        print(ts[ts.iloc[:,0] == True].head(3))  # first days as member
except Exception as e:
    print(f"  FAIL: {e}")

# Test 7: price for recent AAPL
print("\nTest 7: AAPL price (recent, within 2yr trial window)")
try:
    px = ng.price_timeseries('AAPL',
        start_date='2026-01-01', end_date=TODAY,
        timeseriesformat='pandas-dataframe')
    print(f"  OK  rows={len(px)}  cols={list(px.columns)}")
    if len(px) > 0:
        print(px.tail(2))
except Exception as e:
    print(f"  FAIL: {e}")

# Test 8: list symbols in 'S&P 500 Current & Past'
print("\nTest 8: watchlist_symbols('S&P 500 Current & Past')")
try:
    syms = ng.watchlist_symbols('S&P 500 Current & Past')
    print(f"  OK  count={len(syms)}  sample={list(syms)[:5]}")
except Exception as e:
    print(f"  FAIL: {e}")
