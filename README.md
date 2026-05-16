# ml-stock-predictor

A minimal machine-learning-style stock predictor built with pure Python.

## What it does

The predictor fits a simple linear regression model against historical closing
prices and estimates a future closing price.

## Usage

```bash
python -m ml_stock_predictor 100 102 104 106 --days-ahead 2
```

Example output:

```text
Predicted closing price in 2 day(s): 110.00
```
