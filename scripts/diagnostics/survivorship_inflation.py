"""Measure the survivorship/anachronism premium in the SP500 universe.

Compares, per month since 2010, the equal-weight mean 20d forward return of:
  A) SNAPSHOT pool — today's 503 members, any date they have data
     (= how the panel builds universes today: the present projected backwards)
  B) PIT pool      — actual S&P 500 members on that date (membership intervals)

A − B is the bias the training labels currently absorb. Note B is still
optimistic: dead members with no local CSV (yfinance purged them) can't be
included, so TRUE PIT would be worse than B. That gap is counted too.
"""
import logging
logging.disable(logging.CRITICAL)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from pipeline.config.paths import PATHS
from pipeline.universe import load_membership_intervals, normalize_ticker

MEMBERSHIP = r"C:/Victor/Learning_charts/stock_lists/membership_sp500.csv"
FWD = 20

iv = load_membership_intervals(MEMBERSHIP)
current = set(iv.loc[iv.end_date.isna(), "ticker"])
all_members = set(iv.ticker)

# ── load close series for every membership ticker with local data ───────────
closes = {}
missing = []
for tk in sorted(all_members):
    f = PATHS.stock_data.us / f"{tk}-1d.csv"
    if not f.exists():
        missing.append(tk)
        continue
    s = pd.read_csv(f, parse_dates=["Date"], usecols=["Date", "Close"]
                    ).set_index("Date")["Close"].sort_index()
    s = s[~s.index.duplicated(keep="last")]   # some local CSVs carry dup dates
    if len(s) > 100:
        closes[tk] = s

px = pd.DataFrame(closes)
# CSVs mix naive and tz-aware dates -> after UTC-coercion the same trading day
# can appear as both 00:00 and 05:00 rows. Floor to calendar date and merge.
px.index = pd.to_datetime(px.index, errors="coerce", utc=True).tz_localize(None).normalize()
px = px[px.index.notna()]
px = px.groupby(level=0).last().sort_index()
px = px[px.index >= pd.Timestamp("2009-06-01")]
print(f"price matrix: {px.shape[0]} days x {px.shape[1]} tickers")
fwd_ret = px.shift(-FWD) / px - 1.0

print(f"membership tickers: {len(all_members)} | with local data: {len(closes)} "
      f"| MISSING (mostly dead, unpurchasable from yfinance): {len(missing)}")

# membership lookup
imap = {}
for r in iv.itertuples(index=False):
    end = r.end_date if pd.notna(r.end_date) else pd.Timestamp.max
    imap.setdefault(r.ticker, []).append((r.start_date, end))

def members_on(d):
    return {t for t, ws in imap.items() if any(s <= d < e for s, e in ws)}

# ── monthly comparison ───────────────────────────────────────────────────────
# sample only real common trading days (tz-tainted stray index rows are mostly
# NaN across tickers and would otherwise eat the sample)
valid_days = fwd_ret.index[fwd_ret.notna().sum(axis=1) >= 300]
dates = valid_days[::21]              # ~monthly
rows = []
for d in dates:
    if d > fwd_ret.index[-1] - pd.Timedelta(days=40):
        break
    r = fwd_ret.loc[d]
    alive = set(r.dropna().index)
    pool_a = current & alive                  # snapshot projected back
    pool_b = members_on(d) & alive            # PIT (data-limited)
    if len(pool_a) < 50 or len(pool_b) < 50:
        continue
    anachronistic = pool_a - pool_b           # in snapshot, NOT yet/no-longer members
    dead_gap = len(members_on(d) - alive)     # true members we have no data for
    rows.append({
        "date": d,
        "ret_snapshot": r[list(pool_a)].mean(),
        "ret_pit": r[list(pool_b)].mean(),
        "n_snapshot": len(pool_a),
        "n_pit": len(pool_b),
        "n_anachronistic": len(anachronistic),
        "n_dead_missing": dead_gap,
    })

df = pd.DataFrame(rows).set_index("date")
df["bias"] = df.ret_snapshot - df.ret_pit

ann = 252 / FWD
print(f"\n{len(df)} monthly observations  {df.index[0].date()} -> {df.index[-1].date()}")
print(f"mean 20d fwd return  SNAPSHOT pool: {df.ret_snapshot.mean():+.4%}")
print(f"mean 20d fwd return  PIT pool     : {df.ret_pit.mean():+.4%}")
print(f"mean bias (snapshot - PIT)        : {df.bias.mean():+.4%} per 20d "
      f"(~{df.bias.mean()*ann:+.2%}/yr)")
t = df.bias.mean() / (df.bias.std() / np.sqrt(len(df)))
print(f"t-stat of bias                    : {t:.2f}")
print(f"bias positive in                  : {(df.bias > 0).mean():.0%} of months")

print(f"\navg pool sizes: snapshot={df.n_snapshot.mean():.0f}  pit={df.n_pit.mean():.0f}")
print(f"avg anachronistic tickers per date (in snapshot, not PIT): {df.n_anachronistic.mean():.0f}")
print(f"avg TRUE members missing from data (dead, need Norgate)  : {df.n_dead_missing.mean():.0f}")

print("\nby era:")
for era, g in df.groupby(df.index.year // 4 * 4):
    print(f"  {era}-{era+3}: bias {g.bias.mean():+.4%}/20d  "
          f"(anachronistic {g.n_anachronistic.mean():.0f}, dead-missing {g.n_dead_missing.mean():.0f})")
