"""
Tests for pipeline/features/fundamental_features.py.

Covers the traps that make PIT fundamentals subtly wrong:
  - Q4 reconstruction (FY - Q1 - Q2 - Q3), since most filers drop standalone Q4
  - TTM as a rolling 4-quarter sum with a gap guard
  - split adjustment (a 7:1 / 4:1 split must NOT read as share issuance)
  - YoY matched by DATE, not a fixed 4-row shift
  - negative-base earnings/EPS growth collapsing to NaN, not exploding
  - a known margin / leverage change producing the correct feature sign
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.features import fundamental_features as ff


def _q(concept, start, end, filed, value, ptype="Q"):
    return {"concept": concept, "period_type": ptype, "start": pd.Timestamp(start),
            "end": pd.Timestamp(end), "filed": pd.Timestamp(filed), "value": float(value)}


# ─────────────────────────────────────────────────────────────────────────
# Q4 reconstruction
# ─────────────────────────────────────────────────────────────────────────
def test_q4_reconstructed_as_fy_minus_three_quarters():
    flow = pd.DataFrame([
        _q("revenue", "2020-01-01", "2020-03-31", "2020-04-20", 20),
        _q("revenue", "2020-04-01", "2020-06-30", "2020-07-20", 25),
        _q("revenue", "2020-07-01", "2020-09-30", "2020-10-20", 25),
        _q("revenue", "2020-01-01", "2020-12-31", "2021-02-15", 100, ptype="FY"),
    ])
    out = ff._discrete_quarters(flow)
    q4 = out[out["end"] == pd.Timestamp("2020-12-31")]
    assert len(q4) == 1
    # 100 - (20 + 25 + 25) = 30
    assert q4["value"].iloc[0] == pytest.approx(30.0)


def test_q4_skipped_when_not_exactly_three_interior_quarters():
    flow = pd.DataFrame([
        _q("revenue", "2020-01-01", "2020-03-31", "2020-04-20", 20),
        _q("revenue", "2020-04-01", "2020-06-30", "2020-07-20", 25),  # only 2 quarters
        _q("revenue", "2020-01-01", "2020-12-31", "2021-02-15", 100, ptype="FY"),
    ])
    out = ff._discrete_quarters(flow)
    # no reconstructed Q4 -> the FY-end date carries no quarterly row
    assert (out["end"] == pd.Timestamp("2020-12-31")).sum() == 0


def test_cashflow_ytd_ladder_differenced_into_quarters():
    # Cash-flow style: ONLY YTD cumulatives (all sharing the fiscal-year
    # start), no discrete Q2/Q3. Must be differenced back into 4 quarters.
    fy_start = "2020-01-01"
    flow = pd.DataFrame([
        _q("operating_cash_flow", fy_start, "2020-03-31", "2020-04-20", 10, ptype="Q"),     # YTD1 = Q1
        _q("operating_cash_flow", fy_start, "2020-06-30", "2020-07-20", 30, ptype="YTD2"),   # 6-mo
        _q("operating_cash_flow", fy_start, "2020-09-30", "2020-10-20", 60, ptype="YTD3"),   # 9-mo
        _q("operating_cash_flow", fy_start, "2020-12-31", "2021-02-15", 100, ptype="FY"),    # FY
    ])
    out = ff._discrete_quarters(flow).set_index("end")["value"]
    assert out[pd.Timestamp("2020-03-31")] == pytest.approx(10)   # Q1
    assert out[pd.Timestamp("2020-06-30")] == pytest.approx(20)   # 30 - 10
    assert out[pd.Timestamp("2020-09-30")] == pytest.approx(30)   # 60 - 30
    assert out[pd.Timestamp("2020-12-31")] == pytest.approx(40)   # 100 - 60
    assert out.sum() == pytest.approx(100)   # quarters reconstitute the FY


# ─────────────────────────────────────────────────────────────────────────
# TTM
# ─────────────────────────────────────────────────────────────────────────
def test_ttm_is_rolling_four_quarter_sum():
    quarters = pd.DataFrame({
        "start": pd.to_datetime(["2019-10-01", "2020-01-01", "2020-04-01", "2020-07-01"]),
        "end": pd.to_datetime(["2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30"]),
        "filed": pd.to_datetime(["2020-02-01", "2020-05-01", "2020-08-01", "2020-11-01"]),
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    out = ff._ttm(quarters)
    assert np.isnan(out["ttm"].iloc[2])          # not enough history yet
    assert out["ttm"].iloc[3] == pytest.approx(100.0)   # 10+20+30+40
    # known-date is the latest of the 4 filings
    assert out["ttm_filed"].iloc[3] == pd.Timestamp("2020-11-01")


def test_ttm_nan_across_a_missing_quarter():
    # a ~6-month gap between q2 and q3 -> the 4-window spans too many days
    quarters = pd.DataFrame({
        "start": pd.to_datetime(["2019-10-01", "2020-01-01", "2020-07-01", "2020-10-01"]),
        "end": pd.to_datetime(["2019-12-31", "2020-03-31", "2020-09-30", "2020-12-31"]),
        "filed": pd.to_datetime(["2020-02-01", "2020-05-01", "2020-11-01", "2021-02-01"]),
        "value": [10.0, 20.0, 30.0, 40.0],
    })
    out = ff._ttm(quarters)
    # end[0]=2019-12-31 .. end[3]=2020-12-31 spans 366d, outside the ~270d
    # 4-consecutive-quarter window -> guarded to NaN
    assert np.isnan(out["ttm"].iloc[3])


# ─────────────────────────────────────────────────────────────────────────
# split adjustment — the headline correctness property
# ─────────────────────────────────────────────────────────────────────────
def test_split_not_mistaken_for_issuance():
    # AAPL-like: steady count, then 7:1, then 4:1. In the latest basis every
    # row should be equal -> zero net issuance across the splits.
    shares = pd.DataFrame({
        "end": pd.to_datetime(["2013-12-31", "2014-06-30", "2014-12-31",
                               "2020-06-30", "2020-12-31"]),
        "filed": pd.to_datetime(["2014-01-20", "2014-07-20", "2015-01-20",
                                 "2020-07-20", "2021-01-20"]),
        "shares_outstanding": [100.0, 700.0, 700.0, 2800.0, 2800.0],  # 7:1 then 4:1
    })
    out = ff._split_adjust_shares(shares)
    adj = out["adj_shares"].to_numpy()
    assert np.allclose(adj, adj[0]), f"splits leaked into adj_shares: {adj}"


def test_genuine_buyback_survives_adjustment():
    # gradual ~3%/yr shrink, no split -> adjustment must leave it essentially
    # untouched (a real buyback is signal, not a split).
    raw = [1000.0, 985.0, 970.0, 955.0]
    shares = pd.DataFrame({
        "end": pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"]),
        "filed": pd.to_datetime(["2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01"]),
        "shares_outstanding": raw,
    })
    out = ff._split_adjust_shares(shares)
    assert np.allclose(out["adj_shares"].to_numpy(), raw)


# ─────────────────────────────────────────────────────────────────────────
# YoY date-matching, not row-shift
# ─────────────────────────────────────────────────────────────────────────
def test_prior_year_index_matches_by_date_not_position():
    # 3 quarters in 2020, then a GAP, then a quarter in 2021. A fixed shift(4)
    # would compare across the gap; date-matching finds the true ~1yr prior.
    ends = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30", "2021-06-30"]).to_numpy()
    idx = ff._prior_year_index(ends)
    # 2021-06-30's ~1yr-prior is 2020-06-30 (index 1)
    assert idx[3] == 1
    # the early rows have no prior year
    assert idx[0] == -1


# ─────────────────────────────────────────────────────────────────────────
# negative-base growth
# ─────────────────────────────────────────────────────────────────────────
def test_growth_nan_on_nonpositive_base():
    cur = np.array([5.0, 5.0, 5.0])
    prior = np.array([2.0, -2.0, 0.0])
    g = ff._growth(cur, prior)
    assert g[0] == pytest.approx(1.5)     # 5/2 - 1
    assert np.isnan(g[1])                 # negative base -> NaN, not -3.5
    assert np.isnan(g[2])                 # zero base -> NaN


def test_delta_defined_even_near_zero_base():
    cur = np.array([0.05, 0.05])
    prior = np.array([-0.01, np.nan])
    d = ff._delta(cur, prior)
    assert d[0] == pytest.approx(0.06)    # margin went from -1% to +5%
    assert np.isnan(d[1])


# ─────────────────────────────────────────────────────────────────────────
# integration: known margin / leverage change -> correct sign
# ─────────────────────────────────────────────────────────────────────────
def _two_years_of_quarters(concept, values_by_end, ptype_fy_value=None):
    """8 quarterly rows (2 fiscal years) for one flow concept."""
    rows = []
    ends = list(values_by_end.keys())
    for end in ends:
        start = (pd.Timestamp(end) - pd.offsets.QuarterBegin(startingMonth=1)).normalize()
        filed = pd.Timestamp(end) + pd.Timedelta(days=30)
        rows.append(_q(concept, start, end, filed, values_by_end[end]))
    return rows


def _toy_panel():
    dates = pd.date_range("2023-01-02", periods=6, freq="30D")
    idx = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    return pd.DataFrame({"close": 100.0, "sector": "Tech"}, index=idx)


def _toy_features(tmp_path):
    # one filing per ticker, filed mid-window (2023-03-01). Values distinct so
    # we can tell which side leaked.
    rows = []
    for tkr, val in [("AAA", 0.10), ("BBB", 0.20)]:
        rows.append({"ticker": tkr, "filed": pd.Timestamp("2023-03-01"),
                     "features_fund_revenue_growth_yoy": val,
                     "features_fund_eps_growth_yoy": val,
                     "features_fund_operating_margin": val,
                     "features_fund_net_margin": val, "features_fund_roe": val,
                     "features_fund_operating_margin_delta_yoy": val,
                     "features_fund_debt_to_equity": val,
                     "features_fund_debt_to_equity_delta_yoy": val,
                     "features_fund_asset_growth_yoy": val,
                     "features_fund_asset_turnover": val,
                     "features_fund_accruals_to_assets": val,
                     "features_fund_net_share_issuance_yoy": val,
                     "features_fund_book_value": 50.0, "features_fund_earnings_ttm": 5.0,
                     "features_fund_sales_ttm": 80.0, "features_fund_fcf_ttm": 4.0,
                     "features_fund_adj_shares": 10.0})
    f = pd.DataFrame(rows).set_index(["ticker", "filed"])
    path = tmp_path / "fund_feats.parquet"
    f.to_parquet(path)
    return path


def test_attach_is_causal_no_future_leak(tmp_path, monkeypatch):
    monkeypatch.setenv("FUNDAMENTAL_FEATURES", "1")
    panel = _toy_panel()
    path = _toy_features(tmp_path)
    # min_names_per_sector=1 so the 2-name toy sector isn't nulled by the guard
    out = ff.attach_fundamental_features(panel, features_path=path, min_names_per_sector=1)
    col = "features_fund_operating_margin"
    # dates before the 2023-03-01 filing must be NaN (nothing known yet);
    # dates on/after must be populated.
    om = out[col].reset_index()
    before = om[om["date"] < pd.Timestamp("2023-03-01")]
    after = om[om["date"] >= pd.Timestamp("2023-03-01")]
    assert before[col].isna().all(), "a future filing leaked onto an earlier day"
    assert after[col].notna().any(), "filed value never attached on/after its filed date"


def test_toggle_off_returns_panel_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("FUNDAMENTAL_FEATURES", raising=False)
    panel = _toy_panel()
    out = ff.attach_fundamental_features(panel, features_path=_toy_features(tmp_path))
    pd.testing.assert_frame_equal(out, panel)


def test_rising_margin_and_leverage_have_correct_signs():
    # Construct a ticker whose operating margin RISES YoY and whose D/E RISES
    # YoY, then assert the delta features are positive.
    ends = ["2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
            "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31"]
    # revenue flat at 100/qtr; operating income 10/qtr in 2021, 20/qtr in 2022
    rev = {e: 100.0 for e in ends}
    opi = {e: (10.0 if e < "2022" else 20.0) for e in ends}
    ni = {e: (8.0 if e < "2022" else 16.0) for e in ends}
    rows = []
    for c, vals in [("revenue", rev), ("operating_income", opi), ("net_income", ni)]:
        rows += _two_years_of_quarters(c, vals)
    # instants: equity flat 200; total debt rises 100 -> 200 (D/E 0.5 -> 1.0)
    for e in ends:
        filed = pd.Timestamp(e) + pd.Timedelta(days=30)
        rows.append(_q("stockholders_equity", e, e, filed, 200.0, ptype="PIT"))
        rows.append(_q("total_assets", e, e, filed, 1000.0, ptype="PIT"))
        debt = 100.0 if e < "2022" else 200.0
        rows.append(_q("long_term_debt", e, e, filed, debt, ptype="PIT"))
    fund = pd.DataFrame(rows)

    out = ff._ticker_frame(fund, shares=None)
    assert out is not None
    # take the last row (2022-12-31 period, TTM available, ~1yr prior exists)
    last = out.iloc[-1]
    assert last["operating_margin_delta_yoy"] > 0, "op margin doubled YoY -> delta must be +"
    assert last["debt_to_equity_delta_yoy"] > 0, "D/E rose 0.5->1.0 YoY -> delta must be +"
    assert last["operating_margin"] == pytest.approx(0.20, abs=1e-6)   # 80/400 TTM
    assert last["debt_to_equity"] == pytest.approx(1.0, abs=1e-6)      # 200/200


def test_attach_survives_mismatched_datetime_units(tmp_path, monkeypatch):
    # Reproduces the Hetzner failure: the saved feature parquet's `filed` index
    # round-trips as datetime64[us] on some pandas/pyarrow versions while the
    # panel's `date` index is datetime64[ns]. merge_asof raises MergeError on
    # a unit mismatch even though both sides are genuinely datetime — attach()
    # must normalize both sides rather than assume they already agree.
    monkeypatch.setenv("FUNDAMENTAL_FEATURES", "1")
    panel = _toy_panel()
    path = _toy_features(tmp_path)

    feats = pd.read_parquet(path).reset_index()
    feats["filed"] = feats["filed"].astype("datetime64[us]")  # force the mismatch
    feats = feats.set_index(["ticker", "filed"])
    assert feats.index.get_level_values("filed").dtype == np.dtype("datetime64[us]")
    feats.to_parquet(path)

    out = ff.attach_fundamental_features(panel, features_path=path, min_names_per_sector=1)
    assert out["features_fund_operating_margin"].notna().any()
