"""
Filing-level fundamental feature builder.

Turns the point-in-time EDGAR extracts (fundamentals_pit.parquet +
shares_outstanding_pit.parquet) into a tidy frame of fundamental features,
one row per (ticker, filed) — i.e. keyed on the date each value became
public. That frame is designed to be merged onto the daily price panel with
`pd.merge_asof(direction="backward")`, exactly like the ICT HTF features in
engineer.py: every trading day picks up the most recent fundamentals that
were *already known* on that day. Nothing here looks forward.

WHAT THIS MODULE DOES (filing granularity):
  - Reconstructs the missing Q4 quarter (FY - Q1 - Q2 - Q3) so every fiscal
    year has 4 discrete quarters — most filers stop reporting a standalone
    Q4 after ~2020 (Apple included); the annual 10-K carries it instead.
  - Builds TTM (trailing-twelve-month) series for every flow concept as a
    rolling 4-quarter sum — smooths the lumpiness (e.g. capex) that a single
    quarter shows, and is the right base for margins/growth/valuation.
  - Split-adjusts the raw EDGAR share count so a 7:1 or 4:1 stock split is
    NOT mistaken for a 600%/300% share issuance (verified against AAPL's
    2014 7:1 and 2020 4:1 splits).
  - Computes the fully-defined fundamental features (growth, margins, ROE,
    leverage + their YoY changes, asset growth, turnover, accruals, EPS
    growth, net share issuance). YoY is date-matched (~365d prior), never a
    fixed 4-row shift — the row cadence is not a reliable 4/year once Q4 is
    reconstruction-derived, and a naive shift silently compares the wrong
    quarters (a bug caught while spot-checking AAPL revenue).

WHAT THIS MODULE DELIBERATELY DEFERS TO THE PANEL WIRING (daily granularity):
  - Valuation ratios (book_to_price, earnings_yield, fcf_yield) and
    log_market_cap: these move every day because *price* moves every day,
    even though the fundamental numerator only updates at a filing. Computing
    them here would freeze price at the filing date — wrong. So this module
    emits the NUMERATORS (book_value, earnings_ttm, sales_ttm, fcf_ttm) plus
    the split-adjusted share count, and the panel step divides by that day's
    market cap.
  - Within-sector percentile-rank standardization: it is a cross-sectional
    (per date, per sector) transform and must run on the assembled daily
    panel, not on this per-filing frame.

See docs discussion / session notes for the agreed feature set and the
financials handling (banks structurally lack operating income, current
assets, etc. -> those features come out NaN and are handled downstream by
within-sector ranking, no special-casing here).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config.paths import PATHS

FEATURE_PREFIX = "features_"

EDGAR_DIR = PATHS.edgar_pit
FUNDAMENTALS_PATH = EDGAR_DIR / "fundamentals_pit.parquet"
SHARES_PATH = EDGAR_DIR / "shares_outstanding_pit.parquet"
# Pre-built filing-level feature frame (built offline by this module's
# __main__); attach_fundamental_features() loads it at run time.
FEATURES_PARQUET = EDGAR_DIR / "fundamental_features.parquet"


def fundamental_features_enabled() -> bool:
    """Env toggle, read at call time — same pattern as PHASE4_FEATURES /
    PIVOT_FEATURES in engineer.py. Default OFF: unless FUNDAMENTAL_FEATURES is
    set, attach_fundamental_features() returns the panel untouched, so the
    baseline is bit-identical to a run without this family."""
    return os.environ.get("FUNDAMENTAL_FEATURES", "0").strip().lower() in {"1", "true", "on", "yes"}

# Flow concepts get Q4-reconstruction + TTM; instants are used as-of.
FLOW_CONCEPTS = ["revenue", "operating_income", "net_income", "operating_cash_flow", "capex"]
INSTANT_CONCEPTS = [
    "stockholders_equity", "cash", "assets_current", "liabilities_current",
    "long_term_debt", "current_debt", "total_assets",
]

# YoY match window: a "one year prior" period end must land in this band.
YOY_MIN_DAYS, YOY_MAX_DAYS = 330, 400
# A valid TTM = 4 consecutive quarters whose end dates span roughly 3 quarters.
TTM_SPAN_MIN_DAYS, TTM_SPAN_MAX_DAYS = 240, 300
# Common split ratios to snap sudden share-count jumps to (and their inverses
# for reverse splits). A jump close to one of these is a split, not issuance.
SPLIT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 10, 20, 3 / 2, 5 / 2, 5 / 3, 5 / 4]
SPLIT_SNAP_TOL = 0.05  # within 5% of a round ratio => treat as split

# Features fully computed here (pure fundamentals, no price needed).
PURE_FEATURES = [
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "operating_margin",
    "net_margin",
    "roe",
    "operating_margin_delta_yoy",
    "debt_to_equity",
    "debt_to_equity_delta_yoy",
    "asset_growth_yoy",
    "asset_turnover",
    "accruals_to_assets",
    "net_share_issuance_yoy",
]
# Numerators emitted for the panel step to divide by daily market cap.
VALUATION_NUMERATORS = ["book_value", "earnings_ttm", "sales_ttm", "fcf_ttm", "adj_shares", "raw_shares"]


# ─────────────────────────────────────────────────────────────────────────
# TTM / Q4 reconstruction
# ─────────────────────────────────────────────────────────────────────────
def _discrete_quarters(flow: pd.DataFrame) -> pd.DataFrame:
    """
    flow: one ticker, one flow concept. Columns: period_type (Q / YTD2 / YTD3
    / FY), start, end, filed, value. Returns a discrete-quarter series
    (columns start, end, filed, value), sorted by end.

    Three sources, combined and de-duplicated by period end (earlier source
    wins):
      (a) direct discrete Q facts — a true "three months ended" column, the
          way income statements report.
      (b) YTD-ladder differencing — cash-flow statements report only
          cumulatives sharing the fiscal-year start (Q1, 6-mo, 9-mo, FY), so
          Qn = cum_n - cum_{n-1}. Only consecutive rungs ~90d apart are
          differenced; a missing rung leaves a hole rather than a 2-quarter
          lump.
      (c) Q4 fallback = FY - (the three interior quarters) — covers income
          years where the YTD rungs are absent but the discrete quarters exist.
    """
    parts: list[tuple] = []  # (end, start, filed, value, priority)

    for _, r in flow[flow["period_type"] == "Q"].iterrows():
        parts.append((r["end"], r["start"], r["filed"], r["value"], 0))

    fy = flow[flow["period_type"] == "FY"]
    for _, yr in fy.iterrows():
        cum = flow[(flow["start"] == yr["start"]) & (flow["end"] <= yr["end"])].sort_values("end")
        prev = None
        for _, r in cum.iterrows():
            if prev is None:
                parts.append((r["end"], r["start"], r["filed"], r["value"], 1))
            else:
                gap = (r["end"] - prev["end"]).days
                if 80 <= gap <= 100:
                    parts.append((r["end"], prev["end"], max(r["filed"], prev["filed"]),
                                  r["value"] - prev["value"], 1))
            prev = r

    if not parts:
        empty = pd.DataFrame(columns=["start", "end", "filed", "value"])
        for col in ("start", "end", "filed"):
            empty[col] = pd.to_datetime(empty[col])
        return empty

    q = pd.DataFrame(parts, columns=["end", "start", "filed", "value", "prio"])
    q = q.sort_values(["end", "prio"]).drop_duplicates(subset=["end"], keep="first")

    have_ends = set(q["end"])
    extra = []
    for _, yr in fy.iterrows():
        if yr["end"] in have_ends:
            continue
        interior = q[(q["end"] > yr["start"]) & (q["end"] <= yr["end"])]
        if len(interior) == 3:
            extra.append({
                "start": interior["end"].max(), "end": yr["end"], "filed": yr["filed"],
                "value": yr["value"] - interior["value"].sum(), "prio": 2,
            })
    if extra:
        q = pd.concat([q, pd.DataFrame(extra)], ignore_index=True)

    for col in ("start", "end", "filed"):
        q[col] = pd.to_datetime(q[col])
    return q.sort_values("end")[["start", "end", "filed", "value"]].reset_index(drop=True)


def _ttm(quarters: pd.DataFrame) -> pd.DataFrame:
    """
    quarters: discrete-quarter series (from _reconstruct_quarterly). Adds a
    `ttm` column = rolling sum of the current + 3 prior quarters, and a
    `ttm_filed` column = the date that TTM became known (the latest of the 4
    filings). NaN where 4 clean consecutive quarters aren't available.
    """
    q = quarters.sort_values("end").reset_index(drop=True)
    ttm_val = np.full(len(q), np.nan)
    ttm_filed = np.full(len(q), np.datetime64("NaT"), dtype="datetime64[ns]")

    for i in range(len(q)):
        if i < 3:
            continue
        window = q.iloc[i - 3:i + 1]
        span = (window["end"].iloc[-1] - window["end"].iloc[0]).days
        if TTM_SPAN_MIN_DAYS <= span <= TTM_SPAN_MAX_DAYS:
            ttm_val[i] = window["value"].sum()
            ttm_filed[i] = window["filed"].max()

    q["ttm"] = ttm_val
    q["ttm_filed"] = ttm_filed
    return q


# ─────────────────────────────────────────────────────────────────────────
# split adjustment
# ─────────────────────────────────────────────────────────────────────────
def _split_adjust_shares(shares: pd.DataFrame) -> pd.DataFrame:
    """
    shares: one ticker. Columns end, filed, shares_outstanding (raw EDGAR
    count). Returns the same rows plus `adj_shares` — the raw count divided
    by the cumulative forward split factor, so all rows are expressed in the
    latest share basis. A sudden consecutive-filing ratio close to a round
    split ratio is treated as a split; gradual changes are real issuance and
    left alone.
    """
    s = shares.sort_values("end").reset_index(drop=True).copy()
    raw = s["shares_outstanding"].to_numpy(dtype=float)
    n = len(raw)
    # split factor going FORWARD in time at each step (1.0 = no split).
    step_factor = np.ones(n)
    for i in range(1, n):
        if raw[i - 1] <= 0 or not np.isfinite(raw[i]) or not np.isfinite(raw[i - 1]):
            continue
        ratio = raw[i] / raw[i - 1]
        for r in SPLIT_RATIOS:
            if abs(ratio - r) / r <= SPLIT_SNAP_TOL or abs(ratio - 1 / r) / (1 / r) <= SPLIT_SNAP_TOL:
                step_factor[i] = r if abs(ratio - r) < abs(ratio - 1 / r) else 1 / r
                break

    # cumulative factor from each row forward to the latest row: to express an
    # early raw count in the latest basis, multiply by every split that
    # happened AFTER it.
    cum_forward = np.cumprod(step_factor)          # basis at row i relative to row 0
    s["adj_shares"] = raw * (cum_forward[-1] / cum_forward)
    return s


# ─────────────────────────────────────────────────────────────────────────
# YoY helpers
# ─────────────────────────────────────────────────────────────────────────
def _prior_year_index(ends: np.ndarray) -> np.ndarray:
    """
    For each period-end date, index of the row whose end is ~1 year earlier
    (within [YOY_MIN_DAYS, YOY_MAX_DAYS]); -1 if none. `ends` must be sorted
    ascending. Date-matched, NOT a fixed row shift.
    """
    ends = ends.astype("datetime64[D]")
    out = np.full(len(ends), -1, dtype=int)
    for i, e in enumerate(ends):
        lo = e - np.timedelta64(YOY_MAX_DAYS, "D")
        hi = e - np.timedelta64(YOY_MIN_DAYS, "D")
        cand = np.where((ends >= lo) & (ends <= hi))[0]
        if len(cand):
            out[i] = cand[-1]  # closest to `e` (largest end within the band)
    return out


def _growth(cur: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """cur/prior - 1, NaN where prior <= 0 or missing (avoids the negative-
    base explosion for earnings/EPS growth)."""
    prior = np.where((prior > 0) & np.isfinite(prior), prior, np.nan)
    return cur / prior - 1.0


def _delta(cur: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """cur - prior; NaN where prior missing. For level features (margins,
    D/E) whose change is meaningful even when the base is near zero."""
    prior = np.where(np.isfinite(prior), prior, np.nan)
    return cur - prior


# ─────────────────────────────────────────────────────────────────────────
# per-ticker assembly
# ─────────────────────────────────────────────────────────────────────────
def _ticker_frame(fund: pd.DataFrame, shares: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Build the per-(period end) aligned table for one ticker and compute all
    filing-level features on it. Returns a frame keyed by `filed`, or None if
    there isn't enough to build anything.
    """
    # TTM series per flow concept, keyed by period end.
    ttm_by_concept: dict[str, pd.DataFrame] = {}
    for concept in FLOW_CONCEPTS:
        c = fund[fund["concept"] == concept]
        if c.empty:
            continue
        ttm_by_concept[concept] = _ttm(_discrete_quarters(c))

    # Instant value per concept, keyed by period end (earliest filed already
    # ensured upstream by the extractor's dedup).
    instant_by_concept: dict[str, pd.DataFrame] = {}
    for concept in INSTANT_CONCEPTS:
        c = fund[fund["concept"] == concept][["end", "filed", "value"]]
        if not c.empty:
            instant_by_concept[concept] = c.sort_values("end").reset_index(drop=True)

    # Master period-end axis = union of all end dates seen for this ticker.
    ends = set()
    for df in ttm_by_concept.values():
        ends.update(df["end"].tolist())
    for df in instant_by_concept.values():
        ends.update(df["end"].tolist())
    if not ends:
        return None
    axis = pd.DataFrame({"end": pd.to_datetime(sorted(ends))})

    # As-of merge each series onto the axis (backward: most recent value known
    # at that period end). filed is tracked to derive the row's known-date.
    def _asof(series: pd.DataFrame, val_col: str, filed_col: str, out_val: str, out_filed: str):
        if series.empty:
            axis[out_val] = np.nan
            axis[out_filed] = pd.NaT
            return
        s = series.copy()
        s["end"] = pd.to_datetime(s["end"])
        s = s.sort_values("end")
        merged = pd.merge_asof(axis, s, on="end", direction="backward")
        axis[out_val] = merged[val_col].to_numpy()
        axis[out_filed] = merged[filed_col].to_numpy()

    for concept, df in ttm_by_concept.items():
        _asof(df[["end", "ttm", "ttm_filed"]].dropna(subset=["ttm"]),
              "ttm", "ttm_filed", f"{concept}_ttm", f"{concept}_filed")
    for concept, df in instant_by_concept.items():
        _asof(df, "value", "filed", concept, f"{concept}_filed")

    # Split-adjusted shares, as-of the period end.
    if shares is not None and not shares.empty:
        adj = _split_adjust_shares(shares)
        m = pd.merge_asof(axis, adj.sort_values("end")[["end", "shares_outstanding", "adj_shares"]],
                          on="end", direction="backward")
        axis["raw_shares"] = m["shares_outstanding"].to_numpy()
        axis["adj_shares"] = m["adj_shares"].to_numpy()
    else:
        axis["raw_shares"] = np.nan
        axis["adj_shares"] = np.nan

    axis = axis.sort_values("end").reset_index(drop=True)
    g = lambda col: axis[col].to_numpy(dtype=float) if col in axis else np.full(len(axis), np.nan)

    rev = g("revenue_ttm"); opi = g("operating_income_ttm"); ni = g("net_income_ttm")
    ocf = g("operating_cash_flow_ttm"); capex = g("capex_ttm")
    eq = g("stockholders_equity"); assets = g("total_assets")
    ltd = g("long_term_debt"); cd = g("current_debt")
    adj_sh = g("adj_shares")

    # ── level features ────────────────────────────────────────────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        axis["operating_margin"] = opi / rev
        axis["net_margin"] = ni / rev
        axis["roe"] = ni / eq
        total_debt = np.nansum(np.vstack([ltd, cd]), axis=0)
        # nansum treats all-NaN as 0; restore NaN when BOTH debt legs missing.
        both_missing = ~np.isfinite(ltd) & ~np.isfinite(cd)
        total_debt = np.where(both_missing, np.nan, total_debt)
        axis["debt_to_equity"] = total_debt / eq
        axis["asset_turnover"] = rev / assets
        eps_ttm = ni / adj_sh

    # ── YoY growth / delta features (date-matched) ──────────────────────────
    pidx = _prior_year_index(axis["end"].to_numpy())
    have = pidx >= 0

    def prior_of(arr: np.ndarray) -> np.ndarray:
        p = np.full(len(arr), np.nan)
        p[have] = arr[pidx[have]]
        return p

    axis["revenue_growth_yoy"] = _growth(rev, prior_of(rev))
    axis["eps_growth_yoy"] = _growth(eps_ttm, prior_of(eps_ttm))
    axis["asset_growth_yoy"] = _growth(assets, prior_of(assets))
    axis["operating_margin_delta_yoy"] = _delta(
        axis["operating_margin"].to_numpy(), prior_of(axis["operating_margin"].to_numpy()))
    axis["debt_to_equity_delta_yoy"] = _delta(
        axis["debt_to_equity"].to_numpy(), prior_of(axis["debt_to_equity"].to_numpy()))
    axis["net_share_issuance_yoy"] = _growth(adj_sh, prior_of(adj_sh))

    # ── accruals: (NI_ttm - OCF_ttm) / avg total assets (Sloan) ─────────────
    avg_assets = (assets + prior_of(assets)) / 2.0
    avg_assets = np.where(prior_of(assets) > 0, avg_assets, assets)  # fall back to level if no prior
    with np.errstate(divide="ignore", invalid="ignore"):
        axis["accruals_to_assets"] = (ni - ocf) / avg_assets

    # ── valuation numerators (division deferred to the panel step) ──────────
    axis["book_value"] = eq
    axis["earnings_ttm"] = ni
    axis["sales_ttm"] = rev
    axis["fcf_ttm"] = ocf - capex

    # ── row known-date = latest filed among the inputs actually used ────────
    filed_cols = [c for c in axis.columns if c.endswith("_filed")]
    axis["filed"] = axis[filed_cols].max(axis=1)
    axis = axis.dropna(subset=["filed"])
    if axis.empty:
        return None

    keep = ["filed", "end"] + PURE_FEATURES + VALUATION_NUMERATORS
    out = axis[[c for c in keep if c in axis.columns]].copy()
    # One row per filed date (latest period wins if two share a filing date).
    out = out.sort_values(["filed", "end"]).drop_duplicates(subset=["filed"], keep="last")
    return out


# ─────────────────────────────────────────────────────────────────────────
# public entry point
# ─────────────────────────────────────────────────────────────────────────
def build_fundamental_features(
    fundamentals_path: Path | str = FUNDAMENTALS_PATH,
    shares_path: Path | str = SHARES_PATH,
    prefix: str = FEATURE_PREFIX,
) -> pd.DataFrame:
    """
    Build the filing-level fundamental feature frame for every ticker.

    Returns a DataFrame with MultiIndex (ticker, filed) and one column per
    feature/numerator, prefixed with `prefix`. Merge onto the daily panel
    with merge_asof(direction="backward") on the filed date.
    """
    fund = pd.read_parquet(fundamentals_path)
    for col in ("end", "filed", "start"):
        if col in fund.columns:
            fund[col] = pd.to_datetime(fund[col])

    try:
        shares_all = pd.read_parquet(shares_path)
        shares_all["end"] = pd.to_datetime(shares_all["end"])
        shares_all["filed"] = pd.to_datetime(shares_all["filed"])
    except FileNotFoundError:
        shares_all = pd.DataFrame(columns=["ticker", "end", "filed", "shares_outstanding"])

    shares_by_ticker = dict(tuple(shares_all.groupby("ticker")))

    frames = []
    for ticker, fdf in fund.groupby("ticker"):
        out = _ticker_frame(fdf, shares_by_ticker.get(ticker))
        if out is None or out.empty:
            continue
        out.insert(0, "ticker", ticker)
        frames.append(out)

    if not frames:
        raise ValueError("No fundamental features built — check the input parquets.")

    result = pd.concat(frames, ignore_index=True)
    feat_cols = PURE_FEATURES + VALUATION_NUMERATORS
    result = result.rename(columns={c: f"{prefix}fund_{c}" for c in feat_cols})
    result = result.set_index(["ticker", "filed"]).sort_index()
    # `end` kept as a diagnostic column (period the row describes).
    return result


# Final daily feature columns (post-wiring): pure features carried through +
# valuation ratios computed against daily market cap + log size.
DAILY_VALUATION = ["book_to_price", "earnings_yield", "fcf_yield", "sales_to_price", "log_market_cap"]
DAILY_FUND_FEATURES = PURE_FEATURES + DAILY_VALUATION


# ─────────────────────────────────────────────────────────────────────────
# panel wiring — merge onto the daily panel (causal), value ratios, rank
# ─────────────────────────────────────────────────────────────────────────
def attach_fundamental_features(
    panel: pd.DataFrame,
    features_path: Path | str | None = None,
    sector_col: str = "sector",
    close_col: str = "close",
    min_names_per_sector: int = 15,
    prefix: str = FEATURE_PREFIX,
) -> pd.DataFrame:
    """
    Attach the fundamental feature family to a daily panel and return it.

    Gated by FUNDAMENTAL_FEATURES: if the toggle is OFF, the panel is returned
    unchanged (zero baseline impact). If ON:

      1. merge_asof(direction="backward") the pre-built filing-level frame onto
         each ticker's trading days, keyed on filed date — a value is visible
         only on/after the day it was filed. This is what makes using a parquet
         extracted "through today" safe for a run that starts in 2023.
      2. compute valuation ratios against THAT DAY's market cap
         (close x split-adjusted shares) — these move daily even though the
         numerator only updates at a filing; and log_market_cap.
      3. within-sector percentile-rank every fundamental feature per date, so a
         bank is ranked only against banks and structurally-absent metrics stay
         NaN. Sectors with < min_names_per_sector names on a date yield NaN
         rather than a meaningless small-n rank.

    panel: MultiIndex (date, ticker), must carry `close` and `sector`.
    """
    if not fundamental_features_enabled():
        return panel

    # Resolve the parquet location: explicit arg > FUNDAMENTAL_FEATURES_PATH env
    # > paths.yaml edgar_pit (default). The default follows data_root, so on
    # Hetzner just set data_root / ML_EDGAR_PIT and this resolves automatically.
    # Fail loud if the toggle is ON but the file is absent — a silent skip would
    # produce a feature-less panel that looks identical to a baseline run.
    path = Path(features_path or os.environ.get("FUNDAMENTAL_FEATURES_PATH", FEATURES_PARQUET))
    if not path.exists():
        raise FileNotFoundError(
            f"FUNDAMENTAL_FEATURES is ON but {path} not found. Build it with "
            f"`python -m pipeline.features.fundamental_features`, or point it via "
            f"paths.yaml edgar_pit / ML_EDGAR_PIT / FUNDAMENTAL_FEATURES_PATH."
        )

    feats = pd.read_parquet(path)  # index (ticker, filed), prefixed cols
    p = prefix + "fund_"
    pure = [p + c for c in PURE_FEATURES]
    num = {c: p + c for c in ("book_value", "earnings_ttm", "sales_ttm", "fcf_ttm", "adj_shares")}
    carry = pure + list(num.values())

    date_idx = panel.index.get_level_values("date")
    tkr_idx = panel.index.get_level_values("ticker")

    # ── 1. causal backward merge, per ticker ────────────────────────────────
    merged_cols = {c: np.full(len(panel), np.nan) for c in carry}
    for ticker in pd.unique(tkr_idx):
        if ticker not in feats.index.get_level_values("ticker"):
            continue
        rows = np.where(tkr_idx == ticker)[0]
        left = pd.DataFrame({"date": date_idx[rows]}).reset_index(drop=True)
        order = left["date"].argsort().to_numpy()
        left_sorted = left.iloc[order]

        right = (feats.xs(ticker, level="ticker").reset_index()
                 .rename(columns={"filed": "date"})[["date"] + carry]
                 .sort_values("date"))
        m = pd.merge_asof(left_sorted, right, on="date", direction="backward")
        # scatter back to original row positions
        for c in carry:
            vals = np.full(len(rows), np.nan)
            vals[order] = m[c].to_numpy()
            merged_cols[c][rows] = vals

    add = pd.DataFrame(merged_cols, index=panel.index)

    # ── 2. valuation ratios vs daily market cap ─────────────────────────────
    close = panel[close_col].to_numpy(dtype=float)
    mcap = close * add[num["adj_shares"]].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        add[p + "book_to_price"] = add[num["book_value"]].to_numpy() / mcap
        add[p + "earnings_yield"] = add[num["earnings_ttm"]].to_numpy() / mcap
        add[p + "fcf_yield"] = add[num["fcf_ttm"]].to_numpy() / mcap
        add[p + "sales_to_price"] = add[num["sales_ttm"]].to_numpy() / mcap
        add[p + "log_market_cap"] = np.log(np.where(mcap > 0, mcap, np.nan))

    # numerators were scaffolding for the ratios — drop them.
    add = add.drop(columns=list(num.values()))

    # ── 3. within-sector percentile-rank, per date ──────────────────────────
    feat_cols = [p + c for c in DAILY_FUND_FEATURES]
    sector = panel[sector_col].astype("object").to_numpy()
    grouper = pd.DataFrame({"date": date_idx, "sector": sector}, index=panel.index)
    ranked = {}
    for c in feat_cols:
        s = add[c]
        grp = s.groupby([grouper["date"], grouper["sector"]])
        pct = grp.rank(pct=True)
        n = grp.transform("count")
        ranked[c] = pct.where(n >= min_names_per_sector)
    ranked = pd.DataFrame(ranked, index=panel.index)

    return pd.concat([panel, ranked], axis=1)


def build_and_save(out_path: Path | str = FEATURES_PARQUET) -> pd.DataFrame:
    """Build the filing-level frame and persist it for run-time attach."""
    df = build_fundamental_features()
    df.to_parquet(out_path)
    return df


if __name__ == "__main__":
    df = build_and_save()
    print(f"Built {len(df):,} filing-level rows for {df.index.get_level_values('ticker').nunique()} tickers")
    print(f"saved -> {FEATURES_PARQUET}")
    feat = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    print(f"{len(feat)} feature columns:")
    for c in feat:
        nonnull = df[c].notna().mean() * 100
        print(f"  {c:<45} {nonnull:5.1f}% populated")
