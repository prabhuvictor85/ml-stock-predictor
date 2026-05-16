# sr_levels.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# Optional imports for density-based/ML-ish methods
try:
    from scipy.stats import gaussian_kde

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from sklearn.cluster import DBSCAN

    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ---------------- Models ----------------


@dataclass
class Level:
    price: float
    strength: float
    touches: int
    method: str
    meta: dict


# ---------------- Utilities ----------------


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
    return out


def _slice_from_date(df: pd.DataFrame, from_date: str | None) -> pd.DataFrame:
    if from_date and "Date" in df.columns:
        return df[df["Date"] >= pd.to_datetime(from_date)].copy()
    return df.copy()


def _dedupe_close_prices(levels: list[Level], band_frac: float = 0.001) -> list[Level]:
    """De-duplicate levels that are within ~0.1% by keeping the strongest."""
    if not levels:
        return levels
    levels_sorted = sorted(levels, key=lambda x: x.price)
    out: list[Level] = []
    for lvl in levels_sorted:
        if any(
            abs(lvl.price - kept.price) / ((lvl.price + kept.price) / 2) < band_frac for kept in out
        ):
            # If close to an existing level, keep the stronger one
            # Replace if current stronger
            for i, kept in enumerate(out):
                if abs(lvl.price - kept.price) / ((lvl.price + kept.price) / 2) < band_frac:
                    if lvl.strength > kept.strength:
                        out[i] = lvl
                    break
        else:
            out.append(lvl)
    # Return sorted by strength desc to be nice
    out.sort(key=lambda x: x.strength, reverse=True)
    return out


# ---------------- 1) Swings (pivots) ----------------


def find_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """
    Return pivot highs/lows with their index, price and type.
    Pivot high at i: High[i] > High[i-left:i] and High[i] >= High[i+1:i+1+right]
    Pivot low  at i: Low[i]  < Low[i-left:i]  and Low[i]  <= Low[i+1:i+1+right]
    """
    df = _validate_df(df)
    H, L = df["High"].values, df["Low"].values
    n = len(df)
    pivots = []

    for i in range(left, n - right):
        h = H[i]
        l = L[i]
        if all(h > H[i - left : i]) and all(h >= H[i + 1 : i + 1 + right]):
            pivots.append({"idx": i, "price": float(h), "type": "res"})
        if all(l < L[i - left : i]) and all(l <= L[i + 1 : i + 1 + right]):
            pivots.append({"idx": i, "price": float(l), "type": "sup"})

    return pd.DataFrame(pivots)


def cluster_levels_by_distance(
    prices: np.ndarray,
    types: np.ndarray,
    dates: np.ndarray | None = None,
    eps_frac: float = 0.002,  # merge band ~0.2% of median price
    min_touches: int = 2,
    recency_halflife: int = 100,  # bars for exp weighting (larger = flatter)
) -> list[Level]:
    if prices.size == 0:
        return []
    pmed = float(np.median(prices))
    eps = max(1e-12, pmed * eps_frac)

    # Make a recency weight vector
    if dates is None:
        t = np.arange(len(prices))
    else:
        order = np.argsort(dates)
        t = np.empty_like(order)
        t[order] = np.arange(len(order))
    if recency_halflife > 0:
        lam = np.log(2) / recency_halflife
        recency_w = np.exp(lam * (t - t.min()))
    else:
        recency_w = np.ones_like(prices, dtype=float)

    # 1D agglomeration by sorted price banding
    order = np.argsort(prices)
    prices_s = prices[order]
    types_s = types[order]
    rec_w_s = recency_w[order]

    clusters: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, len(prices_s)):
        if abs(prices_s[i] - prices_s[i - 1]) > eps:
            clusters.append((prices_s[start:i], types_s[start:i], rec_w_s[start:i]))
            start = i
    clusters.append((prices_s[start:], types_s[start:], rec_w_s[start:]))

    levels: list[Level] = []
    for prc, typ, w in clusters:
        if len(prc) < min_touches:
            continue
        # Weighted median representative
        sort_idx = np.argsort(prc)
        prc_sorted = prc[sort_idx]
        w_sorted = w[sort_idx]
        csum = np.cumsum(w_sorted)
        median_idx = np.searchsorted(csum, csum[-1] / 2)
        rep = float(prc_sorted[min(median_idx, len(prc_sorted) - 1)])

        t_sup = int(np.sum(typ == "sup"))
        t_res = int(np.sum(typ == "res"))
        balance_bonus = 1.0 + min(t_sup, t_res) / max(t_sup + t_res, 1)
        strength = float(len(prc) * (w.mean()) * balance_bonus)

        levels.append(
            Level(
                price=rep,
                strength=strength,
                touches=len(prc),
                method="swing_cluster",
                meta={"touch_sup": t_sup, "touch_res": t_res},
            )
        )
    levels.sort(key=lambda x: x.strength, reverse=True)
    return levels


def swing_cluster_levels(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    eps_frac: float = 0.002,
    max_levels: int = 10,
    from_date: str | None = None,
) -> list[Level]:
    df = _slice_from_date(_validate_df(df), from_date)
    piv = find_pivots(df, left, right)
    if piv.empty:
        return []
    dates = df["Date"].values[piv["idx"].values] if "Date" in df.columns else np.arange(len(piv))
    prices = piv["price"].values
    types = piv["type"].values
    lvls = cluster_levels_by_distance(prices, types, dates=dates, eps_frac=eps_frac)
    return lvls[:max_levels]


def swing_multiscale_levels(
    df: pd.DataFrame,
    scales: list[tuple[int, int]] = ((2, 2), (3, 3), (5, 5)),
    eps_frac: float = 0.0025,
    max_levels: int = 15,
    from_date: str | None = None,
) -> list[Level]:
    """Run swings at multiple (left,right) and merge."""
    all_levels: list[Level] = []
    for left, right in scales:
        all_levels.extend(
            swing_cluster_levels(
                df, left, right, eps_frac=eps_frac, max_levels=999, from_date=from_date
            )
        )
    # de-dup close prices keeping stronger
    merged = _dedupe_close_prices(all_levels, band_frac=eps_frac)
    return merged[:max_levels]


def build_zigzag_path(
    df: pd.DataFrame, left: int = 2, right: int = 2, from_date: str | None = None
) -> list[dict]:
    """Alternating H/L zigzag across the whole span (good for overlay)."""
    df2 = _slice_from_date(_validate_df(df), from_date)
    piv = find_pivots(df2, left, right)
    if piv.empty:
        return []
    piv = piv.sort_values("idx").reset_index(drop=True)
    path = [piv.iloc[0].to_dict()]
    for i in range(1, len(piv)):
        prev = path[-1]
        cur = piv.iloc[i].to_dict()
        if prev["type"] == cur["type"]:
            # keep more extreme of the same type
            if cur["type"] == "res":
                if cur["price"] > prev["price"]:
                    path[-1] = cur
            else:
                if cur["price"] < prev["price"]:
                    path[-1] = cur
        else:
            path.append(cur)
    out = []
    for p in path:
        i = int(p["idx"])
        if "Date" in df2.columns:
            t = df2.loc[i, "Date"]
            out.append(
                {
                    "time": {"year": int(t.year), "month": int(t.month), "day": int(t.day)},
                    "value": float(p["price"]),
                    "type": p["type"],
                    "idx": i,
                }
            )
        else:
            out.append({"time": i, "value": float(p["price"]), "type": p["type"], "idx": i})
    return out


# ---------------- 2) Pivot Points ----------------


def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    P = (high + low + close) / 3.0
    R1 = 2 * P - low
    S1 = 2 * P - high
    R2 = P + (high - low)
    S2 = P - (high - low)
    R3 = high + 2 * (P - low)
    S3 = low - 2 * (high - P)
    return {"P": P, "R1": R1, "S1": S1, "R2": R2, "S2": S2, "R3": R3, "S3": S3}


def camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    rng = high - low
    L1 = close - 1.1 * rng / 12
    L2 = close - 1.1 * rng / 6
    L3 = close - 1.1 * rng / 4
    L4 = close - 1.1 * rng / 2
    H1 = close + 1.1 * rng / 12
    H2 = close + 1.1 * rng / 6
    H3 = close + 1.1 * rng / 4
    H4 = close + 1.1 * rng / 2
    return {"H1": H1, "H2": H2, "H3": H3, "H4": H4, "L1": L1, "L2": L2, "L3": L3, "L4": L4}


def pivot_levels_from_last_period(
    df: pd.DataFrame, period: Literal["D", "W", "M"] = "D", from_date: str | None = None
) -> dict[str, dict[str, float]]:
    """
    Compute Classic + Camarilla from the last COMPLETED period (daily/weekly/monthly).
    """
    df = _slice_from_date(_validate_df(df), from_date)
    if "Date" in df.columns:
        df = df.set_index("Date")
    ohlc = (
        df[["Open", "High", "Low", "Close"]]
        .resample(period)
        .agg({"High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )
    if len(ohlc) < 2:
        raise ValueError("Not enough data to compute previous period pivots")
    prev = ohlc.iloc[-2]
    c = float(prev["Close"])
    h = float(prev["High"])
    l = float(prev["Low"])
    return {"classic": classic_pivots(h, l, c), "camarilla": camarilla_pivots(h, l, c)}


# ---------------- 3) Density (KDE / DBSCAN on pivots) ----------------


def kde_levels(
    prices: np.ndarray, grid_points: int = 400, top_k: int = 10, bandwidth: float | None = None
) -> list[Level]:
    if not _HAS_SCIPY or prices.size < 3:
        return []
    pmin, pmax = float(np.min(prices)), float(np.max(prices))
    if pmax <= pmin:
        return []
    grid = np.linspace(pmin, pmax, grid_points)
    kde = gaussian_kde(prices, bw_method=bandwidth)
    dens = kde(grid)

    peaks: list[tuple[float, float]] = []
    for i in range(1, len(grid) - 1):
        if dens[i] > dens[i - 1] and dens[i] > dens[i + 1]:
            peaks.append((grid[i], dens[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)

    return [
        Level(price=float(px), strength=float(d), touches=0, method="kde", meta={})
        for px, d in peaks[:top_k]
    ]


def dbscan_levels(
    prices: np.ndarray, eps_frac: float = 0.002, min_samples: int = 2, top_k: int = 10
) -> list[Level]:
    if not _HAS_SKLEARN or prices.size == 0:
        return []
    pmed = float(np.median(prices))
    eps = max(1e-9, pmed * eps_frac)
    X = prices.reshape(-1, 1)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    outs: list[Level] = []
    for lab in set(labels):
        if lab == -1:  # noise
            continue
        members = prices[labels == lab]
        rep = float(np.median(members))
        outs.append(
            Level(
                price=rep,
                strength=float(len(members)),
                touches=int(len(members)),
                method="dbscan",
                meta={"cluster_label": int(lab)},
            )
        )
    outs.sort(key=lambda x: x.strength, reverse=True)
    return outs[:top_k]


def density_levels_from_pivots(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
    use_kde_first: bool = True,
    max_levels: int = 12,
    from_date: str | None = None,
) -> list[Level]:
    df2 = _slice_from_date(_validate_df(df), from_date)
    piv = find_pivots(df2, left, right)
    if piv.empty:
        return []
    prices = piv["price"].values.astype(float)
    out: list[Level] = []
    if use_kde_first and _HAS_SCIPY:
        out.extend(kde_levels(prices, top_k=max_levels))
    if _HAS_SKLEARN:
        out.extend(dbscan_levels(prices, top_k=max_levels))
    return _dedupe_close_prices(out, band_frac=0.001)[:max_levels]


# ---------------- 4) Rolling Extremes (Donchian-like) ----------------


def rolling_extreme_levels(
    df: pd.DataFrame, windows: list[int] = (20, 50, 100), from_date: str | None = None
) -> list[Level]:
    """
    Highest high / lowest low over rolling windows. Each window contributes 2 levels.
    Strength scales with window length (longer = stronger).
    """
    df2 = _slice_from_date(_validate_df(df), from_date)
    outs: list[Level] = []
    for w in windows:
        if len(df2) < w:
            continue
        hh = float(df2["High"].rolling(w).max().iloc[-1])
        ll = float(df2["Low"].rolling(w).min().iloc[-1])
        outs.append(Level(price=hh, strength=w, touches=0, method="roll_high", meta={"window": w}))
        outs.append(Level(price=ll, strength=w, touches=0, method="roll_low", meta={"window": w}))
    return _dedupe_close_prices(outs, band_frac=0.0015)


# ---------------- 5) Volume Profile (HVNs) ----------------


def volume_profile_hvn_levels(
    df: pd.DataFrame, bins: int = 60, use_typical_price: bool = True, from_date: str | None = None
) -> list[Level]:
    """
    Approximate HVNs by summing volume into price bins.
    If Volume not available, returns [].
    """
    df2 = _slice_from_date(_validate_df(df), from_date)
    if "Volume" not in df2.columns:
        return []
    if len(df2) < 5:
        return []

    if use_typical_price:
        price = (df2["High"] + df2["Low"] + df2["Close"]) / 3.0
    else:
        price = df2["Close"]

    pmin, pmax = float(price.min()), float(price.max())
    if pmax <= pmin:
        return []

    vol = df2["Volume"].astype(float).values
    price_arr = price.astype(float).values
    hist, edges = np.histogram(price_arr, bins=bins, range=(pmin, pmax), weights=vol)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # pick local maxima
    peaks: list[tuple[float, float]] = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append((centers[i], hist[i]))
    peaks.sort(key=lambda x: x[1], reverse=True)

    # scale strength by relative histogram height
    if not peaks:
        return []
    maxh = max(h for _, h in peaks)
    levels = [
        Level(
            price=float(px),
            strength=float(h / (maxh + 1e-9)) * 10.0,
            touches=0,
            method="vprofile",
            meta={},
        )
        for px, h in peaks[:12]
    ]
    return _dedupe_close_prices(levels, band_frac=0.0015)


# ---------------- Public API ----------------
