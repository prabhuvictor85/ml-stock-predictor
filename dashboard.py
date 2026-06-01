"""
NSE & US ML Stock Predictor — Streamlit Dashboard
Run: .venv/Scripts/streamlit run dashboard.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

# ── Market config ─────────────────────────────────────────────────────────────
from pipeline.config.paths import PATHS
PROJECT_DIR = PATHS.project_root
PYTHON      = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"

MARKET_CONFIG = {
    "NSE (India)": {
        "script":    "run_nse_local.py",
        "output":    PROJECT_DIR / "output"    / "nse_local",
        "log_dir":   PROJECT_DIR / "artefacts" / "nse_local" / "logs",
        "shap_img":  PROJECT_DIR / "reports"   / "shap_global_nse_local.png",
        "pid_file":  PROJECT_DIR / ".dashboard_run_nse.pid",
        "watchlist_prefix": "watchlist_momentum_bull_",
        "benchmark": "^NSEI",
        "icon": "🇮🇳",
    },
    "US Stocks (SP500 + NASDAQ)": {
        "script":    "run_sp500_local.py",
        "output":    PROJECT_DIR / "output"    / "us_local",
        "log_dir":   PROJECT_DIR / "artefacts" / "us_local"  / "logs",
        "shap_img":  PROJECT_DIR / "reports"   / "shap_global_us_local.png",
        "pid_file":  PROJECT_DIR / ".dashboard_run_us.pid",
        "watchlist_prefix": "watchlist_momentum_bull_",
        "benchmark": "^GSPC / ^NDX",
        "icon": "🇺🇸",
    },
    "NSE TradingView": {
        "script":    "run_nse_tradingv_local.py",
        "output":    PROJECT_DIR / "output"    / "nse_tradingv",
        "log_dir":   PROJECT_DIR / "artefacts" / "nse_tradingv" / "logs",
        "shap_img":  PROJECT_DIR / "reports"   / "shap_global_nse_tradingv.png",
        "pid_file":  PROJECT_DIR / ".dashboard_run_nse_tv.pid",
        "watchlist_prefix": "watchlist_reversal_bull_",
        "benchmark": "NIFTY (TradingView)",
        "icon": "📺",
    },
}

st.set_page_config(
    page_title="ML Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Market selector (persisted in session state) ──────────────────────────────
if "market" not in st.session_state:
    st.session_state["market"] = "NSE (India)"


# ── Process helpers ───────────────────────────────────────────────────────────

def _pid_info(cfg: dict) -> tuple[int | None, str, Path | None]:
    """Return (pid, run_type, log_path) if a managed run is active, else (None, '', None)."""
    pid_file = cfg["pid_file"]
    if not pid_file.exists():
        return None, "", None
    try:
        parts    = pid_file.read_text().strip().split("\n")
        pid      = int(parts[0])
        run_type = parts[1] if len(parts) > 1 else "unknown"
        log_path = Path(parts[2]) if len(parts) > 2 else None
        if psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            return pid, run_type, log_path
    except Exception:
        pass
    pid_file.unlink(missing_ok=True)
    return None, "", None


def is_running(cfg: dict) -> bool:
    return _pid_info(cfg)[0] is not None


def launch(cfg: dict, run_type: str, extra_args: list[str]) -> Path:
    """Start the market script as a detached subprocess; return log path."""
    log_dir = cfg["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    tag      = "skip" if "--skip_train" in extra_args else "full"
    log_path = log_dir / f"dashboard_{tag}_{int(time.time())}.log"
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            [str(PYTHON), cfg["script"]] + extra_args,
            stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    cfg["pid_file"].write_text(f"{proc.pid}\n{run_type}\n{log_path}\n")
    return log_path


def stop_run(cfg: dict):
    pid, _, _ = _pid_info(cfg)
    if pid:
        try:
            psutil.Process(pid).terminate()
        except Exception:
            pass
    cfg["pid_file"].unlink(missing_ok=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

import re as _re
_DATE_RE = _re.compile(r"\d{4}-\d{2}-\d{2}$")


def _watchlist_dir(cfg: dict, date_str: str) -> Path:
    """
    Return the directory that holds watchlist CSVs for a given date.
    Supports two layouts:
      • Flat:   output/us_local/watchlist_*_YYYY-MM-DD.csv
      • Nested: output/us_local/YYYY-MM-DD/output/watchlist_*_YYYY-MM-DD.csv
    """
    output_dir = cfg["output"]
    nested = output_dir / date_str / "output"
    if nested.exists():
        return nested
    return output_dir


def available_run_dates(cfg: dict) -> list[str]:
    """
    Scan the output directory and return all scoring dates found, newest first.
    Looks for both flat CSV files and date-named sub-folders.
    """
    output_dir = cfg["output"]
    prefix     = cfg["watchlist_prefix"]
    _TIER_KEYS = ("large_", "mid_", "small_", "micro_")
    dates: set[str] = set()

    # Flat files in output_dir itself
    for f in output_dir.glob(f"{prefix}*.csv"):
        stem_suffix = f.stem.replace(prefix, "")
        if not any(stem_suffix.startswith(t) for t in _TIER_KEYS):
            if _DATE_RE.match(stem_suffix):
                dates.add(stem_suffix)

    # Nested: output_dir/<date>/output/watchlist_*.csv
    for sub in output_dir.iterdir():
        if sub.is_dir() and _DATE_RE.match(sub.name):
            nested = sub / "output"
            search_dir = nested if nested.exists() else sub
            for f in search_dir.glob(f"{prefix}*.csv"):
                stem_suffix = f.stem.replace(prefix, "")
                if not any(stem_suffix.startswith(t) for t in _TIER_KEYS):
                    if _DATE_RE.match(stem_suffix):
                        dates.add(stem_suffix)

    return sorted(dates, reverse=True)   # newest first


def load_watchlists_for_date(cfg: dict, date_str: str) -> dict[str, pd.DataFrame]:
    """Load all watchlist CSVs for a specific date string."""
    wl_dir = _watchlist_dir(cfg, date_str)
    result: dict[str, pd.DataFrame] = {}
    for label, pattern in [
        ("Momentum Bull",       f"watchlist_momentum_bull_{date_str}.csv"),
        ("Momentum Bear",       f"watchlist_momentum_bear_{date_str}.csv"),
        ("Reversal Bull",       f"watchlist_reversal_bull_{date_str}.csv"),
        ("Reversal Bear",       f"watchlist_reversal_bear_{date_str}.csv"),
        ("Momentum Bull large", f"watchlist_momentum_bull_large_{date_str}.csv"),
        ("Momentum Bull mid",   f"watchlist_momentum_bull_mid_{date_str}.csv"),
        ("Momentum Bull small", f"watchlist_momentum_bull_small_{date_str}.csv"),
        ("Momentum Bull micro", f"watchlist_momentum_bull_micro_{date_str}.csv"),
        ("Momentum Bear large", f"watchlist_momentum_bear_large_{date_str}.csv"),
        ("Momentum Bear mid",   f"watchlist_momentum_bear_mid_{date_str}.csv"),
        ("Momentum Bear small", f"watchlist_momentum_bear_small_{date_str}.csv"),
        ("Momentum Bear micro", f"watchlist_momentum_bear_micro_{date_str}.csv"),
        ("Reversal Bull large", f"watchlist_reversal_bull_large_{date_str}.csv"),
        ("Reversal Bull mid",   f"watchlist_reversal_bull_mid_{date_str}.csv"),
        ("Reversal Bull small", f"watchlist_reversal_bull_small_{date_str}.csv"),
        ("Reversal Bull micro", f"watchlist_reversal_bull_micro_{date_str}.csv"),
        ("Reversal Bear large", f"watchlist_reversal_bear_large_{date_str}.csv"),
        ("Reversal Bear mid",   f"watchlist_reversal_bear_mid_{date_str}.csv"),
        ("Reversal Bear small", f"watchlist_reversal_bear_small_{date_str}.csv"),
        ("Reversal Bear micro", f"watchlist_reversal_bear_micro_{date_str}.csv"),
    ]:
        p = wl_dir / pattern
        if p.exists():
            result[label] = pd.read_csv(p)
    return result


def latest_watchlists(cfg: dict) -> tuple[str, dict[str, pd.DataFrame]]:
    """Backward-compat wrapper: returns the most recent date and its watchlists."""
    dates = available_run_dates(cfg)
    if not dates:
        return "", {}
    date_str = dates[0]
    return date_str, load_watchlists_for_date(cfg, date_str)


def active_log_path(cfg: dict) -> Path | None:
    _, _, log_path = _pid_info(cfg)
    if log_path and log_path.exists():
        return log_path
    log_dir = cfg["log_dir"]
    logs = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def tail_log(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def parse_progress(log_text: str) -> tuple[int, int, int | None]:
    done = total = 0
    eta  = None
    for line in reversed(log_text.splitlines()):
        if "Feature engineering:" in line and "tickers done" in line:
            try:
                part  = line.split("Feature engineering:")[1]
                done  = int(part.split("/")[0].strip())
                total = int(part.split("/")[1].split()[0])
                eta   = int(part.split("ETA=")[1].rstrip("s").split("}")[0]) if "ETA=" in part else None
            except Exception:
                pass
            break
    return done, total, eta


def drift_summary(log_text: str):
    alerts   = [l for l in log_text.splitlines() if "Feature drift ALERT" in l]
    retrain  = any("RETRAIN TRIGGER" in l for l in log_text.splitlines())
    return alerts, retrain


# ── Explain helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_scores_detail(output_dir_str: str, date_str: str) -> dict:
    """
    Load scores_detail_momentum_{date}.json + scores_detail_reversal_{date}.json.
    Returns {ticker: {"momentum": {...}, "reversal": {...}}} merged dict.
    """
    output_dir = Path(output_dir_str)
    merged: dict = {}
    for model in ["momentum", "reversal"]:
        f = output_dir / f"scores_detail_{model}_{date_str}.json"
        if not f.exists():
            # older single-file fallback
            f = output_dir / f"scores_detail_{date_str}.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            for ticker, info in data.items():
                merged.setdefault(ticker, {})[model] = info
        except Exception:
            pass
    return merged


RESCORE_SCHEMES = {
    "Pure ML":     (1.00, 0.00),
    "Blend 85/15": (0.85, 0.15),
}


@st.cache_data(ttl=3600, show_spinner=False)
@st.cache_data(show_spinner=False)
def load_cap_tier_map(market_label: str) -> dict:
    """
    Return {ticker: tier} for the selected market. Tiers are computed from
    SEBI thresholds at lookup time so we never trust a stale `cap_tier` column.
    Returns {} for markets without a cap-tier source.
    """
    if market_label == "NSE TradingView":
        p = PATHS.stock_lists.nse_cap_tiers
        if not p.exists():
            return {}
        df = pd.read_csv(p)
        if "market_cap_crore" not in df.columns:
            return {}
        key_col = "TV_ticker" if "TV_ticker" in df.columns else "Symbol"
        def _sebi(m):
            if m >= 105_000: return "large"
            if m >= 34_700:  return "mid"
            if m >= 5_000:   return "small"
            return "micro"
        return {
            str(sym).strip(): _sebi(float(mcap))
            for sym, mcap in zip(df[key_col], df["market_cap_crore"])
            if pd.notna(sym) and pd.notna(mcap)
        }
    return {}


def rescore_tickers(output_dir_str: str, date_str: str, model_name: str,
                    side: str, model_wt: float, comp_wt: float, top_n: int) -> pd.DataFrame:
    """Re-rank a model+side using given weights from scores_detail JSON."""
    output_dir = Path(output_dir_str)
    f = output_dir / f"scores_detail_{model_name}_{date_str}.json"
    if not f.exists():
        f = output_dir / f"scores_detail_{date_str}.json"
    if not f.exists():
        return pd.DataFrame()

    data = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for ticker, info in data.items():
        sd = info.get(side, {})
        if not sd:
            continue
        m_sc   = float(sd.get("model_score",     0))
        c_sc   = float(sd.get("composite_score", 0))
        final  = m_sc * model_wt + c_sc * comp_wt
        orig_r = info.get(f"{side}_rank", sd.get("rank_in_universe"))
        in_wl  = info.get(f"in_{side}_watchlist", False)
        rows.append({
            "rank":            0,          # filled below
            "ticker":          ticker,
            "final_score":     round(final, 4),
            "model_score":     round(m_sc,  4),
            "composite_score": round(c_sc,  4),
            "orig_rank":       orig_r,
            "in_orig_wl":      "✅" if in_wl else "",
        })

    df = (pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .reset_index(drop=True))
    df["rank"] = df.index + 1
    return df.head(top_n)


def _render_explain(ticker: str, scores_all: dict, date_str: str):
    """Render the per-ticker explain breakdown panel."""
    data = scores_all.get(ticker)
    if not data:
        st.warning(f"No score detail found for **{ticker}** on {date_str}. "
                   "Re-run scoring to generate scores_detail files.")
        return

    st.markdown(f"### Explain: **{ticker}**  —  scored {date_str}")

    for model_name in ["momentum", "reversal"]:
        model_data = data.get(model_name)
        if not model_data:
            st.caption(f"No {model_name} data for {ticker}")
            continue

        in_bull = model_data.get("in_bull_watchlist", False)
        in_bear = model_data.get("in_bear_watchlist", False)
        bull_icon = "🟢" if in_bull else "⚪"
        bear_icon = "🔴" if in_bear else "⚪"
        header = (
            f"{bull_icon} **{model_name.title()} Bull** "
            f"{'✅ IN WATCHLIST' if in_bull else ''}"
            f"  |  "
            f"{bear_icon} **{model_name.title()} Bear** "
            f"{'✅ IN WATCHLIST' if in_bear else ''}"
        )

        with st.expander(header, expanded=True):
            col_bull, col_bear = st.columns(2)

            for col, side in [(col_bull, "bull"), (col_bear, "bear")]:
                side_data = model_data.get(side, {})
                if not side_data:
                    col.caption(f"No {side} data")
                    continue

                rank          = side_data.get("rank_in_universe", "?")
                universe      = side_data.get("universe_size", "?")
                model_score   = float(side_data.get("model_score",     0))
                composite_sc  = float(side_data.get("composite_score", 0))
                model_wt      = float(side_data.get("model_weight",    0))
                comp_wt       = float(side_data.get("composite_weight",0))
                in_wl         = model_data.get(f"in_{side}_watchlist", False)
                combined      = model_score * model_wt + composite_sc * comp_wt

                icon = "🟢" if side == "bull" else "🔴"
                badge = "  ✅" if in_wl else ""
                col.markdown(f"**{icon} {side.upper()}{badge}**")
                col.markdown(
                    f"Rank **{rank}** / {universe}  "
                    f"({'top ' + str(round(rank/universe*100, 1)) + '%' if isinstance(rank, int) and isinstance(universe, int) else ''})"
                )
                col.metric("Model score",     f"{model_score:.4f}",  help=f"Weight {model_wt:.0%}")
                col.metric("Composite score", f"{composite_sc:.4f}", help=f"Weight {comp_wt:.0%}")
                col.caption(
                    f"Combined = {model_score:.4f} × {model_wt:.0%} "
                    f"+ {composite_sc:.4f} × {comp_wt:.0%} = **{combined:.4f}**"
                )

                # Signal breakdown table
                sig_weights = side_data.get("signal_weights", {})
                sig_values  = side_data.get("signal_values",  {})
                if sig_weights:
                    rows = [
                        {
                            "Signal":       sig,
                            "Weight":       w,
                            "Value":        round(float(sig_values.get(sig, 0)), 4),
                            "Contribution": round(w * float(sig_values.get(sig, 0)), 4),
                        }
                        for sig, w in sig_weights.items()
                    ]
                    df_sig = pd.DataFrame(rows).sort_values("Contribution", ascending=False)
                    col.dataframe(df_sig, use_container_width=True, hide_index=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 ML Stock Predictor")
    st.divider()

    # ── Market selector ────────────────────────────────────────────────────
    st.subheader("Market")
    market_choice = st.radio(
        "Select market",
        options=list(MARKET_CONFIG.keys()),
        index=list(MARKET_CONFIG.keys()).index(st.session_state["market"]),
        label_visibility="collapsed",
    )
    if market_choice != st.session_state["market"]:
        st.session_state["market"] = market_choice
        st.rerun()

    cfg     = MARKET_CONFIG[st.session_state["market"]]
    pid, run_type, run_log = _pid_info(cfg)
    running = pid is not None

    st.caption(f"Benchmark: {cfg['benchmark']}")
    st.divider()

    # ── Status ─────────────────────────────────────────────────────────────
    if running:
        st.error(f"🔄 **{run_type.upper()} in progress**")
        st.caption(f"PID {pid}")
        if st.button("⛔ Stop Run", use_container_width=True):
            stop_run(cfg)
            st.rerun()
    else:
        st.success("✅ Idle — ready to run")

    st.divider()
    st.subheader("Run Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Weekly\n(skip train)", use_container_width=True, disabled=running):
            log = launch(cfg, "skip_train", ["--skip_train"])
            st.session_state["active_log"] = str(log)
            st.rerun()
    with col2:
        if st.button("🔁 Full\nRetrain", use_container_width=True, disabled=running):
            log = launch(cfg, "full_train", [])
            st.session_state["active_log"] = str(log)
            st.rerun()

    st.divider()
    st.subheader("Auto-refresh")
    auto = st.toggle("Live updates (20 min)", value=running)

    st.divider()

    # ── Date selector ──────────────────────────────────────────────────────
    st.subheader("📅 Run Date")
    _run_dates = available_run_dates(cfg)
    if _run_dates:
        # Key includes market so switching market resets the picker
        _date_key = f"selected_date_{st.session_state['market']}"
        if _date_key not in st.session_state or st.session_state[_date_key] not in _run_dates:
            st.session_state[_date_key] = _run_dates[0]

        # Radio list — newest first, most recent pre-selected
        _chosen = st.radio(
            "Select a scoring date",
            options=_run_dates,
            index=_run_dates.index(st.session_state[_date_key]),
            label_visibility="collapsed",
            key=f"date_radio_{st.session_state['market']}",
        )
        if _chosen != st.session_state[_date_key]:
            st.session_state[_date_key] = _chosen
            st.rerun()
    else:
        st.caption("No run dates found yet.")
        _date_key = f"selected_date_{st.session_state['market']}"
        st.session_state.setdefault(_date_key, "")

    st.divider()
    log_dir = cfg["log_dir"]
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if logs:
            import datetime
            st.caption(f"Last log: `{logs[0].name}`")
            mtime = datetime.datetime.fromtimestamp(logs[0].stat().st_mtime)
            st.caption(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")


# ── Header ────────────────────────────────────────────────────────────────────
market_icon  = cfg["icon"]
market_label = st.session_state["market"]
st.markdown(f"## {market_icon} {market_label}")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_watch, tab_log, tab_shap, tab_drift = st.tabs(
    ["📊 Watchlists", "📜 Live Log", "🔬 SHAP Analysis", "⚠️ Drift Monitor"]
)

# ── Tab: Watchlists ───────────────────────────────────────────────────────────
with tab_watch:
    _date_key  = f"selected_date_{st.session_state['market']}"
    _sel_date  = st.session_state.get(_date_key, "")
    if _sel_date:
        date_str   = _sel_date
        watchlists = load_watchlists_for_date(cfg, date_str)
    else:
        date_str, watchlists = latest_watchlists(cfg)

    if not watchlists:
        st.info(f"No watchlists found for {market_label}. Run a scoring pass first.")
    else:
        # ── Ranking method selector ───────────────────────────────────────────
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            rank_method = st.radio(
                "Ranking method",
                ["Original (70/30)", "Pure ML", "Blend 85/15"],
                horizontal=True,
                key="rank_method",
            )
        with rc2:
            top_n = st.number_input("Top N", min_value=10, max_value=200,
                                    value=30, step=10, key="wl_top_n")

        mw, cw = {"Original (70/30)": (0.70, 0.30),
                  "Pure ML":          (1.00, 0.00),
                  "Blend 85/15":      (0.85, 0.15)}[rank_method]

        st.subheader(f"Watchlists — as of {date_str}  |  {rank_method}")

        # Highlight colours — tested against Streamlit dark theme
        # Mid-tone values: visible on dark (#0e1117) and readable on light
        SCHEME_COLORS = {
            "both":    "#1a4731",   # deep green  — in Pure ML AND Blend
            "ml_only": "#17375e",   # deep blue   — Pure ML only
            "bl_only": "#5c3d00",   # deep amber  — Blend 85/15 only
            "orig":    "#2d2d2d",   # dark grey   — Original only / neither
        }

        DISPLAY_COLS = ["rank", "ticker", "side", "score",
                        "adx_14", "return_20d", "vol_contraction",
                        "sector_rs_20d", "zone_htf_confluence",
                        "sdz_htf_score", "ssz_htf_score"]

        wt1, wt2, wt3, wt4 = st.tabs(
            ["🟢 Momentum Bull", "🔴 Momentum Bear",
             "🔵 Reversal Bull",  "🟠 Reversal Bear"]
        )

        TAB_META = [
            (wt1, "Momentum Bull", "momentum", "bull"),
            (wt2, "Momentum Bear", "momentum", "bear"),
            (wt3, "Reversal Bull", "reversal",  "bull"),
            (wt4, "Reversal Bear", "reversal",  "bear"),
        ]

        _scores_dir = str(_watchlist_dir(cfg, date_str))

        for tab_obj, label, model_name, side in TAB_META:
            with tab_obj:
                # Always fetch Pure ML and Blend sets for cross-highlighting
                df_ml = rescore_tickers(_scores_dir, date_str,
                                        model_name, side, 1.00, 0.00, int(top_n))
                df_bl = rescore_tickers(_scores_dir, date_str,
                                        model_name, side, 0.85, 0.15, int(top_n))
                ml_set = set(df_ml["ticker"]) if not df_ml.empty else set()
                bl_set = set(df_bl["ticker"]) if not df_bl.empty else set()

                def _scheme_tag(ticker):
                    in_ml = ticker in ml_set
                    in_bl = ticker in bl_set
                    if in_ml and in_bl:   return "ML + Blend"
                    if in_ml:             return "Pure ML"
                    if in_bl:             return "Blend 85/15"
                    return ""

                def _row_color(row):
                    t = row["ticker"]
                    in_ml = t in ml_set
                    in_bl = t in bl_set
                    if in_ml and in_bl:
                        c = SCHEME_COLORS["both"]
                    elif in_ml:
                        c = SCHEME_COLORS["ml_only"]
                    elif in_bl:
                        c = SCHEME_COLORS["bl_only"]
                    else:
                        c = SCHEME_COLORS["orig"]
                    return [f"background-color: {c}; color: #e8e8e8"] * len(row)

                if rank_method == "Original (70/30)":
                    df = watchlists.get(label, pd.DataFrame())  # noqa: F841 (label used below)
                    if df.empty:
                        st.info(f"No {label} watchlist found.")
                        continue
                    df = df.head(int(top_n)).copy()

                    # Add scheme tag column
                    if "ticker" in df.columns:
                        df.insert(2, "schemes", df["ticker"].apply(_scheme_tag))

                    cols    = ["schemes"] + [c for c in DISPLAY_COLS if c in df.columns]
                    display = df[[c for c in cols if c in df.columns]].copy()
                    for c in ["score", "adx_14", "return_20d", "vol_contraction",
                              "sector_rs_20d", "zone_htf_confluence",
                              "sdz_htf_score", "ssz_htf_score"]:
                        if c in display.columns:
                            display[c] = display[c].apply(
                                lambda x: f"{x:.2f}" if pd.notna(x) else ""
                            )
                    st.dataframe(
                        display.style.apply(_row_color, axis=1),
                        use_container_width=True, hide_index=True,
                    )
                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        f"⬇️ Download {label}",
                        csv,
                        file_name=f"watchlist_{label.lower().replace(' ','_')}_{date_str}.csv",
                        mime="text/csv",
                    )

                else:
                    df = rescore_tickers(_scores_dir, date_str,
                                         model_name, side, mw, cw, int(top_n))
                    if df.empty:
                        st.warning(f"No scores_detail data for {label}.")
                        continue

                    display = df.copy()
                    display.insert(2, "schemes", display["ticker"].apply(_scheme_tag))
                    display["orig_wl"] = display["in_orig_wl"].apply(
                        lambda v: "Yes" if v == "✅" else ""
                    )
                    display = display.drop(columns=["in_orig_wl"])
                    col_order = ["rank", "ticker", "schemes", "orig_wl",
                                 "final_score", "model_score", "composite_score", "orig_rank"]
                    display = display[[c for c in col_order if c in display.columns]]

                    st.dataframe(
                        display.style.apply(_row_color, axis=1),
                        use_container_width=True, hide_index=True,
                    )

                    n_both   = sum(1 for t in df["ticker"] if t in ml_set and t in bl_set)
                    n_ml     = sum(1 for t in df["ticker"] if t in ml_set and t not in bl_set)
                    n_bl     = sum(1 for t in df["ticker"] if t not in ml_set and t in bl_set)
                    n_orig   = (df["in_orig_wl"] == "✅").sum()
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("In original (70/30)", int(n_orig))
                    mc2.metric("ML + Blend",          n_both)
                    mc3.metric("Pure ML only",         n_ml)
                    mc4.metric("Blend 85/15 only",     n_bl)

                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        f"⬇️ Download {label} ({rank_method})",
                        csv,
                        file_name=f"rescore_{model_name}_{side}_{rank_method.lower().replace(' ','_')}_{date_str}.csv",
                        mime="text/csv",
                    )

                # ── Colour legend (always shown) ──────────────────────────────
                st.markdown(
                    "<div style='font-size:0.82rem; margin-top:6px;'>"
                    "<span style='background:#1a4731; color:#e8e8e8; "
                    "padding:2px 8px; border-radius:4px; margin-right:8px;'>"
                    "&#9632; ML + Blend</span>"
                    "<span style='background:#17375e; color:#e8e8e8; "
                    "padding:2px 8px; border-radius:4px; margin-right:8px;'>"
                    "&#9632; Pure ML only</span>"
                    "<span style='background:#5c3d00; color:#e8e8e8; "
                    "padding:2px 8px; border-radius:4px; margin-right:8px;'>"
                    "&#9632; Blend 85/15 only</span>"
                    "<span style='background:#2d2d2d; color:#e8e8e8; "
                    "padding:2px 8px; border-radius:4px;'>"
                    "&#9632; Not in either</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ── Cap-tier breakdown ─────────────────────────────────────────
                # Original (70/30 momentum, 55/45 reversal) → read pre-computed
                # tier CSVs from disk.
                # Pure ML or Blend 85/15 → rescore the full universe with the
                # user's chosen blend, then filter by cap_tier_map.
                _TIER_DEFS = [("large", "🔵 Large Cap"),
                              ("mid",   "🟡 Mid Cap"),
                              ("small", "🟢 Small Cap"),
                              ("micro", "🟣 Micro Cap")]
                _tier_data = {}

                if rank_method == "Original (70/30)":
                    # Pre-computed tier CSVs (saved by run_*.py with config blend)
                    for _tk, _tl in _TIER_DEFS:
                        _key = f"{label} {_tk}"
                        if _key in watchlists and not watchlists[_key].empty:
                            _tier_data[_tl] = watchlists[_key]
                    _tier_show_cols = ["rank", "ticker", "score",
                                       "return_20d", "adx_14", "sector_rs_20d"]
                else:
                    # Re-compute tiers using the user's blend
                    _tier_map = load_cap_tier_map(market_label)
                    if _tier_map:
                        _df_full = rescore_tickers(
                            _scores_dir, date_str,
                            model_name, side, mw, cw, top_n=20_000,
                        )
                        if not _df_full.empty:
                            _df_full["cap_tier"] = _df_full["ticker"].map(_tier_map)
                            for _tk, _tl in _TIER_DEFS:
                                _tdf = (_df_full[_df_full["cap_tier"] == _tk]
                                        .head(10).reset_index(drop=True))
                                if not _tdf.empty:
                                    _tdf["rank"] = _tdf.index + 1
                                    _tier_data[_tl] = _tdf
                    _tier_show_cols = ["rank", "ticker", "final_score",
                                       "model_score", "composite_score", "orig_rank"]

                if _tier_data:
                    _expander_label = (
                        f"📊 By Market Cap Tier — Top 10 each ({rank_method})"
                    )
                    with st.expander(_expander_label, expanded=False):
                        _tier_cols = st.columns(len(_tier_data))
                        for (_tlabel, _tdf), _tcol in zip(_tier_data.items(), _tier_cols):
                            with _tcol:
                                st.markdown(f"**{_tlabel}**")
                                _show = _tdf[[c for c in _tier_show_cols
                                              if c in _tdf.columns]].copy()
                                for _fc in ["score", "final_score", "model_score",
                                             "composite_score", "return_20d",
                                             "adx_14", "sector_rs_20d"]:
                                    if _fc in _show.columns:
                                        _show[_fc] = _show[_fc].apply(
                                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                                        )
                                st.dataframe(_show, use_container_width=True,
                                             hide_index=True)

        # ── Explain Panel ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Explain a Ticker")

        scores_all = load_scores_detail(_scores_dir, date_str)

        # Watchlist tickers first, then all scored tickers
        wl_tickers = sorted({
            str(t) for df in watchlists.values()
            if "ticker" in df.columns
            for t in df["ticker"].dropna().tolist()
        })
        extra = [t for t in sorted(scores_all.keys()) if t not in wl_tickers]
        ticker_options = ["— select —"] + wl_tickers + extra

        explain_ticker = st.selectbox(
            "Select ticker (watchlist tickers listed first)",
            ticker_options,
            key=f"explain_{st.session_state['market']}",
        )
        if explain_ticker and explain_ticker != "— select —":
            _render_explain(explain_ticker, scores_all, date_str)


# ── Tab: Live Log ─────────────────────────────────────────────────────────────
with tab_log:
    log_path = active_log_path(cfg)

    if log_path:
        st.caption(f"📄 `{log_path.name}`")
        log_text = tail_log(log_path)

        done, total, eta = parse_progress(log_text)
        if total > 0:
            pct = done / total
            st.progress(pct, text=f"Feature engineering: {done}/{total} "
                                  f"({'ETA ' + str(eta) + 's' if eta else 'done ✓'})")

        milestones = [l for l in log_text.splitlines()
                      if any(k in l for k in [
                          "Feature engineering:", "MultiTFMerger",
                          "SHAP global", "Files saved", "SCORING",
                          "Drift records", "RETRAIN TRIGGER", "ERROR", "Error"
                      ]) and "DEBUG" not in l]
        if milestones:
            with st.expander("📍 Milestones", expanded=True):
                for m in milestones[-15:]:
                    if '"msg":"' in m:
                        try:
                            msg = json.loads(m)
                            lvl = msg.get("level", "INFO")
                            txt = msg.get("msg", m)
                            t   = msg.get("time", "")[-8:]
                            icon = "⚠️" if lvl == "WARNING" else ("❌" if lvl == "ERROR" else "✅")
                            st.markdown(f"`{t}` {icon} {txt}")
                        except Exception:
                            st.markdown(f"• {m[-120:]}")
                    else:
                        st.markdown(f"• {m[-120:]}")

        with st.expander("📃 Raw log (last 80 lines)"):
            st.code(log_text, language=None)
    else:
        st.info("No log file found yet.")


# ── Tab: SHAP ─────────────────────────────────────────────────────────────────
with tab_shap:
    shap_img = cfg["shap_img"]
    if shap_img.exists():
        import datetime
        mtime = datetime.datetime.fromtimestamp(shap_img.stat().st_mtime)
        st.caption(f"Generated: {mtime.strftime('%Y-%m-%d %H:%M')}")
        st.image(str(shap_img), use_container_width=True)
    else:
        st.info(f"SHAP plot not found for {market_label}. Run a full train or skip_train to generate it.")


# ── Tab: Drift Monitor ────────────────────────────────────────────────────────
with tab_drift:
    log_path = active_log_path(cfg)
    if log_path:
        log_text = tail_log(log_path, n=500)
        alerts, retrain_needed = drift_summary(log_text)

        if retrain_needed:
            st.error("🚨 RETRAIN TRIGGER — significant feature distribution shift detected.")
        elif alerts:
            st.warning(f"⚠️ {len(alerts)} feature drift alerts")
        else:
            st.success("✅ No drift alerts in latest log")

        if alerts:
            rows = []
            for a in alerts:
                try:
                    msg  = json.loads(a).get("msg", a)
                    feat = msg.split("feature='")[1].split("'")[0]
                    psi  = float(msg.split("PSI=")[1].split(" ")[0])
                    rows.append({"feature": feat, "PSI": round(psi, 4),
                                 "severity": "🔴 High" if psi > 1.0 else "🟡 Medium"})
                except Exception:
                    pass
            if rows:
                df_drift = pd.DataFrame(rows).sort_values("PSI", ascending=False)
                st.dataframe(df_drift, use_container_width=True, hide_index=True)
    else:
        st.info("No log available for drift analysis.")


# ── Auto-refresh loop ─────────────────────────────────────────────────────────
if auto and is_running(cfg):
    time.sleep(1200)  # 20 minutes
    st.rerun()
