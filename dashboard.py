"""
NSE & US ML Stock Predictor — Streamlit Dashboard
Run: .venv\Scripts\streamlit run dashboard.py
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
PROJECT_DIR = Path(r"C:\Victor\Project\ml-stock-predictor")
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

def latest_watchlists(cfg: dict) -> tuple[str, dict[str, pd.DataFrame]]:
    output_dir = cfg["output"]
    prefix     = cfg["watchlist_prefix"]
    files = sorted(output_dir.glob(f"{prefix}*.csv"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return "", {}
    date_str = files[0].stem.replace(prefix, "")
    result: dict[str, pd.DataFrame] = {}
    for label, pattern in [
        ("Momentum Bull", f"watchlist_momentum_bull_{date_str}.csv"),
        ("Momentum Bear", f"watchlist_momentum_bear_{date_str}.csv"),
        ("Reversal Bull", f"watchlist_reversal_bull_{date_str}.csv"),
        ("Reversal Bear", f"watchlist_reversal_bear_{date_str}.csv"),
    ]:
        p = output_dir / pattern
        if p.exists():
            result[label] = pd.read_csv(p)
    return date_str, result


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
    auto = st.toggle("Live updates (5s)", value=running)

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
    date_str, watchlists = latest_watchlists(cfg)

    if not watchlists:
        st.info(f"No watchlists found for {market_label}. Run a scoring pass first.")
    else:
        st.subheader(f"Watchlists — as of {date_str}")

        DISPLAY_COLS = ["rank", "ticker", "side", "score",
                        "adx_14", "return_20d", "vol_contraction",
                        "sector_rs_20d", "zone_htf_confluence",
                        "sdz_htf_score", "ssz_htf_score"]

        wt1, wt2, wt3, wt4 = st.tabs(
            ["🟢 Momentum Bull", "🔴 Momentum Bear",
             "🔵 Reversal Bull",  "🟠 Reversal Bear"]
        )

        def show_watchlist(tab, df: pd.DataFrame, side: str):
            with tab:
                cols    = [c for c in DISPLAY_COLS if c in df.columns]
                display = df[cols].copy()
                for c in ["score", "adx_14", "return_20d", "vol_contraction",
                          "sector_rs_20d", "zone_htf_confluence",
                          "sdz_htf_score", "ssz_htf_score"]:
                    if c in display.columns:
                        display[c] = display[c].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
                st.dataframe(display, use_container_width=True, hide_index=True)
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    f"⬇️ Download {side}",
                    csv,
                    file_name=f"watchlist_{side.lower().replace(' ', '_')}_{date_str}.csv",
                    mime="text/csv",
                )

        for tab_obj, (label, df) in zip(
            [wt1, wt2, wt3, wt4], watchlists.items()
        ):
            show_watchlist(tab_obj, df, label)


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
    time.sleep(5)
    st.rerun()
