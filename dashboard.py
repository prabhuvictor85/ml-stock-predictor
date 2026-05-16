"""
NSE ML Stock Predictor — Streamlit Dashboard
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

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path(r"C:\Victor\Project\ml-stock-predictor")
PYTHON       = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
OUTPUT_DIR   = PROJECT_DIR / "output"  / "nse_local"
LOG_DIR      = PROJECT_DIR / "artefacts" / "nse_local" / "logs"
SHAP_IMG     = PROJECT_DIR / "reports" / "shap_global_nse_local.png"
PID_FILE     = PROJECT_DIR / ".dashboard_run.pid"

st.set_page_config(
    page_title="NSE ML Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Process helpers ───────────────────────────────────────────────────────────

def _pid_info() -> tuple[int | None, str, Path | None]:
    """Return (pid, run_type, log_path) if a managed run is active, else (None, '', None)."""
    if not PID_FILE.exists():
        return None, "", None
    try:
        parts = PID_FILE.read_text().strip().split("\n")
        pid      = int(parts[0])
        run_type = parts[1] if len(parts) > 1 else "unknown"
        log_path = Path(parts[2]) if len(parts) > 2 else None
        if psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower():
            return pid, run_type, log_path
    except Exception:
        pass
    PID_FILE.unlink(missing_ok=True)
    return None, "", None


def is_running() -> bool:
    return _pid_info()[0] is not None


def launch(run_type: str, extra_args: list[str]) -> Path:
    """Start run_nse_local.py as a detached subprocess, return log path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag      = "skip" if "--skip_train" in extra_args else "full"
    log_path = LOG_DIR / f"dashboard_{tag}_{int(time.time())}.log"
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(
            [str(PYTHON), "run_nse_local.py"] + extra_args,
            stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_DIR),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    PID_FILE.write_text(f"{proc.pid}\n{run_type}\n{log_path}\n")
    return log_path



def stop_run():
    pid, _, _ = _pid_info()
    if pid:
        try:
            psutil.Process(pid).terminate()
        except Exception:
            pass
    PID_FILE.unlink(missing_ok=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

def latest_watchlists() -> tuple[str, dict[str, pd.DataFrame]]:
    files = sorted(OUTPUT_DIR.glob("watchlist_momentum_bull_*.csv"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return "", {}
    date_str = files[0].stem.replace("watchlist_momentum_bull_", "")
    result: dict[str, pd.DataFrame] = {}
    for label, pattern in [
        ("Momentum Bull",  f"watchlist_momentum_bull_{date_str}.csv"),
        ("Momentum Bear",  f"watchlist_momentum_bear_{date_str}.csv"),
        ("Reversal Bull",  f"watchlist_reversal_bull_{date_str}.csv"),
        ("Reversal Bear",  f"watchlist_reversal_bear_{date_str}.csv"),
    ]:
        p = OUTPUT_DIR / pattern
        if p.exists():
            result[label] = pd.read_csv(p)
    return date_str, result


def active_log_path() -> Path | None:
    _, _, log_path = _pid_info()
    if log_path and log_path.exists():
        return log_path
    # Fall back to most recently modified run log
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def tail_log(path: Path, n: int = 80) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def parse_progress(log_text: str) -> tuple[int, int, int | None]:
    """Extract (done, total, eta_s) from latest Feature engineering line."""
    done = total = eta = 0
    for line in reversed(log_text.splitlines()):
        if "Feature engineering:" in line and "tickers done" in line:
            try:
                part = line.split("Feature engineering:")[1]
                done  = int(part.split("/")[0].strip())
                total = int(part.split("/")[1].split()[0])
                eta   = int(part.split("ETA=")[1].rstrip("s").split("}")[0]) if "ETA=" in part else None
            except Exception:
                pass
            break
    return done, total, eta


def drift_summary(log_text: str) -> list[str]:
    alerts = [l for l in log_text.splitlines() if "Feature drift ALERT" in l]
    retrain = any("RETRAIN TRIGGER" in l for l in log_text.splitlines())
    return alerts, retrain


# ── Sidebar ───────────────────────────────────────────────────────────────────

pid, run_type, run_log = _pid_info()
running = pid is not None

with st.sidebar:
    st.title("📈 NSE ML Predictor")
    st.divider()

    if running:
        st.error(f"🔄 **{run_type.upper()} in progress**")
        st.caption(f"PID {pid}")
        if st.button("⛔ Stop Run", use_container_width=True):
            stop_run()
            st.rerun()
    else:
        st.success("✅ Idle — ready to run")

    st.divider()
    st.subheader("Run Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Weekly\n(skip train)", use_container_width=True, disabled=running):
            log = launch("skip_train", ["--skip_train"])
            st.session_state["active_log"] = str(log)
            st.rerun()
    with col2:
        if st.button("🔁 Full\nRetrain", use_container_width=True, disabled=running):
            log = launch("full_train", [])
            st.session_state["active_log"] = str(log)
            st.rerun()

    st.divider()
    st.subheader("Auto-refresh")
    auto = st.toggle("Live updates (5s)", value=running)

    st.divider()
    # Last run info
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if logs:
        st.caption(f"Last log: `{logs[0].name}`")
        import datetime
        mtime = datetime.datetime.fromtimestamp(logs[0].stat().st_mtime)
        st.caption(f"Modified: {mtime.strftime('%Y-%m-%d %H:%M')}")


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_watch, tab_log, tab_shap, tab_drift = st.tabs(
    ["📊 Watchlists", "📜 Live Log", "🔬 SHAP Analysis", "⚠️ Drift Monitor"]
)

# ── Tab: Watchlists ───────────────────────────────────────────────────────────
with tab_watch:
    date_str, watchlists = latest_watchlists()

    if not watchlists:
        st.info("No watchlists found. Run a scoring pass first.")
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
                cols = [c for c in DISPLAY_COLS if c in df.columns]
                display = df[cols].copy()
                # Format numerics
                for c in ["score", "adx_14", "return_20d", "vol_contraction",
                          "sector_rs_20d", "zone_htf_confluence",
                          "sdz_htf_score", "ssz_htf_score"]:
                    if c in display.columns:
                        display[c] = display[c].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
                st.dataframe(display, use_container_width=True, hide_index=True)

                # Export button
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
    log_path = active_log_path()

    if log_path:
        st.caption(f"📄 `{log_path.name}`")
        log_text = tail_log(log_path)

        # Progress bar
        done, total, eta = parse_progress(log_text)
        if total > 0:
            pct = done / total
            st.progress(pct, text=f"Feature engineering: {done}/{total} "
                                  f"({'ETA ' + str(eta) + 's' if eta else 'done ✓'})")

        # Key milestones
        milestones = [l for l in log_text.splitlines()
                      if any(k in l for k in [
                          "Feature engineering:", "MultiTFMerger",
                          "SHAP global", "Files saved", "SCORING",
                          "Drift records", "RETRAIN TRIGGER", "ERROR", "Error"
                      ]) and "DEBUG" not in l]
        if milestones:
            with st.expander("📍 Milestones", expanded=True):
                for m in milestones[-15:]:
                    # strip json wrapper for readability
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

        # Raw log
        with st.expander("📃 Raw log (last 80 lines)"):
            st.code(log_text, language=None)
    else:
        st.info("No log file found yet.")


# ── Tab: SHAP ─────────────────────────────────────────────────────────────────
with tab_shap:
    if SHAP_IMG.exists():
        import datetime
        mtime = datetime.datetime.fromtimestamp(SHAP_IMG.stat().st_mtime)
        st.caption(f"Generated: {mtime.strftime('%Y-%m-%d %H:%M')}")
        st.image(str(SHAP_IMG), use_container_width=True)
    else:
        st.info("SHAP plot not found. Run a full train or skip_train to generate it.")


# ── Tab: Drift Monitor ────────────────────────────────────────────────────────
with tab_drift:
    log_path = active_log_path()
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
                    msg   = json.loads(a).get("msg", a)
                    feat  = msg.split("feature='")[1].split("'")[0]
                    psi   = float(msg.split("PSI=")[1].split(" ")[0])
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
if auto and is_running():
    time.sleep(5)
    st.rerun()
