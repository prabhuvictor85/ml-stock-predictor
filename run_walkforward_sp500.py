#!/usr/bin/env python3
"""
run_walkforward_sp500.py — Automated walk-forward production simulation for SP500.

WHAT IT DOES
────────────
Starting from an already-trained SP500 model (frozen at its last train date),
this orchestrator walks forward in time on a fixed cadence:

    for each scheduled date D:
        1. Incrementally download price data up to D       (download_us_data.py)
        2. INFERENCE-ONLY: score & pick stocks, no training (run_sp500_local.py --skip_train --as_of D)
        3. Read the drift monitor's PSI output for date D
        4. If drift breaches the retrain threshold  ->  FULL RETRAIN now, then continue
        5. At the quarter-end date                  ->  FULL RETRAIN (scheduled)

This is a *point-in-time* simulation: at every step the model only ever sees
data up to D, so it faithfully reproduces how the system would have behaved in
live production over that window — and measures how fast the model decays
between retrains.

DEFAULT SCHEDULE  (full year 2024 — quarterly retrains, bi-weekly inference)
────────────────
    2024-01-13  inference   ...  2024-03-31  FULL RETRAIN  (Q1)
    2024-01-27  inference   ...  2024-06-30  FULL RETRAIN  (Q2)
    2024-02-10  inference   ...  2024-09-30  FULL RETRAIN  (Q3)
    ...                          2024-12-31  FULL RETRAIN  (Q4)

LAUNCH
──────
    # one-time: model must already be trained, e.g.
    #   python3 run_sp500_local.py --train_start 2010-01-01 --as_of 2023-12-08

    export NTFY_TOPIC="hetzner-victor-ml"       # optional phone alerts
    nohup python3 run_walkforward_sp500.py \
        --train_start 2010-01-01 \
        --start 2024-01-13 \
        --end   2024-12-31 \
        --quarterly_retrain \
        --cadence_days 14 \
        > /mnt/data/artefacts/us_local/walkforward.log 2>&1 &
    echo "Walk-forward PID: $!"

    # preview the schedule without running anything:
    python3 run_walkforward_sp500.py --dry_run --quarterly_retrain

RESUMABILITY
────────────
Progress is written to a JSON state file (--state_file). Re-launching with the
same arguments skips dates already completed, so a crash/restart resumes cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Memory-safety env defaults ───────────────────────────────────────────────
# Child processes (run_sp500_local.py) inherit these. Keeping joblib in-process
# prevents loky from forking one worker (and one panel copy) per CPU core — the
# exact pattern that OOM-killed earlier runs on the 30 GB box.
for _k, _v in {
    "JOBLIB_MULTIPROCESSING": "0",
    "LOKY_MAX_CPU_COUNT": "2",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
}.items():
    os.environ.setdefault(_k, _v)

# ── Project layout ──────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent
DOWNLOADER    = PROJECT_DIR / "scripts" / "data" / "download_us_data.py"
RUNNER        = PROJECT_DIR / "run_sp500_local.py"
MONITORING_DIR = PROJECT_DIR / "monitoring"

# Matches RETRAIN_FEATURE_FRACTION in pipeline/monitoring/drift_monitor.py:
#   a retrain fires when > this fraction of monitored features breach the PSI
#   retrain threshold.
DEFAULT_RETRAIN_FRACTION = 0.20

PY = sys.executable or "python3"

# Filled in from CLI in main(); read by run_cmd() for OOM self-healing.
_OOM_CFG = {"swap_gb": 40, "swapfile": "/swapfile", "max_retries": 1, "ntfy": ""}

# ── Heartbeat (progress ping, interval set via --heartbeat_mins) ─────────────
_STATUS: dict = {
    "market":       "",       # set in main() — "NSE" or "SP500"
    "step":         "",       # current as_of date string e.g. "2024-02-10"
    "action":       "idle",   # "download" | "infer" | "retrain"
    "action_start": None,     # datetime of when current action began
    "step_idx":     0,        # 1-based index into schedule
    "total_steps":  0,        # len(schedule)
}


class HeartbeatThread(threading.Thread):
    """Daemon thread — fires a push notification every `interval` seconds."""

    def __init__(self, interval: int = 900):
        super().__init__(daemon=True, name="heartbeat")
        self._stop = threading.Event()
        self.interval = interval

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            self._ping()

    def _ping(self) -> None:
        topic = _OOM_CFG["ntfy"]
        if not topic:
            return
        s = _STATUS
        elapsed = ""
        if s["action_start"]:
            secs = int((datetime.now() - s["action_start"]).total_seconds())
            elapsed = f" — {secs // 60}m {secs % 60:02d}s elapsed"
        title = f"⏱ {s['market']} wf {s['step_idx']}/{s['total_steps']} — {s['action']}"
        body  = (f"Step {s['step_idx']} of {s['total_steps']}: {s['step']}\n"
                 f"Action: {s['action']}{elapsed}")
        notify(topic, title, body, "hourglass_flowing_sand")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated walk-forward inference + drift-triggered retrain for SP500.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", default="2024-01-13",
                   help="First inference date (YYYY-MM-DD).")
    p.add_argument("--end", default="2024-12-31",
                   help="Last date in the walk-forward window — a FULL RETRAIN always runs here.")
    p.add_argument("--cadence_days", type=int, default=14,
                   help="Days between inference runs (14 = bi-weekly).")
    p.add_argument("--train_start", default="2010-01-01",
                   help="Earliest date in the training panel for (re)training.")
    p.add_argument("--mode", default="all",
                   choices=["all", "momentum", "reversal", "legacy"],
                   help="Ranker mode(s) passed through to run_sp500_local.py.")
    p.add_argument("--retrain_fraction", type=float, default=DEFAULT_RETRAIN_FRACTION,
                   help="Drift retrain trigger: fraction of features breaching PSI retrain "
                        "threshold that forces an early retrain.")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="Passed through to run_sp500_local.py.")
    p.add_argument("--no_drift_retrain", action="store_true",
                   help="Disable mid-quarter drift-triggered retrains "
                        "(only the scheduled quarter-end retrain runs).")
    p.add_argument("--quarterly_retrain", action="store_true",
                   help="Insert Q1/Q2/Q3/Q4 quarter-end dates (Mar-31, Jun-30, Sep-30, Dec-31) "
                        "as full retrain steps in the schedule.")
    p.add_argument("--retrain_dates", nargs="+", default=[],
                   help="Additional dates to force a full retrain (YYYY-MM-DD). "
                        "Combined with --quarterly_retrain if both are given.")
    p.add_argument("--heartbeat_mins", type=int, default=30,
                   help="Minutes between heartbeat push notifications.")
    p.add_argument("--swap_gb", type=int, default=40,
                   help="On an OOM-killed step, ensure at least this much swap (GB), then rerun.")
    p.add_argument("--swapfile", default="/swapfile",
                   help="Swap file path to provision/extend when handling OOM.")
    p.add_argument("--oom_retries", type=int, default=1,
                   help="How many times to rerun a step after provisioning swap (0 disables).")
    p.add_argument("--min_free_gb", type=int, default=15,
                   help="Abort a step before it writes if free disk drops below this (GB). "
                        "A cleanup of regenerable caches is attempted first.")
    p.add_argument("--artefacts_dir", default=None,
                   help="Where regenerable caches live for cleanup (default: --log_dir).")
    p.add_argument("--state_file", default=None,
                   help="Resumability state JSON (default: <log_dir>/walkforward_state.json).")
    p.add_argument("--log_dir", default="/mnt/data/artefacts/us_local",
                   help="Where per-step logs and state are written.")
    p.add_argument("--ntfy_topic", default=os.environ.get("NTFY_TOPIC", "hetzner-victor-ml"),
                   help="ntfy.sh topic for phone alerts (or set $NTFY_TOPIC).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the schedule and the exact commands, then exit.")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


_LAST_NOTIFY_TS: float = 0.0
_NOTIFY_MIN_GAP: float = 3.0   # seconds — prevents burst rate-limiting on ntfy.sh free tier

def notify(topic: str, title: str, body: str, tags: str = "") -> None:
    """Best-effort ntfy.sh push. Never raises. Enforces a minimum gap between sends."""
    global _LAST_NOTIFY_TS
    import time
    if not topic:
        return
    # Respect ntfy.sh free-tier burst limit — wait if last send was too recent
    gap = time.time() - _LAST_NOTIFY_TS
    if gap < _NOTIFY_MIN_GAP:
        time.sleep(_NOTIFY_MIN_GAP - gap)
    try:
        # HTTP headers must be ASCII — replace non-ASCII chars (e.g. em dash) with safe equivalents
        safe_title = title.encode("ascii", errors="replace").decode("ascii")
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": safe_title, "Tags": tags},
        )
        urllib.request.urlopen(req, timeout=10)
        _LAST_NOTIFY_TS = time.time()
    except Exception as e:
        # Log but never raise — notifications must not kill the pipeline
        log(f"  [notify] send failed (topic={topic!r}): {e}")


def _quarter_ends(start: str, end: str) -> List[pd.Timestamp]:
    """Return quarter-end dates (Mar-31, Jun-30, Sep-30, Dec-31) within [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    result: List[pd.Timestamp] = []
    quarter_month_day = [(3, 31), (6, 30), (9, 30), (12, 31)]
    year = start_ts.year
    while True:
        for m, d in quarter_month_day:
            ts = pd.Timestamp(year=year, month=m, day=d)
            if ts < start_ts:
                continue
            if ts > end_ts:
                return result
            result.append(ts)
        year += 1
        if year > end_ts.year + 1:
            break
    return result


def build_schedule(
    start: str,
    end: str,
    cadence_days: int,
    retrain_dates: Optional[List[pd.Timestamp]] = None,
) -> List[Tuple[pd.Timestamp, str]]:
    """
    Build a walk-forward schedule from `start` through `end`.

    Generates bi-weekly inference dates on `cadence_days` intervals.
    Any date in `retrain_dates` (and `end` itself) is marked as a full retrain.
    Retrain dates that don't fall on the cadence are inserted as extra steps.
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError(f"--end ({end}) must be after --start ({start}).")

    retrain_set: set = set(retrain_dates or [])
    retrain_set.add(end_ts)   # end is always a full retrain

    # Generate cadence dates from start up to (and including) end
    all_dates: set = set()
    d = start_ts
    while d <= end_ts:
        all_dates.add(d)
        d += timedelta(days=cadence_days)

    # Insert any retrain dates that sit between cadence steps
    for rt in retrain_set:
        if start_ts <= rt <= end_ts:
            all_dates.add(rt)

    sched: List[Tuple[pd.Timestamp, str]] = [
        (ts, "retrain" if ts in retrain_set else "infer")
        for ts in sorted(all_dates)
    ]
    return sched


def current_swap_gb() -> float:
    """Total system swap in GB (Linux). Returns 0.0 if unknown."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("SwapTotal:"):
                return int(line.split()[1]) / 1024 / 1024   # kB -> GB
    except Exception:
        pass
    return 0.0


def ensure_swap(target_gb: int, swapfile: str) -> None:
    """Provision/extend a swap file to at least target_gb (Linux only, idempotent)."""
    if not platform.system().lower().startswith("linux"):
        log(f"  (ensure_swap skipped — not Linux: {platform.system()})")
        return
    cur = current_swap_gb()
    if cur >= target_gb * 0.95:
        log(f"  Swap already {cur:.1f} GB (>= {target_gb} GB) — no change.")
        return

    sudo = [] if (hasattr(os, "geteuid") and os.geteuid() == 0) else ["sudo"]
    log(f"  Swap is {cur:.1f} GB; provisioning {target_gb} GB at {swapfile} ...")

    def _run(parts: List[str]) -> int:
        return subprocess.run(sudo + parts, check=False,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode

    _run(["swapoff", swapfile])                              # ignore if not on
    if _run(["fallocate", "-l", f"{target_gb}G", swapfile]) != 0:
        # fallocate can fail on some filesystems — fall back to dd
        _run(["dd", "if=/dev/zero", f"of={swapfile}", "bs=1M", f"count={target_gb * 1024}"])
    _run(["chmod", "600", swapfile])
    _run(["mkswap", swapfile])
    if _run(["swapon", swapfile]) != 0:
        log("  WARNING: swapon failed — check permissions / disk space.")
    log(f"  Swap now: {current_swap_gb():.1f} GB")


def free_gb(path: Path) -> float:
    """Free space (GB) on the filesystem holding `path`. Returns large value if unknown."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:
        return 1e9   # unknown — don't block


def clean_regenerable(artefacts_dir: Path) -> float:
    """Delete regenerable caches under artefacts_dir. Returns GB freed.

    Safe to call between steps: panel checkpoints and fold caches are recomputed
    on the next run; only watchlist outputs, drift records, and trained model
    artefacts are kept.
    """
    freed = 0
    targets: List[Path] = []
    ck = artefacts_dir / "checkpoints"
    targets += [ck / "panel_features.pkl", ck / "panel_targets.pkl", ck / "feat_cols.txt"]
    targets += list(artefacts_dir.glob("*/fold_cache/*.pkl"))   # momentum/ + reversal/ fold caches
    for p in targets:
        try:
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
        except Exception:
            pass
    return freed / (1024 ** 3)


def ensure_disk(check_path: Path, min_gb: int, artefacts_dir: Path, ntfy: str) -> bool:
    """Guarantee at least min_gb free before a step writes. Cleans caches if low.

    Returns True if free space is OK (possibly after cleanup), False if it
    remains below min_gb (caller should abort rather than crash mid-write).
    """
    fg = free_gb(check_path)
    log(f"  Disk free on {check_path}: {fg:.1f} GB (min {min_gb} GB)")
    if fg >= min_gb:
        return True
    log(f"  LOW DISK ({fg:.1f} GB < {min_gb} GB) — cleaning regenerable caches ...")
    notify(ntfy, "Low disk -> cleanup",
           f"Only {fg:.1f} GB free on {check_path}. Cleaning regenerable caches.", "warning")
    freed = clean_regenerable(artefacts_dir)
    fg = free_gb(check_path)
    log(f"  Freed ~{freed:.1f} GB; now {fg:.1f} GB free.")
    return fg >= min_gb


def _looks_like_oom(rc: int, step_log: Path) -> bool:
    """Heuristic: was this step killed by the OOM killer?"""
    if rc in (-9, 137):          # SIGKILL (137 = 128 + 9) — classic OOM kill
        return True
    try:
        txt = step_log.read_text(errors="ignore")[-30000:]
        markers = ("MemoryError", "Out of memory", "Killed process",
                   "Cannot allocate memory", "leaked semaphore", "resource_tracker")
        return any(m in txt for m in markers)
    except Exception:
        return False


def _run_once(cmd: List[str], step_log: Path) -> int:
    """Run a subprocess once, tee-ing combined output to step_log. Returns exit code."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    step_log.parent.mkdir(parents=True, exist_ok=True)
    with open(step_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
        proc.wait()
    return proc.returncode


def run_cmd(cmd: List[str], step_log: Path) -> int:
    """Run a step; on OOM kill, provision swap (_OOM_CFG) and rerun up to max_retries."""
    attempt = 0
    while True:
        rc = _run_once(cmd, step_log)
        if rc == 0 or not _looks_like_oom(rc, step_log):
            return rc
        if attempt >= _OOM_CFG["max_retries"]:
            log(f"  OOM persists after {attempt} retr{'y' if attempt == 1 else 'ies'} "
                f"(rc={rc}) — giving up on this step.")
            return rc
        attempt += 1
        log(f"  OOM DETECTED (rc={rc}). Provisioning {_OOM_CFG['swap_gb']} GB swap "
            f"and rerunning (attempt {attempt}/{_OOM_CFG['max_retries']}) ...")
        notify(_OOM_CFG["ntfy"], "OOM -> adding swap + retry",
               f"Step OOM-killed (rc={rc}). Ensuring {_OOM_CFG['swap_gb']} GB swap and rerunning.",
               "warning")
        ensure_swap(_OOM_CFG["swap_gb"], _OOM_CFG["swapfile"])


def download_until(as_of: pd.Timestamp, step_log: Path) -> int:
    """Incrementally extend local CSVs up to and including `as_of`.

    yfinance treats --end as EXCLUSIVE, so pass as_of + 1 day to include the
    as_of bar itself (or the latest trading day on/just before it).
    """
    end_excl = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")
    cmd = [PY, str(DOWNLOADER), "--end", end_excl]
    return run_cmd(cmd, step_log)


def run_inference(as_of: pd.Timestamp, mode: str, n_jobs: int, step_log: Path) -> int:
    cmd = [PY, str(RUNNER),
           "--skip_train",
           "--as_of", as_of.strftime("%Y-%m-%d"),
           "--mode", mode,
           "--n_jobs", str(n_jobs)]
    return run_cmd(cmd, step_log)


def run_retrain(as_of: pd.Timestamp, train_start: str, mode: str,
                n_jobs: int, step_log: Path) -> int:
    cmd = [PY, str(RUNNER),
           "--train_start", train_start,
           "--as_of", as_of.strftime("%Y-%m-%d"),
           "--mode", mode,
           "--n_jobs", str(n_jobs)]
    return run_cmd(cmd, step_log)


def check_drift(mode: str) -> Dict[str, float]:
    """Read the latest drift snapshot per sub-mode and return {submode: breach_fraction}.

    The drift monitor writes monitoring/<submode>/feature_drift.parquet with one
    row per (date, feature) carrying a boolean `retrain_flag`. We read the most
    recent date's rows and compute the fraction of features flagged for retrain.
    """
    submodes = {"all": ["momentum", "reversal"]}.get(mode, [mode])
    fractions: Dict[str, float] = {}
    for sm in submodes:
        path = MONITORING_DIR / sm / "feature_drift.parquet"
        if not path.exists():
            fractions[sm] = 0.0
            continue
        try:
            df = pd.read_parquet(path)
            if df.empty or "retrain_flag" not in df.columns:
                fractions[sm] = 0.0
                continue
            latest = df["date"].max()
            snap = df[df["date"] == latest]
            fractions[sm] = float(snap["retrain_flag"].mean()) if len(snap) else 0.0
        except Exception as e:
            log(f"  drift read failed for '{sm}': {e}")
            fractions[sm] = 0.0
    return fractions


# ── State (resumability) ──────────────────────────────────────────────────────
def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"completed": [], "model_trained_through": None, "events": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    state_file = Path(args.state_file) if args.state_file else (log_dir / "walkforward_state.json")

    # OOM self-healing config (read by run_cmd)
    _OOM_CFG["swap_gb"]     = args.swap_gb
    _OOM_CFG["swapfile"]    = args.swapfile
    _OOM_CFG["max_retries"] = args.oom_retries
    _OOM_CFG["ntfy"]        = args.ntfy_topic

    artefacts_dir = Path(args.artefacts_dir) if args.artefacts_dir else log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build retrain dates from CLI flags
    retrain_dates: List[pd.Timestamp] = [pd.Timestamp(dt) for dt in (args.retrain_dates or [])]
    if args.quarterly_retrain:
        retrain_dates += _quarter_ends(args.start, args.end)

    schedule = build_schedule(
        args.start, args.end, args.cadence_days,
        retrain_dates=retrain_dates if retrain_dates else None,
    )

    # ── Dry run: show the plan and exit ────────────────────────────────────
    end_ts = pd.Timestamp(args.end)
    print("=" * 64)
    print("  SP500 WALK-FORWARD SCHEDULE")
    print("=" * 64)
    for d, action in schedule:
        if action == "retrain":
            label = "FULL RETRAIN (end)" if d == end_ts else "FULL RETRAIN (scheduled)"
        else:
            label = "inference (--skip_train)"
        print(f"  {d.date()}  {'[' + d.strftime('%a') + ']':>5}   {label}")
    n_retrains = sum(1 for _, a in schedule if a == "retrain")
    print(f"\n  cadence       : every {args.cadence_days} days")
    print(f"  total steps   : {len(schedule)} ({n_retrains} retrain{'s' if n_retrains != 1 else ''}, "
          f"{len(schedule) - n_retrains} inference)")
    print(f"  mode          : {args.mode}")
    print(f"  drift retrain : {'DISABLED' if args.no_drift_retrain else f'frac > {args.retrain_fraction:.0%}'}")
    print(f"  heartbeat     : every {args.heartbeat_mins} min")
    print(f"  OOM handling  : {'DISABLED' if args.oom_retries == 0 else f'ensure {args.swap_gb} GB swap @ {args.swapfile}, rerun x{args.oom_retries}'}")
    print(f"  disk guard    : abort if < {args.min_free_gb} GB free on {log_dir} (auto-clean caches first); now {free_gb(log_dir):.1f} GB free")
    print(f"  state file    : {state_file}")
    print(f"  per-step logs : {log_dir}")
    print("=" * 64)
    if args.dry_run:
        print("\n[--dry_run] No commands executed.")
        return

    if not DOWNLOADER.exists():
        sys.exit(f"Downloader not found: {DOWNLOADER}")
    if not RUNNER.exists():
        sys.exit(f"Runner not found: {RUNNER}")

    state = load_state(state_file)
    completed = set(state.get("completed", []))

    notify(args.ntfy_topic, "Walk-forward started",
           f"SP500 walk-forward {args.start} -> {args.end}, {len(schedule)} steps.", "rocket")

    _STATUS["market"]      = "SP500"
    _STATUS["total_steps"] = len(schedule)
    heartbeat = HeartbeatThread(interval=args.heartbeat_mins * 60)
    heartbeat.start()

    for idx, (d, action) in enumerate(schedule):
        dkey = d.strftime("%Y-%m-%d")
        _STATUS["step"]     = dkey
        _STATUS["step_idx"] = idx + 1
        if dkey in completed:
            log(f"SKIP {dkey} — already completed (resumed from state).")
            continue

        log("=" * 60)
        log(f"STEP {dkey} [{d.strftime('%a')}] — planned action: {action}")
        notify(args.ntfy_topic,
               f"[{idx+1}/{len(schedule)}] {dkey} — {action}",
               f"Starting {'full retrain' if action == 'retrain' else 'inference'} as of {dkey}.",
               "arrow_forward")

        # 0. Disk guard — never start a step that could crash mid-write on a full disk
        if not ensure_disk(log_dir, args.min_free_gb, artefacts_dir, args.ntfy_topic):
            msg = (f"Insufficient disk at {dkey}: still < {args.min_free_gb} GB free "
                   f"after cleanup. Aborting before write to avoid corruption.")
            log("X  " + msg)
            notify(args.ntfy_topic, "Walk-forward HALTED (disk)", msg, "x,rotating_light")
            state.setdefault("events", []).append({"date": dkey, "event": "disk_abort"})
            save_state(state_file, state)
            sys.exit(1)

        # 1. Incremental data download up to this date
        _STATUS["action"] = "download"; _STATUS["action_start"] = datetime.now()
        rc = download_until(d, log_dir / f"wf_{dkey}_download.log")
        if rc != 0:
            msg = f"Data download FAILED at {dkey} (exit {rc})."
            log("X  " + msg)
            notify(args.ntfy_topic, "Walk-forward FAILED", msg, "x,rotating_light")
            state.setdefault("events", []).append({"date": dkey, "event": "download_failed", "rc": rc})
            save_state(state_file, state)
            sys.exit(rc)

        did_retrain = False

        if action == "retrain":
            # 2a. Scheduled quarter-end full retrain (also scores & picks stocks)
            log(f"FULL RETRAIN (scheduled quarter end) at {dkey} ...")
            _STATUS["action"] = "retrain"; _STATUS["action_start"] = datetime.now()
            rc = run_retrain(d, args.train_start, args.mode, args.n_jobs,
                             log_dir / f"wf_{dkey}_retrain.log")
            did_retrain = True
        else:
            # 2b. Inference-only — model frozen
            log(f"INFERENCE (model frozen) at {dkey} ...")
            _STATUS["action"] = "infer"; _STATUS["action_start"] = datetime.now()
            rc = run_inference(d, args.mode, args.n_jobs,
                               log_dir / f"wf_{dkey}_infer.log")
            if rc != 0:
                msg = f"Inference FAILED at {dkey} (exit {rc})."
                log("X  " + msg)
                notify(args.ntfy_topic, "Walk-forward FAILED", msg, "x,rotating_light")
                state.setdefault("events", []).append({"date": dkey, "event": "infer_failed", "rc": rc})
                save_state(state_file, state)
                sys.exit(rc)

            # 3. Drift gate
            fractions = check_drift(args.mode)
            frac_str = ", ".join(f"{k}={v:.0%}" for k, v in fractions.items())
            log(f"  Drift breach fractions: {frac_str}")
            breached = (not args.no_drift_retrain) and any(
                v > args.retrain_fraction for v in fractions.values()
            )
            state.setdefault("events", []).append(
                {"date": dkey, "event": "drift_check", "fractions": fractions, "breached": breached}
            )

            if breached:
                # 4. Drift-triggered early retrain, then continue serving from new model
                log(f"  DRIFT TRIGGER at {dkey} ({frac_str}) — running EARLY full retrain ...")
                notify(args.ntfy_topic, "Drift -> early retrain",
                       f"{dkey}: drift breached ({frac_str}). Retraining now.", "warning")
                _STATUS["action"] = "retrain"; _STATUS["action_start"] = datetime.now()
                rc = run_retrain(d, args.train_start, args.mode, args.n_jobs,
                                 log_dir / f"wf_{dkey}_drift_retrain.log")
                did_retrain = True

        if rc != 0:
            msg = f"Retrain FAILED at {dkey} (exit {rc})."
            log("X  " + msg)
            notify(args.ntfy_topic, "Walk-forward FAILED", msg, "x,rotating_light")
            state.setdefault("events", []).append({"date": dkey, "event": "retrain_failed", "rc": rc})
            save_state(state_file, state)
            sys.exit(rc)

        if did_retrain:
            state["model_trained_through"] = dkey
        completed.add(dkey)
        state["completed"] = sorted(completed)
        save_state(state_file, state)
        done_label = "model retrained" if did_retrain else "inference"
        steps_left = len(schedule) - (idx + 1)
        log(f"OK  {dkey} done ({done_label}). {steps_left} steps remaining.")
        notify(args.ntfy_topic,
               f"[{idx+1}/{len(schedule)}] {dkey} done",
               f"{done_label.capitalize()} complete.\n"
               f"Steps remaining: {steps_left}."
               + (f"\nModel trained through: {dkey}" if did_retrain else ""),
               "white_check_mark" if not did_retrain else "brain")

    heartbeat.stop()
    log("=" * 60)
    log("ALL WALK-FORWARD STEPS COMPLETE.")
    log(f"  Model now trained through: {state.get('model_trained_through')}")
    log(f"  Watchlists + drift records under: {PROJECT_DIR} (output/ and monitoring/)")
    notify(args.ntfy_topic, "Walk-forward DONE",
           f"SP500 walk-forward complete through {args.end}.", "tada,white_check_mark")


if __name__ == "__main__":
    main()
