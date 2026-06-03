"""
forward_eval_server.py
─────────────────────
FastAPI server that exposes the forward-performance evaluation script as an
HTTP API. The React dashboard calls this to run 6-month forward evals and
download the resulting Excel file.

Usage (from project root):
    pip install fastapi uvicorn
    python api/forward_eval_server.py

Endpoints
─────────
  GET  /api/forward-eval/stream?market=us_local&date=2024-01-12&months=6[&universe=true]
       universe=false (default) → watchlist tickers only (fast, ~30–100 tickers)
       universe=true            → full stock universe (slow, 1 000–1 600 tickers)
       Server-Sent Events stream — one line per ticker evaluated.
       Final line: __DONE__:<job_id>
       Error line: __ERROR__:<message>

  GET  /api/forward-eval/download/<job_id>
       Download the Excel file produced by the eval run.
"""
from __future__ import annotations

import asyncio
import datetime
import io
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

# ── project path ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# ── import eval logic ──────────────────────────────────────────────────────────
from scripts.tools.evaluate_forward_performance import (
    EVAL_DIR, MARKET_CONFIG,
    fetch_forward_ohlc, load_watchlist_scores,
)
import pandas as pd

# ── market key mapping  (React key → eval script key) ─────────────────────────
MARKET_MAP = {
    "us_local":    "sp500",
    "nse_local":   "nse",
    "nse_tradingv":"nse",
}

# ── full-universe stock list files ─────────────────────────────────────────────
STOCK_LISTS_DIR = Path(r"C:\Victor\Learning_charts\stock_lists")

UNIVERSE_FILES = {
    "us_local":    (STOCK_LISTS_DIR / "constituents_us_combined.csv", "Symbol"),
    "nse_local":   (STOCK_LISTS_DIR / "constituentsi.csv",            "Symbol"),
    "nse_tradingv":(STOCK_LISTS_DIR / "constituentsi.csv",            "Symbol"),
}


def _load_universe_tickers(market_key: str) -> list[str]:
    """Load the full ticker universe from the static stock-list CSV."""
    entry = UNIVERSE_FILES.get(market_key)
    if not entry:
        raise ValueError(f"No universe file configured for market '{market_key}'")
    csv_path, sym_col = entry
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    col = next((c for c in df.columns if c.lower() == sym_col.lower()), None)
    if not col:
        raise ValueError(f"Column '{sym_col}' not found in {csv_path.name}")
    tickers = df[col].dropna().str.strip().tolist()
    # Remove index symbols (e.g. ^NSEBANK)
    tickers = [t for t in tickers if t and not t.startswith("^")]
    return tickers

# ── in-memory job store ────────────────────────────────────────────────────────
# { job_id: {"status": "running"|"done"|"error", "file": Path|None, "lines": [str]} }
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="ML Stock — Forward Eval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_tickers_from_date(market_key: str, date: str) -> list[str]:
    """Load unique tickers from all watchlist CSVs for a given date."""
    output_dir = PROJECT_ROOT / "output" / market_key
    date_dir   = output_dir / date          # nested layout
    search_dir = date_dir if date_dir.is_dir() else output_dir

    # Regex covers: watchlist_<mode>[_<type>]_<side>[_<tier>]_<date>.csv
    pattern = re.compile(
        r"^watchlist_(?:momentum|reversal)(?:_(?:composite|pureml|combined))?"
        r"_(?:bull|bear)(?:_(?:large|mid|small|micro))?_"
        + re.escape(date) + r"\.csv$"
    )

    tickers = set()
    for f in search_dir.glob(f"watchlist_*_{date}.csv"):
        if not pattern.match(f.name):
            continue
        try:
            df = pd.read_csv(f)
            col = next((c for c in df.columns if c.lower() in ("ticker", "symbol")), None)
            if col:
                tickers.update(df[col].dropna().str.strip().tolist())
        except Exception:
            pass
    return sorted(tickers)


def _run_eval_job(job_id: str, market_key: str, date_str: str, months: int,
                  universe: bool = False):
    """Background thread: run eval, emit lines into job store."""

    def emit(line: str):
        with _jobs_lock:
            _jobs[job_id]["lines"].append(line)

    try:
        eval_market = MARKET_MAP.get(market_key)
        if not eval_market:
            raise ValueError(f"Unknown market key: {market_key}")

        base_date = datetime.date.fromisoformat(date_str)

        # Forward date = base + N months
        m = base_date.month + months
        y = base_date.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        forward_date = base_date.replace(year=y, month=m)

        emit(f"Base date    : {base_date}")
        emit(f"Forward date : {forward_date}  (+{months} months)")
        emit(f"Market       : {market_key}")

        if universe:
            tickers = _load_universe_tickers(market_key)
            emit(f"Tickers      : {len(tickers)} (full universe)")
            emit(f"NOTE: Full universe run takes 20-40 min for {len(tickers)} tickers")
        else:
            tickers = _load_tickers_from_date(market_key, date_str)
            if not tickers:
                raise ValueError(
                    f"No watchlist tickers found for {market_key} / {date_str}. "
                    "Make sure the output folder contains watchlist CSV files for this date."
                )
            emit(f"Tickers      : {len(tickers)} from watchlist")
        emit("─" * 48)

        results = []
        total   = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            base_ohlc = fetch_forward_ohlc(ticker, base_date)
            fwd_ohlc  = fetch_forward_ohlc(ticker, forward_date)

            pct = None
            if (base_ohlc and fwd_ohlc
                    and base_ohlc["close"] and base_ohlc["close"] != 0):
                pct = round(
                    (fwd_ohlc["close"] - base_ohlc["close"])
                    / base_ohlc["close"] * 100, 2
                )

            status = f"{pct:+.1f}%" if pct is not None else "no data"
            emit(f"[{i:>3}/{total}] {ticker:<12} {status}")

            row = {
                "ticker":           ticker,
                "base_date":        base_ohlc["date"]   if base_ohlc else None,
                "base_close":       base_ohlc["close"]  if base_ohlc else None,
                "fwd_date":         fwd_ohlc["date"]    if fwd_ohlc  else None,
                "fwd_close":        fwd_ohlc["close"]   if fwd_ohlc  else None,
                "close_pct_change": pct,
            }
            results.append(row)

        if not results:
            raise ValueError("No results generated — check tickers and date.")

        df = pd.DataFrame(results)
        if "close_pct_change" in df.columns:
            df = df.sort_values("close_pct_change", ascending=False)

        # ── Summary stats ──────────────────────────────────────────────────
        valid = df["close_pct_change"].dropna()
        emit("─" * 48)
        emit(f"Gainers : {(valid > 0).sum()}   Losers : {(valid < 0).sum()}")
        emit(f"Avg     : {valid.mean():.2f}%   Median : {valid.median():.2f}%")

        # ── Save Excel ─────────────────────────────────────────────────────
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        scope    = "universe" if universe else "watchlist"
        out_file = EVAL_DIR / (
            f"forward_eval_{market_key}_{scope}_{date_str}_{str(forward_date)}_{int(time.time())}.xlsx"
        )

        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="All Tickers", index=False)

            stats_rows = {
                "Metric": [
                    "Tickers evaluated", "Forward price fetched",
                    "Avg % change", "Median % change",
                    "Gainers", "Losers",
                    "Best performer", "Best %",
                    "Worst performer", "Worst %",
                ],
                "Value": [
                    len(df),
                    df["fwd_close"].notna().sum(),
                    round(valid.mean(), 2),
                    round(valid.median(), 2),
                    int((valid > 0).sum()),
                    int((valid < 0).sum()),
                    df.loc[df["close_pct_change"].idxmax(), "ticker"] if not valid.empty else "N/A",
                    round(valid.max(), 2) if not valid.empty else None,
                    df.loc[df["close_pct_change"].idxmin(), "ticker"] if not valid.empty else "N/A",
                    round(valid.min(), 2) if not valid.empty else None,
                ],
            }
            pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Summary", index=False)
            df.head(50).to_excel(writer, sheet_name="Top 50 Gainers", index=False)
            df.tail(50).to_excel(writer, sheet_name="Top 50 Losers", index=False)

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["file"]   = out_file

        emit(f"__DONE__:{job_id}")

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
        emit(f"__ERROR__:{exc}")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/forward-eval/stream")
async def stream_eval(market: str, date: str, months: int = 6, universe: bool = False):
    """
    SSE stream for a forward-eval run.
    Events are plain text lines; special lines:
      __DONE__:<job_id>   — eval completed, use /download/<job_id>
      __ERROR__:<message> — eval failed
    """
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "file": None, "lines": []}

    # Launch eval in background thread so we don't block the event loop
    t = threading.Thread(
        target=_run_eval_job,
        args=(job_id, market, date, months, universe),
        daemon=True,
    )
    t.start()

    async def event_generator():
        sent = 0
        while True:
            with _jobs_lock:
                lines  = _jobs[job_id]["lines"]
                status = _jobs[job_id]["status"]

            while sent < len(lines):
                line = lines[sent]
                yield f"data: {json.dumps({'line': line})}\n\n"
                sent += 1

            if status in ("done", "error"):
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/forward-eval/download/{job_id}")
def download_eval(job_id: str):
    """Download the Excel file produced by a completed eval job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done":
        raise HTTPException(status_code=202, detail="Job not yet complete")
    if not job["file"] or not Path(job["file"]).exists():
        raise HTTPException(status_code=500, detail="Output file missing")

    return FileResponse(
        path=str(job["file"]),
        filename=Path(job["file"]).name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  ML Stock — Forward Eval API Server")
    print("  http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
