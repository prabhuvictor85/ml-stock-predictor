#!/usr/bin/env bash
#
# Sequential pipeline chain:  NSE  →  clean NSE checkpoints  →  SP500
#
# Launch (fully detached, returns the orchestrator PID):
#     nohup ./run_pipeline_chain.sh > /mnt/data/artefacts/run_chain.log 2>&1 &
#     echo $!
#
# Watch progress:
#     tail -f /mnt/data/artefacts/run_chain.log          # orchestrator
#     tail -f /mnt/data/artefacts/nse_local/run_nse_full.log
#     tail -f /mnt/data/artefacts/us_local/run_sp500_full.log
#
set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────
PROJ=/root/ml-stock-predictor
ART=/mnt/data/artefacts
TRAIN_START=2010-01-01
AS_OF=2023-12-08
N_JOBS=1

NSE_LOG="$ART/nse_local/run_nse_full.log"
SP_LOG="$ART/us_local/run_sp500_full.log"
NSE_CKPT="$ART/nse_local/checkpoints"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

cd "$PROJ" || { echo "[$(ts)] FATAL: cannot cd to $PROJ"; exit 1; }
mkdir -p "$ART/nse_local" "$ART/us_local"

# ── 1. NSE (blocks until complete) ──────────────────────────────────────────
echo "[$(ts)] ▶ NSE starting  → $NSE_LOG"
python3 run_nse_local.py --train_start "$TRAIN_START" --as_of "$AS_OF" --n_jobs "$N_JOBS" > "$NSE_LOG" 2>&1
NSE_RC=$?
echo "[$(ts)] ◀ NSE finished  rc=$NSE_RC"

if [ "$NSE_RC" -ne 0 ]; then
    echo "[$(ts)] ✖ NSE FAILED (rc=$NSE_RC) — aborting before SP500. See $NSE_LOG"
    exit "$NSE_RC"
fi

# ── 2. Clean NSE checkpoints (disk hygiene; SP500 uses its own us_local dir) ─
if [ -d "$NSE_CKPT" ]; then
    echo "[$(ts)] ✓ Cleaning NSE checkpoints: $NSE_CKPT"
    rm -rf "$NSE_CKPT"
else
    echo "[$(ts)] (no NSE checkpoint dir at $NSE_CKPT — skipping clean)"
fi

# ── 3. SP500 (blocks until complete) ────────────────────────────────────────
echo "[$(ts)] ▶ SP500 starting → $SP_LOG"
python3 run_sp500_local.py --train_start "$TRAIN_START" --as_of "$AS_OF" --n_jobs "$N_JOBS" > "$SP_LOG" 2>&1
SP_RC=$?
echo "[$(ts)] ◀ SP500 finished rc=$SP_RC"

echo "[$(ts)] ■ Chain complete  (NSE rc=$NSE_RC, SP500 rc=$SP_RC)"
exit "$SP_RC"
