#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline_hetzner.sh
#
# Sequential pipeline runner for Hetzner — fully unattended.
# Runs NSE, cleans panel checkpoints to free disk, then runs SP500.
# Sends a push notification to your phone on completion or failure.
#
# ── One-time phone setup ──────────────────────────────────────────────────────
#   1. Install "ntfy" app on Android/iOS  (https://ntfy.sh)
#   2. Subscribe to a topic name of your choice, e.g.  hetzner-victor-ml
#   3. Export the topic before running this script:
#        export NTFY_TOPIC="hetzner-victor-ml"
#   (If NTFY_TOPIC is not set, notifications are silently skipped.)
#
# ── Launch ────────────────────────────────────────────────────────────────────
#   chmod +x run_pipeline_hetzner.sh
#   export NTFY_TOPIC="hetzner-victor-ml"        # optional but recommended
#   nohup bash run_pipeline_hetzner.sh \
#       > /mnt/data/artefacts/pipeline_master.log 2>&1 &
#   echo "Pipeline PID: $!"
#
# ── Monitor from your laptop ──────────────────────────────────────────────────
#   ssh root@<server> "tail -f /mnt/data/artefacts/pipeline_master.log"
#   ssh root@<server> "tail -f /mnt/data/artefacts/nse_local/run_nse_full.log"
#   ssh root@<server> "tail -f /mnt/data/artefacts/us_local/run_sp500_full.log"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Memory safety: prevent joblib/loky from forking one worker per CPU core ────
# Each loky worker gets its OWN copy of the feature panel, which multiplies RAM
# by the core count and triggers the OOM killer on this 30 GB box. Keeping
# joblib in-process (no forks) and capping native thread pools keeps the peak
# footprint to a single copy of the panel.
export JOBLIB_MULTIPROCESSING=0     # joblib stays in-process — no loky, no semaphore leak
export LOKY_MAX_CPU_COUNT=2
export OMP_NUM_THREADS=4            # LightGBM / XGBoost OpenMP threads
export OPENBLAS_NUM_THREADS=4

PROJECT_DIR="/root/ml-stock-predictor"
ARTEFACTS_ROOT="/mnt/data/artefacts"

NSE_LOG="${ARTEFACTS_ROOT}/nse_local/run_nse_full.log"
SP500_LOG="${ARTEFACTS_ROOT}/us_local/run_sp500_full.log"
STATUS_FILE="${ARTEFACTS_ROOT}/pipeline_status.txt"

NSE_CKPT="${ARTEFACTS_ROOT}/nse_local/checkpoints"

# ── OOM self-healing ──────────────────────────────────────────────────────────
# If a pipeline step is killed by the kernel OOM killer, provision swap up to
# ${SWAP_GB} GB and rerun the step ${OOM_RETRIES} time(s) before giving up.
SWAP_GB=40
SWAPFILE="/swapfile"
OOM_RETRIES=1

# ── Disk safety ───────────────────────────────────────────────────────────────
# Never start a heavy step that could fill /mnt and crash mid-write. Below this
# free-space floor we first delete regenerable caches, then abort if still low.
MIN_FREE_GB=15

PIPELINE_START=$(date +%s)

# ── helpers ───────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

elapsed_hm() {
    local secs=$(( $(date +%s) - PIPELINE_START ))
    printf "%dh %02dm" $(( secs / 3600 )) $(( (secs % 3600) / 60 ))
}

# Send push notification via ntfy.sh (silently skipped if NTFY_TOPIC unset)
notify() {
    local title="$1"
    local body="$2"
    local tags="${3:-}"            # e.g. "white_check_mark" or "x"

    if [ -z "${NTFY_TOPIC:-}" ]; then
        return 0
    fi

    curl -s \
        -H "Title: ${title}" \
        -H "Tags: ${tags}" \
        -d "${body}" \
        "https://ntfy.sh/${NTFY_TOPIC}" \
        > /dev/null 2>&1 || true   # never let a notification kill the pipeline
}

# ── OOM self-healing helpers ───────────────────────────────────────────────────

current_swap_gb() {
    awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0
}

ensure_swap() {
    local target="$1"
    local cur
    cur=$(current_swap_gb)
    if [ "${cur:-0}" -ge "${target}" ]; then
        log "  Swap already ${cur}GB (>= ${target}GB) — no change."
        return 0
    fi
    log "  Swap is ${cur}GB; provisioning ${target}GB at ${SWAPFILE} ..."
    swapoff "${SWAPFILE}" 2>/dev/null || true
    fallocate -l "${target}G" "${SWAPFILE}" 2>/dev/null \
        || dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$(( target * 1024 )) status=none
    chmod 600 "${SWAPFILE}"
    mkswap "${SWAPFILE}" > /dev/null 2>&1 || true
    swapon "${SWAPFILE}" 2>/dev/null || log "  WARNING: swapon failed (check permissions/disk)."
    log "  Swap now: $(current_swap_gb)GB"
}

# looks_like_oom <exit_code> <logfile>  → returns 0 (true) if step looks OOM-killed
looks_like_oom() {
    local rc="$1" logf="$2"
    [ "${rc}" -eq 137 ] && return 0        # 128 + 9 (SIGKILL) — classic OOM kill
    if [ -f "${logf}" ] && tail -c 30000 "${logf}" 2>/dev/null \
        | grep -qE "MemoryError|Out of memory|Killed|Cannot allocate memory|leaked semaphore|resource_tracker"; then
        return 0
    fi
    return 1
}

# run_with_oom_retry <logfile> <command...>
# Runs the command (output → logfile). On an OOM kill, adds swap and reruns.
# Returns the command's final exit code.
run_with_oom_retry() {
    local logf="$1"; shift
    local attempt=0 rc
    while true; do
        set +e
        "$@" > "${logf}" 2>&1
        rc=$?
        set -e
        if [ "${rc}" -eq 0 ] || ! looks_like_oom "${rc}" "${logf}"; then
            return "${rc}"
        fi
        if [ "${attempt}" -ge "${OOM_RETRIES}" ]; then
            log "  OOM persists after ${attempt} retry(s) (rc=${rc}) — giving up on this step."
            return "${rc}"
        fi
        attempt=$(( attempt + 1 ))
        log "  OOM DETECTED (rc=${rc}). Provisioning ${SWAP_GB}GB swap and rerunning (attempt ${attempt}/${OOM_RETRIES}) ..."
        notify "⚠️ OOM → swap + retry" \
            "Pipeline step OOM-killed (rc=${rc}). Ensuring ${SWAP_GB}GB swap and rerunning." \
            "warning"
        ensure_swap "${SWAP_GB}"
    done
}

# ── Disk safety helpers ─────────────────────────────────────────────────────────

free_gb() {
    # Free GB on the filesystem containing $1
    df -BG "$1" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4+0}'
}

clean_regenerable() {
    # Delete large regenerable caches under the artefacts root. Safe between
    # steps — panel checkpoints + fold caches are recomputed on the next run.
    log "  Cleaning regenerable caches to free disk ..."
    rm -f  "${ARTEFACTS_ROOT}"/nse_local/checkpoints/panel_features.pkl \
           "${ARTEFACTS_ROOT}"/nse_local/checkpoints/panel_targets.pkl  \
           "${ARTEFACTS_ROOT}"/nse_local/checkpoints/feat_cols.txt      \
           "${ARTEFACTS_ROOT}"/us_local/checkpoints/panel_features.pkl  \
           "${ARTEFACTS_ROOT}"/us_local/checkpoints/panel_targets.pkl   \
           "${ARTEFACTS_ROOT}"/us_local/checkpoints/feat_cols.txt 2>/dev/null || true
    rm -rf "${ARTEFACTS_ROOT}"/nse_local/*/fold_cache \
           "${ARTEFACTS_ROOT}"/us_local/*/fold_cache 2>/dev/null || true
}

# ensure_disk <path> <label> → 0 if OK (possibly after cleanup), 1 if still low
ensure_disk() {
    local path="$1" label="$2" free
    free=$(free_gb "${path}")
    log "  Disk free on ${path}: ${free}GB (min ${MIN_FREE_GB}GB)  [before ${label}]"
    if [ "${free:-0}" -ge "${MIN_FREE_GB}" ]; then
        return 0
    fi
    log "  ⚠ LOW DISK (${free}GB < ${MIN_FREE_GB}GB) — attempting cleanup before ${label} ..."
    notify "⚠️ Low disk" "Only ${free}GB free on ${path}. Cleaning regenerable caches." "warning"
    clean_regenerable
    free=$(free_gb "${path}")
    log "  Disk after cleanup: ${free}GB"
    [ "${free:-0}" -ge "${MIN_FREE_GB}" ]
}

# ── sanity checks ─────────────────────────────────────────────────────────────

cd "${PROJECT_DIR}"

mkdir -p "${ARTEFACTS_ROOT}/nse_local"
mkdir -p "${ARTEFACTS_ROOT}/us_local"

echo "RUNNING" > "${STATUS_FILE}"

log "════════════════════════════════════════════════════"
log "Pipeline starting — $(date)"
log "Python: $(python3 --version 2>&1)"
log "Disk:   $(df -h /mnt/data | tail -1)"
log "════════════════════════════════════════════════════"

notify "🚀 Pipeline started" \
    "NSE + SP500 pipeline started on Hetzner.  Will notify on done/fail." \
    "rocket"

# ── STEP 1: Run NSE ───────────────────────────────────────────────────────────

NSE_START=$(date +%s)

log ""
log "STEP 1 — Starting NSE pipeline"
log "Log: ${NSE_LOG}"

ensure_disk "${ARTEFACTS_ROOT}" "NSE" || {
    log "❌  Aborting before NSE — insufficient disk after cleanup."
    echo "DISK_FULL" > "${STATUS_FILE}"
    notify "❌ Disk full" "Not enough free space on ${ARTEFACTS_ROOT} to start NSE safely." "x,rotating_light"
    exit 1
}

run_with_oom_retry "${NSE_LOG}" \
    python3 run_nse_local.py \
        --train_start 2010-01-01 \
        --as_of       2023-12-08 \
        --n_jobs      1 \
    || {
        NSE_ELAPSED=$(( $(date +%s) - NSE_START ))
        NSE_HM=$(printf "%dh %02dm" $(( NSE_ELAPSED / 3600 )) $(( (NSE_ELAPSED % 3600) / 60 )))
        log "❌  NSE FAILED after ${NSE_HM}.  SP500 will NOT start."
        echo "NSE_FAILED" > "${STATUS_FILE}"
        notify "❌ NSE FAILED" \
            "NSE pipeline failed after ${NSE_HM}.  Check run_nse_full.log for details." \
            "x,rotating_light"
        exit 1
    }

NSE_ELAPSED=$(( $(date +%s) - NSE_START ))
NSE_HM=$(printf "%dh %02dm" $(( NSE_ELAPSED / 3600 )) $(( (NSE_ELAPSED % 3600) / 60 )))

log "✅  NSE completed in ${NSE_HM}."

notify "✅ NSE done (${NSE_HM})" \
    "NSE pipeline finished in ${NSE_HM}.  Cleaning checkpoints and starting SP500..." \
    "white_check_mark"

# ── STEP 2: Clean NSE panel checkpoints ───────────────────────────────────────

log ""
log "STEP 2 — Cleaning NSE panel checkpoints to free disk"

FREED=0
for CKPT_FILE in \
    "${NSE_CKPT}/panel_features.pkl" \
    "${NSE_CKPT}/panel_targets.pkl"  \
    "${NSE_CKPT}/feat_cols.txt"
do
    if [ -f "${CKPT_FILE}" ]; then
        SIZE_BYTES=$(du -sb "${CKPT_FILE}" 2>/dev/null | cut -f1)
        SIZE_HUMAN=$(du -sh "${CKPT_FILE}" 2>/dev/null | cut -f1)
        rm -f "${CKPT_FILE}"
        FREED=$(( FREED + SIZE_BYTES ))
        log "  Removed $(basename "${CKPT_FILE}")  (${SIZE_HUMAN})"
    fi
done

FREED_GB=$(echo "scale=1; ${FREED} / 1073741824" | bc)
log "  Freed ~${FREED_GB} GB.  Disk now: $(df -h /mnt/data | tail -1)"

# Also drop NSE fold caches — large and no longer needed once NSE is done.
if compgen -G "${NSE_CKPT%/checkpoints}"/*/fold_cache > /dev/null 2>&1 || \
   compgen -G "${ARTEFACTS_ROOT}/nse_local/*/fold_cache" > /dev/null 2>&1; then
    rm -rf "${ARTEFACTS_ROOT}"/nse_local/*/fold_cache 2>/dev/null || true
    log "  Removed NSE fold_cache dirs.  Disk now: $(df -h /mnt/data | tail -1)"
fi

# ── STEP 3: Run SP500 ─────────────────────────────────────────────────────────

SP500_START=$(date +%s)

log ""
log "STEP 3 — Starting SP500 pipeline"
log "Log: ${SP500_LOG}"

ensure_disk "${ARTEFACTS_ROOT}" "SP500" || {
    log "❌  Aborting before SP500 — insufficient disk after cleanup."
    echo "DISK_FULL" > "${STATUS_FILE}"
    notify "❌ Disk full" "Not enough free space on ${ARTEFACTS_ROOT} to start SP500 safely." "x,rotating_light"
    exit 1
}

run_with_oom_retry "${SP500_LOG}" \
    python3 run_sp500_local.py \
        --train_start 2010-01-01 \
        --as_of       2023-12-08 \
        --n_jobs      1 \
    || {
        SP500_ELAPSED=$(( $(date +%s) - SP500_START ))
        SP500_HM=$(printf "%dh %02dm" $(( SP500_ELAPSED / 3600 )) $(( (SP500_ELAPSED % 3600) / 60 )))
        log "❌  SP500 FAILED after ${SP500_HM}."
        echo "SP500_FAILED" > "${STATUS_FILE}"
        notify "❌ SP500 FAILED" \
            "SP500 failed after ${SP500_HM} (NSE was OK).  Check run_sp500_full.log." \
            "x,rotating_light"
        exit 1
    }

SP500_ELAPSED=$(( $(date +%s) - SP500_START ))
SP500_HM=$(printf "%dh %02dm" $(( SP500_ELAPSED / 3600 )) $(( (SP500_ELAPSED % 3600) / 60 )))

TOTAL_HM=$(elapsed_hm)

log ""
log "════════════════════════════════════════════════════"
log "✅  ALL DONE — total wall time: ${TOTAL_HM}"
log "  NSE   took: ${NSE_HM}"
log "  SP500 took: ${SP500_HM}"
log "  NSE   artefacts: ${ARTEFACTS_ROOT}/nse_local/"
log "  SP500 artefacts: ${ARTEFACTS_ROOT}/us_local/"
log "════════════════════════════════════════════════════"

echo "ALL_DONE" > "${STATUS_FILE}"

notify "🏁 Both pipelines done! (${TOTAL_HM})" \
    "NSE: ${NSE_HM} | SP500: ${SP500_HM}. Artefacts ready on Hetzner." \
    "tada,white_check_mark"
