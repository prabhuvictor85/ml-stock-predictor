#!/usr/bin/env bash
# run_lockbox_parallel.sh -- parallel lockbox inference.
#
# The frozen-model lockbox walk has NO inter-date dependency: every date uses the
# same seed model, there are no drift retrains, and (with the causality fix) each
# date scores on a panel truncated to <= as_of. So the dates are embarrassingly
# parallel -- unlike the live walk-forward harness, which is sequential by design.
#
# This dispatches N concurrent `run_sp500_local.py --skip_train --as_of D` jobs.
# All score/watchlist files are date-stamped (no collision), and --no_drift_save
# avoids the one shared write. Resumable: dates whose score file already exists
# are skipped, so a crash/kill just re-runs the missing ones.
#
# Prereqs: seed model already trained (step 1), causality fix pulled.
#
# Usage:
#   ML_ARTEFACTS_ROOT=/mnt/data/artefacts/us_lockbox \
#   scripts/tools/run_lockbox_parallel.sh 2024-01-12 2026-05-04 14 4 momentum
#                                          START      END        CAD JOBS MODE
set -u

START="${1:-2024-01-12}"
END="${2:-2026-05-04}"
CADENCE="${3:-14}"
JOBS="${4:-4}"          # concurrent processes; 4 is safe on 32 GB (each holds a panel)
MODE="${5:-momentum}"

export ML_ARTEFACTS_ROOT="${ML_ARTEFACTS_ROOT:-/mnt/data/artefacts/us_lockbox}"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ML_ARTEFACTS_ROOT/us_local/output"
LOGD="$ML_ARTEFACTS_ROOT/parallel_logs"
mkdir -p "$LOGD"
cd "$PROJ"

echo "project=$PROJ"
echo "artefacts=$ML_ARTEFACTS_ROOT  out=$OUT"
echo "range=$START..$END cadence=${CADENCE}d jobs=$JOBS mode=$MODE"

# cadence dates
mapfile -t DATES < <(python3 -c "
import pandas as pd
d=pd.Timestamp('$START'); e=pd.Timestamp('$END')
while d<=e:
    print(d.date()); d+=pd.Timedelta(days=$CADENCE)
")
echo "dates: ${#DATES[@]}"

run_one() {
  local D="$1"
  if ls "$OUT"/scores_detail_${MODE}_${D}.json >/dev/null 2>&1; then
    echo "SKIP $D (exists)"; return 0
  fi
  if python3 run_sp500_local.py --skip_train --mode "$MODE" --as_of "$D" \
        --n_jobs 1 --no_drift_save > "$LOGD/infer_${D}.log" 2>&1; then
    echo "DONE $D"
  else
    echo "FAIL $D (see $LOGD/infer_${D}.log)"
  fi
}
export -f run_one
export OUT LOGD MODE ML_ARTEFACTS_ROOT

printf '%s\n' "${DATES[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo "=== all dispatched. score files in: $OUT ==="
echo "Re-run this script to retry any FAIL dates (resumable)."
