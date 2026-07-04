#!/usr/bin/env bash
# run_lockbox_v3_zone.sh — Pure-zone lockbox (--feature_set zone)
#
# Identical procedure to the lockbox v2 script.
# Feature engineering runs in full; HPO/FeatureSelector/training see only
# features_sdz_* / ssz_* / dz_* / sz_* / zone_* columns.
#
# Usage:
#   cd /root/ml-stock-predictor
#   nohup bash run_lockbox_v3_zone.sh > /tmp/lockbox_v3_zone.log 2>&1 &
#   echo "PID: $!"
#   tail -f /tmp/lockbox_v3_zone.log
#
set -uo pipefail

REPO=/root/ml-stock-predictor
ROOT=/mnt/data/artefacts/us_lockbox_v3
DATA_DIR=/mnt/data/Learning_charts/stock_data/us_stocks
FENCE=2023-12-31
HEARTBEAT=1800
NTFY_TOPIC=""

mkdir -p "$ROOT"
export ML_ARTEFACTS_ROOT="$ROOT"
STATUS="$ROOT/run_status.log"

notify() {
    local l="[$(date '+%F %T')] $1"
    echo "$l" | tee -a "$STATUS"
    [ -n "$NTFY_TOPIC" ] && curl -s -m10 -d "$l" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}
fail() { notify "ABORT: $1"; kill "${HB_PID:-0}" 2>/dev/null; exit 1; }
set_step() { echo "$1" > "$ROOT/.step"; }

cd "$REPO" || fail "no repo at $REPO"
# Do NOT auto-pull — caller controls the exact commit to pin to.
# Pull manually before running this script, verify the hash, then launch.
HASH=$(git rev-parse --short HEAD)
notify "PREFLIGHT OK | commit=$HASH | root=$ROOT | fence=$FENCE | ZONE-ONLY (--feature_set zone) | NO auto-pull"

set_step init
heartbeat() {
    local t0=$SECONDS s
    while sleep "$HEARTBEAT"; do
        s=$(cat "$ROOT/.step" 2>/dev/null || echo '?')
        notify "HEARTBEAT $(( (SECONDS - t0) / 60 ))m | step=$s | $(tail -n1 "$ROOT/$s.log" 2>/dev/null | cut -c1-160)"
    done
}
heartbeat & HB_PID=$!
trap 'kill $HB_PID 2>/dev/null' EXIT

# ── STEP 1: Fenced HPO + feature selection + train (zone only) ──────────────
set_step step1
notify "STEP 1 START — fenced HPO+selection+train (<= $FENCE) with --feature_set zone"
python3 run_sp500_local.py \
    --mode momentum \
    --train_start 2010-01-01 \
    --train_end "$FENCE" \
    --as_of 2023-12-29 \
    --n_trials 40 \
    --feature_set zone \
    > "$ROOT/step1.log" 2>&1 || fail "step1 failed — see $ROOT/step1.log"

grep -q "LOCKBOX FENCE ACTIVE" "$ROOT/step1.log" \
    || fail "fence banner missing in step1.log — training NOT capped at $FENCE"
grep -q "\[feature_set=zone\]" "$ROOT/step1.log" \
    || fail "feature_set=zone banner missing — zone filter may not have applied"
notify "STEP 1 DONE — fence verified, zone-only confirmed"

# ── STEP 2: Walk forward 2024-01-12 → 2026-05-04 ────────────────────────────
set_step step2
notify "STEP 2 START — walk 2024-01-12 -> 2026-05-04 (cadence 14d)"
python3 run_walkforward_sp500.py \
    --start 2024-01-12 \
    --end 2026-05-04 \
    --cadence_days 14 \
    --mode momentum \
    --train_end "$FENCE" \
    --no_drift_retrain \
    --log_dir "$ROOT/us_local" \
    > "$ROOT/step2.log" 2>&1 || fail "step2 failed — see $ROOT/step2.log"
notify "STEP 2 DONE"

# ── STEP 3: Independent verdict ──────────────────────────────────────────────
set_step step3
notify "STEP 3 START — independent verdict"
python3 scripts/tools/validate_lockbox.py \
    --scores_dir "$ROOT/us_local/output" \
    --data_dir   "$DATA_DIR" \
    --mode momentum \
    --side bull \
    --score_field model_score \
    --start 2024-01-01 \
    --end 2026-05-06 \
    --out "$ROOT/lockbox_verdict.json" \
    > "$ROOT/step3.log" 2>&1 || fail "step3 failed — see $ROOT/step3.log"

kill $HB_PID 2>/dev/null
notify "LOCKBOX V3 COMPLETE — verdict: $ROOT/lockbox_verdict.json"

echo
echo "===== VERDICT ====="
cat "$ROOT/lockbox_verdict.json"
