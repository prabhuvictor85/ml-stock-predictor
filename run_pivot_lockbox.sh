#!/usr/bin/env bash
# run_pivot_lockbox.sh — MODEL_D pivot-only isolation (PIVOT_FEATURES=1)
#
# Three stages, gated:
#   1. Build a pivot-enabled features+targets checkpoint (full history, no
#      fence — the fence is applied INSIDE the experiment scripts, same as
#      MODEL_A/MODEL_C).
#   2. Tuning-era walk-forward CV (2018-2023, yearly expanding folds) —
#      the pre-registered gate. See PROTOCOL.md §3.1 for the criteria:
#      adopt iff mean IC >= 0.03 AND t-stat >= 2.0 AND >=4/6 folds positive.
#   3. Lockbox static split (2024 -> last realized-return date) — runs
#      ONLY if step 2's gate passed. One-shot: do not re-run this step on
#      a disappointing result (PROTOCOL.md §6).
#
# Isolated via a FRESH ML_ARTEFACTS_ROOT — never touches us_lockbox_v2 (the
# panel MODEL_A/MODEL_C read) or any production artefact root.
#
# PREFLIGHT — do this before running, not automated here:
#   Confirm the seed flags used to build us_lockbox_v2 (esp. --pit_universe)
#   from its own run logs, and match them below. If MODEL_D's panel isn't
#   built the same way, its IC isn't apples-to-apples with MODEL_A/MODEL_C
#   (in_universe changes both the label and the breadth).
#
# Usage:
#   cd /root/ml-stock-predictor
#   nohup bash run_pivot_lockbox.sh > /tmp/pivot_lockbox.log 2>&1 &
#   echo "PID: $!"
#   tail -f /tmp/pivot_lockbox.log
#
set -uo pipefail

REPO=/root/ml-stock-predictor
ROOT=/mnt/data/artefacts/us_pivot_v1
HEARTBEAT=1800
NTFY_TOPIC=""

mkdir -p "$ROOT" "$ROOT/experiments"
export ML_ARTEFACTS_ROOT="$ROOT"
export PIVOT_FEATURES=1
STATUS="$ROOT/run_status.log"

PANEL="$ROOT/us_local/checkpoints/panel_targets.pkl"
CV_OUT="$ROOT/experiments/model_d_results.json"
LOCKBOX_OUT="$ROOT/experiments/model_d_lockbox_results.json"

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
notify "PREFLIGHT OK | commit=$HASH | root=$ROOT | PIVOT_FEATURES=1 | mode=momentum | NO auto-pull"

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

# ── STEP 1: Build pivot-enabled features+targets checkpoint ─────────────────
# No --train_end here: the checkpoint is full-history; MODEL_D's own scripts
# fence to <= 2023-12-31 for CV and to the 2024+ window for the lockbox split.
set_step step1
notify "STEP 1 START — pivot-enabled panel build (--stop_after_targets)"
python3 run_sp500_local.py \
    --mode momentum \
    --pit_universe \
    --train_start 2010-01-01 \
    --stop_after_targets \
    > "$ROOT/step1.log" 2>&1 || fail "step1 failed — see $ROOT/step1.log"

grep -q "stop_after_targets: checkpoint ready" "$ROOT/step1.log" \
    || fail "checkpoint-ready banner missing in step1.log — panel build may not have completed"
notify "STEP 1 DONE — checkpoint at $PANEL"

# ── STEP 2: Tuning-era CV — the pre-registered gate ──────────────────────────
set_step step2
notify "STEP 2 START — MODEL_D pivot-only CV (2018-2023)"
python3 scripts/experiments/model_d_pivot_only.py \
    --panel "$PANEL" \
    --out "$CV_OUT" \
    > "$ROOT/step2.log" 2>&1 || fail "step2 failed — see $ROOT/step2.log"
notify "STEP 2 DONE — results at $CV_OUT"

# Parse the gate MODEL_D already computed (mean_ic>=0.03 AND t>=2.0 AND
# >=4/6 folds positive). Exit code 2 = the JSON/key itself is malformed
# (a real bug — abort). Exit code 1 = the script ran fine and the gate
# genuinely failed (a valid experimental outcome — stop gracefully, do
# NOT abort loudly, and do NOT touch the lockbox).
GATE_MSG=$(python3 -c "
import json, sys
try:
    d = json.load(open('$CV_OUT'))
    g = d['preregistered_gate']
    ok = bool(g['gate_pass'])
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    sys.exit(2)
s = d['summary']
print(f\"{'PASS' if ok else 'FAIL'} | mean_ic={s['mean_ic']:+.4f} t={s['t_stat']:+.2f} \"
      f\"folds_positive={s['n_folds_positive']}/{s['n_folds']}\")
sys.exit(0 if ok else 1)
")
GATE_RC=$?
if [ "$GATE_RC" -eq 2 ]; then
    fail "could not parse MODEL_D gate result: $GATE_MSG"
fi
notify "MODEL_D CV gate: $GATE_MSG"

if [ "$GATE_RC" -ne 0 ]; then
    notify "GATE FAIL — stopping BEFORE the lockbox (one-shot rule, PROTOCOL.md §6). Record this CV result in PROTOCOL.md §3.1. Pivots stay OFF."
    kill $HB_PID 2>/dev/null
    exit 0
fi

# ── STEP 3: Lockbox static split — ONLY reached if the gate passed ──────────
set_step step3
notify "STEP 3 START — gate PASSED, running lockbox static split (2024 -> last realized return)"
python3 scripts/experiments/model_d_pivot_only_lockbox.py \
    --panel "$PANEL" \
    --out "$LOCKBOX_OUT" \
    > "$ROOT/step3.log" 2>&1 || fail "step3 failed — see $ROOT/step3.log"
notify "STEP 3 DONE — results at $LOCKBOX_OUT"

kill $HB_PID 2>/dev/null
notify "PIVOT LOCKBOX COMPLETE — commit=$HASH | CV: $CV_OUT | lockbox: $LOCKBOX_OUT"

echo
echo "===== MODEL_D CV SUMMARY ====="
python3 -c "import json; print(json.dumps(json.load(open('$CV_OUT'))['summary'], indent=2))"
echo
echo "===== MODEL_D LOCKBOX SUMMARY ====="
python3 -c "import json; print(json.dumps(json.load(open('$LOCKBOX_OUT'))['summary'], indent=2))"
echo
echo "Next: record both results in PROTOCOL.md §3.1 (commit=$HASH), including the"
echo "consistency check (lockbox IC >= 50% of CV IC, same sign) per the pre-registration."
