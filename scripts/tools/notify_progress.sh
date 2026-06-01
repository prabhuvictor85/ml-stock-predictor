#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  notify_progress.sh
#  Monitors a pipeline log and sends ntfy updates every 15 minutes.
#  Survives PuTTY closure via nohup.
#
#  Usage:
#    chmod +x notify_progress.sh
#    nohup bash notify_progress.sh \
#        /mnt/data/artefacts/us_local/run_sp500_full.log \
#        "SP500" \
#        > /tmp/notify_progress.out 2>&1 &
#    echo "Monitor PID: $!"
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="${1:-/mnt/data/artefacts/us_local/run_sp500_full.log}"
LABEL="${2:-Pipeline}"
NTFY_TOPIC="ntfy.sh/hetzner-victor-ml"
INTERVAL=900   # 15 minutes

send() {
    local title="$1"
    local msg="$2"
    local priority="${3:-default}"
    curl -s \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Tags: chart_with_upwards_trend" \
        -d "$msg" \
        "https://${NTFY_TOPIC}" > /dev/null 2>&1
}

extract_status() {
    # Pull last 200 lines for context
    local tail200
    tail200=$(tail -200 "$LOG_FILE" 2>/dev/null)

    # Current stage
    local stage
    stage=$(echo "$tail200" | grep -oP '\[\d+/\d+\][^\]]+' | tail -1)

    # HPO progress
    local hpo
    hpo=$(echo "$tail200" | grep -oP '\d+%\|.*\|\s+\d+/\d+' | tail -1)

    # Best trial
    local best_trial
    best_trial=$(echo "$tail200" | grep "Best trial:" | tail -1 | grep -oP 'Best trial:.*')

    # Latest fold
    local fold
    fold=$(echo "$tail200" | grep -oP 'Fold \d+:.*NDCG@10=[\d.]+' | tail -1)

    # Latest trial complete
    local trial
    trial=$(echo "$tail200" | grep "Trial.*COMPLETE" | tail -1 | grep -oP 'Trial \d+ COMPLETE.*objective=[\d.]+')

    # Errors
    local errors
    errors=$(echo "$tail200" | grep -c "ERROR\|Traceback\|Error" || true)

    # Run complete check
    local done
    done=$(echo "$tail200" | grep -c "PERFORMANCE SUMMARY\|Watchlist saved\|watchlist.*csv" || true)

    # Build message
    local msg=""
    [ -n "$stage"       ] && msg+="Stage: $stage\n"
    [ -n "$hpo"         ] && msg+="HPO: $hpo\n"
    [ -n "$best_trial"  ] && msg+="$best_trial\n"
    [ -n "$trial"       ] && msg+="Latest: $trial\n"
    [ -n "$fold"        ] && msg+="$fold\n"
    [ "$errors" -gt 0   ] && msg+="⚠️ Errors detected: $errors\n"
    [ "$done"   -gt 0   ] && msg+="✅ Run appears complete!\n"

    echo -e "${msg:-No progress lines found yet}"
    echo "DONE=$done"
    echo "ERRORS=$errors"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
echo "[$(date)] notify_progress.sh started. Monitoring: $LOG_FILE"
send "🚀 $LABEL Monitor Started" "Watching: $LOG_FILE\nUpdates every 15 min." "low"

iteration=0
while true; do
    sleep "$INTERVAL"
    iteration=$((iteration + 1))

    if [ ! -f "$LOG_FILE" ]; then
        send "⚠️ $LABEL — Log Missing" "Log file not found:\n$LOG_FILE" "high"
        continue
    fi

    status=$(extract_status)
    done_flag=$(echo "$status"  | grep "^DONE="   | cut -d= -f2)
    error_flag=$(echo "$status" | grep "^ERRORS=" | cut -d= -f2)
    clean_msg=$(echo "$status"  | grep -v "^DONE=\|^ERRORS=")

    elapsed=$(( iteration * INTERVAL / 60 ))
    title="$LABEL Update (+${elapsed}m)"

    if [ "${error_flag:-0}" -gt 0 ]; then
        send "🔴 $LABEL — Error Detected" "$clean_msg" "urgent"
    elif [ "${done_flag:-0}" -gt 0 ]; then
        send "✅ $LABEL — Run Complete!" "$clean_msg" "high"
        echo "[$(date)] Run complete — sending final notification and exiting."
        exit 0
    else
        send "$title" "$clean_msg" "default"
    fi

    echo "[$(date)] Notification #$iteration sent."
done
