#!/usr/bin/env bash
# =============================================================================
# run_full_training.sh — Train one or more markets
#
# Usage:
#   bash scripts/run_full_training.sh nse              # NSE local only
#   bash scripts/run_full_training.sh sp500            # SP500 only
#   bash scripts/run_full_training.sh tv               # NSE TradingView only
#   bash scripts/run_full_training.sh nse sp500 tv     # all three
#   bash scripts/run_full_training.sh nse sp500        # NSE + SP500
#
# Options:
#   --n_folds  N    walk-forward CV folds   (default: 8)
#   --n_trials N    Optuna HPO trials       (default: 25)
#
# Examples:
#   bash scripts/run_full_training.sh nse --n_trials 1 --n_folds 2   # quick test
#   bash scripts/run_full_training.sh nse sp500 tv                    # full run
#
# Recommended: run inside tmux so it survives SSH disconnect:
#   tmux new -s training
#   bash scripts/run_full_training.sh nse sp500 tv
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VENV_DIR="/root/venv"
PROJECT_DIR="/root/ml-stock-predictor"
LOG_DIR="${PROJECT_DIR}/logs"
N_FOLDS=8
N_TRIALS=25

# ── Parse arguments ───────────────────────────────────────────────────────────
RUN_NSE=0
RUN_SP500=0
RUN_TV=0

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/run_full_training.sh <market(s)> [--n_folds N] [--n_trials N]"
    echo "  Markets : nse  sp500  tv  (space-separated, any combination)"
    echo "  Example : bash scripts/run_full_training.sh nse sp500 tv"
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        nse)       RUN_NSE=1;        shift ;;
        sp500)     RUN_SP500=1;      shift ;;
        tv)        RUN_TV=1;         shift ;;
        --n_folds)  N_FOLDS="$2";   shift 2 ;;
        --n_trials) N_TRIALS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1  (valid markets: nse sp500 tv)"; exit 1 ;;
    esac
done

if [ $((RUN_NSE + RUN_SP500 + RUN_TV)) -eq 0 ]; then
    echo "No markets selected. Specify at least one: nse  sp500  tv"
    exit 1
fi

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${CYAN}  $*${NC}"
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

elapsed() { local s=$(( $(date +%s) - $1 )); printf "%dm %ds" $(( s/60 )) $(( s%60 )); }

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/training_${RUN_TS}.log"
START_TOTAL=$(date +%s)

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found at ${VENV_DIR}. Run server_setup.sh first."
    exit 1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

git pull origin master --quiet

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Training Run — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  Markets  : $([ $RUN_NSE -eq 1 ] && echo -n 'NSE  ') $([ $RUN_SP500 -eq 1 ] && echo -n 'SP500  ') $([ $RUN_TV -eq 1 ] && echo -n 'TradingView')"
echo "  n_folds  : ${N_FOLDS}"
echo "  n_trials : ${N_TRIALS}"
echo "  Log      : ${MASTER_LOG}"
echo ""

exec > >(tee -a "${MASTER_LOG}") 2>&1

declare -A MARKET_STATUS
declare -A MARKET_TIME
STEP=0
TOTAL=$(( RUN_NSE + RUN_SP500 + RUN_TV ))

# ── NSE Local ─────────────────────────────────────────────────────────────────
if [ "${RUN_NSE}" -eq 1 ]; then
    STEP=$(( STEP + 1 ))
    section "STEP ${STEP}/${TOTAL}  NSE Local  (momentum + reversal)"
    T=$(date +%s)
    if python run_nse_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[nse]="✅ PASSED"
    else
        MARKET_STATUS[nse]="❌ FAILED"
        error "NSE training failed — continuing"
    fi
    MARKET_TIME[nse]=$(elapsed $T)
fi

# ── SP500 / NASDAQ ────────────────────────────────────────────────────────────
if [ "${RUN_SP500}" -eq 1 ]; then
    STEP=$(( STEP + 1 ))
    section "STEP ${STEP}/${TOTAL}  SP500 / NASDAQ  (momentum + reversal)"
    T=$(date +%s)
    if python run_sp500_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[sp500]="✅ PASSED"
    else
        MARKET_STATUS[sp500]="❌ FAILED"
        error "SP500 training failed — continuing"
    fi
    MARKET_TIME[sp500]=$(elapsed $T)
fi

# ── NSE TradingView ───────────────────────────────────────────────────────────
if [ "${RUN_TV}" -eq 1 ]; then
    STEP=$(( STEP + 1 ))
    section "STEP ${STEP}/${TOTAL}  NSE TradingView  (momentum + reversal)"
    T=$(date +%s)
    if python run_nse_tradingv_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[tv]="✅ PASSED"
    else
        MARKET_STATUS[tv]="❌ FAILED"
        error "TradingView training failed"
    fi
    MARKET_TIME[tv]=$(elapsed $T)
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Complete — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
printf "  %-22s  %-14s  %s\n" "Market" "Status" "Time"
printf "  %-22s  %-14s  %s\n" "──────────────────────" "──────────────" "──────"
[ "${RUN_NSE}"   -eq 1 ] && printf "  %-22s  %-14s  %s\n" "NSE Local"       "${MARKET_STATUS[nse]}"   "${MARKET_TIME[nse]}"
[ "${RUN_SP500}" -eq 1 ] && printf "  %-22s  %-14s  %s\n" "SP500 / NASDAQ"  "${MARKET_STATUS[sp500]}" "${MARKET_TIME[sp500]}"
[ "${RUN_TV}"    -eq 1 ] && printf "  %-22s  %-14s  %s\n" "NSE TradingView" "${MARKET_STATUS[tv]}"    "${MARKET_TIME[tv]}"
echo ""
echo -e "  Total elapsed : ${GREEN}$(elapsed $START_TOTAL)${NC}"
echo -e "  Log           : ${MASTER_LOG}"
echo ""

for market in nse sp500 tv; do
    if [[ "${MARKET_STATUS[$market]:-}" == "❌ FAILED" ]]; then exit 1; fi
done
exit 0
