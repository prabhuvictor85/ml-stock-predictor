#!/usr/bin/env bash
# =============================================================================
# run_full_training.sh — Train all three markets
#
# Markets:
#   1. NSE Local         (run_nse_local.py)
#   2. S&P 500 / NASDAQ  (run_sp500_local.py)
#   3. NSE TradingView   (run_nse_tradingv_local.py)
#
# Data download is handled separately — run download scripts manually first.
#
# Usage:
#   bash scripts/run_full_training.sh                    # all markets, defaults
#   bash scripts/run_full_training.sh --skip_nse         # skip NSE local
#   bash scripts/run_full_training.sh --skip_sp500       # skip SP500
#   bash scripts/run_full_training.sh --skip_tv          # skip TradingView
#   bash scripts/run_full_training.sh --n_trials 5 --n_folds 3   # quick test run
#
# Recommended: run inside tmux so it survives SSH disconnect:
#   tmux new -s training
#   bash scripts/run_full_training.sh
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
VENV_DIR="/root/venv"
PROJECT_DIR="/root/ml-stock-predictor"
LOG_DIR="${PROJECT_DIR}/logs"

# Training defaults (match run_*.py defaults)
N_FOLDS=8
N_TRIALS=25

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_NSE=0
SKIP_SP500=0
SKIP_TV=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_nse)   SKIP_NSE=1;    shift ;;
        --skip_sp500) SKIP_SP500=1;  shift ;;
        --skip_tv)    SKIP_TV=1;     shift ;;
        --n_folds)    N_FOLDS="$2";  shift 2 ;;
        --n_trials)   N_TRIALS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
            echo -e "${CYAN}  $*${NC}"; \
            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── Timing helpers ────────────────────────────────────────────────────────────
START_TOTAL=$(date +%s)
elapsed() {
    local secs=$(( $(date +%s) - $1 ))
    printf "%dm %ds" $(( secs/60 )) $(( secs%60 ))
}

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/full_training_${RUN_TS}.log"

# Activate venv
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    error "venv not found at ${VENV_DIR}. Run server_setup.sh first."
    exit 1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

# Pull latest code
info "Pulling latest code from master ..."
git pull origin master --quiet

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Full Training Run — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  n_folds  : ${N_FOLDS}"
echo "  n_trials : ${N_TRIALS}"
echo "  Log      : ${MASTER_LOG}"
echo ""

# Tee all output to master log
exec > >(tee -a "${MASTER_LOG}") 2>&1

# ── Track per-market results ───────────────────────────────────────────────────
declare -A MARKET_STATUS
declare -A MARKET_TIME

# =============================================================================
# STEP 1: NSE LOCAL TRAINING
# =============================================================================
if [ "${SKIP_NSE}" -eq 0 ]; then
    section "STEP 1/3  NSE Local Training  (momentum + reversal)"
    T=$(date +%s)
    if python run_nse_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[nse]="✅ PASSED"
    else
        MARKET_STATUS[nse]="❌ FAILED"
        error "NSE training failed — continuing with SP500"
    fi
    MARKET_TIME[nse]=$(elapsed $T)
else
    MARKET_STATUS[nse]="⏭  SKIPPED"
    MARKET_TIME[nse]="—"
    warn "Skipping NSE local (--skip_nse)"
fi

# =============================================================================
# STEP 2: SP500 / NASDAQ TRAINING
# =============================================================================
if [ "${SKIP_SP500}" -eq 0 ]; then
    section "STEP 2/3  SP500 / NASDAQ Training  (momentum + reversal)"
    T=$(date +%s)
    if python run_sp500_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[sp500]="✅ PASSED"
    else
        MARKET_STATUS[sp500]="❌ FAILED"
        error "SP500 training failed — continuing with TradingView"
    fi
    MARKET_TIME[sp500]=$(elapsed $T)
else
    MARKET_STATUS[sp500]="⏭  SKIPPED"
    MARKET_TIME[sp500]="—"
    warn "Skipping SP500 (--skip_sp500)"
fi

# =============================================================================
# STEP 3: NSE TRADINGVIEW TRAINING
# =============================================================================
if [ "${SKIP_TV}" -eq 0 ]; then
    section "STEP 3/3  NSE TradingView Training  (momentum + reversal)"
    T=$(date +%s)
    if python run_nse_tradingv_local.py --n_folds "${N_FOLDS}" --n_trials "${N_TRIALS}"; then
        MARKET_STATUS[tv]="✅ PASSED"
    else
        MARKET_STATUS[tv]="❌ FAILED"
        error "TradingView training failed"
    fi
    MARKET_TIME[tv]=$(elapsed $T)
else
    MARKET_STATUS[tv]="⏭  SKIPPED"
    MARKET_TIME[tv]="—"
    warn "Skipping TradingView (--skip_tv)"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Training Complete — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
printf "  %-22s  %-14s  %s\n" "Market" "Status" "Time"
printf "  %-22s  %-14s  %s\n" "──────────────────────" "──────────────" "──────"
printf "  %-22s  %-14s  %s\n" "NSE Local"        "${MARKET_STATUS[nse]:-⏭  SKIPPED}"   "${MARKET_TIME[nse]:-—}"
printf "  %-22s  %-14s  %s\n" "SP500 / NASDAQ"   "${MARKET_STATUS[sp500]:-⏭  SKIPPED}" "${MARKET_TIME[sp500]:-—}"
printf "  %-22s  %-14s  %s\n" "NSE TradingView"  "${MARKET_STATUS[tv]:-⏭  SKIPPED}"    "${MARKET_TIME[tv]:-—}"
echo ""
echo -e "  Total elapsed : ${GREEN}$(elapsed $START_TOTAL)${NC}"
echo -e "  Log file      : ${MASTER_LOG}"
echo ""
echo "  Output files:"
echo "    output/nse_local/watchlist_*.csv"
echo "    output/us_local/watchlist_*.csv"
echo "    output/nse_tradingv/watchlist_*.csv"
echo ""
echo -e "${YELLOW}  Next: detach volume in Hetzner console, then delete server.${NC}"
echo ""

# Exit with failure if any market failed
for market in nse sp500 tv; do
    if [[ "${MARKET_STATUS[$market]:-}" == "❌ FAILED" ]]; then
        exit 1
    fi
done
exit 0
