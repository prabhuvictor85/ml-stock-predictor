#!/usr/bin/env bash
# =============================================================================
# server_setup.sh — One-shot Hetzner server bootstrap for ml-stock-predictor
#
# Run this once after every new server creation:
#   bash server_setup.sh
#
# What it does:
#   1. Mounts the persistent Hetzner Volume (/dev/sdb → /mnt/data)
#   2. Creates required directory structure on the volume
#   3. Clones / updates ml-stock-predictor from GitHub (master branch)
#   4. Installs Python dependencies
#   5. Writes paths.yaml pointing to /mnt/data
#
# Prerequisites:
#   - Volume "ml-data" already exists on Hetzner and is attached to this server
#   - GitHub repo is accessible (public or SSH key pre-loaded)
#   - Python 3.10+ already installed (Hetzner Ubuntu images include it)
# =============================================================================

set -euo pipefail

# ── Config — edit these if they change ──────────────────────────────────────
GITHUB_REPO="https://github.com/prabhuvictor85/ml-stock-predictor.git"
GIT_BRANCH="master"
PROJECT_DIR="/root/ml-stock-predictor"
VOLUME_DEVICE="/dev/sdb"
MOUNT_POINT="/mnt/data"

# Paths on the volume (survive server deletion)
DATA_ROOT="${MOUNT_POINT}/Learning_charts"
ARTEFACTS_ROOT="${MOUNT_POINT}/artefacts"
STOCK_DATA_DIR="${DATA_ROOT}/stock_data"
STOCK_LISTS_DIR="${DATA_ROOT}/stock_lists"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${GREEN}===== $* =====${NC}"; }

# =============================================================================
# 1. MOUNT VOLUME
# =============================================================================
section "1/5  Mounting Hetzner Volume"

if mountpoint -q "${MOUNT_POINT}"; then
    info "Volume already mounted at ${MOUNT_POINT} — skipping"
else
    if [ ! -b "${VOLUME_DEVICE}" ]; then
        echo -e "${RED}[ERROR]${NC} Device ${VOLUME_DEVICE} not found."
        echo "  → Make sure the Volume is attached to this server in Hetzner console."
        exit 1
    fi

    # Format only if the device has no filesystem yet
    FS_TYPE=$(blkid -o value -s TYPE "${VOLUME_DEVICE}" 2>/dev/null || true)
    if [ -z "${FS_TYPE}" ]; then
        info "No filesystem detected — formatting ${VOLUME_DEVICE} as ext4 ..."
        mkfs.ext4 -F "${VOLUME_DEVICE}"
    else
        info "Existing filesystem (${FS_TYPE}) found — skipping format"
    fi

    mkdir -p "${MOUNT_POINT}"
    mount "${VOLUME_DEVICE}" "${MOUNT_POINT}"
    info "Mounted ${VOLUME_DEVICE} → ${MOUNT_POINT}"

    # Persist mount across reboots (add only if not already in fstab)
    if ! grep -q "${VOLUME_DEVICE}" /etc/fstab; then
        echo "${VOLUME_DEVICE} ${MOUNT_POINT} ext4 discard,nofail,defaults 0 0" >> /etc/fstab
        info "Added to /etc/fstab for auto-mount on reboot"
    fi
fi

# =============================================================================
# 2. CREATE DIRECTORY STRUCTURE ON VOLUME
# =============================================================================
section "2/5  Creating directory structure on volume"

mkdir -p "${STOCK_DATA_DIR}"
mkdir -p "${STOCK_LISTS_DIR}"
mkdir -p "${ARTEFACTS_ROOT}"
info "Directories ready:"
info "  ${STOCK_DATA_DIR}"
info "  ${STOCK_LISTS_DIR}"
info "  ${ARTEFACTS_ROOT}"

# =============================================================================
# 3. CLONE / UPDATE GITHUB REPO
# =============================================================================
section "3/5  Cloning / updating repository"

if [ -d "${PROJECT_DIR}/.git" ]; then
    info "Repo already exists — pulling latest ${GIT_BRANCH} ..."
    git -C "${PROJECT_DIR}" fetch origin
    git -C "${PROJECT_DIR}" checkout "${GIT_BRANCH}"
    git -C "${PROJECT_DIR}" pull origin "${GIT_BRANCH}"
else
    info "Cloning ${GITHUB_REPO} (branch: ${GIT_BRANCH}) ..."
    git clone --branch "${GIT_BRANCH}" "${GITHUB_REPO}" "${PROJECT_DIR}"
fi

info "Repo ready at ${PROJECT_DIR}"

# Re-exec from the repo's copy of this script so we always run the latest version.
# Guard against infinite loop with SETUP_REEXECED env var.
REPO_SCRIPT="${PROJECT_DIR}/scripts/server_setup.sh"
if [ -z "${SETUP_REEXECED:-}" ] && [ -f "${REPO_SCRIPT}" ]; then
    if ! diff -q "$0" "${REPO_SCRIPT}" &>/dev/null; then
        info "Newer version of setup script found — re-executing from repo ..."
        export SETUP_REEXECED=1
        exec bash "${REPO_SCRIPT}" "$@"
    fi
fi

# =============================================================================
# 4. INSTALL PYTHON DEPENDENCIES
# =============================================================================
section "4/5  Installing Python dependencies"

cd "${PROJECT_DIR}"

# Ensure pip is available (Hetzner Ubuntu images may not include it)
if ! python3 -m pip --version &>/dev/null; then
    info "pip not found — installing via apt ..."
    apt-get update -qq && apt-get install -y python3-pip python3-venv
fi

# Ensure pip is up to date
# --break-system-packages is safe here: Hetzner servers are ephemeral/throwaway
python3 -m pip install --upgrade pip --quiet --break-system-packages

if [ -f "requirements.txt" ]; then
    info "Installing from requirements.txt ..."
    python3 -m pip install -r requirements.txt --quiet --break-system-packages
else
    warn "requirements.txt not found — skipping pip install"
fi

info "Python dependencies installed"

# =============================================================================
# 5. WRITE paths.yaml
# =============================================================================
section "5/5  Writing paths.yaml"

PATHS_YAML="${PROJECT_DIR}/paths.yaml"

cat > "${PATHS_YAML}" <<EOF
# paths.yaml — auto-generated by server_setup.sh
# Edit GITHUB_REPO in server_setup.sh to match your repo.
# All data lives on the persistent Hetzner Volume at /mnt/data.

data_root:    ${DATA_ROOT}
project_root: ${PROJECT_DIR}

stock_lists:
  nse_local:     ${STOCK_LISTS_DIR}/constituentsi.csv
  nse_tv:        ${STOCK_LISTS_DIR}/constituents_nse_tradingv.csv
  nse_cap_tiers: ${STOCK_LISTS_DIR}/nse_cap_tiers.csv
  us_combined:   ${STOCK_LISTS_DIR}/constituents_us_combined.csv
  lists_dir:     ${STOCK_LISTS_DIR}

stock_data:
  nse_local: ${STOCK_DATA_DIR}
  nse_tv:    ${STOCK_DATA_DIR}/tradingview
  us:        ${STOCK_DATA_DIR}/us_stocks
  us_alt:    ${DATA_ROOT}/us_data

artefacts_root: ${ARTEFACTS_ROOT}
EOF

info "paths.yaml written to ${PATHS_YAML}"
cat "${PATHS_YAML}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Server setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Volume  : ${MOUNT_POINT}  (persists after server deletion)"
echo "  Project : ${PROJECT_DIR}"
echo "  Data    : ${DATA_ROOT}"
echo "  Models  : ${ARTEFACTS_ROOT}"
echo ""
echo "Next steps:"
echo "  # Download NSE data (first time or delta update):"
echo "  cd ${PROJECT_DIR}"
echo "  python scripts/data/download_nse_data.py"
echo ""
echo "  # Run full training:"
echo "  python run_nse_local.py"
echo ""
echo "  # When done — detach volume in Hetzner console, then delete server."
echo ""
