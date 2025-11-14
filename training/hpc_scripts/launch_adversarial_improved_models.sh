#!/bin/bash
# Launch script for adversarial training on all improved models
# Submits 5 jobs: UNet32 (d3), UNet48 (d3), UNet48 (d4), UNet64 (d3), UNet64 (d4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Adversarial Training for Improved Models"
echo "=========================================="
echo ""
echo "Submitting 5 jobs:"
echo "  [1] UNet32 (d3, improved)"
echo "  [2] UNet48 (d3, improved)"
echo "  [3] UNet48 (d4, improved)"
echo "  [4] UNet64 (d3, improved)"
echo "  [5] UNet64 (d4, improved)"
echo ""
echo "All jobs use:"
echo "  - Queue: gpul40s"
echo "  - 3 adversarial iterations"
echo "  - 10000 fake samples per iteration"
echo "  - Discriminator loss weight: 0.1"
echo "  - Starting from improved model checkpoints"
echo ""
echo "Log directory: ${LOG_DIR}"
echo ""

# Verify script files exist
SCRIPTS=(
    "run_adversarial_unet32_d3_improved.sh"
    "run_adversarial_unet48_d3_improved.sh"
    "run_adversarial_unet48_d4_improved.sh"
    "run_adversarial_unet64_d3_improved.sh"
    "run_adversarial_unet64_d4_improved.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [[ ! -f "${SCRIPT_DIR}/${script}" ]]; then
        echo "ERROR: Script not found: ${SCRIPT_DIR}/${script}" >&2
        exit 1
    fi
    if [[ ! -r "${SCRIPT_DIR}/${script}" ]]; then
        echo "ERROR: Script not readable: ${SCRIPT_DIR}/${script}" >&2
        exit 1
    fi
done

# Submit all 5 jobs one by one
echo "Submitting UNet32 (d3, improved)..."
if ! bsub < "${SCRIPT_DIR}/run_adversarial_unet32_d3_improved.sh"; then
    echo "  ✗ Failed to submit UNet32 (d3)"
    exit 1
fi
sleep 2

echo ""
echo "Submitting UNet48 (d3, improved)..."
if ! bsub < "${SCRIPT_DIR}/run_adversarial_unet48_d3_improved.sh"; then
    echo "  ✗ Failed to submit UNet48 (d3)"
    exit 1
fi
sleep 2

echo ""
echo "Submitting UNet48 (d4, improved)..."
if ! bsub < "${SCRIPT_DIR}/run_adversarial_unet48_d4_improved.sh"; then
    echo "  ✗ Failed to submit UNet48 (d4)"
    exit 1
fi
sleep 2

echo ""
echo "Submitting UNet64 (d3, improved)..."
if ! bsub < "${SCRIPT_DIR}/run_adversarial_unet64_d3_improved.sh"; then
    echo "  ✗ Failed to submit UNet64 (d3)"
    exit 1
fi
sleep 2

echo ""
echo "Submitting UNet64 (d4, improved)..."
if ! bsub < "${SCRIPT_DIR}/run_adversarial_unet64_d4_improved.sh"; then
    echo "  ✗ Failed to submit UNet64 (d4)"
    exit 1
fi

echo ""
echo "=========================================="
echo "All 5 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with:"
echo "  bjobs -u $USER"
echo ""
echo "Check logs:"
echo "  ${LOG_DIR}/adversarial_*.out"
echo "  ${LOG_DIR}/adversarial_*.err"
echo ""

