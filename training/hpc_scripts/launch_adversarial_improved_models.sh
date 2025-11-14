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
echo "  - Queue: gpuv100"
echo "  - 3 adversarial iterations"
echo "  - 10000 fake samples per iteration"
echo "  - Discriminator loss weight: 0.1"
echo "  - Starting from improved model checkpoints"
echo ""
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit all 5 jobs
JOB_IDS=()

echo "Submitting UNet32 (d3, improved)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_adversarial_unet32_d3_improved.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    JOB_IDS+=("$JOBID")
    echo "  ✓ Job submitted: $JOBID"
else
    echo "  ✗ Failed to submit"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "Submitting UNet48 (d3, improved)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_adversarial_unet48_d3_improved.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    JOB_IDS+=("$JOBID")
    echo "  ✓ Job submitted: $JOBID"
else
    echo "  ✗ Failed to submit"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "Submitting UNet48 (d4, improved)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_adversarial_unet48_d4_improved.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    JOB_IDS+=("$JOBID")
    echo "  ✓ Job submitted: $JOBID"
else
    echo "  ✗ Failed to submit"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "Submitting UNet64 (d3, improved)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_adversarial_unet64_d3_improved.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    JOB_IDS+=("$JOBID")
    echo "  ✓ Job submitted: $JOBID"
else
    echo "  ✗ Failed to submit"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "Submitting UNet64 (d4, improved)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_adversarial_unet64_d4_improved.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    JOB_IDS+=("$JOBID")
    echo "  ✓ Job submitted: $JOBID"
else
    echo "  ✗ Failed to submit"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "=========================================="
echo "All jobs submitted successfully!"
echo "=========================================="
echo "Job IDs:"
for i in "${!JOB_IDS[@]}"; do
    echo "  [$((i+1))] ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor with:"
echo "  bjobs -u $USER"
echo ""
echo "Check logs:"
echo "  ${LOG_DIR}/adversarial_*.out"
echo "  ${LOG_DIR}/adversarial_*.err"
echo ""

