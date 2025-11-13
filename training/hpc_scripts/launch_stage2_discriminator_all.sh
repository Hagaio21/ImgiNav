#!/bin/bash
# Launch script for all Stage 2 Discriminator training jobs
# Submits 3 jobs: UNet32 (d3), UNet48 (d4), UNet64 (d4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Stage 2 Discriminator Training"
echo "=========================================="
echo ""
echo "Submitting 3 jobs:"
echo "  [1] UNet32 (d3)"
echo "  [2] UNet48 (d4)"
echo "  [3] UNet64 (d4)"
echo ""
echo "All jobs use:"
echo "  - Queue: gpuv100"
echo "  - 3 iterations (adversarial training)"
echo "  - 5000 samples per iteration"
echo "  - Discriminator loss weight: 0.3"
echo ""
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit all 3 jobs
JOB_IDS=()

echo "Submitting UNet32 (d3)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_stage2_discriminator_unet32_d3.sh" 2>&1)
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
echo "Submitting UNet48 (d4)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_stage2_discriminator_unet48_d4.sh" 2>&1)
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
echo "Submitting UNet64 (d4)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_stage2_discriminator_unet64_d4.sh" 2>&1)
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
    case $i in
        0) echo "  [1] UNet32 (d3): ${JOB_IDS[$i]}" ;;
        1) echo "  [2] UNet48 (d4): ${JOB_IDS[$i]}" ;;
        2) echo "  [3] UNet64 (d4): ${JOB_IDS[$i]}" ;;
    esac
done
echo ""
echo "Check status with:"
echo "  bjobs ${JOB_IDS[0]} ${JOB_IDS[1]} ${JOB_IDS[2]}"
echo ""
echo "Monitor logs in: ${LOG_DIR}"
echo "  - stage2_discriminator_unet32_d3.*.out"
echo "  - stage2_discriminator_unet48_d4.*.out"
echo "  - stage2_discriminator_unet64_d4.*.out"
echo ""
echo "Each job will run 3 iterations of adversarial training."
echo "Expected duration: ~24 hours per job (72 hours total for all 3)"
echo ""

