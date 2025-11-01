#!/bin/bash
# Launch script for Phase 1.1: Channel × Spatial Resolution Sweep
# Runs 10 experiments on V100 and 2 on L40s (12 total)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.1: Channel × Spatial Resolution Sweep"
echo "=========================================="
echo "V100 jobs: 10 experiments (S1-S10) - will queue automatically"
echo "L40s jobs: 2 experiments (S11-S12) - will queue automatically"
echo "Total: 12 experiments covering full latent shape sweep"
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit V100 jobs (10 experiments - jobs will queue automatically)
echo "Submitting V100 jobs (S1-S10)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_1_v100.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    echo "✓ V100 job array submitted: $JOBID"
    echo "  Will run 10 experiments (S1-S10) as slots become available"
else
    echo "✗ Failed to submit V100 jobs"
    echo "$OUTPUT"
fi

# Submit L40s jobs (2 experiments)
echo ""
echo "Submitting L40s jobs (S11-S12)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_1_l40s.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    echo "✓ L40s job array submitted: $JOBID"
    echo "  Will run 2 experiments (S11-S12) as slots become available"
else
    echo "✗ Failed to submit L40s jobs"
    echo "$OUTPUT"
fi

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Check status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Phase 1.1 experiments (12 total - channel × spatial resolution sweep):"
echo "  V100 (S1-S10):"
echo "    - S1: ch16_ds4 (32×32×16 = 16K)"
echo "    - S2: ch8_ds3 (64×64×8 = 32K)"
echo "    - S3: ch4_ds3 (64×64×4 = 16K)"
echo "    - S4: ch8_ds4 (32×32×8 = 8K)"
echo "    - S5: ch16_ds5 (16×16×16 = 4K)"
echo "    - S6: ch32_ds4 (32×32×32 = 32K)"
echo "    - S7: ch4_ds4 (32×32×4 = 4K)"
echo "    - S8: ch8_ds5 (16×16×8 = 2K)"
echo "    - S9: ch16_ds3 (64×64×16 = 65K)"
echo "    - S10: ch2_ds3 (64×64×2 = 8K)"
echo "  L40s (S11-S12):"
echo "    - S11: ch8_ds6 (8×8×8 = 512)"
echo "    - S12: ch4_ds2 (128×128×4 = 65K)"
echo ""

