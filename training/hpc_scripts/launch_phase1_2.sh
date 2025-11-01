#!/bin/bash
# Launch script for Phase 1.2: Spatial Resolution Test
# Runs 2 experiments on V100 and 1 on L40s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.2: Spatial Resolution Test"
echo "=========================================="
echo "V100 jobs: 2 experiments (R1, R2)"
echo "L40s jobs: 1 experiment (R3)"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Using Phase 1.1 winner: latent_channels=16, base_channels=32"
echo ""

# Submit V100 jobs (2 experiments)
echo "Submitting V100 jobs..."
bsub < "${SCRIPT_DIR}/run_phase1_2_v100.sh"
V100_JOBID=$?
echo "V100 job submitted (if successful, check job queue)"

# Submit L40s jobs (1 experiment)
echo ""
echo "Submitting L40s jobs..."
bsub < "${SCRIPT_DIR}/run_phase1_2_l40s.sh"
L40S_JOBID=$?
echo "L40s job submitted (if successful, check job queue)"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Check status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Phase 1.2 experiments:"
echo "  - phase1_2_AE_R1_ds3.yaml (V100) - 64×64×16 latent"
echo "  - phase1_2_AE_R2_ds4.yaml (V100) - 32×32×16 latent"
echo "  - phase1_2_AE_R3_ds5.yaml (L40s) - 16×16×16 latent"
echo ""

