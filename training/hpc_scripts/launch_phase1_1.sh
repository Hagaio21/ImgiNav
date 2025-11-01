#!/bin/bash
# Launch script for Phase 1.1: Latent Channel Sweep
# Runs 4 experiments on V100 and 1 on L40s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.1: Latent Channel Sweep"
echo "=========================================="
echo "V100 jobs: 4 experiments (L1-L4)"
echo "L40s jobs: 1 experiment (L5)"
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit V100 jobs (4 experiments)
echo "Submitting V100 jobs..."
bsub < "${SCRIPT_DIR}/run_phase1_1_v100.sh"
V100_JOBID=$?
echo "V100 job submitted (if successful, check job queue)"

# Submit L40s jobs (1 experiment)
echo ""
echo "Submitting L40s jobs..."
bsub < "${SCRIPT_DIR}/run_phase1_1_l40s.sh"
L40S_JOBID=$?
echo "L40s job submitted (if successful, check job queue)"

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Check status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Phase 1.1 experiments:"
echo "  - phase1_1_AE_L1_ch4_base32.yaml (V100)"
echo "  - phase1_1_AE_L2_ch8_base32.yaml (V100)"
echo "  - phase1_1_AE_L3_ch16_base32.yaml (V100)"
echo "  - phase1_1_AE_L4_ch8_base64.yaml (V100)"
echo "  - phase1_1_AE_L5_ch4_base64.yaml (L40s)"
echo ""

