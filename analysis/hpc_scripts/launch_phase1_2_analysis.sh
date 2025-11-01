#!/bin/bash
# Launch script for Phase 1.2 Analysis
# Runs analysis script on GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.2 Analysis"
echo "=========================================="
echo "This will:"
echo "  1. Load metrics from all 3 experiments"
echo "  2. Create comparison plots and analysis"
echo "  3. Load autoencoder checkpoints"
echo "  4. Run visual comparisons on test samples"
echo ""
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit analysis job
echo "Submitting analysis job (V100 GPU)..."
JOB_OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_2_analysis.sh" 2>&1)
EXIT_CODE=$?
echo "${JOB_OUTPUT}"

# Extract job ID if available (LSF format: "Job <12345> is submitted...")
JOB_ID=$(echo "${JOB_OUTPUT}" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*' | head -1)

echo ""
echo "=========================================="
if [ -n "${JOB_ID}" ]; then
    echo "Analysis job submitted successfully!"
    echo "Job ID: ${JOB_ID}"
else
    echo "Analysis job submission attempted"
fi
echo "=========================================="
echo "Check status with: bjobs"
if [ -n "${JOB_ID}" ]; then
    echo "Or specifically: bjobs ${JOB_ID}"
fi
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Results will be saved to:"
echo "  outputs/phase1_2_spatial_resolution/analysis/"
echo ""

