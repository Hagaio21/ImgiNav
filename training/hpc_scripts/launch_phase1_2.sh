#!/bin/bash
# Launch script for Phase 1.2: VAE Test
# Runs 2 experiments: V1 (deterministic) on V100, V2 (VAE) on L40s

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.2: VAE Test"
echo "=========================================="
echo "V100 jobs: 1 experiment (V1 - Deterministic)"
echo "L40s jobs: 1 experiment (V2 - VAE)"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Using Phase 1.1 winner: 32×32×16 = 16,384 dims"
echo "  - latent_channels: 16"
echo "  - downsampling_steps: 4"
echo "  - base_channels: 32"
echo ""

# Submit V100 jobs (V1 - Deterministic)
echo "Submitting V100 job (V1 - Deterministic)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_2_v100.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    echo "✓ V100 job submitted: $JOBID"
    echo "  Running: phase1_2_AE_V1_deterministic"
else
    echo "✗ Failed to submit V100 job"
    echo "$OUTPUT"
fi

# Submit L40s jobs (V2 - VAE)
echo ""
echo "Submitting L40s job (V2 - VAE)..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_2_l40s.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    echo "✓ L40s job submitted: $JOBID"
    echo "  Running: phase1_2_AE_V2_vae_light"
else
    echo "✗ Failed to submit L40s job"
    echo "$OUTPUT"
fi

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Check status with: bjobs"
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Phase 1.2 experiments:"
echo "  - phase1_2_AE_V1_deterministic.yaml (V100) - Deterministic encoder"
echo "  - phase1_2_AE_V2_vae_light.yaml (L40s) - VAE with KL=0.0001"
echo ""

