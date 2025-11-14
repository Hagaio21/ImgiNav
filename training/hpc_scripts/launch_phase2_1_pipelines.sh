#!/bin/bash
# Launch Phase 2.1 pipelines for all UNet sizes (32, 64, 128)
# Submits 3 separate jobs to LSF queue
#
# Each pipeline includes:
# 1. Autoencoder training (with per-channel latent standardization)
# 2. Dataset embedding
# 3. Diffusion training (500 steps, linear scheduler)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Phase 2.1 Pipelines"
echo "UNet sizes: 32, 64, 128"
echo "=========================================="
echo ""
echo "Submitting jobs to LSF queue..."
echo ""

# Submit UNet32 pipeline
echo "Submitting UNet32 pipeline..."
bsub < "${SCRIPT_DIR}/run_phase2_1_pipeline_unet32.sh"
sleep 2

# Submit UNet64 pipeline
echo "Submitting UNet64 pipeline..."
bsub < "${SCRIPT_DIR}/run_phase2_1_pipeline_unet64.sh"
sleep 2

# Submit UNet128 pipeline
echo "Submitting UNet128 pipeline..."
bsub < "${SCRIPT_DIR}/run_phase2_1_pipeline_unet128.sh"
sleep 2

echo ""
echo "=========================================="
echo "All 3 pipelines submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/phase2_1_pipeline_*.out"
echo ""
echo "Pipeline breakdown:"
echo "  - UNet32: base_channels=32, batch_size=128"
echo "  - UNet64: base_channels=64, batch_size=64"
echo "  - UNet128: base_channels=128, batch_size=64"
echo ""
echo "All pipelines use:"
echo "  - 500 noise steps with LinearScheduler"
echo "  - SNRWeightedNoiseLoss + LatentStructuralLoss"
echo "  - Same autoencoder (trained once, reused)"
echo "=========================================="

