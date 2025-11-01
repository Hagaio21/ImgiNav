#!/bin/bash
# Combined launch script for Phase 1.2 (VAE Test) + Phase 1.3 (Loss Tuning)
# Total: 8 experiments (2 from Phase 1.2 + 6 from Phase 1.3 with both VAE/deterministic)
# All experiments run as a single job array on V100 queue

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Launching Phase 1.2 + Phase 1.3 Combined (All 8 experiments)"
echo "=========================================="
echo ""
echo "Phase 1.2: VAE Test (2 experiments)"
echo "  [1] V1 - Deterministic"
echo "  [2] V2 - VAE"
echo ""
echo "Phase 1.3: Loss Tuning (6 experiments)"
echo "  [3] F1 - RGB only (Deterministic)"
echo "  [4] F2 - RGB + Seg (Deterministic)"
echo "  [5] F3 - RGB + Seg + Cls (Deterministic)"
echo "  [6] F1_vae - RGB only (VAE)"
echo "  [7] F2_vae - RGB + Seg (VAE)"
echo "  [8] F3_vae - RGB + Seg + Cls (VAE)"
echo ""
echo "Using Phase 1.1 winner: 32×32×16 = 16,384 dims"
echo "  - latent_channels: 16"
echo "  - downsampling_steps: 4"
echo "  - base_channels: 32"
echo ""
echo "All 8 experiments will run as a single job array on V100 queue"
echo "Log directory: ${LOG_DIR}"
echo ""

# Submit single job array for all 8 experiments
echo "Submitting job array for all 8 experiments..."
OUTPUT=$(bsub < "${SCRIPT_DIR}/run_phase1_2_and_1_3_all.sh" 2>&1)
if [[ $? -eq 0 ]]; then
    JOBID=$(echo "$OUTPUT" | grep -oP '<\d+>' | tr -d '<>')
    echo "✓ Job array submitted: $JOBID"
    echo "  Will run all 8 experiments as slots become available"
else
    echo "✗ Failed to submit job array"
    echo "$OUTPUT"
    exit 1
fi

echo ""
echo "=========================================="
echo "Job array submitted successfully!"
echo "=========================================="
echo "Job ID: $JOBID"
echo "Check status with: bjobs $JOBID"
echo "Monitor logs in: ${LOG_DIR}"
echo ""
echo "Experiments in array:"
echo "  [1] phase1_2_AE_V1_deterministic.yaml"
echo "  [2] phase1_2_AE_V2_vae_light.yaml"
echo "  [3] phase1_3_AE_F1_rgb_only.yaml"
echo "  [4] phase1_3_AE_F2_rgb_seg.yaml"
echo "  [5] phase1_3_AE_F3_rgb_seg_cls.yaml"
echo "  [6] phase1_3_AE_F1_rgb_only_vae.yaml"
echo "  [7] phase1_3_AE_F2_rgb_seg_vae.yaml"
echo "  [8] phase1_3_AE_F3_rgb_seg_cls_vae.yaml"
echo ""
echo "All experiments will run overnight and should be complete by morning."
echo ""

