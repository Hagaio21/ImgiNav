#!/bin/bash
# Launch Stage 2 fine-tuning experiments for UNet128 variants
# These experiments fine-tune Stage 1 checkpoints with semantic losses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Stage 2 Diffusion Fine-tuning Experiments"
echo "Focus: UNet128 variants with semantic losses"
echo "=========================================="
echo ""

echo "Submitting Stage 2 experiments to V100 queue..."
echo ""

# Stage 2: UNet128 depth 4 (2 resblocks)
echo "  - stage2_unet128_d4 (depth 4, 2 resblocks)"
bsub < "${SCRIPT_DIR}/run_stage2_unet128_d4.sh"
sleep 2

# Stage 2: UNet128 depth 5 (3 resblocks)
echo "  - stage2_unet128_d5_rb3 (depth 5, 3 resblocks)"
bsub < "${SCRIPT_DIR}/run_stage2_unet128_d5_rb3.sh"
sleep 2

echo ""
echo "=========================================="
echo "All Stage 2 experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/stage2_*.out"
echo ""
echo ""
echo "Experiment breakdown:"
echo "  Stage 2 Fine-tuning (2 experiments):"
echo "    - stage2_unet128_d4 (depth 4, 2 resblocks)"
echo "      → Loads: capacity_unet128_d4 checkpoint"
echo "      → Fine-tunes with semantic losses"
echo ""
echo "    - stage2_unet128_d5_rb3 (depth 5, 3 resblocks)"
echo "      → Loads: capacity_unet128_d5_rb3 checkpoint"
echo "      → Fine-tunes with semantic losses"
echo ""
echo "Note: Both experiments use semantic losses (segmentation + perceptual)"
echo "      to ensure decoded layouts are viable."

