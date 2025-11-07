#!/bin/bash
# Launch Stage 2 fine-tuning experiments for UNet32 and UNet64 variants
# These experiments fine-tune Stage 1 checkpoints with semantic losses

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Stage 2 Diffusion Fine-tuning Experiments"
echo "Focus: UNet32 and UNet64 variants with semantic losses"
echo "=========================================="
echo ""

echo "Submitting Stage 2 experiments to V100 queue..."
echo ""

# Stage 2: UNet32 depth 3
echo "  - stage2_unet32_d3 (base_channels 32, depth 3, 2 resblocks)"
bsub < "${SCRIPT_DIR}/run_stage2_unet32_d3.sh"
sleep 2

# Stage 2: UNet64 depth 4
echo "  - stage2_unet64_d4 (base_channels 64, depth 4, 2 resblocks)"
bsub < "${SCRIPT_DIR}/run_stage2_unet64_d4.sh"
sleep 2

# Stage 2: UNet64 depth 5
echo "  - stage2_unet64_d5 (base_channels 64, depth 5, 2 resblocks)"
bsub < "${SCRIPT_DIR}/run_stage2_unet64_d5.sh"
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
echo "  Stage 2 Fine-tuning (3 experiments):"
echo "    - stage2_unet32_d3 (base_channels 32, depth 3, 2 resblocks)"
echo "      → Loads: capacity_unet32_d3 checkpoint"
echo "      → Fine-tunes with semantic losses"
echo ""
echo "    - stage2_unet64_d4 (base_channels 64, depth 4, 2 resblocks)"
echo "      → Loads: capacity_unet64_d4 checkpoint"
echo "      → Fine-tunes with semantic losses"
echo ""
echo "    - stage2_unet64_d5 (base_channels 64, depth 5, 2 resblocks)"
echo "      → Loads: capacity_unet64_d5 checkpoint"
echo "      → Fine-tunes with semantic losses"
echo ""
echo "Note: All experiments use semantic losses (segmentation + perceptual)"
echo "      to ensure decoded layouts are viable."

