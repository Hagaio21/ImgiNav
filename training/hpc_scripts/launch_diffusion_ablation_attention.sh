#!/bin/bash
# Launch Stage 1 diffusion ablation experiments with attention
# These experiments test self-attention blocks in UNet architecture

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Diffusion Ablation Experiments"
echo "Focus: UNet with Self-Attention (Stage 1)"
echo "=========================================="
echo ""

echo "Submitting attention experiments to V100 queue..."
echo ""

# UNet32 with attention (depth 3)
echo "  - capacity_unet32_d3_attn (base_channels=32, depth=3, with attention)"
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet32_d3_attn.sh"
sleep 2

# UNet64 with attention (depth 4)
echo "  - capacity_unet64_d4_attn (base_channels=64, depth=4, with attention)"
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d4_attn.sh"
sleep 2

echo ""
echo "=========================================="
echo "All attention experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/diffusion_capacity_unet*_attn.*.out"
echo ""
echo ""
echo "Experiment breakdown:"
echo "  Stage 1 with Attention (2 experiments):"
echo "    - capacity_unet32_d3_attn (base_channels=32, depth=3)"
echo "      → Tests attention impact on smaller model"
echo "      → Attention at: bottleneck, downs, ups"
echo ""
echo "    - capacity_unet64_d4_attn (base_channels=64, depth=4)"
echo "      → Tests attention impact on medium model"
echo "      → Attention at: bottleneck, downs, ups"
echo ""
echo "Note: These experiments compare against baseline models without attention"
echo "      to measure the improvement from self-attention blocks."

