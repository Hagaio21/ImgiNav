#!/bin/bash
# Launch improved diffusion experiments (32-64 capacity range)
# Focus: Anti-memorization techniques for low-diversity layouts
#
# Improvements over baseline:
# - SNR-weighted noise loss (focuses on harder timesteps)
# - Latent structural loss (preserves spatial relationships)
# - Increased dropout (0.35)
# - Increased weight decay (0.2)
# - Reduced model capacity (32, 48, 64 base_channels)
# - Reduced diffusion steps (300 instead of 500)
# - Lower learning rate (0.00008)
# - Early stopping (patience=15)
# - Non-uniform timestep sampling (favors high-noise timesteps)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Improved Diffusion Experiments (5 total)"
echo "Focus: Anti-Memorization for Low-Diversity Layouts"
echo "=========================================="
echo ""

# Small to medium capacity experiments (32-64 range)
echo "Submitting capacity experiments (V100)..."
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet32_d3_improved.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet48_d3_improved.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet48_d4_improved.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d3_improved.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d4_improved.sh"

echo ""
echo "=========================================="
echo "All 5 experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/diffusion_*.out"
echo ""
echo ""
echo "Experiment breakdown:"
echo ""
echo "  Small Capacity (V100):"
echo "    - capacity_unet32_d3_improved (32 channels, depth 3)"
echo "    - capacity_unet48_d3_improved (48 channels, depth 3)"
echo "    - capacity_unet48_d4_improved (48 channels, depth 4)"
echo "    - capacity_unet64_d3_improved (64 channels, depth 3)"
echo "    - capacity_unet64_d4_improved (64 channels, depth 4)"
echo ""
echo "Anti-memorization improvements (all experiments):"
echo "  - SNR-weighted noise loss (focuses on harder timesteps)"
echo "  - Latent structural loss (weight: 0.2, preserves spatial relationships)"
echo "  - Dropout: 0.35 (increased from 0.2)"
echo "  - Weight decay: 0.2 (increased from 0.1)"
echo "  - Reduced diffusion steps: 300 (from 500)"
echo "  - Lower learning rate: 0.00008 (from 0.00015)"
echo "  - Early stopping: patience=15, min_delta=0.0001"
echo "  - Non-uniform timestep sampling (favors high-noise timesteps)"
echo "  - Reduced model capacity (num_res_blocks: 1 instead of 2)"
echo "  - More frequent evaluation: every 3 epochs"
echo ""

