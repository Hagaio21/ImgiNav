#!/bin/bash
# Launch diffusion ablation experiments (32-64 capacity range + linear scheduler)
# Focus: Capacity ablation with memorization checks and anti-memorization measures
#
# Anti-memorization measures:
# - Timestep importance sampling (weights higher timesteps)
# - Increased dropout (0.2)
# - Increased weight decay (0.1)
# - Reduced network capacity (32, 48, 64 base_channels)
# - Memorization checks every 5 epochs
# - Reduced num_steps (500 instead of 1000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Diffusion Ablation Experiments (6 total)"
echo "Focus: Capacity Ablation (32-64) + Memorization Prevention"
echo "=========================================="
echo ""

# Small to medium capacity experiments (32-64 range)
echo "Submitting capacity experiments (V100)..."
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet32_d3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet48_d3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet48_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d4.sh"
sleep 2

echo ""
echo "Submitting to L40s queue..."
# L40s experiments
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d5.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_scheduler_linear.sh"

echo ""
echo "=========================================="
echo "All 6 experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/diffusion_ablation.*.out"
echo ""
echo ""
echo "Experiment breakdown:"
echo ""
echo "  Small Capacity (V100):"
echo "    - capacity_unet32_d3 (32 channels, depth 3)"
echo "    - capacity_unet48_d3 (48 channels, depth 3)"
echo "    - capacity_unet48_d4 (48 channels, depth 4)"
echo "    - capacity_unet64_d4 (64 channels, depth 4)"
echo ""
echo "  L40s Queue:"
echo "    - capacity_unet64_d5 (64 channels, depth 5)"
echo "    - scheduler_linear (Linear scheduler, 48 channels, depth 4)"
echo ""
echo "Anti-memorization measures (all experiments):"
echo "  - Timestep importance sampling (weights higher timesteps)"
echo "  - Dropout: 0.2 (increased from 0.1)"
echo "  - Weight decay: 0.1 (increased from 0.05)"
echo "  - Memorization checks every 5 epochs"
echo "  - Reduced num_steps: 500 (faster training)"
echo "  - All use Cosine scheduler (except scheduler_linear)"
echo ""
