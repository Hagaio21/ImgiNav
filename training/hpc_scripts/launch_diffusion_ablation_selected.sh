#!/bin/bash
# Launch selected 6 diffusion ablation experiments
# Focus: UNet capacity for frozen ControlNet backbone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching Selected Diffusion Ablation Experiments (6 total)"
echo "Focus: UNet Capacity for ControlNet"
echo "=========================================="
echo ""

# V100 experiments (4 total)
echo "Submitting to V100 queue..."
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet64_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet128_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet256_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_diffusion_capacity_unet128_d3.sh"
sleep 2

echo ""
echo "Submitting to L40s queue..."
# L40s experiments (2 total)
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
echo "  V100 (4 experiments):"
echo "    - capacity_unet64_d4 (Small, depth 4)"
echo "    - capacity_unet128_d4 (Medium, depth 4) ⭐ BASELINE"
echo "    - capacity_unet256_d4 (Large, depth 4)"
echo "    - capacity_unet128_d3 (Medium, depth 3)"
echo ""
echo "  L40s (2 experiments):"
echo "    - capacity_unet64_d5 (Small, depth 5)"
echo "    - scheduler_linear (Linear scheduler)"
echo ""
echo "Note: All capacity experiments use Cosine scheduler except scheduler_linear"

