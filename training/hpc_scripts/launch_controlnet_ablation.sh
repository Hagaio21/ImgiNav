#!/bin/bash
# Launch ControlNet ablation experiments
# These train ControlNet adapters on frozen diffusion UNets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Launching ControlNet Ablation Experiments"
echo "=========================================="
echo ""

# V100 experiments (smaller models)
echo "Submitting to V100 queue (smaller models)..."
bsub < "${SCRIPT_DIR}/run_controlnet_unet32_d3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet48_d3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet48_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet64_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet64_d5.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet128_d3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet128_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet128_d5_rb3.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet32_d3_attn.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet64_d4_attn.sh"
sleep 2

echo ""
echo "Submitting to L40s queue (larger models)..."
# L40s experiments (larger models)
bsub < "${SCRIPT_DIR}/run_controlnet_unet256_d4.sh"
sleep 2
bsub < "${SCRIPT_DIR}/run_controlnet_unet256_d4_rb3.sh"
sleep 2

echo ""
echo "=========================================="
echo "All ControlNet experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor with: bjobs"
echo "Check logs: ${SCRIPT_DIR}/logs/controlnet_*.out"
echo ""
echo "Experiment breakdown:"
echo "  V100 (10 experiments):"
echo "    - controlnet_unet32_d3 (Small, depth 3)"
echo "    - controlnet_unet48_d3 (Small-medium, depth 3)"
echo "    - controlnet_unet48_d4 (Small-medium, depth 4)"
echo "    - controlnet_unet64_d4 (Medium, depth 4)"
echo "    - controlnet_unet64_d5 (Medium, depth 5)"
echo "    - controlnet_unet128_d3 (Medium-large, depth 3)"
echo "    - controlnet_unet128_d4 (Medium-large, depth 4) ⭐ BASELINE"
echo "    - controlnet_unet128_d5_rb3 (Medium-large, depth 5, 3 res blocks)"
echo "    - controlnet_unet32_d3_attn (Small with attention)"
echo "    - controlnet_unet64_d4_attn (Medium with attention)"
echo ""
echo "  L40s (2 experiments):"
echo "    - controlnet_unet256_d4 (Large, depth 4)"
echo "    - controlnet_unet256_d4_rb3 (Large, depth 4, 3 res blocks)"
echo ""
echo "Note: Make sure to update checkpoint paths in configs before training!"
echo "      Each config has PLACEHOLDER_PATH_TO_DIFFUSION_CHECKPOINT that needs updating."

