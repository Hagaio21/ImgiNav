#!/bin/bash
# Launch selected 6 diffusion ablation experiments
# Focus: UNet capacity for frozen ControlNet backbone

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/diffusion/ablation"

echo "=========================================="
echo "Launching Selected Diffusion Ablation Experiments (6 total)"
echo "Focus: UNet Capacity for ControlNet"
echo "=========================================="
echo "Config directory: ${CONFIG_DIR}"
echo ""

# Selected experiments (6 total)
# Focus: UNet capacity for frozen ControlNet backbone
# Distribution: 4 on V100, 2 on L40s

V100_CONFIGS=(
    # UNet Capacity (4 experiments on V100)
    "capacity_unet64_d4.yaml"       # Small UNet (64 base, depth 4) - lower bound
    "capacity_unet128_d4.yaml"      # Medium UNet (128 base, depth 4) - BASELINE ⭐
    "capacity_unet256_d4.yaml"      # Large UNet (256 base, depth 4) - upper bound
    "capacity_unet128_d3.yaml"      # Medium UNet, shallow (128 base, depth 3)
)

L40S_CONFIGS=(
    # Remaining experiments on L40s
    "capacity_unet64_d5.yaml"       # Small UNet, deep (64 base, depth 5) - tests depth vs width
    "scheduler_linear.yaml"         # Alternative scheduler (others use cosine)
)

echo "Submitting experiments..."
echo "  V100 queue: ${#V100_CONFIGS[@]} experiments"
echo "  L40s queue: ${#L40S_CONFIGS[@]} experiments"
echo "  Total: $((${#V100_CONFIGS[@]} + ${#L40S_CONFIGS[@]})) experiments"
echo ""

# Submit to V100 (4 experiments)
echo "=========================================="
echo "Submitting to V100 queue..."
echo "=========================================="
for config in "${V100_CONFIGS[@]}"; do
    if [ -f "${CONFIG_DIR}/${config}" ]; then
        CONFIG_PATH="${CONFIG_DIR}/${config}"
        echo "  Submitting: ${config} (V100)"
        # Export config path and submit script with input redirection to read #BSUB directives
        export DIFFUSION_CONFIG="${CONFIG_PATH}"
        bsub < "${SCRIPT_DIR}/run_diffusion_ablation_v100.sh"
        unset DIFFUSION_CONFIG
        sleep 2  # Small delay to avoid overwhelming the scheduler
    else
        echo "  WARNING: Config not found: ${config}" >&2
    fi
done

echo ""
echo "=========================================="
echo "Submitting to L40s queue..."
echo "=========================================="
# Submit to L40s (2 experiments)
for config in "${L40S_CONFIGS[@]}"; do
    if [ -f "${CONFIG_DIR}/${config}" ]; then
        CONFIG_PATH="${CONFIG_DIR}/${config}"
        echo "  Submitting: ${config} (L40s)"
        # Export config path and submit script with input redirection to read #BSUB directives
        export DIFFUSION_CONFIG="${CONFIG_PATH}"
        bsub < "${SCRIPT_DIR}/run_diffusion_ablation_l40s.sh"
        unset DIFFUSION_CONFIG
        sleep 2  # Small delay to avoid overwhelming the scheduler
    else
        echo "  WARNING: Config not found: ${config}" >&2
    fi
done

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

