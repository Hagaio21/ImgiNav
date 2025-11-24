#!/bin/bash
# Launch script for medium size experiments with full cross attention
# Runs rooms, scenes, and both (regular)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_train_diff_clip.sh"

# Configs to launch (medium size with full cross attention)
CONFIGS=(
    "experiments/diffusion/clip/regular_rooms/medium_all.yaml"
    "experiments/diffusion/clip/regular_scenes/medium_all.yaml"
    "experiments/diffusion/clip/regular/medium_all.yaml"
)

echo "=============================================================================="
echo "Launching Medium Size Experiments with Full Cross Attention"
echo "=============================================================================="
echo ""
echo "Configs to launch:"
for config in "${CONFIGS[@]}"; do
    if [ -f "${BASE_DIR}/${config}" ]; then
        echo "  - ${config} ✓"
    else
        echo "  - ${config} ✗ (NOT FOUND - will be skipped)"
    fi
done
echo ""
echo "=============================================================================="

# Make launch script executable
chmod +x "${LAUNCH_SCRIPT}"

# Launch all configs
bash "${LAUNCH_SCRIPT}" "${CONFIGS[@]}"

