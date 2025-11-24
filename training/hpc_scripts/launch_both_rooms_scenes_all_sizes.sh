#!/bin/bash
# Launch script for both rooms and scenes experiments with full cross attention
# Runs small, medium, and large sizes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCH_SCRIPT="${SCRIPT_DIR}/launch_train_diff_clip.sh"

# Configs to launch (both rooms and scenes = regular/ directory)
CONFIGS=(
    "experiments/diffusion/clip/regular/small_all.yaml"
    "experiments/diffusion/clip/regular/medium_all.yaml"
    "experiments/diffusion/clip/regular/large_all.yaml"
)

echo "=============================================================================="
echo "Launching Both Rooms and Scenes Experiments (Full Cross Attention)"
echo "=============================================================================="
echo ""
echo "Configs to launch:"
for config in "${CONFIGS[@]}"; do
    echo "  - ${config}"
done
echo ""
echo "=============================================================================="

# Make launch script executable
chmod +x "${LAUNCH_SCRIPT}"

# Launch all configs
bash "${LAUNCH_SCRIPT}" "${CONFIGS[@]}"

