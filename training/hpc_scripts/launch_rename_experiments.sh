#!/bin/bash
# Launch script to rename existing experiments on HPC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
RENAME_SCRIPT="${SCRIPT_DIR}/rename_existing_experiments.sh"

echo "=============================================================================="
echo "Launching Experiment Rename Job"
echo "=============================================================================="
echo "This will rename the following experiments:"
echo "  - diff_clip_regular_medium_all -> diff_clip_regular_both_medium_all"
echo "  - diff_clip_regular_rooms_large_bottleneck -> diff_clip_regular_rooms_large_down_bottleneck"
echo "  - diff_clip_regular_rooms_medium_bottleneck -> diff_clip_regular_rooms_medium_down_bottleneck"
echo "  - diff_clip_regular_rooms_small_bottleneck -> diff_clip_regular_rooms_small_down_bottleneck"
echo "  - diff_clip_regular_scenes_large_bottleneck -> diff_clip_regular_scenes_large_down_bottleneck"
echo "  - diff_clip_regular_scenes_medium_bottleneck -> diff_clip_regular_scenes_medium_down_bottleneck"
echo "  - diff_clip_regular_scenes_small_bottleneck -> diff_clip_regular_scenes_small_down_bottleneck"
echo "  - diff_clip_regular_small_all -> diff_clip_regular_both_small_all"
echo ""
echo "Submitting job..."

bsub -J "rename_experiments" \
    -o "${BASE_DIR}/training/hpc_scripts/logs/launch_rename_experiments.%J.out" \
    -e "${BASE_DIR}/training/hpc_scripts/logs/launch_rename_experiments.%J.err" \
    -n 1 \
    -R "rusage[mem=2000]" \
    -W 1:00 \
    -q normal \
    bash "${RENAME_SCRIPT}"

echo "Job submitted! Check logs for progress."

