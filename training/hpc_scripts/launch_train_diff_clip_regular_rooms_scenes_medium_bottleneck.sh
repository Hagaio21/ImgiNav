#!/bin/bash
# Launch script - just bsubs the two run scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting 2 jobs..."
bsub < "${SCRIPT_DIR}/run_train_diff_clip_regular_rooms_medium_bottleneck.sh"
sleep 1
bsub < "${SCRIPT_DIR}/run_train_diff_clip_regular_scenes_medium_bottleneck.sh"
echo "Done!"

