#!/bin/bash
# Find all best checkpoints in experiment directories
#
# Usage:
#   ./find_checkpoints.sh /path/to/experiments > checkpoints.txt
#   ./find_checkpoints.sh  # Uses default path

set -euo pipefail

EXPERIMENTS_DIR="${1:-/work3/s233249/ImgiNav/ImgiNav/experiments/diffusion/v2}"

echo "# Checkpoints found in: ${EXPERIMENTS_DIR}" >&2
echo "# Generated on: $(date)" >&2
echo "" >&2

count=0

# Find best_checkpoint.pt files
while IFS= read -r -d '' ckpt; do
    # Get relative info
    exp_dir=$(dirname "$(dirname "${ckpt}")")
    exp_name=$(basename "${exp_dir}")
    
    # Check file size (skip empty/corrupt)
    size=$(stat -f%z "${ckpt}" 2>/dev/null || stat -c%s "${ckpt}" 2>/dev/null || echo "0")
    if [ "${size}" -lt 1000 ]; then
        echo "# SKIP (too small): ${ckpt}" >&2
        continue
    fi
    
    echo "${ckpt}"
    ((count++))
    
done < <(find "${EXPERIMENTS_DIR}" -name "best_checkpoint.pt" -print0 2>/dev/null)

# If no best checkpoints found, look for latest
if [ ${count} -eq 0 ]; then
    echo "# No best_checkpoint.pt found, looking for checkpoint_*.pt" >&2
    
    while IFS= read -r -d '' checkpoints_dir; do
        # Find most recent checkpoint in each directory
        latest=$(ls -t "${checkpoints_dir}"/checkpoint_*.pt 2>/dev/null | head -1)
        if [ -n "${latest}" ]; then
            echo "${latest}"
            ((count++))
        fi
    done < <(find "${EXPERIMENTS_DIR}" -type d -name "checkpoints" -print0 2>/dev/null)
fi

echo "" >&2
echo "# Found ${count} checkpoints" >&2
