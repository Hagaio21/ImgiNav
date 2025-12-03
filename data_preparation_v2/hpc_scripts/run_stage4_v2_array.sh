#!/bin/bash
#BSUB -J stage4v2[1-500]
#BSUB -o logs/stage4v2_%J_%I.out
#BSUB -e logs/stage4v2_%J_%I.err
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"

# Stage 4 v2: Improved POV Rendering with Layout Rotation
# ========================================================
# This script:
# 1. Renders POV images with improved camera (stepped back, wider FOV)
# 2. Rotates existing layouts to match POV orientation (not re-rendering!)
# 3. Generates POV-normalized graphs with descriptive naming
# 
# Outputs (per scene):
# - povs/tex/{scene}_{room}_{pov}_tex_pov.png
# - povs/seg/{scene}_{room}_{pov}_seg_pov.png
# - layouts_pov/tex/{scene}_{room}_{pov}_tex_layout.png
# - layouts_pov/seg/{scene}_{room}_{pov}_seg_layout.png
# - graphs/jsons/{scene}_{room}_{pov}_room_graph.json
# - graphs/texts/{scene}_{room}_{pov}_room_description.txt
#
# Outputs (per shard):
# - povs/pov_info_shard_XXXX.json
#
# After all shards complete, merge with:
#   cat povs/pov_info_shard_*.json | jq -s 'add' > povs/pov_info.json
#   OR use: python -c "import json, glob; data=[x for f in glob.glob('povs/pov_info_shard_*.json') for x in json.load(open(f))]; json.dump(data, open('povs/pov_info.json','w'))"

set -e

# Configuration
DATASET_ROOT="${DATASET_ROOT:-/path/to/structured3d}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$(dirname "$0")/..}"
SHARDS_DIR="${SHARDS_DIR:-${DATASET_ROOT}/shards}"
NUM_SHARDS="${NUM_SHARDS:-500}"

# Camera parameters (adjustable)
FOV="${FOV:-80.0}"
STEP_BACK="${STEP_BACK:-0.8}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1.6}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"

# Create logs directory
mkdir -p logs

# Get shard file for this array task (LSB_JOBINDEX is 1-based, convert to 0-based for shard ID)
SHARD_ID=$(printf "%04d" $((LSB_JOBINDEX - 1)))
SHARD_FILE="${SHARDS_DIR}/shard_${SHARD_ID}.txt"

if [[ ! -f "$SHARD_FILE" ]]; then
    echo "Shard file not found: $SHARD_FILE (this is OK if fewer shards exist)"
    exit 0
fi

SCENE_COUNT=$(wc -l < "$SHARD_FILE")
echo "=========================================="
echo "Stage 4 v2: Improved POV + Layout Rotation"
echo "=========================================="
echo "Shard: ${SHARD_ID} / ${NUM_SHARDS}"
echo "Scenes: ${SCENE_COUNT}"
echo "Dataset: ${DATASET_ROOT}"
echo "Camera: FOV=${FOV}°, step_back=${STEP_BACK}m, height=${CAMERA_HEIGHT}m"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "=========================================="

# Load modules
module load python/3.10 2>/dev/null || true
module load cuda/11.8 2>/dev/null || true

# Activate virtual environment if exists
if [[ -f "${SCRIPTS_DIR}/venv/bin/activate" ]]; then
    source "${SCRIPTS_DIR}/venv/bin/activate"
fi

# Run stage 4 v2
python "${SCRIPTS_DIR}/stage4_render_povs_v2.py" \
    --dataset-root "${DATASET_ROOT}" \
    --scene-list "${SHARD_FILE}" \
    --shard-id "${SHARD_ID}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --fov "${FOV}" \
    --step-back "${STEP_BACK}" \
    --camera-height "${CAMERA_HEIGHT}" \
    --hpc \
    --skip-existing

EXIT_CODE=$?

echo "=========================================="
echo "Stage 4 v2 shard ${SHARD_ID} completed with exit code: ${EXIT_CODE}"
echo "Output: povs/pov_info_shard_${SHARD_ID}.json"
echo "=========================================="

exit ${EXIT_CODE}
