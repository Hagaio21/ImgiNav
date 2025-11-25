#!/bin/bash
#BSUB -J render[1-20]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/render_%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/render_%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 07:00
#BSUB -q hpc

set -euo pipefail

# Pipeline v2: Render 3D-FRONT scenes to layouts and POVs
# Job array with maximum 20 jobs

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2"
PROJECT_ROOT="/work3/s233249/ImgiNav"

N_SHARDS=20                                          # Must match [1-20] above
PYTHON_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/render_worker.py"
# =============================================================================

# Create logs directory
mkdir -p "${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs"

IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
TMPDIR_LOCAL="${TMPDIR:-/tmp}"
ALL_LIST="${TMPDIR_LOCAL}/all_scenes_pipeline_v2.$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_pipeline_v2_"
SHARD_TXT=""                                        # will set below

echo "Starting Pipeline v2 rendering - Task ${IDX}/${N_SHARDS}"
echo "Scenes root: ${SCENES_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

# 1) Collect all *.json scene files, deterministic order
echo "Finding scene files in ${SCENES_ROOT}..."
if [ ! -d "${SCENES_ROOT}" ]; then
  echo "ERROR: SCENES_ROOT directory does not exist: ${SCENES_ROOT}" >&2
  exit 1
fi

find "${SCENES_ROOT}" -type f -name '*.json' -print | sort > "${ALL_LIST}" || {
  echo "ERROR: Failed to find scene files" >&2
  exit 1
}

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} total scene files"

if [ ${TOTAL_SCENES} -eq 0 ]; then
  echo "ERROR: No scene files found!" >&2
  exit 1
fi

# 2) Split into balanced shards using GNU split
echo "Splitting into ${N_SHARDS} shards..."
split -d -n l/${N_SHARDS} "${ALL_LIST}" "${SHARD_PREFIX}" || {
  echo "ERROR: Failed to split scene list" >&2
  exit 1
}

# 3) Pick this task's shard file
SUFFIX=$(printf "%02d" $((IDX-1)))
SHARD_TXT="${SHARD_PREFIX}${SUFFIX}"

# Safety: ensure shard not empty
if [ ! -s "${SHARD_TXT}" ]; then
  echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
  exit 2
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"

# 4) Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# 5) Robust conda activation (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate scenefactor || {
    echo "WARNING: Failed to activate scenefactor environment" >&2
  }
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda activate scenefactor || {
    echo "WARNING: Failed to activate scenefactor environment" >&2
  }
fi

# 6) Check if required files exist before processing
if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: taxonomy.json not found at: ${TAXONOMY_FILE}" >&2
  exit 1
fi

# 7) Check required dependencies
echo "Checking Python dependencies..."
python -c "import trimesh, pyrender, numpy, scipy, PIL" || {
  echo "ERROR: Required Python packages not available" >&2
  echo "Trying to import individually to identify missing package..." >&2
  python -c "import trimesh" || echo "  - trimesh missing" >&2
  python -c "import pyrender" || echo "  - pyrender missing" >&2
  python -c "import numpy" || echo "  - numpy missing" >&2
  python -c "import scipy" || echo "  - scipy missing" >&2
  python -c "import PIL" || echo "  - PIL missing" >&2
  exit 1
}
echo "All dependencies available"

# 8) Process each scene in the shard
echo "Starting processing at $(date)"

echo "Changing to project directory: ${PROJECT_ROOT}/ImgiNav"
cd "${PROJECT_ROOT}/ImgiNav" || {
  echo "ERROR: Failed to change to project directory: ${PROJECT_ROOT}/ImgiNav" >&2
  exit 1
}

echo "Current directory: $(pwd)"
echo "Python script path: ${PYTHON_SCRIPT}"
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi
echo "Python script exists and is readable"

while IFS= read -r JSON_PATH; do
  if [ -z "${JSON_PATH}" ] || [ ! -f "${JSON_PATH}" ]; then
    echo "Warning: Scene file not found or invalid: ${JSON_PATH}"
    continue
  fi
  
  SCENE_ID=$(basename "${JSON_PATH}" .json)
  echo "Processing scene: ${SCENE_ID}"
  
  python "${PYTHON_SCRIPT}" \
    --scene_json "${JSON_PATH}" \
    --future_root "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --num_povs 6 \
    --seed 42 || {
    echo "ERROR: Failed to process scene ${SCENE_ID}" >&2
    continue
  }
  
  echo "✓ Successfully processed ${SCENE_ID}"
done < "${SHARD_TXT}"

echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"

# 9) Cleanup temporary files
rm -f "${ALL_LIST}" "${SHARD_PREFIX}"*

echo "Pipeline v2 rendering complete for task ${IDX}"

