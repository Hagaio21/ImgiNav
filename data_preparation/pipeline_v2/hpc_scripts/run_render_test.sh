#!/bin/bash
#BSUB -J render_test[1-2]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/render_test_%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/render_test_%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 07:00
#BSUB -q hpc

set -euo pipefail

# Pipeline v2: Test rendering for 10 scenes
# Job array with 2 jobs for testing

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2_test"
PROJECT_ROOT="/work3/s233249/ImgiNav"

N_SHARDS=2                                           # Must match [1-2] above
MAX_SCENES=10                                        # Limit to 10 scenes for testing
PYTHON_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/render_worker.py"
# =============================================================================

# Create logs directory
mkdir -p "${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs"

IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
TMPDIR_LOCAL="${TMPDIR:-/tmp}"

# Ensure temp directory exists and is writable
if [ ! -d "${TMPDIR_LOCAL}" ] || [ ! -w "${TMPDIR_LOCAL}" ]; then
  echo "WARNING: TMPDIR ${TMPDIR_LOCAL} not accessible, using /tmp instead" >&2
  TMPDIR_LOCAL="/tmp"
fi

# Create temp directory if it doesn't exist
mkdir -p "${TMPDIR_LOCAL}" || {
  echo "ERROR: Cannot create or access temp directory: ${TMPDIR_LOCAL}" >&2
  exit 1
}

ALL_LIST="${TMPDIR_LOCAL}/all_scenes_pipeline_v2_test.$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_pipeline_v2_test_"
SHARD_TXT=""                                        # will set below

echo "Starting Pipeline v2 TEST rendering - Task ${IDX}/${N_SHARDS}"
echo "Scenes root: ${SCENES_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "MAX_SCENES: ${MAX_SCENES}"

# 1) Collect all *.json scene files, deterministic order, limit to MAX_SCENES
echo "Finding scene files in ${SCENES_ROOT}..."
if [ ! -d "${SCENES_ROOT}" ]; then
  echo "ERROR: SCENES_ROOT directory does not exist: ${SCENES_ROOT}" >&2
  exit 1
fi

echo "SCENES_ROOT exists, searching for JSON files..."
echo "Searching in: ${SCENES_ROOT}"
FIND_RESULT=$(find "${SCENES_ROOT}" -type f -name '*.json' 2>&1)
FIND_EXIT=$?

if [ ${FIND_EXIT} -ne 0 ]; then
  echo "ERROR: find command failed with exit code ${FIND_EXIT}" >&2
  echo "find error output: ${FIND_RESULT}" >&2
  exit 1
fi

JSON_COUNT=$(echo "${FIND_RESULT}" | grep -v '^$' | wc -l)
echo "Found ${JSON_COUNT} JSON files (before limiting)"

if [ ${JSON_COUNT} -eq 0 ]; then
  echo "ERROR: No JSON files found in ${SCENES_ROOT}" >&2
  echo "Trying to list directory contents..." >&2
  echo "Top-level contents:" >&2
  ls -la "${SCENES_ROOT}" | head -20 >&2
  echo "" >&2
  echo "Looking for subdirectories..." >&2
  find "${SCENES_ROOT}" -maxdepth 2 -type d | head -10 >&2
  echo "" >&2
  echo "Looking for any JSON files recursively (showing first 5):" >&2
  find "${SCENES_ROOT}" -type f -name '*.json' 2>&1 | head -5 >&2
  exit 1
fi

echo "Sample of found files (first 3):"
echo "${FIND_RESULT}" | head -3

echo "${FIND_RESULT}" | grep -v '^$' | sort | head -n ${MAX_SCENES} > "${ALL_LIST}" || {
  echo "ERROR: Failed to write scene list to ${ALL_LIST}" >&2
  echo "Temp directory: ${TMPDIR_LOCAL}" >&2
  echo "Temp directory exists: $([ -d "${TMPDIR_LOCAL}" ] && echo 'yes' || echo 'no')" >&2
  echo "Temp directory writable: $([ -w "${TMPDIR_LOCAL}" ] && echo 'yes' || echo 'no')" >&2
  echo "Disk space:" >&2
  df -h "${TMPDIR_LOCAL}" >&2
  exit 1
}

echo "Wrote ${MAX_SCENES} scene paths to ${ALL_LIST}"
echo "First few lines of scene list:"
head -3 "${ALL_LIST}"

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} total scene files (limited to ${MAX_SCENES} for testing)"

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

SCENE_COUNT=0
while IFS= read -r JSON_PATH; do
  if [ -z "${JSON_PATH}" ] || [ ! -f "${JSON_PATH}" ]; then
    echo "Warning: Scene file not found or invalid: ${JSON_PATH}"
    continue
  fi
  
  SCENE_ID=$(basename "${JSON_PATH}" .json)
  SCENE_COUNT=$((SCENE_COUNT + 1))
  echo "=========================================="
  echo "Processing scene ${SCENE_COUNT}/${SHARD_COUNT}: ${SCENE_ID}"
  echo "=========================================="
  
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
  echo ""
done < "${SHARD_TXT}"

echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"
echo "Processed ${SCENE_COUNT} scenes"

# 9) Cleanup temporary files
rm -f "${ALL_LIST}" "${SHARD_PREFIX}"*

echo "Pipeline v2 TEST rendering complete for task ${IDX}"

