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
echo "Found ${JSON_COUNT} JSON files"

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
echo "${FIND_RESULT}" | head -3 || true  # Ignore SIGPIPE from head

# Write scene list - handle pipe errors gracefully
echo "Writing scene list to ${ALL_LIST}..."
# Use set +e temporarily to avoid SIGPIPE issues
set +e
echo "${FIND_RESULT}" | grep -v '^$' | sort > "${ALL_LIST}" 2>&1
WRITE_EXIT=$?
set -e

if [ ${WRITE_EXIT} -ne 0 ] && [ ${WRITE_EXIT} -ne 141 ]; then
  echo "ERROR: Failed to write scene list to ${ALL_LIST} (exit code: ${WRITE_EXIT})" >&2
  echo "Temp directory: ${TMPDIR_LOCAL}" >&2
  echo "Temp directory exists: $([ -d "${TMPDIR_LOCAL}" ] && echo 'yes' || echo 'no')" >&2
  echo "Temp directory writable: $([ -w "${TMPDIR_LOCAL}" ] && echo 'yes' || echo 'no')" >&2
  echo "Disk space:" >&2
  df -h "${TMPDIR_LOCAL}" >&2
  exit 1
fi

# Verify file was written
if [ ! -f "${ALL_LIST}" ]; then
  echo "ERROR: Scene list file was not created: ${ALL_LIST}" >&2
  exit 1
fi

FILE_SIZE=$(wc -l < "${ALL_LIST}")
echo "Wrote scene list to ${ALL_LIST}"
echo "Total scenes: ${FILE_SIZE}"
if [ ${FILE_SIZE} -eq 0 ]; then
  echo "ERROR: Scene list file is empty!" >&2
  exit 1
fi

echo "First few lines of scene list:"
head -3 "${ALL_LIST}"

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} total scene files"

if [ ${TOTAL_SCENES} -eq 0 ]; then
  echo "ERROR: No scene files found!" >&2
  exit 1
fi

# 2) Split into balanced shards using GNU split
echo "Splitting into ${N_SHARDS} shards..."
echo "Input file: ${ALL_LIST}"
echo "Input file size: $(wc -l < "${ALL_LIST}") lines"
echo "Shard prefix: ${SHARD_PREFIX}"

split -d -n l/${N_SHARDS} "${ALL_LIST}" "${SHARD_PREFIX}" || {
  echo "ERROR: Failed to split scene list" >&2
  echo "Split command: split -d -n l/${N_SHARDS} ${ALL_LIST} ${SHARD_PREFIX}" >&2
  echo "Checking if split command exists:" >&2
  which split >&2
  exit 1
}

echo "Split completed, checking shard files..."
ls -lh "${SHARD_PREFIX}"* 2>&1 || echo "No shard files found" >&2

# 3) Pick this task's shard file
SUFFIX=$(printf "%02d" $((IDX-1)))
SHARD_TXT="${SHARD_PREFIX}${SUFFIX}"

echo "Looking for shard file: ${SHARD_TXT}"

# Safety: ensure shard not empty
if [ ! -f "${SHARD_TXT}" ]; then
  echo "ERROR: shard file does not exist: ${SHARD_TXT}" >&2
  echo "Available shard files:" >&2
  ls -la "${SHARD_PREFIX}"* >&2 || echo "No shard files found" >&2
  exit 2
fi

if [ ! -s "${SHARD_TXT}" ]; then
  echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
  echo "File exists but is empty. Size: $(wc -c < "${SHARD_TXT}") bytes" >&2
  exit 2
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo "Shard file: ${SHARD_TXT}"
echo "First scene in shard: $(head -1 "${SHARD_TXT}")"

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

