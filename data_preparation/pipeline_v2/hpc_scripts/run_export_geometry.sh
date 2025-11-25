#!/bin/bash
#BSUB -J export_geometry[1-20]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/export_geometry_%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/export_geometry_%I.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Pipeline v2: Export Geometry Only (Fast - No Rendering)
# This script exports GLB files and metadata JSON for scenes.
# Run this first before rendering to create geometry files.

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2"
PROJECT_ROOT="/work3/s233249/ImgiNav"

N_SHARDS=20                                          # Must match [1-20] above
PYTHON_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/export_geometry.py"
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

ALL_LIST="${TMPDIR_LOCAL}/all_scenes_geometry_export.$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_geometry_export_"
SHARD_TXT=""                                        # will set below

echo "Starting Pipeline v2 geometry export - Task ${IDX}/${N_SHARDS}"
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
  exit 1
fi

echo "Sample of found files (first 3):"
echo "${FIND_RESULT}" | head -3 || true  # Ignore SIGPIPE from head

# Write scene list - handle pipe errors gracefully
echo "${FIND_RESULT}" | sort > "${ALL_LIST}" || {
  EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 141 ]; then
    # SIGPIPE from head, but file was written - this is OK
    echo "Note: Received SIGPIPE (exit 141) but scene list should be complete"
  else
    echo "ERROR: Failed to write scene list, exit code: ${EXIT_CODE}" >&2
    exit 1
  fi
}

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Wrote ${TOTAL_SCENES} scene paths to ${ALL_LIST}"

if [ ${TOTAL_SCENES} -eq 0 ]; then
  echo "ERROR: Scene list is empty" >&2
  exit 1
fi

# 2) Split into shards
echo "Splitting into ${N_SHARDS} shards..."
split -a 2 -n l/${IDX}/${N_SHARDS} "${ALL_LIST}" "${SHARD_PREFIX}" || {
  echo "ERROR: split command failed" >&2
  exit 1
}

SHARD_TXT="${SHARD_PREFIX}$(printf "%02d" $((IDX - 1)))"
if [ ! -f "${SHARD_TXT}" ]; then
  echo "ERROR: Shard file not found: ${SHARD_TXT}" >&2
  exit 1
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo "Shard file: ${SHARD_TXT}"

# 3) Activate conda environment
echo "Activating conda environment..."
if command -v conda &> /dev/null; then
  if conda env list | grep -q "^imginav "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate imginav
    echo "Activated conda environment: imginav"
  elif conda env list | grep -q "^scenefactor "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scenefactor
    echo "Activated conda environment: scenefactor"
  else
    echo "WARNING: Neither 'imginav' nor 'scenefactor' conda environment found" >&2
  fi
fi

# 4) Check Python dependencies
echo "Checking Python dependencies..."
python -c "import trimesh, numpy, json" || {
  echo "ERROR: Required Python packages not available" >&2
  exit 1
}
echo "All dependencies available"

# 5) Process each scene in the shard
echo "Starting processing at $(date)"
cd "${PROJECT_ROOT}/ImgiNav" || {
  echo "ERROR: Failed to change to project directory" >&2
  exit 1
}

SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r JSON_PATH; do
  if [ -z "${JSON_PATH}" ] || [ ! -f "${JSON_PATH}" ]; then
    echo "Warning: Scene file not found or invalid: ${JSON_PATH}"
    ((FAIL_COUNT++)) || true
    continue
  fi
  
  SCENE_ID=$(basename "${JSON_PATH}" .json)
  echo ""
  echo "=========================================="
  echo "Processing scene: ${SCENE_ID}"
  echo "=========================================="
  
  python "${PYTHON_SCRIPT}" \
    --scene_json "${JSON_PATH}" \
    --future_root "${MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" || {
    echo "ERROR: Failed to process scene ${SCENE_ID}" >&2
    ((FAIL_COUNT++)) || true
    continue
  }
  
  ((SUCCESS_COUNT++)) || true
  echo "✓ Successfully processed ${SCENE_ID}"
done < "${SHARD_TXT}"

echo ""
echo "Task ${IDX}/${N_SHARDS} completed at $(date)"
echo "Success: ${SUCCESS_COUNT}, Failed: ${FAIL_COUNT}"

