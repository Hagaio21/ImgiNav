#!/bin/bash
#BSUB -J build_graphs[1-20]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/build_graphs_%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/build_graphs_%I.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 2:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Pipeline v2: Build graphs from metadata
# Job array with maximum 20 jobs

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2"
PROJECT_ROOT="/work3/s233249/ImgiNav"

N_SHARDS=20                                          # Must match [1-20] above
PYTHON_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/build_graphs_from_metadata.py"
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

ALL_LIST="${TMPDIR_LOCAL}/all_scenes_graphs_$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_graphs_"
SHARD_TXT=""                                        # will set below

echo "Starting Pipeline v2 graph building - Task ${IDX}/${N_SHARDS}"
echo "Output dir: ${OUTPUT_DIR}"

# 1) Collect all scene IDs from geometry directory (scenes that have been exported)
echo "Finding scenes with metadata files in ${OUTPUT_DIR}/geometry..."
GEOMETRY_DIR="${OUTPUT_DIR}/geometry"
if [ ! -d "${GEOMETRY_DIR}" ]; then
  echo "ERROR: Geometry directory does not exist: ${GEOMETRY_DIR}" >&2
  echo "Run geometry export first!" >&2
  exit 1
fi

# Find all metadata files
FIND_RESULT=$(find "${GEOMETRY_DIR}" -type f -name '*_metadata.json' 2>&1)
FIND_EXIT=$?

if [ ${FIND_EXIT} -ne 0 ]; then
  echo "ERROR: find command failed with exit code ${FIND_EXIT}" >&2
  echo "find error output: ${FIND_RESULT}" >&2
  exit 1
fi

# Extract scene IDs from metadata filenames
echo "${FIND_RESULT}" | sed 's|.*/||' | sed 's|_metadata\.json$||' | sort > "${ALL_LIST}" || {
  EXIT_CODE=$?
  if [ ${EXIT_CODE} -eq 141 ]; then
    echo "Note: Received SIGPIPE (exit 141) but scene list should be complete"
  else
    echo "ERROR: Failed to write scene list, exit code: ${EXIT_CODE}" >&2
    exit 1
  fi
}

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} scenes with metadata files"

if [ ${TOTAL_SCENES} -eq 0 ]; then
  echo "ERROR: No scenes found with metadata files. Run geometry export first!" >&2
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

echo "Looking for shard file: ${SHARD_TXT}"

if [ ! -f "${SHARD_TXT}" ]; then
  echo "ERROR: shard file does not exist: ${SHARD_TXT}" >&2
  exit 2
fi

if [ ! -s "${SHARD_TXT}" ]; then
  echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
  exit 2
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"

# 4) Robust conda activation (non-interactive safe)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "WARNING: Failed to activate imginav, trying scenefactor..." >&2
    conda activate scenefactor || {
      echo "WARNING: Failed to activate scenefactor environment" >&2
    }
  }
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda activate imginav || {
    echo "WARNING: Failed to activate imginav, trying scenefactor..." >&2
    conda activate scenefactor || {
      echo "WARNING: Failed to activate scenefactor environment" >&2
    }
  }
fi

# 5) Check required dependencies
echo "Checking Python dependencies..."
python -c "import json, numpy" || {
  echo "ERROR: Required Python packages not available" >&2
  exit 1
}
echo "All dependencies available"

echo "Changing to project directory: ${PROJECT_ROOT}/ImgiNav"
cd "${PROJECT_ROOT}/ImgiNav" || {
  echo "ERROR: Failed to change to project directory" >&2
  exit 1
}

# 6) Process each scene in the shard
echo "Starting processing at $(date)"

SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r SCENE_ID; do
  if [ -z "${SCENE_ID}" ]; then
    continue
  fi
  
  echo ""
  echo "=========================================="
  echo "Processing scene: ${SCENE_ID}"
  echo "=========================================="
  
  # Check if metadata file exists
  METADATA_PATH="${OUTPUT_DIR}/geometry/${SCENE_ID}_metadata.json"
  
  if [ ! -f "${METADATA_PATH}" ]; then
    echo "WARNING: Metadata file not found: ${METADATA_PATH}, skipping"
    ((FAIL_COUNT++)) || true
    continue
  fi
  
  python "${PYTHON_SCRIPT}" \
    --scene_id "${SCENE_ID}" \
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

# 7) Cleanup temporary files
rm -f "${ALL_LIST}" "${SHARD_PREFIX}"*

echo "Pipeline v2 graph building complete for task ${IDX}"

