#!/bin/bash
#BSUB -J export_geometry_test[1-2]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/export_geometry_test_%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs/export_geometry_test_%I.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 1:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Pipeline v2: Export Geometry Only - TEST VERSION
# Tests geometry export on 10 scenes

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/dataset_v2_test"
PROJECT_ROOT="/work3/s233249/ImgiNav"

MAX_SCENES=10                                        # Limit for testing
N_SHARDS=2                                           # Must match [1-2] above
PYTHON_SCRIPT="${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/export_geometry.py"
# =============================================================================

# Create logs directory
mkdir -p "${PROJECT_ROOT}/ImgiNav/data_preparation/pipeline_v2/hpc_scripts/logs"

IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
TMPDIR_LOCAL="${TMPDIR:-/tmp}"

if [ ! -d "${TMPDIR_LOCAL}" ] || [ ! -w "${TMPDIR_LOCAL}" ]; then
  echo "WARNING: TMPDIR ${TMPDIR_LOCAL} not accessible, using /tmp instead" >&2
  TMPDIR_LOCAL="/tmp"
fi

mkdir -p "${TMPDIR_LOCAL}" || {
  echo "ERROR: Cannot create or access temp directory: ${TMPDIR_LOCAL}" >&2
  exit 1
}

ALL_LIST="${TMPDIR_LOCAL}/all_scenes_geometry_test.$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_geometry_test_"
SHARD_TXT=""

echo "Starting Pipeline v2 geometry export TEST - Task ${IDX}/${N_SHARDS}"
echo "Scenes root: ${SCENES_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "MAX_SCENES: ${MAX_SCENES}"

# 1) Collect scene files
echo "Finding scene files in ${SCENES_ROOT}..."
if [ ! -d "${SCENES_ROOT}" ]; then
  echo "ERROR: SCENES_ROOT directory does not exist: ${SCENES_ROOT}" >&2
  exit 1
fi

FIND_RESULT=$(find "${SCENES_ROOT}" -type f -name '*.json' 2>&1)
FIND_EXIT=$?

if [ ${FIND_EXIT} -ne 0 ]; then
  echo "ERROR: find command failed" >&2
  exit 1
fi

echo "${FIND_RESULT}" | sort | head -${MAX_SCENES} > "${ALL_LIST}" || {
  EXIT_CODE=$?
  if [ ${EXIT_CODE} -ne 141 ]; then
    echo "ERROR: Failed to write scene list" >&2
    exit 1
  fi
}

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} total scene files (limited to ${MAX_SCENES} for testing)"

# 2) Split into balanced shards using GNU split
echo "Splitting into ${N_SHARDS} shards..."
echo "Input file: ${ALL_LIST}"
echo "Input file size: $(wc -l < "${ALL_LIST}") lines"
echo "Shard prefix: ${SHARD_PREFIX}"

split -d -n l/${N_SHARDS} "${ALL_LIST}" "${SHARD_PREFIX}" || {
  echo "ERROR: Failed to split scene list" >&2
  echo "Split command: split -d -n l/${N_SHARDS} ${ALL_LIST} ${SHARD_PREFIX}" >&2
  exit 1
}

echo "Split completed, checking shard files..."
ls -lh "${SHARD_PREFIX}"* 2>&1 || echo "No shard files found" >&2

# 3) Pick this task's shard file
SUFFIX=$(printf "%02d" $((IDX-1)))
SHARD_TXT="${SHARD_PREFIX}${SUFFIX}"

echo "Looking for shard file: ${SHARD_TXT}"

if [ ! -f "${SHARD_TXT}" ]; then
  echo "ERROR: shard file does not exist: ${SHARD_TXT}" >&2
  echo "Available shard files:" >&2
  ls -la "${SHARD_PREFIX}"* >&2 || echo "No shard files found" >&2
  exit 2
fi

if [ ! -s "${SHARD_TXT}" ]; then
  echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
  exit 2
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo "Shard file: ${SHARD_TXT}"
echo "First scene in shard: $(head -1 "${SHARD_TXT}")"

# 4) Activate conda
echo "Activating conda environment..."
if command -v conda &> /dev/null; then
  if conda env list | grep -q "^imginav "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate imginav
  elif conda env list | grep -q "^scenefactor "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate scenefactor
  fi
fi

# 5) Check dependencies
echo "Checking Python dependencies..."
python -c "import trimesh, numpy, json" || {
  echo "ERROR: Required Python packages not available" >&2
  exit 1
}

# 6) Process scenes
echo "Starting processing at $(date)"
cd "${PROJECT_ROOT}/ImgiNav" || exit 1

SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r JSON_PATH; do
  if [ -z "${JSON_PATH}" ] || [ ! -f "${JSON_PATH}" ]; then
    echo "Warning: Scene file not found: ${JSON_PATH}"
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

