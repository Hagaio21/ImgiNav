#!/bin/bash
#BSUB -J stage3_layouts[1-10]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage3_layouts.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage3_layouts.%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Cleanup function
cleanup() {
  local exit_code=$?
  echo "Cleaning up temporary files..."
  rm -f "${SHARD_TXT:-}" 2>/dev/null || true
  if [ ${exit_code} -ne 0 ]; then
    exit ${exit_code}
  fi
}
trap cleanup EXIT
# =============================================================================
# CONFIGURATION
# =============================================================================
GEOMETRY_DIR="/work3/s233249/ImgiNav/dataset_v2/geometry"
METADATA_DIR="/work3/s233249/ImgiNav/dataset_v2/metadata"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/taxonomy.json"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
OUTPUT_LAYOUTS_DIR="/work3/s233249/ImgiNav/dataset_v2/layouts"
RESOLUTION=512

N_SHARDS=10
STAGE3_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/stage3_render_layouts.py"
# =============================================================================

# Accept shard info from previous stage if provided
if [ $# -ge 2 ]; then
  IDX="$1"
  SHARD_TXT="$2"
  echo "Received shard info from previous stage: IDX=${IDX}, SHARD_TXT=${SHARD_TXT}"
else
  IDX=${LSB_JOBINDEX}
  SHARD_TXT=""
fi

TMPDIR_LOCAL="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_LOCAL}"

STAGE4_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/run_stage4_array.sh"
SCRIPT_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts"

echo "=============================================================================="
echo "Stage 3 Layout Rendering - Task ${IDX}/${N_SHARDS}"
echo "=============================================================================="

# Extract shard if not provided
if [ -z "${SHARD_TXT}" ] || [ ! -f "${SHARD_TXT}" ]; then
  if [ ! -f "${VALID_SCENES_FILE}" ]; then
    echo "ERROR: valid_scenes.txt not found: ${VALID_SCENES_FILE}" >&2
    exit 1
  fi
  
  TOTAL_LINES=$(wc -l < "${VALID_SCENES_FILE}")
  LINES_PER_SHARD=$(( (TOTAL_LINES + N_SHARDS - 1) / N_SHARDS ))
  START_LINE=$(( (IDX - 1) * LINES_PER_SHARD + 1 ))
  END_LINE=$(( IDX * LINES_PER_SHARD ))
  
  SHARD_TXT="${TMPDIR_LOCAL}/shard_${IDX}_$$.txt"
  sed -n "${START_LINE},${END_LINE}p" "${VALID_SCENES_FILE}" > "${SHARD_TXT}"
  
  if [ ! -s "${SHARD_TXT}" ]; then
    echo "ERROR: Shard ${IDX} is empty" >&2
    exit 1
  fi
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo ""

# Create output directory
mkdir -p "${OUTPUT_LAYOUTS_DIR}"

# Activate conda environment
echo "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/zhome/c7/6/155569/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/zhome/c7/6/155569/miniconda3/etc/profile.d/conda.sh"
fi

conda activate imginav || conda activate scenefactor || {
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
}

# Check dependencies
echo "Checking dependencies..."
python -c "import trimesh, numpy, PIL" || {
    echo "ERROR: Missing Python dependencies" >&2
    exit 1
}

# Check required files
if [ ! -f "${TAXONOMY_FILE}" ]; then
    echo "ERROR: Taxonomy file not found: ${TAXONOMY_FILE}" >&2
    exit 1
fi

if [ ! -d "${GEOMETRY_DIR}" ]; then
    echo "ERROR: Geometry directory not found: ${GEOMETRY_DIR}" >&2
    exit 1
fi

# Run Stage 3 with --hpc flag for Xvfb
echo "=============================================================================="
echo "Running Stage 3: Render Layouts"
echo "=============================================================================="
echo "Started at $(date)"

python "${STAGE3_SCRIPT}" \
    --geometry-dir "${GEOMETRY_DIR}" \
    --metadata-dir "${METADATA_DIR}" \
    --taxonomy "${TAXONOMY_FILE}" \
    --output-dir "${OUTPUT_LAYOUTS_DIR}" \
    --scene-list "${SHARD_TXT}" \
    --resolution "${RESOLUTION}" \
    --hpc

EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Stage 3 failed with exit code ${EXIT_CODE}" >&2
    rm -f "${SHARD_TXT}"
    exit ${EXIT_CODE}
fi

# Submit Stage 4 for this shard (preserve shard file for next stage)
echo "Submitting Stage 4 for shard ${IDX}..."
bsub -J "stage4_${IDX}" \
  -o "${SCRIPT_DIR}/logs/stage4.${IDX}.%J.out" \
  -e "${SCRIPT_DIR}/logs/stage4.${IDX}.%J.err" \
  -n 8 \
  -R "rusage[mem=8000]" \
  -W 10:00 \
  -q hpc \
  bash "${STAGE4_SCRIPT}" "${IDX}" "${SHARD_TXT}" || {
  echo "WARNING: Failed to submit Stage 4 for shard ${IDX}" >&2
}

# Cleanup after submitting next stage (trap will also run)
rm -f "${SHARD_TXT}" 2>/dev/null || true

echo ""
echo "=============================================================================="
echo "Task ${IDX}/${N_SHARDS} completed at $(date)"
echo "=============================================================================="