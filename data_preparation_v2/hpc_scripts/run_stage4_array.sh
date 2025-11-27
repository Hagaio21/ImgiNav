#!/bin/bash
#BSUB -J stage4_povs[1-10]                    # 10 parallel workers
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage4_povs.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage4_povs.%I.%J.err
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
  rm -rf "${TMP_METADATA_DIR:-}" 2>/dev/null || true
  rm -f "${SHARD_PREFIX:-}"* 2>/dev/null || true
  if [ ${exit_code} -ne 0 ]; then
    exit ${exit_code}
  fi
}
trap cleanup EXIT

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
GEOMETRY_DIR="/work3/s233249/ImgiNav/dataset_v2/geometry"
METADATA_DIR="/work3/s233249/ImgiNav/dataset_v2/metadata"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
OUTPUT_POVS_DIR="/work3/s233249/ImgiNav/dataset_v2/povs"
WIDTH=1280
HEIGHT=720

N_SHARDS=10                                          # Must match [1-10] above
STAGE4_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/stage4_render_povs.py"
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
JOB_UNIQUE_ID="${IDX}_$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_${JOB_UNIQUE_ID}_"
TMP_METADATA_DIR="${TMPDIR_LOCAL}/metadata_shard_${JOB_UNIQUE_ID}"

STAGE5_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/run_stage5_array.sh"
SCRIPT_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts"

echo "=============================================================================="
echo "Starting Stage 4 POV Rendering - Task ${IDX}/${N_SHARDS}"
echo "=============================================================================="
echo "Geometry dir: ${GEOMETRY_DIR}"
echo "Metadata dir: ${METADATA_DIR}"
echo "Output povs dir: ${OUTPUT_POVS_DIR}"
echo "Valid scenes file: ${VALID_SCENES_FILE}"
echo ""

# 1) Check if valid_scenes.txt exists
if [ ! -f "${VALID_SCENES_FILE}" ]; then
  echo "ERROR: valid_scenes.txt not found at: ${VALID_SCENES_FILE}" >&2
  exit 1
fi

# 2) Extract shard if not provided
if [ -z "${SHARD_TXT}" ] || [ ! -f "${SHARD_TXT}" ]; then
  TOTAL_LINES=$(wc -l < "${VALID_SCENES_FILE}")
  LINES_PER_SHARD=$(( (TOTAL_LINES + N_SHARDS - 1) / N_SHARDS ))
  START_LINE=$(( (IDX - 1) * LINES_PER_SHARD + 1 ))
  END_LINE=$(( IDX * LINES_PER_SHARD ))
  
  echo "Extracting shard ${IDX} (lines ${START_LINE}-${END_LINE} from ${TOTAL_LINES} total lines)..."
  SHARD_TXT="${SHARD_PREFIX}${IDX}.txt"
  mkdir -p "$(dirname "${SHARD_TXT}")"
  sed -n "${START_LINE},${END_LINE}p" "${VALID_SCENES_FILE}" > "${SHARD_TXT}" || {
    echo "ERROR: Failed to extract shard from ${VALID_SCENES_FILE}" >&2
    exit 2
  }
  
  if [ ! -s "${SHARD_TXT}" ]; then
    echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
    exit 2
  fi
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo ""

# 3) Create temporary metadata directory structure with hardlinks
echo "Creating temporary metadata directory structure..."
mkdir -p "${TMP_METADATA_DIR}/scenes"
mkdir -p "${TMP_METADATA_DIR}/rooms"

# 4) Create hardlinks for scene and room metadata files for this shard
echo "Creating hardlinks for scene and room metadata files..."
LINKED_SCENES=0
LINKED_ROOMS=0
MISSING_SCENES=0

while IFS= read -r scene_id; do
  scene_id=$(echo "${scene_id}" | tr -d '\r\n' | xargs)
  if [ -z "${scene_id}" ]; then
    continue
  fi
  
  # Hardlink scene metadata
  scene_meta_src="${METADATA_DIR}/scenes/${scene_id}.json"
  scene_meta_dst="${TMP_METADATA_DIR}/scenes/${scene_id}.json"
  
  if [ -f "${scene_meta_src}" ]; then
    ln "${scene_meta_src}" "${scene_meta_dst}" 2>/dev/null || true
    LINKED_SCENES=$((LINKED_SCENES + 1))
  else
    MISSING_SCENES=$((MISSING_SCENES + 1))
  fi
  
  # Hardlink room metadata files
  for room_meta_src in "${METADATA_DIR}/rooms/${scene_id}"_*.json; do
    if [ -f "${room_meta_src}" ]; then
      room_filename=$(basename "${room_meta_src}")
      room_meta_dst="${TMP_METADATA_DIR}/rooms/${room_filename}"
      ln "${room_meta_src}" "${room_meta_dst}" 2>/dev/null || true
      LINKED_ROOMS=$((LINKED_ROOMS + 1))
    fi
  done
done < "${SHARD_TXT}"

echo "Created ${LINKED_SCENES} scene metadata hardlinks (${MISSING_SCENES} missing)"
echo "Created ${LINKED_ROOMS} room metadata hardlinks"
if [ ${LINKED_SCENES} -eq 0 ]; then
  echo "ERROR: No scene metadata files found for shard ${IDX}" >&2
  rm -rf "${TMP_METADATA_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 3
fi

# 5) Create output directory if it doesn't exist
mkdir -p "${OUTPUT_POVS_DIR}"

# 6) Robust conda activation (non-interactive safe)
echo "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "WARNING: Failed to activate imginav environment, trying scenefactor..." >&2
    conda activate scenefactor || {
      echo "ERROR: Failed to activate any conda environment" >&2
      exit 1
    }
  }
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda activate imginav || {
    echo "WARNING: Failed to activate imginav environment, trying scenefactor..." >&2
    conda activate scenefactor || {
      echo "ERROR: Failed to activate any conda environment" >&2
      exit 1
    }
  }
fi

# 7) Check if required directories exist
if [ ! -d "${GEOMETRY_DIR}" ]; then
  echo "ERROR: Geometry directory not found: ${GEOMETRY_DIR}" >&2
  exit 1
fi

# 8) Check required dependencies
echo "Checking Python dependencies..."
python -c "import trimesh, numpy, PIL, json" || {
  echo "ERROR: Required Python packages not available" >&2
  exit 1
}

# 9) Run Stage 4 processing
echo ""
echo "=============================================================================="
echo "Running Stage 4: Render POVs"
echo "=============================================================================="
echo "Starting at $(date)"

python "${STAGE4_SCRIPT}" \
  --geometry-dir "${GEOMETRY_DIR}" \
  --metadata-dir "${TMP_METADATA_DIR}" \
  --output-dir "${OUTPUT_POVS_DIR}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --hpc || {
  echo "ERROR: Stage 4 failed for task ${IDX}" >&2
  exit 1
}

echo "Stage 4 completed at $(date)"
echo ""

# Submit Stage 5 for this shard (pass shard file, then cleanup)
echo "Submitting Stage 5 for shard ${IDX}..."
bsub -J "stage5_${IDX}" \
  -o "${SCRIPT_DIR}/logs/stage5.${IDX}.%J.out" \
  -e "${SCRIPT_DIR}/logs/stage5.${IDX}.%J.err" \
  -n 8 \
  -R "rusage[mem=8000]" \
  -W 10:00 \
  -q hpc \
  bash "${STAGE5_SCRIPT}" "${IDX}" "${SHARD_TXT}" || {
  echo "WARNING: Failed to submit Stage 5 for shard ${IDX}" >&2
}

# Cleanup after submitting next stage
rm -rf "${TMP_METADATA_DIR}" 2>/dev/null || true
rm -f "${SHARD_PREFIX}"* 2>/dev/null || true

echo ""
echo "=============================================================================="
echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"
echo "=============================================================================="

