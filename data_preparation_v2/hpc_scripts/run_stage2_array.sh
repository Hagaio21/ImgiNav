#!/bin/bash
#BSUB -J stage2[1-10]                    # 10 parallel workers
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage2.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage2.%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"  # Original 3D-FRONT scenes directory
MODEL_INFO="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model/model_info.json"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/taxonomy.json"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
OUTPUT_METADATA_DIR="/work3/s233249/ImgiNav/dataset_v2/metadata"

N_SHARDS=10                                          # Must match [1-10] above
STAGE2_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/stage2_compile_metadata.py"
# =============================================================================

# Accept shard info from previous stage if provided, otherwise extract
if [ $# -ge 2 ]; then
  IDX="$1"
  SHARD_TXT="$2"
  echo "Received shard info from previous stage: IDX=${IDX}, SHARD_TXT=${SHARD_TXT}"
else
  IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
  SHARD_TXT=""
fi

TMPDIR_LOCAL="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_LOCAL}"
JOB_UNIQUE_ID="${IDX}_$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_${JOB_UNIQUE_ID}_"
SHARD_SCENES_DIR="${TMPDIR_LOCAL}/filtered_scenes_shard_${JOB_UNIQUE_ID}"

STAGE3_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/run_stage3_array.sh"
SCRIPT_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts"

echo "=============================================================================="
echo "Starting Stage 2 processing - Task ${IDX}/${N_SHARDS}"
echo "=============================================================================="
echo "Scenes root: ${SCENES_ROOT}"
echo "Output metadata dir: ${OUTPUT_METADATA_DIR}"
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

# 3) Create temporary directory for this shard's scene files
echo "Creating temporary scenes directory: ${SHARD_SCENES_DIR}"
mkdir -p "${SHARD_SCENES_DIR}"

# 4) Create symlinks for scene files for this shard (more efficient than copying)
echo "Creating symlinks for scene files for shard ${IDX}..."
LINKED=0
MISSING=0
while IFS= read -r scene_id; do
  scene_id=$(echo "${scene_id}" | tr -d '\r\n' | xargs)  # Trim whitespace
  if [ -z "${scene_id}" ]; then
    continue
  fi
  
  # Search for scene file recursively in SCENES_ROOT
  scene_file=$(find "${SCENES_ROOT}" -type f -name "${scene_id}.json" 2>/dev/null | head -1)
  
  if [ -n "${scene_file}" ] && [ -f "${scene_file}" ]; then
    # Use absolute path for symlink to avoid issues
    scene_file_abs=$(readlink -f "${scene_file}" 2>/dev/null || echo "${scene_file}")
    ln -sf "${scene_file_abs}" "${SHARD_SCENES_DIR}/${scene_id}.json" 2>/dev/null || {
      # Fallback to copy if symlink fails (e.g., cross-filesystem)
      cp "${scene_file}" "${SHARD_SCENES_DIR}/${scene_id}.json"
    }
    LINKED=$((LINKED + 1))
  else
    echo "WARNING: Scene file not found: ${scene_id}.json (searched in ${SCENES_ROOT})" >&2
    MISSING=$((MISSING + 1))
  fi
done < "${SHARD_TXT}"

echo "Linked ${LINKED} scene files (${MISSING} missing)"
if [ ${LINKED} -eq 0 ]; then
  echo "ERROR: No scene files were linked for shard ${IDX}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 3
fi

# 5) Create output directories if they don't exist
mkdir -p "${OUTPUT_METADATA_DIR}"

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

# 7) Check if required files exist before processing
if [ ! -f "${MODEL_INFO}" ]; then
  echo "ERROR: model_info.json not found at: ${MODEL_INFO}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
fi

if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: taxonomy.json not found at: ${TAXONOMY_FILE}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
fi

# 8) Check required dependencies
echo "Checking Python dependencies..."
python -c "import numpy, scipy, json" || {
  echo "ERROR: Required Python packages not available" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

# 9) Run Stage 2 processing
echo ""
echo "=============================================================================="
echo "Running Stage 2: Compile Metadata"
echo "=============================================================================="
echo "Starting at $(date)"

python "${STAGE2_SCRIPT}" \
  --scenes-dir "${SHARD_SCENES_DIR}" \
  --model-info "${MODEL_INFO}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --output-dir "${OUTPUT_METADATA_DIR}" || {
  echo "ERROR: Stage 2 failed for task ${IDX}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

echo "Stage 2 completed at $(date)"
echo ""

# 10) Cleanup temporary files
rm -rf "${SHARD_SCENES_DIR}"
rm -f "${SHARD_PREFIX}"*

# Submit Stage 3 for this shard
echo "Submitting Stage 3 for shard ${IDX}..."
bsub -J "stage3_${IDX}" \
  -o "${SCRIPT_DIR}/logs/stage3.${IDX}.%J.out" \
  -e "${SCRIPT_DIR}/logs/stage3.${IDX}.%J.err" \
  -n 8 \
  -R "rusage[mem=8000]" \
  -W 10:00 \
  -q hpc \
  bash "${STAGE3_SCRIPT}" "${IDX}" "${SHARD_TXT}" || {
  echo "WARNING: Failed to submit Stage 3 for shard ${IDX}" >&2
}

echo ""
echo "=============================================================================="
echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"
echo "=============================================================================="

