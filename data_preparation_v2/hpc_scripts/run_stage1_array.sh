#!/bin/bash
#BSUB -J stage1[1-10]                    # 10 parallel workers
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage1.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage1.%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model/3D-FUTURE-model"
MODEL_INFO="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model/model_info.json"
TAXONOMY_FILE="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/taxonomy.json"
TEXTURE_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FRONT-texture"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
OUTPUT_GEOMETRY_DIR="/work3/s233249/ImgiNav/dataset_v2/geometry"

N_SHARDS=10
STAGE1_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/stage1_reconstruct_geometry.py"
STAGE2_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/run_stage2_array.sh"
SCRIPT_DIR="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts"
# =============================================================================

IDX=${LSB_JOBINDEX}
TMPDIR_LOCAL="${TMPDIR:-/tmp}"
mkdir -p "${TMPDIR_LOCAL}"
JOB_UNIQUE_ID="${IDX}_$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_${JOB_UNIQUE_ID}_"
SHARD_TXT=""
SHARD_SCENES_DIR="${TMPDIR_LOCAL}/filtered_scenes_shard_${JOB_UNIQUE_ID}"

echo "=============================================================================="
echo "Starting Stage 1 processing - Task ${IDX}/${N_SHARDS}"
echo "=============================================================================="

# Extract shard
if [ ! -f "${VALID_SCENES_FILE}" ]; then
  echo "ERROR: valid_scenes.txt not found at: ${VALID_SCENES_FILE}" >&2
  exit 1
fi

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

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo ""

# Create temporary directory and hardlinks
echo "Creating temporary scenes directory: ${SHARD_SCENES_DIR}"
mkdir -p "${SHARD_SCENES_DIR}"

echo "Creating hardlinks for scene files for shard ${IDX}..."
declare -A SCENE_FILE_MAP
while IFS= read -r -d '' scene_file; do
  scene_basename=$(basename "${scene_file}")
  scene_id="${scene_basename%.json}"
  SCENE_FILE_MAP["${scene_id}"]="${scene_file}"
done < <(find "${SCENES_ROOT}" -type f -name "*.json" -print0 2>/dev/null)

echo "Found ${#SCENE_FILE_MAP[@]} scene files in ${SCENES_ROOT}"

LINKED=0
MISSING=0
while IFS= read -r scene_id; do
  scene_id=$(echo "${scene_id}" | tr -d '\r\n' | xargs)
  if [ -z "${scene_id}" ]; then
    continue
  fi
  
  scene_file="${SCENE_FILE_MAP[${scene_id}]:-}"
  
  if [ -n "${scene_file}" ] && [ -f "${scene_file}" ]; then
    ln "${scene_file}" "${SHARD_SCENES_DIR}/${scene_id}.json" 2>/dev/null || true
    LINKED=$((LINKED + 1))
  else
    MISSING=$((MISSING + 1))
  fi
done < "${SHARD_TXT}"

echo "Created ${LINKED} hardlinks (${MISSING} missing)"
if [ ${LINKED} -eq 0 ]; then
  echo "ERROR: No scene files found for shard ${IDX}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 3
fi

mkdir -p "${OUTPUT_GEOMETRY_DIR}"

# Conda activation
echo "Activating conda environment..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || conda activate scenefactor || {
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
  }
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda activate imginav || conda activate scenefactor || {
    echo "ERROR: Failed to activate conda environment" >&2
    exit 1
  }
fi

# Verify paths
if [ ! -f "${MODEL_INFO}" ]; then
  echo "ERROR: model_info.json not found at: ${MODEL_INFO}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  ALTERNATIVE_MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
  if [ -d "${ALTERNATIVE_MODEL_DIR}" ]; then
    MODEL_DIR="${ALTERNATIVE_MODEL_DIR}"
  else
    echo "ERROR: MODEL_DIR not found" >&2
    rm -rf "${SHARD_SCENES_DIR}"
    rm -f "${SHARD_PREFIX}"*
    exit 1
  fi
fi

if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: taxonomy.json not found at: ${TAXONOMY_FILE}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
fi

python -c "import trimesh, numpy, scipy, json" || {
  echo "ERROR: Required Python packages not available" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

# Run Stage 1
echo ""
echo "=============================================================================="
echo "Running Stage 1: Reconstruct Geometry"
echo "=============================================================================="
echo "Starting at $(date)"

STAGE1_ARGS=(
  "${STAGE1_SCRIPT}"
  --scenes-dir "${SHARD_SCENES_DIR}"
  --model-dir "${MODEL_DIR}"
  --model-info "${MODEL_INFO}"
  --taxonomy "${TAXONOMY_FILE}"
  --output-dir "${OUTPUT_GEOMETRY_DIR}"
)

if [ -n "${TEXTURE_DIR:-}" ] && [ -d "${TEXTURE_DIR}" ]; then
  STAGE1_ARGS+=(--texture-dir "${TEXTURE_DIR}")
fi

python "${STAGE1_ARGS[@]}" || {
  echo "ERROR: Stage 1 failed for task ${IDX}" >&2
  rm -rf "${SHARD_SCENES_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

echo "Stage 1 completed at $(date)"
echo ""

# Cleanup
rm -rf "${SHARD_SCENES_DIR}"
rm -f "${SHARD_PREFIX}"*

# Submit Stage 2 for this shard
echo "Submitting Stage 2 for shard ${IDX}..."
bsub -J "stage2_${IDX}" \
  -o "${SCRIPT_DIR}/logs/stage2.${IDX}.%J.out" \
  -e "${SCRIPT_DIR}/logs/stage2.${IDX}.%J.err" \
  -n 8 \
  -R "rusage[mem=8000]" \
  -W 10:00 \
  -q hpc \
  bash "${STAGE2_SCRIPT}" "${IDX}" "${SHARD_TXT}" || {
  echo "WARNING: Failed to submit Stage 2 for shard ${IDX}" >&2
}

echo ""
echo "=============================================================================="
echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"
echo "=============================================================================="
