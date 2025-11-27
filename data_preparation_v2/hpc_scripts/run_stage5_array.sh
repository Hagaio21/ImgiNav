#!/bin/bash
#BSUB -J stage5_graphs[1-10]                    # 10 parallel workers
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage5_graphs.%I.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage5_graphs.%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=8000]"
#BSUB -W 10:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
METADATA_DIR="/work3/s233249/ImgiNav/dataset_v2/metadata"
VALID_SCENES_FILE="/work3/s233249/ImgiNav/ImgiNav/valid_scenes.txt"
OUTPUT_GRAPHS_DIR="/work3/s233249/ImgiNav/dataset_v2/graphs"

N_SHARDS=10                                          # Must match [1-10] above
STAGE5_SCRIPT="/work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/stage5_build_graphs.py"
# =============================================================================

IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
TMPDIR_LOCAL="${TMPDIR:-/tmp}"
# Ensure TMPDIR exists
mkdir -p "${TMPDIR_LOCAL}"
JOB_UNIQUE_ID="${IDX}_$$"                           # Unique ID for this job instance
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_${JOB_UNIQUE_ID}_"
SHARD_TXT=""                                        # will set below
TMP_METADATA_DIR="${TMPDIR_LOCAL}/metadata_shard_${JOB_UNIQUE_ID}"

echo "=============================================================================="
echo "Starting Stage 5 Graph Building - Task ${IDX}/${N_SHARDS}"
echo "=============================================================================="
echo "Metadata dir: ${METADATA_DIR}"
echo "Output graphs dir: ${OUTPUT_GRAPHS_DIR}"
echo "Valid scenes file: ${VALID_SCENES_FILE}"
echo ""

# 1) Check if valid_scenes.txt exists
if [ ! -f "${VALID_SCENES_FILE}" ]; then
  echo "ERROR: valid_scenes.txt not found at: ${VALID_SCENES_FILE}" >&2
  exit 1
fi

# 2) Extract this job's shard from valid_scenes.txt
# Calculate line ranges for this shard
TOTAL_LINES=$(wc -l < "${VALID_SCENES_FILE}")
LINES_PER_SHARD=$(( (TOTAL_LINES + N_SHARDS - 1) / N_SHARDS ))
START_LINE=$(( (IDX - 1) * LINES_PER_SHARD + 1 ))
END_LINE=$(( IDX * LINES_PER_SHARD ))

echo "Extracting shard ${IDX} (lines ${START_LINE}-${END_LINE} from ${TOTAL_LINES} total lines)..."
SHARD_TXT="${SHARD_PREFIX}${IDX}.txt"
# Ensure parent directory exists
mkdir -p "$(dirname "${SHARD_TXT}")"
sed -n "${START_LINE},${END_LINE}p" "${VALID_SCENES_FILE}" > "${SHARD_TXT}" || {
  echo "ERROR: Failed to extract shard from ${VALID_SCENES_FILE}" >&2
  exit 2
}

# Safety: ensure shard not empty
if [ ! -s "${SHARD_TXT}" ]; then
  echo "ERROR: shard ${IDX} is empty (file: ${SHARD_TXT})." >&2
  exit 2
fi

SHARD_COUNT=$(wc -l < "${SHARD_TXT}")
echo "Task ${IDX}/${N_SHARDS}: processing ${SHARD_COUNT} scenes"
echo ""

# 3) Create temporary metadata directory structure with symlinks
echo "Creating temporary metadata directory structure..."
mkdir -p "${TMP_METADATA_DIR}/scenes"
mkdir -p "${TMP_METADATA_DIR}/rooms"

# 4) Create symlinks for scene and room metadata files for this shard
echo "Creating symlinks for scene and room metadata files..."
LINKED_SCENES=0
LINKED_ROOMS=0
MISSING_SCENES=0
MISSING_ROOMS=0

while IFS= read -r scene_id; do
  scene_id=$(echo "${scene_id}" | tr -d '\r\n' | xargs)  # Trim whitespace
  if [ -z "${scene_id}" ]; then
    continue
  fi
  
  # Link scene metadata
  scene_meta_src="${METADATA_DIR}/scenes/${scene_id}.json"
  scene_meta_dst="${TMP_METADATA_DIR}/scenes/${scene_id}.json"
  
  if [ -f "${scene_meta_src}" ]; then
    # Use absolute path for symlink to avoid issues
    scene_meta_abs=$(readlink -f "${scene_meta_src}" 2>/dev/null || echo "${scene_meta_src}")
    ln -sf "${scene_meta_abs}" "${scene_meta_dst}" 2>/dev/null || {
      # Fallback to copy if symlink fails (e.g., cross-filesystem)
      cp "${scene_meta_src}" "${scene_meta_dst}"
    }
    LINKED_SCENES=$((LINKED_SCENES + 1))
  else
    echo "WARNING: Scene metadata not found: ${scene_meta_src}" >&2
    MISSING_SCENES=$((MISSING_SCENES + 1))
  fi
  
  # Link room metadata files (there can be multiple rooms per scene)
  for room_meta_src in "${METADATA_DIR}/rooms/${scene_id}"_*.json; do
    if [ -f "${room_meta_src}" ]; then
      room_filename=$(basename "${room_meta_src}")
      room_meta_dst="${TMP_METADATA_DIR}/rooms/${room_filename}"
      # Use absolute path for symlink
      room_meta_abs=$(readlink -f "${room_meta_src}" 2>/dev/null || echo "${room_meta_src}")
      ln -sf "${room_meta_abs}" "${room_meta_dst}" 2>/dev/null || {
        # Fallback to copy if symlink fails (e.g., cross-filesystem)
        cp "${room_meta_src}" "${room_meta_dst}"
      }
      LINKED_ROOMS=$((LINKED_ROOMS + 1))
    fi
  done
  
  # Check if any rooms were found for this scene
  room_count=$(find "${METADATA_DIR}/rooms" -maxdepth 1 -name "${scene_id}_*.json" 2>/dev/null | wc -l)
  if [ "${room_count}" -eq 0 ]; then
    MISSING_ROOMS=$((MISSING_ROOMS + 1))
  fi
done < "${SHARD_TXT}"

echo "Linked ${LINKED_SCENES} scene metadata files (${MISSING_SCENES} missing)"
echo "Linked ${LINKED_ROOMS} room metadata files (${MISSING_ROOMS} scenes with no rooms)"
if [ ${LINKED_SCENES} -eq 0 ]; then
  echo "ERROR: No scene metadata files were linked for shard ${IDX}" >&2
  rm -rf "${TMP_METADATA_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 3
fi

# 5) Create output directory if it doesn't exist
mkdir -p "${OUTPUT_GRAPHS_DIR}"

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
if [ ! -d "${METADATA_DIR}" ]; then
  echo "ERROR: Metadata directory not found: ${METADATA_DIR}" >&2
  rm -rf "${TMP_METADATA_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
fi

# 8) Check required dependencies
echo "Checking Python dependencies..."
python -c "import json" || {
  echo "ERROR: Required Python packages not available" >&2
  rm -rf "${TMP_METADATA_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

# 9) Run Stage 5 processing
echo ""
echo "=============================================================================="
echo "Running Stage 5: Build Graphs"
echo "=============================================================================="
echo "Starting at $(date)"

python "${STAGE5_SCRIPT}" \
  --metadata-dir "${TMP_METADATA_DIR}" \
  --output-dir "${OUTPUT_GRAPHS_DIR}" || {
  echo "ERROR: Stage 5 failed for task ${IDX}" >&2
  rm -rf "${TMP_METADATA_DIR}"
  rm -f "${SHARD_PREFIX}"*
  exit 1
}

echo "Stage 5 completed at $(date)"
echo ""

# 10) Cleanup temporary files
echo "Cleaning up temporary files..."
rm -rf "${TMP_METADATA_DIR}"
rm -f "${SHARD_PREFIX}"*

echo ""
echo "=============================================================================="
echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"
echo "=============================================================================="

