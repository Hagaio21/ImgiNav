#!/bin/bash
#BSUB -J stage1_[1-10]                    # 8 parallel workers
#BSUB -o /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage1_process.%I.%J.out
#BSUB -e /zhome/62/5/203350/ws/ImgiNav/data_preperation/hpc_scripts/logs/stage1_process.%I.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=4000]"
#BSUB -W 07:00
#BSUB -q hpc

set -euo pipefail

# =============================================================================
# CONFIGURATION - YOUR PATHS
# =============================================================================
SCENES_ROOT="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT"
MODEL_DIR="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model"
MODEL_INFO="/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTURE-model/model_info.json"
TAXONOMY_FILE="/zhome/62/5/203350/ws/ImgiNav/config/taxonomy.json"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/scenes"

N_SHARDS=10                                          # Must match [1-10] above
PYTHON_SCRIPT="/zhome/62/5/203350/ws/ImgiNav/data_preperation/stage1_build_scenes.py"
# =============================================================================

IDX=${LSB_JOBINDEX}                                 # 1..N_SHARDS
TMPDIR_LOCAL="${TMPDIR:-/tmp}"
ALL_LIST="${TMPDIR_LOCAL}/all_scenes.$$"
SHARD_PREFIX="${TMPDIR_LOCAL}/scenes_shard_"
SHARD_TXT=""                                        # will set below

echo "Starting Stage 1 processing - Task ${IDX}/${N_SHARDS}"
echo "Scenes root: ${SCENES_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

# 1) Collect all *.json scene files, deterministic order
find "${SCENES_ROOT}" -type f -name '*.json' -print | sort > "${ALL_LIST}"

# Filter out non-scene JSONs if needed (uncomment and adapt as necessary)
# grep -v '/(readme|metadata|config|model_info)\.json$' "${ALL_LIST}" > "${ALL_LIST}.tmp" && mv "${ALL_LIST}.tmp" "${ALL_LIST}"

TOTAL_SCENES=$(wc -l < "${ALL_LIST}")
echo "Found ${TOTAL_SCENES} total scene files"

# 2) Split into balanced shards using GNU split
split -d -n l/${N_SHARDS} "${ALL_LIST}" "${SHARD_PREFIX}"

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
if [ ! -f "${MODEL_INFO}" ]; then
  echo "ERROR: model_info.json not found at: ${MODEL_INFO}" >&2
  echo "Checking for model_info.json in model directory..." >&2
  find "${MODEL_DIR}" -name "model_info.json" -type f
  exit 1
fi

if [ ! -f "${TAXONOMY_FILE}" ]; then
  echo "ERROR: taxonomy.json not found at: ${TAXONOMY_FILE}" >&2
  exit 1
fi

# 7) Check required dependencies
python -c "import trimesh, pandas, scipy" || {
  echo "ERROR: Required Python packages not available" >&2
  exit 1
}

# 7) Run Stage 1 processing with the shard
#    Using scene_list parameter to pass the list of scenes
echo "Starting processing at $(date)"

python "${PYTHON_SCRIPT}" \
  --scene_list "${SHARD_TXT}" \
  --out_dir "${OUTPUT_DIR}" \
  --model_dir "${MODEL_DIR}" \
  --model_info "${MODEL_INFO}" \
  --taxonomy "${TAXONOMY_FILE}" \
  --total_points 500000 \
  --min_pts_per_mesh 100 \
  --max_pts_per_mesh 0 \
  --ppsm 0.0 \
  --save_parquet \
  --per_scene_subdir || {
  echo "ERROR: Python script failed for task ${IDX}" >&2
  exit 1
}

echo "Task ${IDX}/${N_SHARDS} completed successfully at $(date)"

# 8) Cleanup temporary files
rm -f "${ALL_LIST}" "${SHARD_PREFIX}"*

echo "Stage 1 processing complete for task ${IDX}"