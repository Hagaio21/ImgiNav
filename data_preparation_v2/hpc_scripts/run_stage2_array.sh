#!/bin/bash
#BSUB -J stage2_metadata[1-10]
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage2.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/stage2.%J.%I.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 4:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DATASET_ROOT="/work3/s233249/ImgiNav/datasets"
# 3D-FRONT scenes directory - where the original JSON files are located
# The script will search recursively in this directory for scene files matching shard IDs
# Can be overridden with FRONT3D_SCENES_DIR environment variable
SCENES_DIR="${FRONT3D_SCENES_DIR:-/dtu/datasets2/ScanNet/FutureFront3D/3D-FUTUR_FRONT}"
MODEL_INFO="/work3/s233249/ImgiNav/datasets/3D-FUTURE-model/model_info.json"
SHARDS_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/shards"
LOG_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/logs"
STAGE3_SCRIPT="${BASE_DIR}/data_preparation_v2/hpc_scripts/run_stage3_array.sh"

# Job array indexing
if [ -z "${LSB_JOBINDEX:-}" ]; then
  echo "ERROR: LSB_JOBINDEX is not set. This script must be run as a job array."
  exit 1
fi

JOB_ID=${LSB_JOBINDEX}
SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Stage 2: Compile Metadata - Shard ${JOB_ID}"
echo "Job ID: ${LSB_JOBID:-unknown}"
echo "Job Index: ${JOB_ID}"
echo "Shard file: ${SHARD_FILE}"
echo "Scenes directory (will search recursively): ${SCENES_DIR}"
echo "=========================================="

# Verify shard file exists
if [ ! -f "${SHARD_FILE}" ]; then
  echo "ERROR: Shard file not found: ${SHARD_FILE}"
  exit 1
fi

# Count scenes in shard
SCENE_COUNT=$(wc -l < "${SHARD_FILE}")
echo "Processing ${SCENE_COUNT} scenes from shard ${JOB_ID}"

if [ ${SCENE_COUNT} -eq 0 ]; then
  echo "No scenes in this shard, skipping"
  exit 0
fi

# Conda activation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate imginav || {
    echo "Failed to activate conda environment 'imginav'" >&2
    exit 1
  }
fi

# Change to base directory
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

# Run stage 2 with scene list
echo "Running stage 2..."
python data_preparation_v2/stage2_compile_metadata.py \
  --dataset-root "${DATASET_ROOT}" \
  --scenes-dir "${SCENES_DIR}" \
  --scene-list "${SHARD_FILE}" \
  --model-info "${MODEL_INFO}"

STAGE2_EXIT_CODE=$?

if [ ${STAGE2_EXIT_CODE} -ne 0 ]; then
  echo "ERROR: Stage 2 failed with exit code ${STAGE2_EXIT_CODE}"
  exit ${STAGE2_EXIT_CODE}
fi

echo "Stage 2 completed successfully for shard ${JOB_ID}"

# Submit stage 3 with the same shard
echo "Submitting stage 3 for shard ${JOB_ID}..."
bsub -J "stage3_shard${JOB_ID}" \
     -o "${LOG_DIR}/stage3_shard${JOB_ID}.%J.out" \
     -e "${LOG_DIR}/stage3_shard${JOB_ID}.%J.err" \
     -n 4 \
     -R "rusage[mem=8000]" \
     -W 4:00 \
     -q hpc \
     -w "ended(${LSB_JOBID})" \
     bash "${STAGE3_SCRIPT}" "${JOB_ID}"

echo "Stage 3 job submitted for shard ${JOB_ID}"
echo "Stage 2 complete for shard ${JOB_ID}"

