#!/bin/bash
# This script can be called directly with a shard ID, or as a job array
# Usage: bash run_stage5_array.sh <shard_id>

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DATASET_ROOT="/work3/s233249/ImgiNav/datasets"
SHARDS_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/shards"
LOG_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/logs"
STAGE6_SCRIPT="${BASE_DIR}/data_preparation_v2/hpc_scripts/run_stage6_array.sh"

# Determine shard ID
if [ $# -ge 1 ]; then
  # Called directly with shard ID
  JOB_ID=$1
  PARENT_JOB_ID=""
else
  # Called as job array
  if [ -z "${LSB_JOBINDEX:-}" ]; then
    echo "ERROR: LSB_JOBINDEX is not set and no shard ID provided"
    exit 1
  fi
  JOB_ID=${LSB_JOBINDEX}
  PARENT_JOB_ID="${LSB_JOBID:-}"
fi

SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Stage 5: Build Graphs - Shard ${JOB_ID}"
echo "Job ID: ${LSB_JOBID:-unknown}"
echo "Shard file: ${SHARD_FILE}"
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

# Run stage 5 with scene list
echo "Running stage 5..."
python data_preparation_v2/stage5_build_graphs.py \
  --dataset-root "${DATASET_ROOT}" \
  --scene-list "${SHARD_FILE}"

STAGE5_EXIT_CODE=$?

if [ ${STAGE5_EXIT_CODE} -ne 0 ]; then
  echo "ERROR: Stage 5 failed with exit code ${STAGE5_EXIT_CODE}"
  exit ${STAGE5_EXIT_CODE}
fi

echo "Stage 5 completed successfully for shard ${JOB_ID}"

# Submit stage 6 with the same shard
echo "Submitting stage 6 for shard ${JOB_ID}..."
if [ -n "${PARENT_JOB_ID}" ]; then
  # Wait for parent job if we have one
  bsub -J "stage6_shard${JOB_ID}" \
       -o "${LOG_DIR}/stage6_shard${JOB_ID}.%J.out" \
       -e "${LOG_DIR}/stage6_shard${JOB_ID}.%J.err" \
       -n 1 \
       -R "rusage[mem=2000]" \
       -W 1:00 \
       -q hpc \
       -w "ended(${PARENT_JOB_ID})" \
       bash "${STAGE6_SCRIPT}" "${JOB_ID}"
else
  bsub -J "stage6_shard${JOB_ID}" \
       -o "${LOG_DIR}/stage6_shard${JOB_ID}.%J.out" \
       -e "${LOG_DIR}/stage6_shard${JOB_ID}.%J.err" \
       -n 1 \
       -R "rusage[mem=2000]" \
       -W 1:00 \
       -q hpc \
       bash "${STAGE6_SCRIPT}" "${JOB_ID}"
fi

echo "Stage 6 job submitted for shard ${JOB_ID}"
echo "Stage 5 complete for shard ${JOB_ID}"

