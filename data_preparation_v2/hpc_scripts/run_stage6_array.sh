#!/bin/bash
# This script can be called directly with a shard ID, or as a job array
# Usage: bash run_stage6_array.sh <shard_id>

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
DATASET_ROOT="/work3/s233249/ImgiNav/datasets"
SHARDS_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/shards"
LOG_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/logs"

# Determine shard ID
if [ $# -ge 1 ]; then
  # Called directly with shard ID
  JOB_ID=$1
else
  # Called as job array
  if [ -z "${LSB_JOBINDEX:-}" ]; then
    echo "ERROR: LSB_JOBINDEX is not set and no shard ID provided"
    exit 1
  fi
  JOB_ID=${LSB_JOBINDEX}
fi

SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"

# Ensure directories exist
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Stage 6: Generate Manifests - Shard ${JOB_ID}"
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

# Run stage 6 with scene list
# Note: Stage 6 generates manifests for all scenes, but we can filter by scene list
echo "Running stage 6..."
python data_preparation_v2/stage6_generate_manifests.py \
  --dataset-root "${DATASET_ROOT}" \
  --scene-list "${SHARD_FILE}"

STAGE6_EXIT_CODE=$?

if [ ${STAGE6_EXIT_CODE} -ne 0 ]; then
  echo "ERROR: Stage 6 failed with exit code ${STAGE6_EXIT_CODE}"
  exit ${STAGE6_EXIT_CODE}
fi

echo "Stage 6 completed successfully for shard ${JOB_ID}"
echo "Pipeline complete for shard ${JOB_ID}!"

