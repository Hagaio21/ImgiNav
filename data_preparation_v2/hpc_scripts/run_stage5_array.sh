#!/bin/bash
# Stage 5: Build Graphs
# Only needs OUTPUT_DATASET_ROOT from config

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/paths.yaml}"

# ==============================================================================
# JOB SETUP
# ==============================================================================
if [ $# -ge 1 ]; then
    JOB_ID=$1
elif [ -n "${LSB_JOBINDEX:-}" ]; then
    JOB_ID=${LSB_JOBINDEX}
else
    echo "ERROR: No job index."
    exit 1
fi

# ==============================================================================
# LOAD CONFIGURATION
# ==============================================================================
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

get_config() {
    python3 -c "
import yaml
with open('${CONFIG_FILE}') as f:
    config = yaml.safe_load(f)
print(config.get('$1', '') or '')
"
}

BASE_DIR="$(get_config base_dir)"
OUTPUT_DATASET_ROOT="$(get_config output_dataset_root)"
SHARDS_DIR="$(get_config shards_dir)"
LOG_DIR="$(get_config log_dir)"

if [[ "${SHARDS_DIR}" != /* ]]; then
    SHARDS_DIR="${BASE_DIR}/${SHARDS_DIR}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${BASE_DIR}/${LOG_DIR}"
fi

SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Stage 5: Build Graphs - Shard ${JOB_ID}"
echo "=========================================="
echo "Output Dataset Root: ${OUTPUT_DATASET_ROOT}"
echo "Shard file: ${SHARD_FILE}"
echo "=========================================="

if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}"
    exit 1
fi

SCENE_COUNT=$(wc -l < "${SHARD_FILE}")
echo "Processing ${SCENE_COUNT} scenes"

if [ ${SCENE_COUNT} -eq 0 ]; then
    exit 0
fi

# ==============================================================================
# CONDA ACTIVATION
# ==============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate imginav || { echo "Failed to activate conda" >&2; exit 1; }
fi

# ==============================================================================
# RUN STAGE 5
# ==============================================================================
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "Running stage 5..."
python data_preparation_v2/stage5_build_graphs.py \
    --dataset-root "${OUTPUT_DATASET_ROOT}" \
    --scene-list "${SHARD_FILE}"

STAGE5_EXIT_CODE=$?

if [ ${STAGE5_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Stage 5 failed with exit code ${STAGE5_EXIT_CODE}"
    exit ${STAGE5_EXIT_CODE}
fi

echo "Stage 5 completed successfully for shard ${JOB_ID}"

# ==============================================================================
# CHAIN TO STAGE 6
# ==============================================================================
STAGE6_SCRIPT="${SCRIPT_DIR}/run_stage6_array.sh"

if [ -f "${STAGE6_SCRIPT}" ]; then
    echo "Submitting stage 6 for shard ${JOB_ID}..."
    
    bsub -J "stage6_shard${JOB_ID}" \
         -o "${LOG_DIR}/stage6_shard${JOB_ID}.%J.out" \
         -e "${LOG_DIR}/stage6_shard${JOB_ID}.%J.err" \
         -n 1 \
         -R "rusage[mem=2000]" \
         -W 1:00 \
         -q hpc \
         -env "CONFIG_FILE=${CONFIG_FILE}" \
         bash "${STAGE6_SCRIPT}" "${JOB_ID}"
    
    echo "Stage 6 job submitted"
fi

echo "Stage 5 complete for shard ${JOB_ID}"
