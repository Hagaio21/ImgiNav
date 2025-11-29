#!/bin/bash
# Clean Dataset - Sharded Processing
# Processes dataset cleaning in parallel using shards

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
    echo "ERROR: No job index. Provide shard ID as argument or run as array job."
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
HPC_SCRIPTS_DIR="$(get_config hpc_scripts_dir)"
PYTHON_SCRIPTS_DIR="$(get_config python_scripts_dir)"
SHARDS_DIR="$(get_config shards_dir)"
LOG_DIR="$(get_config log_dir)"

if [[ "${HPC_SCRIPTS_DIR}" != /* ]]; then
    HPC_SCRIPTS_DIR="${BASE_DIR}/${HPC_SCRIPTS_DIR}"
fi
if [[ "${PYTHON_SCRIPTS_DIR}" != /* ]]; then
    PYTHON_SCRIPTS_DIR="${BASE_DIR}/${PYTHON_SCRIPTS_DIR}"
fi
if [[ "${SHARDS_DIR}" != /* ]]; then
    SHARDS_DIR="${BASE_DIR}/${SHARDS_DIR}"
fi
if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${BASE_DIR}/${LOG_DIR}"
fi

SHARD_FILE="${SHARDS_DIR}/shard_${JOB_ID}.txt"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Clean Dataset - Shard ${JOB_ID}"
echo "=========================================="
echo "Config file: ${CONFIG_FILE}"
echo "Output Dataset Root: ${OUTPUT_DATASET_ROOT}"
echo "Shard file: ${SHARD_FILE}"
echo "=========================================="

# Verify shard file exists
if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}"
    exit 1
fi

SCENE_COUNT=$(wc -l < "${SHARD_FILE}")
echo "Processing ${SCENE_COUNT} scenes from shard ${JOB_ID}"

if [ ${SCENE_COUNT} -eq 0 ]; then
    echo "No scenes in this shard, skipping"
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
# RUN CLEANING
# ==============================================================================
cd "${BASE_DIR}"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

echo "Running dataset cleaning for shard ${JOB_ID}..."
python "${PYTHON_SCRIPTS_DIR}/clean_dataset.py" \
    --dataset-root "${OUTPUT_DATASET_ROOT}" \
    --shard-file "${SHARD_FILE}" \
    "${@:2}"  # Pass through any additional arguments

CLEAN_EXIT_CODE=$?

if [ ${CLEAN_EXIT_CODE} -ne 0 ]; then
    echo "ERROR: Dataset cleaning failed with exit code ${CLEAN_EXIT_CODE}"
    exit ${CLEAN_EXIT_CODE}
fi

echo "Successfully completed cleaning for shard ${JOB_ID}"
exit 0

