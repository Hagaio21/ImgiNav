#!/bin/bash
#BSUB -J clean_dataset
#BSUB -n 2
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
CONFIG_FILE="${BASE_DIR}/dataset_v2/shards/clean_config.sh"

# =============================================================================
# LOAD CONFIG
# =============================================================================
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
source "${CONFIG_FILE}"

# =============================================================================
# DETERMINE SHARD FROM ARRAY INDEX
# =============================================================================
# LSB_JOBINDEX is 1-indexed, shard files are 0-indexed (shard_000, shard_001, ...)
SHARD_INDEX=$(printf "%03d" $((LSB_JOBINDEX - 1)))
SHARD_FILE="${SHARDS_DIR}/shard_${SHARD_INDEX}.txt"
OUTPUT_FILE="${OUTPUT_DIR}/rejections_shard_${SHARD_INDEX}.csv"

echo "=========================================="
echo "Clean Dataset - Array Job"
echo "=========================================="
echo "LSB_JOBINDEX: ${LSB_JOBINDEX}"
echo "Shard: ${SHARD_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "Start: $(date)"
echo ""

# =============================================================================
# VALIDATE SHARD FILE
# =============================================================================
if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}" >&2
    exit 1
fi

NUM_SCENES=$(wc -l < "${SHARD_FILE}")
echo "Scenes in shard: ${NUM_SCENES}"
echo ""

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        exit 1
    }
    echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
    echo ""
fi

# =============================================================================
# RUN
# =============================================================================
cd "${BASE_DIR}"

echo "Running clean_dataset.py..."
echo ""

python "${SCRIPTS_DIR}/clean_dataset.py" \
    --dataset-root "${DATASET_ROOT}" \
    --shard-file "${SHARD_FILE}" \
    --output "${OUTPUT_FILE}" \
    --min-pixels "${MIN_PIXELS}" \
    --max-black-fraction "${MAX_BLACK_FRACTION}" \
    --min-content-fraction "${MIN_CONTENT_FRACTION}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "Shard ${SHARD_INDEX} COMPLETE"
echo "Output: ${OUTPUT_FILE}"
echo "End: $(date)"
echo "=========================================="