#!/bin/bash
#BSUB -J clean_dataset
#BSUB -n 2
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

# Job Array Script for Dataset Cleaning
# 
# This script is submitted as an array job. Each array element processes one shard.
# LSF_JOBINDEX (1-indexed) determines which shard to process.
#
# Shard mapping:
#   LSF_JOBINDEX=1  -> shard_000.txt
#   LSF_JOBINDEX=2  -> shard_001.txt
#   etc.

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64
# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
CONFIG_FILE="${BASE_DIR}/dataset_v2/shards/clean_config.sh"

# Load config (written by launcher)
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
OUTPUT_FILE="${REJECTIONS_DIR}/rejections_shard_${SHARD_INDEX}.csv"

if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}" >&2
    exit 1
fi

NUM_SCENES=$(wc -l < "${SHARD_FILE}")

echo "=========================================="
echo "Clean Dataset - Array Job"
echo "=========================================="
echo "LSB_JOBINDEX: ${LSB_JOBINDEX}"
echo "Shard: ${SHARD_FILE}"
echo "Scenes: ${NUM_SCENES}"
echo "Output: ${OUTPUT_FILE}"
echo "Start: $(date)"
echo "=========================================="

# =============================================================================
# CONDA ENVIRONMENT
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        exit 1
    }
fi

# =============================================================================
# RUN
# =============================================================================
cd "${BASE_DIR}"

python "${SCRIPTS_DIR}/clean_dataset.py" \
    --dataset-root "${DATASET_ROOT}" \
    --shard-file "${SHARD_FILE}" \
    --output "${OUTPUT_FILE}" \
    --min-pixels "${MIN_PIXELS}" \
    --max-black-fraction "${MAX_BLACK_FRACTION}" \
    --min-content-fraction "${MIN_CONTENT_FRACTION}" \
    --max-floor-fraction "${MAX_FLOOR_FRACTION}" \
    --max-wall-fraction "${MAX_WALL_FRACTION}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Shard ${SHARD_INDEX} COMPLETE"
    if [ -f "${OUTPUT_FILE}" ]; then
        ROWS=$(wc -l < "${OUTPUT_FILE}")
        echo "Output: ${OUTPUT_FILE} (${ROWS} rows)"
    fi
else
    echo "Shard ${SHARD_INDEX} FAILED: exit code ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE