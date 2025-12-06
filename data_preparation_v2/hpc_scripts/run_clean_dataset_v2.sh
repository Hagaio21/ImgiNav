#!/bin/bash
#BSUB -J clean_dataset_v2
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset_v2.%J.%I.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset_v2.%J.%I.err
#BSUB -n 2
#BSUB -R "rusage[mem=4000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"
SHARDS_DIR="${BASE_DIR}/dataset_v2/shards_clean_dataset"

# Load config if available
if [ -f "${SHARDS_DIR}/config.sh" ]; then
    source "${SHARDS_DIR}/config.sh"
fi

# Defaults (if config not found)
DATASET_ROOT="${DATASET_ROOT:-${BASE_DIR}/dataset_v2}"
MIN_PIXELS="${MIN_PIXELS:-100}"
MAX_BLACK_FRACTION="${MAX_BLACK_FRACTION:-0.95}"
MIN_CONTENT_FRACTION="${MIN_CONTENT_FRACTION:-0.05}"

# =============================================================================
# GET SHARD ID
# =============================================================================
if [ -n "${LSB_JOBINDEX:-}" ]; then
    SHARD_ID=$((LSB_JOBINDEX - 1))  # LSB_JOBINDEX is 1-based, convert to 0-based
elif [ $# -ge 1 ]; then
    SHARD_ID=$1
else
    echo "ERROR: No job index. Run as array job or provide shard ID as argument." >&2
    exit 1
fi

SHARD_FILE="${SHARDS_DIR}/manifest_shard_$(printf "%04d" ${SHARD_ID}).csv"
OUTPUT_SHARD_FILE="${SHARDS_DIR}/manifest_shard_$(printf "%04d" ${SHARD_ID})_cleaned.csv"

# =============================================================================
# VALIDATION
# =============================================================================
if [ ! -f "${SHARD_FILE}" ]; then
    echo "Shard file not found: ${SHARD_FILE} (OK if fewer shards exist)" >&2
    exit 0
fi

# Check if shard is empty (only header)
if [ $(wc -l < "${SHARD_FILE}") -le 1 ]; then
    echo "Shard ${SHARD_ID} is empty, skipping..."
    # Create empty output shard
    cp "${SHARD_FILE}" "${OUTPUT_SHARD_FILE}"
    exit 0
fi

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "ERROR: Failed to activate conda environment" >&2
        set -u
        exit 1
    }
    set -u
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Clean Dataset v2 - Shard ${SHARD_ID}"
echo "=========================================="
echo "Shard file: ${SHARD_FILE}"
echo "Output: ${OUTPUT_SHARD_FILE}"
echo "Dataset Root: ${DATASET_ROOT}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Run clean_dataset.py on this shard (process all rows in the shard file)
python "${SCRIPTS_DIR}/clean_dataset.py" \
    --manifest "${SHARD_FILE}" \
    --dataset-root "${DATASET_ROOT}" \
    --output "${OUTPUT_SHARD_FILE}" \
    --min-pixels ${MIN_PIXELS} \
    --max-black-fraction ${MAX_BLACK_FRACTION} \
    --min-content-fraction ${MIN_CONTENT_FRACTION}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    if [ -f "${OUTPUT_SHARD_FILE}" ]; then
        line_count=$(wc -l < "${OUTPUT_SHARD_FILE}")
        echo "✓ Shard ${SHARD_ID} completed successfully"
        echo "  Output: ${OUTPUT_SHARD_FILE} (${line_count} lines)"
    else
        echo "✗ Shard ${SHARD_ID} completed but output file not found"
        EXIT_CODE=1
    fi
else
    echo "✗ Shard ${SHARD_ID} failed with exit code ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit ${EXIT_CODE}

