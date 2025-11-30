#!/bin/bash
#BSUB -J clean_shard
#BSUB -n 2
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

# Note: -o, -e, -J are typically overridden by the launcher script
# This script is submitted by launch_clean_dataset.sh with environment variables:
#   SHARD_FILE   - Path to shard file with scene IDs
#   OUTPUT_FILE  - Path to output rejections CSV
#   EXTRA_ARGS   - Additional arguments (e.g., --skip-povs)

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/clean_dataset.py"
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# These come from environment (set by launcher)
SHARD_FILE="${SHARD_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Quality thresholds (can be overridden via bsub -env)
BG_THRESHOLD="${BG_THRESHOLD:-0.85}"
DOMINANT_COLOR_THRESHOLD="${DOMINANT_COLOR_THRESHOLD:-0.90}"
MIN_UNIQUE_COLORS="${MIN_UNIQUE_COLORS:-3}"
MIN_ROOM_AREA="${MIN_ROOM_AREA:-0.05}"
POV_MONOCOLOR_THRESHOLD="${POV_MONOCOLOR_THRESHOLD:-0.02}"
POV_BLACK_THRESHOLD="${POV_BLACK_THRESHOLD:-0.95}"
POV_WHITE_THRESHOLD="${POV_WHITE_THRESHOLD:-0.95}"
POV_ENTROPY_THRESHOLD="${POV_ENTROPY_THRESHOLD:-2.0}"

# =============================================================================
# VALIDATION
# =============================================================================
if [ -z "${SHARD_FILE}" ]; then
    echo "ERROR: SHARD_FILE not set" >&2
    exit 1
fi

if [ -z "${OUTPUT_FILE}" ]; then
    echo "ERROR: OUTPUT_FILE not set" >&2
    exit 1
fi

if [ ! -f "${SHARD_FILE}" ]; then
    echo "ERROR: Shard file not found: ${SHARD_FILE}" >&2
    exit 1
fi

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || {
        echo "Failed to activate conda environment" >&2
        exit 1
    }
fi

# =============================================================================
# RUN
# =============================================================================
SHARD_NAME=$(basename "${SHARD_FILE}" .txt)
NUM_SCENES=$(wc -l < "${SHARD_FILE}")

echo "=========================================="
echo "Cleaning Dataset Shard"
echo "=========================================="
echo "Shard: ${SHARD_NAME}"
echo "Scenes: ${NUM_SCENES}"
echo "Output: ${OUTPUT_FILE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

python "${PYTHON_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --shard-file "${SHARD_FILE}" \
    --output "${OUTPUT_FILE}" \
    --background-threshold "${BG_THRESHOLD}" \
    --dominant-color-threshold "${DOMINANT_COLOR_THRESHOLD}" \
    --min-unique-colors "${MIN_UNIQUE_COLORS}" \
    --min-room-area "${MIN_ROOM_AREA}" \
    --pov-monocolor-threshold "${POV_MONOCOLOR_THRESHOLD}" \
    --pov-black-threshold "${POV_BLACK_THRESHOLD}" \
    --pov-white-threshold "${POV_WHITE_THRESHOLD}" \
    --pov-entropy-threshold "${POV_ENTROPY_THRESHOLD}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Shard ${SHARD_NAME} COMPLETE"
    if [ -f "${OUTPUT_FILE}" ]; then
        ROWS=$(wc -l < "${OUTPUT_FILE}")
        echo "Output: ${OUTPUT_FILE} (${ROWS} rows)"
    fi
else
    echo "Shard ${SHARD_NAME} FAILED: ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE