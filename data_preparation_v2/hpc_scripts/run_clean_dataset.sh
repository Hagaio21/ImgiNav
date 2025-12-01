#!/bin/bash
#BSUB -J clean_dataset
#BSUB -n 2
#BSUB -R "rusage[mem=4000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

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
if [ -z "${LSB_JOBINDEX:-}" ]; then
    echo "ERROR: LSB_JOBINDEX not set. This script must be run as a job array." >&2
    echo "  Use launch_clean_dataset.sh to submit as an array job." >&2
    exit 1
fi

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

# Build command
CMD_ARGS=(
    --dataset-root "${DATASET_ROOT}"
    --shard-file "${SHARD_FILE}"
    --output "${OUTPUT_FILE}"
    --min-pixels "${MIN_PIXELS}"
    --max-black-fraction "${MAX_BLACK_FRACTION}"
    --min-content-fraction "${MIN_CONTENT_FRACTION}"
)

# Add manifest if provided (required for sample-based checking)
if [ -n "${MANIFEST_PATH:-}" ]; then
    if [ -f "${MANIFEST_PATH}" ]; then
        echo "Using manifest: ${MANIFEST_PATH}"
        CMD_ARGS+=(--manifest "${MANIFEST_PATH}")
        
        # Add POV palette checking if enabled
        if [ "${CHECK_POV_PALETTE:-0}" = "1" ]; then
            echo "POV palette checking enabled"
            CMD_ARGS+=(
                --check-pov-palette
                --pov-color-tolerance "${POV_COLOR_TOLERANCE:-20}"
                --pov-min-match-ratio "${POV_MIN_MATCH_RATIO:-0.3}"
                --palette-color-tolerance "${PALETTE_COLOR_TOLERANCE:-10}"
            )
        fi
    else
        echo "WARNING: Manifest file not found: ${MANIFEST_PATH}"
        echo "  Will fall back to layout-only checking"
    fi
fi

python "${SCRIPTS_DIR}/clean_dataset.py" "${CMD_ARGS[@]}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "Shard ${SHARD_INDEX} COMPLETE"
echo "Output: ${OUTPUT_FILE}"
echo "End: $(date)"
echo "=========================================="