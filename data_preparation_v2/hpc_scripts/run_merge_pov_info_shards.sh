#!/bin/bash
#BSUB -J merge_pov_info_shards
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/merge_pov_info_shards.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/merge_pov_info_shards.%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_merge_pov_info_shards.sh
#   bsub -env "CLEAN=1" < run_merge_pov_info_shards.sh  # Delete shards after merge

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/merge_pov_info_shards.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# Options (can be overridden via bsub -env)
CLEAN="${CLEAN:-0}"  # Set to 1 to delete shard files after merge

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# CONDA ENV
# =============================================================================
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "Failed to activate conda environment 'imginav'" >&2
        conda activate scenefactor || {
            echo "Failed to activate any conda environment" >&2
            exit 1
        }
    }
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Merging POV Info Shard Files"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Clean shards: ${CLEAN}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# Validate dataset root exists
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: Dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

# Build arguments
EXTRA_ARGS=""
if [ "${CLEAN}" = "1" ]; then
    EXTRA_ARGS="--clean"
fi

# Run merge
echo ""
echo "Merging POV info shards..."
echo "=========================================="

python "${PYTHON_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "POV Info Merge COMPLETE - SUCCESS"
    echo "=========================================="
    echo "Output:"
    echo "  ${DATASET_ROOT}/povs/pov_info.json"
    echo ""
    if [ "${CLEAN}" = "1" ]; then
        echo "Shard files have been deleted."
    else
        echo "Shard files preserved."
    fi
    echo ""
    echo "Next step: Collect POV-normalized manifest"
    echo "  bsub < run_collect_manifest_pov_normalized.sh"
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "POV Info Merge FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

