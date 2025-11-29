#!/bin/bash
#BSUB -J clean_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/clean_dataset.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_clean_dataset.sh                              # Clean all (dry-run)
#   bsub -env "DRY_RUN=0" < run_clean_dataset.sh             # Actually clean
#   bsub -env "MOVE_REJECTED=1" < run_clean_dataset.sh       # Move to *_rejected folders
#   bsub -env "SKIP_POVS=1" < run_clean_dataset.sh           # Only clean layouts
#   bsub -env "SKIP_LAYOUTS=1" < run_clean_dataset.sh        # Only clean POVs
#
# Threshold overrides:
#   bsub -env "BG_THRESHOLD=0.80" < run_clean_dataset.sh     # Stricter background

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/clean_dataset.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# Options (can be overridden via bsub -env)
DRY_RUN="${DRY_RUN:-1}"              # 1 = dry run (default), 0 = actually delete/move
MOVE_REJECTED="${MOVE_REJECTED:-1}"  # 1 = move to *_rejected, 0 = delete permanently
SKIP_POVS="${SKIP_POVS:-0}"          # 1 = skip POV cleaning
SKIP_LAYOUTS="${SKIP_LAYOUTS:-0}"    # 1 = skip layout cleaning
VERBOSE="${VERBOSE:-0}"              # 1 = verbose output

# Quality thresholds (defaults match script defaults)
BG_THRESHOLD="${BG_THRESHOLD:-0.85}"
DOMINANT_COLOR_THRESHOLD="${DOMINANT_COLOR_THRESHOLD:-0.90}"
MIN_UNIQUE_COLORS="${MIN_UNIQUE_COLORS:-3}"
MIN_ROOM_AREA="${MIN_ROOM_AREA:-0.05}"
POV_MONOCOLOR_THRESHOLD="${POV_MONOCOLOR_THRESHOLD:-0.02}"
POV_BLACK_THRESHOLD="${POV_BLACK_THRESHOLD:-0.95}"
POV_WHITE_THRESHOLD="${POV_WHITE_THRESHOLD:-0.95}"
POV_ENTROPY_THRESHOLD="${POV_ENTROPY_THRESHOLD:-2.0}"

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
# BUILD ARGUMENTS
# =============================================================================
EXTRA_ARGS=""

if [ "${DRY_RUN}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --dry-run"
fi

if [ "${MOVE_REJECTED}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --move-to-rejected"
fi

if [ "${SKIP_POVS}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-povs"
fi

if [ "${SKIP_LAYOUTS}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-layouts"
fi

if [ "${VERBOSE}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --verbose"
fi

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Cleaning Dataset - Quality Filter"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Dry Run: ${DRY_RUN}"
echo "Move to Rejected: ${MOVE_REJECTED}"
echo "Skip POVs: ${SKIP_POVS}"
echo "Skip Layouts: ${SKIP_LAYOUTS}"
echo ""
echo "Layout Thresholds:"
echo "  Background: ${BG_THRESHOLD}"
echo "  Dominant Color: ${DOMINANT_COLOR_THRESHOLD}"
echo "  Min Unique Colors: ${MIN_UNIQUE_COLORS}"
echo "  Min Room Area: ${MIN_ROOM_AREA}"
echo ""
echo "POV Thresholds:"
echo "  Monocolor: ${POV_MONOCOLOR_THRESHOLD}"
echo "  Black: ${POV_BLACK_THRESHOLD}"
echo "  White: ${POV_WHITE_THRESHOLD}"
echo "  Entropy: ${POV_ENTROPY_THRESHOLD}"
echo ""
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

# Run dataset cleaning
echo ""
echo "Running quality checks..."
echo "=========================================="

python "${PYTHON_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
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

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Dataset Cleaning COMPLETE - SUCCESS"
    echo "=========================================="
    if [ "${DRY_RUN}" = "1" ]; then
        echo "This was a DRY RUN. To actually clean:"
        echo "  bsub -env \"DRY_RUN=0\" < run_clean_dataset.sh"
    else
        if [ "${MOVE_REJECTED}" = "1" ]; then
            echo "Rejected files moved to *_rejected folders"
        else
            echo "Bad files permanently deleted"
        fi
        echo ""
        echo "Next step: Re-collect manifest"
        echo "  bsub < run_collect_manifest.sh"
    fi
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Dataset Cleaning FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi