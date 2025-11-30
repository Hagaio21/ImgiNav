#!/bin/bash
#BSUB -J clean_pov_palette
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 02:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
CONFIG_FILE="${BASE_DIR}/dataset_v2/rejections/pov_palette_config.sh"

# =============================================================================
# LOAD CONFIG
# =============================================================================
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi
source "${CONFIG_FILE}"

echo "=========================================="
echo "POV Palette Checking"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "POV Color Tolerance: ${POV_COLOR_TOLERANCE}"
echo "POV Min Match Ratio: ${POV_MIN_MATCH_RATIO}"
echo "Palette Color Tolerance: ${PALETTE_COLOR_TOLERANCE}"
echo "Start: $(date)"
echo ""

# =============================================================================
# VALIDATE FILES
# =============================================================================
if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "ERROR: Manifest file not found: ${MANIFEST_PATH}" >&2
    exit 1
fi

NUM_ROWS=$(tail -n +2 "${MANIFEST_PATH}" | wc -l)
echo "Rows in manifest: ${NUM_ROWS}"
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

echo "Running clean_dataset.py with POV palette checking..."
echo ""

python "${SCRIPTS_DIR}/clean_dataset.py" \
    --dataset-root "${DATASET_ROOT}" \
    --manifest "${MANIFEST_PATH}" \
    --output "${OUTPUT_FILE}" \
    --check-pov-palette \
    --pov-color-tolerance "${POV_COLOR_TOLERANCE}" \
    --pov-min-match-ratio "${POV_MIN_MATCH_RATIO}" \
    --palette-color-tolerance "${PALETTE_COLOR_TOLERANCE}"

# =============================================================================
# DONE
# =============================================================================
echo ""
echo "=========================================="
echo "POV Palette Checking COMPLETE"
echo "Output: ${OUTPUT_FILE}"
echo "End: $(date)"
echo "=========================================="

