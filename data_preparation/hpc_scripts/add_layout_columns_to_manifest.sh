#!/bin/bash
#BSUB -J add_layout_columns
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/add_layout_columns.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/add_layout_columns.%J.err
#BSUB -n 16
#BSUB -R "rusage[mem=8000]"
#BSUB -W 04:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/add_layout_columns_to_manifest.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Configuration - UPDATE THESE PATHS
MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest_with_layout_columns.csv"

# Parameters
LAYOUT_COLUMN="layout_path"
WORKERS=16
MIN_PIXEL_THRESHOLD=0
EXCLUDE_BACKGROUND=true

mkdir -p "${LOG_DIR}"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    exit 1
fi


if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

cd "${BASE_DIR}"

echo "========================================="
echo "Adding content_category to Manifest"
echo "========================================="
echo "Input manifest: ${MANIFEST}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo "Workers: ${WORKERS}"
echo "Exclude background: ${EXCLUDE_BACKGROUND}"
echo "Min pixel threshold: ${MIN_PIXEL_THRESHOLD}"
echo ""

if [ "${EXCLUDE_BACKGROUND}" = "true" ]; then
    EXCLUDE_FLAG=""
else
    EXCLUDE_FLAG="--include_background"
fi

python "${SCRIPT_PATH}" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT_MANIFEST}" \
    --layout_column "${LAYOUT_COLUMN}" \
    --workers "${WORKERS}" \
    --min_pixel_threshold "${MIN_PIXEL_THRESHOLD}" \
    ${EXCLUDE_FLAG}

echo ""
echo "========================================="
echo "Layout columns addition completed"
echo "========================================="
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo ""

