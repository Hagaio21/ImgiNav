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
ANALYSIS_SCRIPT_PATH="${BASE_DIR}/analysis/analyze_column_distribution.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Configuration - UPDATE THESE PATHS
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
OUTPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_with_content_category.csv"
TAXONOMY="${BASE_DIR}/config/taxonomy.json"
ANALYSIS_OUTPUT_DIR="/work3/s233249/ImgiNav/analysis/content_category_distribution"

# Parameters
LAYOUT_COLUMN="layout_path"
WORKERS=16
MIN_PIXEL_THRESHOLD=50

mkdir -p "${LOG_DIR}"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${ANALYSIS_SCRIPT_PATH}" ]; then
    echo "ERROR: Analysis script not found: ${ANALYSIS_SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

if [ ! -f "${TAXONOMY}" ]; then
    echo "ERROR: Taxonomy not found: ${TAXONOMY}" >&2
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
echo "Taxonomy: ${TAXONOMY}"
echo "Workers: ${WORKERS}"
echo "Min pixel threshold: ${MIN_PIXEL_THRESHOLD}"
echo ""

python "${SCRIPT_PATH}" \
    --manifest "${MANIFEST}" \
    --output "${OUTPUT_MANIFEST}" \
    --taxonomy "${TAXONOMY}" \
    --layout_column "${LAYOUT_COLUMN}" \
    --workers "${WORKERS}" \
    --min_pixel_threshold "${MIN_PIXEL_THRESHOLD}"

echo ""
echo "========================================="
echo "Layout columns addition completed"
echo "========================================="
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo ""

# Step 2: Analyze the content_category distribution
echo "========================================="
echo "Analyzing content_category distribution"
echo "========================================="
echo "Manifest: ${OUTPUT_MANIFEST}"
echo "Output directory: ${ANALYSIS_OUTPUT_DIR}"
echo ""

mkdir -p "${ANALYSIS_OUTPUT_DIR}"

python "${ANALYSIS_SCRIPT_PATH}" \
    --manifest "${OUTPUT_MANIFEST}" \
    --column "content_category" \
    --output_dir "${ANALYSIS_OUTPUT_DIR}" \
    --rare_threshold_percentile 10.0 \
    --min_samples_threshold 50 \
    --weighting_method "inverse_frequency" \
    --max_weight 10.0

echo ""
echo "========================================="
echo "Analysis completed"
echo "========================================="
echo "Analysis results saved to: ${ANALYSIS_OUTPUT_DIR}"
echo ""

