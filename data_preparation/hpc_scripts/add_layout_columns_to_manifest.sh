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
TAXONOMY="${BASE_DIR}/config/taxonomy.json"
ANALYSIS_OUTPUT_DIR="/work3/s233249/ImgiNav/analysis/content_category_distribution"

# Manifests to update in-place
MANIFESTS=(
    "/work3/s233249/ImgiNav/datasets/layouts.csv"
    "/work3/s233249/ImgiNav/datasets/layouts_with_embeddings.csv"
    "/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"
)

# Parameters
LAYOUT_COLUMN="layout_path"
WORKERS=16

mkdir -p "${LOG_DIR}"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${ANALYSIS_SCRIPT_PATH}" ]; then
    echo "ERROR: Analysis script not found: ${ANALYSIS_SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${TAXONOMY}" ]; then
    echo "ERROR: Taxonomy not found: ${TAXONOMY}" >&2
    exit 1
fi

# Check all manifests exist
for MANIFEST in "${MANIFESTS[@]}"; do
    if [ ! -f "${MANIFEST}" ]; then
        echo "WARNING: Manifest not found: ${MANIFEST}" >&2
        echo "Skipping this manifest..."
    fi
done

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

cd "${BASE_DIR}"

# Process each manifest
for MANIFEST in "${MANIFESTS[@]}"; do
    if [ ! -f "${MANIFEST}" ]; then
        echo "Skipping ${MANIFEST} (not found)"
        continue
    fi
    
    MANIFEST_NAME=$(basename "${MANIFEST}" .csv)
    
    echo ""
    echo "========================================="
    echo "Adding content_category to ${MANIFEST_NAME}"
    echo "========================================="
    echo "Manifest: ${MANIFEST}"
    echo "Taxonomy: ${TAXONOMY}"
    echo "Workers: ${WORKERS}"
    echo "Updating manifest in-place..."
    echo ""

    python "${SCRIPT_PATH}" \
        --manifest "${MANIFEST}" \
        --taxonomy "${TAXONOMY}" \
        --layout_column "${LAYOUT_COLUMN}" \
        --workers "${WORKERS}" \
        --overwrite

    echo ""
    echo "========================================="
    echo "Layout columns addition completed for ${MANIFEST_NAME}"
    echo "========================================="
    echo ""
done

echo ""
echo "========================================="
echo "All manifests updated successfully!"
echo "========================================="
echo "Updated manifests:"
for MANIFEST in "${MANIFESTS[@]}"; do
    if [ -f "${MANIFEST}" ]; then
        echo "  - ${MANIFEST}"
    fi
done
echo ""

