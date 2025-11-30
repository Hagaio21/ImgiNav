#!/bin/bash
#BSUB -J merge_rejections
#BSUB -n 1
#BSUB -R "rusage[mem=2000]"
#BSUB -W 00:30
#BSUB -q hpc

# This script is submitted by launch_clean_dataset.sh with dependency on all shard jobs
# Environment variables:
#   REJECTIONS_DIR - Directory containing shard rejection CSVs

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/merge_rejections.py"

# From environment (set by launcher)
REJECTIONS_DIR="${REJECTIONS_DIR:-${BASE_DIR}/dataset_v2/rejections}"
OUTPUT_FILE="${REJECTIONS_DIR}/rejections_merged.csv"

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
echo "=========================================="
echo "Merging Rejection CSVs"
echo "=========================================="
echo "Input dir: ${REJECTIONS_DIR}"
echo "Output: ${OUTPUT_FILE}"
echo "Start: $(date)"
echo "=========================================="

# Count input files
NUM_FILES=$(find "${REJECTIONS_DIR}" -name "rejections_shard_*.csv" | wc -l)
echo "Found ${NUM_FILES} shard files to merge"

if [ "${NUM_FILES}" -eq 0 ]; then
    echo "ERROR: No shard files found in ${REJECTIONS_DIR}" >&2
    exit 1
fi

cd "${BASE_DIR}"

python "${PYTHON_SCRIPT}" \
    --input-pattern "${REJECTIONS_DIR}/rejections_shard_*.csv" \
    --output "${OUTPUT_FILE}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Merge COMPLETE"
    if [ -f "${OUTPUT_FILE}" ]; then
        ROWS=$(wc -l < "${OUTPUT_FILE}")
        SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "Output: ${OUTPUT_FILE}"
        echo "  Rows: ${ROWS}"
        echo "  Size: ${SIZE}"
    fi
    echo ""
    echo "Next step: Update manifest"
    echo "  python ${BASE_DIR}/ImgiNav/data_preparation_v2/update_manifest_rejections.py \\"
    echo "      --manifest ${BASE_DIR}/dataset_v2/manifests/manifest_tex.csv \\"
    echo "      --rejections ${OUTPUT_FILE} \\"
    echo "      --output ${BASE_DIR}/dataset_v2/manifests/manifest_tex_filtered.csv"
else
    echo "Merge FAILED: ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE