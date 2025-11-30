#!/bin/bash
#BSUB -J add_rejections
#BSUB -n 1
#BSUB -R "rusage[mem=4000]"
#BSUB -W 00:30
#BSUB -q hpc
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/add_rejections.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/add_rejections.%J.err

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav"
DATASET_ROOT="${BASE_DIR}/dataset_v2"
SCRIPTS_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2"

REJECTIONS_FILE="${DATASET_ROOT}/rejections/rejections_merged.csv"
MANIFEST_SEG="${DATASET_ROOT}/manifests/manifest_seg.csv"
MANIFEST_TEX="${DATASET_ROOT}/manifests/manifest_tex.csv"
OUTPUT_DIR="${DATASET_ROOT}/manifests"

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
echo "Add Rejections to Manifests"
echo "=========================================="
echo "Rejections: ${REJECTIONS_FILE}"
echo "Manifest seg: ${MANIFEST_SEG}"
echo "Manifest tex: ${MANIFEST_TEX}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "=========================================="

if [ ! -f "${REJECTIONS_FILE}" ]; then
    echo "ERROR: Rejections file not found: ${REJECTIONS_FILE}" >&2
    exit 1
fi

cd "${BASE_DIR}"

python "${SCRIPTS_DIR}/add_rejections_to_manifest.py" \
    --rejections "${REJECTIONS_FILE}" \
    --manifest-seg "${MANIFEST_SEG}" \
    --manifest-tex "${MANIFEST_TEX}" \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "COMPLETE"
    echo ""
    echo "Outputs:"
    for f in "${OUTPUT_DIR}/manifest_seg_with_rejections.csv" "${OUTPUT_DIR}/manifest_tex_with_rejections.csv"; do
        if [ -f "$f" ]; then
            ROWS=$(wc -l < "$f")
            REJECTED=$(grep -c ",True," "$f" || echo "0")
            echo "  $(basename $f): ${ROWS} rows, ${REJECTED} rejected"
        fi
    done
else
    echo "FAILED: ${EXIT_CODE}"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE