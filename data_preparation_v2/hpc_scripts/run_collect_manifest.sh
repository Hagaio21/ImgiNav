#!/bin/bash
#BSUB -J collect_manifest
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/collect_manifest.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/collect_manifest.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -W 01:00
#BSUB -q hpc

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64
# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_collect_manifest.sh                    # Both tex and seg
#   bsub -env "VARIANT=tex" < run_collect_manifest.sh # Only tex
#   bsub -env "VARIANT=seg" < run_collect_manifest.sh # Only seg

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/data_preparation_v2/collect_manifest.py"
LOG_DIR="${BASE_DIR}/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# Options (can be overridden via bsub -env)
VARIANT="${VARIANT:-both}"  # tex, seg, or both

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
# No GPU needed for manifest collection (just file scanning)
# Load Python if needed
module load python3/3.10

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
case ${VARIANT} in
    tex)
        EXTRA_ARGS="--tex-only"
        ;;
    seg)
        EXTRA_ARGS="--seg-only"
        ;;
    both)
        # No extra args - generate both
        ;;
    *)
        echo "ERROR: Invalid variant '${VARIANT}'. Use: tex, seg, or both" >&2
        exit 1
        ;;
esac

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Collecting Manifest CSV Files"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Variant: ${VARIANT}"
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

# Run manifest collection
echo ""
echo "Collecting manifest data..."
echo "=========================================="

python "${PYTHON_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Manifest Collection COMPLETE - SUCCESS"
    echo "=========================================="
    echo "Output manifests:"
    echo "  Tex: ${DATASET_ROOT}/manifests/manifest_tex.csv"
    echo "  Seg: ${DATASET_ROOT}/manifests/manifest_seg.csv"
    echo ""
    echo "Next step: Create embeddings"
    echo "  bsub < run_embeddings.sh"
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Manifest Collection FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

