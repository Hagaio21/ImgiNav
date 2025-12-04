#!/bin/bash
#BSUB -J embed_pov_normalized
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/embed_pov_normalized.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/embed_pov_normalized.%J.err
#BSUB -n 2
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_embed_pov_normalized.sh                    # Both tex and seg
#   bsub -env "VARIANT=tex" < run_embed_pov_normalized.sh # Only tex
#   bsub -env "VARIANT=seg" < run_embed_pov_normalized.sh # Only seg

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/embed_for_training.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"

# Options (can be overridden via bsub -env)
VARIANT="${VARIANT:-both}"  # tex, seg, or both

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "Embedding POV-Normalized POVs and Graphs"
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

# Process variants
case ${VARIANT} in
    tex)
        MANIFEST="${DATASET_ROOT}/manifests/manifest_tex_pov_normalized.csv"
        if [ ! -f "${MANIFEST}" ]; then
            echo "ERROR: Manifest not found: ${MANIFEST}" >&2
            exit 1
        fi
        echo ""
        echo "Embedding tex POVs and graphs..."
        echo "=========================================="
        python "${PYTHON_SCRIPT}" \
            --manifest "${MANIFEST}" \
            --dataset-root "${DATASET_ROOT}" \
            --batch-size 128 \
            --device cuda \
            --overwrite
        ;;
    seg)
        MANIFEST="${DATASET_ROOT}/manifests/manifest_seg_pov_normalized.csv"
        if [ ! -f "${MANIFEST}" ]; then
            echo "ERROR: Manifest not found: ${MANIFEST}" >&2
            exit 1
        fi
        echo ""
        echo "Embedding seg POVs and graphs..."
        echo "=========================================="
        python "${PYTHON_SCRIPT}" \
            --manifest "${MANIFEST}" \
            --dataset-root "${DATASET_ROOT}" \
            --batch-size 128 \
            --device cuda \
            --overwrite
        ;;
    both)
        # Process tex
        MANIFEST_TEX="${DATASET_ROOT}/manifests/manifest_tex_pov_normalized.csv"
        if [ -f "${MANIFEST_TEX}" ]; then
            echo ""
            echo "Embedding tex POVs and graphs..."
            echo "=========================================="
            python "${PYTHON_SCRIPT}" \
                --manifest "${MANIFEST_TEX}" \
                --dataset-root "${DATASET_ROOT}" \
                --batch-size 128 \
                --device cuda
        else
            echo "WARNING: Tex manifest not found: ${MANIFEST_TEX}" >&2
        fi
        
        # Process seg
        MANIFEST_SEG="${DATASET_ROOT}/manifests/manifest_seg_pov_normalized.csv"
        if [ -f "${MANIFEST_SEG}" ]; then
            echo ""
            echo "Embedding seg POVs and graphs..."
            echo "=========================================="
            python "${PYTHON_SCRIPT}" \
                --manifest "${MANIFEST_SEG}" \
                --dataset-root "${DATASET_ROOT}" \
                --batch-size 128 \
                --device cuda
        else
            echo "WARNING: Seg manifest not found: ${MANIFEST_SEG}" >&2
        fi
        ;;
    *)
        echo "ERROR: Invalid variant '${VARIANT}'. Use: tex, seg, or both" >&2
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "POV-Normalized Embedding COMPLETE - SUCCESS"
    echo "=========================================="
    echo "Embeddings saved to:"
    echo "  POVs: ${DATASET_ROOT}/povs/embeddings_*/"
    echo "  Graphs: ${DATASET_ROOT}/graphs/embeddings/"
    echo ""
    echo "Updated manifests:"
    echo "  Tex: ${DATASET_ROOT}/manifests/manifest_tex_pov_normalized.csv"
    echo "  Seg: ${DATASET_ROOT}/manifests/manifest_seg_pov_normalized.csv"
    echo ""
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "POV-Normalized Embedding FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

