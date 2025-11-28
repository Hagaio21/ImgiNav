#!/bin/bash
#BSUB -J embed_training
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/logs/embed_training.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/logs/embed_training.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1"
#BSUB -W 02:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_embeddings.sh                          # Both tex and seg
#   bsub -env "VARIANT=tex" < run_embeddings.sh       # Only tex
#   bsub -env "VARIANT=seg" < run_embeddings.sh       # Only seg
#   bsub -env "SKIP_POV=1" < run_embeddings.sh        # Skip POV embeddings
#   bsub -env "SKIP_GRAPH=1" < run_embeddings.sh      # Skip graph embeddings

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation/embed_for_training.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"
MANIFEST_DIR="${DATASET_ROOT}/manifests"

# Options (can be overridden via bsub -env)
VARIANT="${VARIANT:-both}"          # tex, seg, or both
SKIP_POV="${SKIP_POV:-0}"           # Set to 1 to skip POV embeddings
SKIP_GRAPH="${SKIP_GRAPH:-0}"       # Set to 1 to skip graph embeddings
BATCH_SIZE="${BATCH_SIZE:-64}"
OVERWRITE="${OVERWRITE:-0}"         # Set to 1 to overwrite existing

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
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
# BUILD ARGUMENTS
# =============================================================================
EXTRA_ARGS=""
if [ "${SKIP_POV}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-pov"
fi
if [ "${SKIP_GRAPH}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-graph"
fi
if [ "${OVERWRITE}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --overwrite"
fi

# =============================================================================
# FUNCTIONS
# =============================================================================
run_embedding() {
    local variant=$1
    local manifest="${MANIFEST_DIR}/manifest_${variant}.csv"
    
    if [ ! -f "${manifest}" ]; then
        echo "ERROR: Manifest not found: ${manifest}" >&2
        echo "Run collect_manifest.py first" >&2
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: ${variant}"
    echo "Manifest: ${manifest}"
    echo "=========================================="
    
    python "${PYTHON_SCRIPT}" \
        --manifest "${manifest}" \
        --dataset-root "${DATASET_ROOT}" \
        --batch-size "${BATCH_SIZE}" \
        --device cuda \
        ${EXTRA_ARGS}
}

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Embedding Creation for Training"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Variant: ${VARIANT}"
echo "Skip POV: ${SKIP_POV}"
echo "Skip Graph: ${SKIP_GRAPH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# Run for selected variant(s)
case ${VARIANT} in
    tex)
        run_embedding "tex"
        ;;
    seg)
        run_embedding "seg"
        ;;
    both)
        run_embedding "tex"
        run_embedding "seg"
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
    echo "Embedding Creation COMPLETE - SUCCESS"
    echo "=========================================="
    echo "Output:"
    echo "  POV: ${DATASET_ROOT}/pov/embeddings_*/"
    echo "  Graph: ${DATASET_ROOT}/graphs/embeddings/"
    echo ""
    echo "Next step: Train VAE"
    echo "  bsub < run_train_vae.sh"
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Embedding Creation FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi