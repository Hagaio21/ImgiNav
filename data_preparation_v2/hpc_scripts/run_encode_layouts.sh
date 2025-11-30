#!/bin/bash
#BSUB -J encode_layouts
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/encode_layouts.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/encode_layouts.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
# Usage:
#   bsub < run_encode_layouts.sh                                    # Both tex and seg
#   bsub -env "VARIANT=tex" < run_encode_layouts.sh                 # Only tex
#   bsub -env "VARIANT=seg" < run_encode_layouts.sh                 # Only seg
#   bsub -env "OVERWRITE=1" < run_encode_layouts.sh                  # Overwrite existing latents
#   bsub -env "BATCH_SIZE=64" < run_encode_layouts.sh               # Custom batch size

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/encode_layouts.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset paths
DATASET_ROOT="${BASE_DIR}/dataset_v2"
MANIFEST_DIR="${DATASET_ROOT}/manifests"

# VAE checkpoint (can be overridden via bsub -env)
VAE_CHECKPOINT="${VAE_CHECKPOINT:-/work3/s233249/ImgiNav/experiments/v2/autoencoders/vae_seg_256_clip/checkpoints/vae_seg_256_clip_checkpoint_best.pt}"

# Options (can be overridden via bsub -env)
VARIANT="${VARIANT:-both}"          # tex, seg, or both
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OVERWRITE="${OVERWRITE:-0}"         # Set to 1 to overwrite existing latents
DEVICE="${DEVICE:-cuda}"

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
# VALIDATION
# =============================================================================
if [ ! -f "${VAE_CHECKPOINT}" ]; then
    echo "ERROR: VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# =============================================================================
# BUILD ARGUMENTS
# =============================================================================
EXTRA_ARGS=""
if [ "${OVERWRITE}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --overwrite"
fi

# =============================================================================
# FUNCTIONS
# =============================================================================
encode_layouts() {
    local variant=$1
    local manifest="${MANIFEST_DIR}/manifest_${variant}.csv"
    
    if [ ! -f "${manifest}" ]; then
        echo "ERROR: Manifest not found: ${manifest}" >&2
        echo "Run collect_manifest.py first" >&2
        return 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Encoding Layouts: ${variant}"
    echo "VAE Checkpoint: ${VAE_CHECKPOINT}"
    echo "Manifest: ${manifest}"
    echo "=========================================="
    
    python "${PYTHON_SCRIPT}" \
        --vae-checkpoint "${VAE_CHECKPOINT}" \
        --manifest "${manifest}" \
        --dataset-root "${DATASET_ROOT}" \
        --variant "${variant}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        ${EXTRA_ARGS}
}

# =============================================================================
# RUN
# =============================================================================
echo "=========================================="
echo "Layout Latent Encoding"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "VAE Checkpoint: ${VAE_CHECKPOINT}"
echo "Variant: ${VARIANT}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Num Workers: ${NUM_WORKERS}"
echo "Overwrite: ${OVERWRITE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Run for selected variant(s)
case ${VARIANT} in
    tex)
        encode_layouts "tex"
        ;;
    seg)
        encode_layouts "seg"
        ;;
    both)
        encode_layouts "tex"
        encode_layouts "seg"
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
    echo "Layout Encoding COMPLETE - SUCCESS"
    echo "=========================================="
    echo "Output:"
    echo "  Latents: ${DATASET_ROOT}/layouts/latents/*/"
    echo "  Manifests: ${MANIFEST_DIR}/*_latent_manifest_*.csv"
    echo ""
    echo "Next step: Train diffusion model"
    echo "  bsub < training/hpc_scripts/regular/run_train_diff_clip_regular_rooms_bottleneck.sh"
    echo "=========================================="
    echo "End: $(date)"
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Layout Encoding FAILED: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

