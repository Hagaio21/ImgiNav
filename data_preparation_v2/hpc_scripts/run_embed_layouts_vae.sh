#!/bin/bash
#BSUB -J embed_layouts_vae
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/embed_layouts_vae.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation_v2/hpc_scripts/logs/embed_layouts_vae.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR="/work3/s233249/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/ImgiNav/data_preparation_v2/embed_layouts_with_vae.py"
LOG_DIR="${BASE_DIR}/ImgiNav/data_preparation_v2/hpc_scripts/logs"

# Dataset and VAE paths
DATASET_ROOT="/work3/s233249/ImgiNav/dataset_v2"
VAE_CHECKPOINT="/work3/s233249/ImgiNav/experiments/v2/autoencoders/vae_clip_v2/checkpoints/vae_clip_v2_checkpoint_best.pt"

# Manifest and output options
MANIFEST="/work3/s233249/ImgiNav/dataset_v2/manifests/manifest_seg_pov_normalized.csv"
OUTPUT_MANIFEST="${DATASET_ROOT}/manifests/manifest_seg_pov_normalized_with_latents.csv"

# Batch size
BATCH_SIZE="${BATCH_SIZE:-16}"

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
# VALIDATION
# =============================================================================
echo "=========================================="
echo "Layout VAE Embedding Job"
echo "=========================================="
echo "Dataset Root: ${DATASET_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "VAE Checkpoint: ${VAE_CHECKPOINT}"
echo "Output Manifest: ${OUTPUT_MANIFEST}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

# Validate dataset root
if [ ! -d "${DATASET_ROOT}" ]; then
    echo "ERROR: Dataset root not found: ${DATASET_ROOT}" >&2
    exit 1
fi

# Validate manifest
if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

# Validate VAE checkpoint
if [ ! -f "${VAE_CHECKPOINT}" ]; then
    echo "ERROR: VAE checkpoint not found: ${VAE_CHECKPOINT}" >&2
    exit 1
fi

# =============================================================================
# RUN
# =============================================================================
echo ""
echo "Starting VAE embedding process..."
echo ""

python "${PYTHON_SCRIPT}" \
    --manifest "${MANIFEST}" \
    --dataset-root "${DATASET_ROOT}" \
    --vae-checkpoint "${VAE_CHECKPOINT}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --vae-name "vae_clip_v2" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda

exit_code=$?

echo ""
echo "=========================================="
if [ ${exit_code} -eq 0 ]; then
    echo "✓ VAE embedding completed successfully"
    if [ -f "${OUTPUT_MANIFEST}" ]; then
        line_count=$(wc -l < "${OUTPUT_MANIFEST}")
        echo "  Output manifest: ${OUTPUT_MANIFEST} (${line_count} lines)"
    fi
else
    echo "✗ VAE embedding failed with exit code ${exit_code}"
fi
echo "End: $(date)"
echo "=========================================="

exit ${exit_code}
