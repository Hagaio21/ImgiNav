#!/bin/bash
#BSUB -J embed_layouts_clip
#BSUB -q gpu
#BSUB -n 1
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o logs/run_embed_layouts_clip_autoencoder_%J.out
#BSUB -e logs/run_embed_layouts_clip_autoencoder_%J.err

export MKL_INTERFACE_LAYER=LP64
set -euo pipefail

# ----------------------------------------------------------------------
# Create Log Directory
# ----------------------------------------------------------------------
LOG_DIR="/work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] LSF Job $LSB_JOBID started on $(hostname)."

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"

# Non-spatial CLIP autoencoder config and checkpoint
AE_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml"
AE_EXP_NAME="new_layouts_VAE_32x32_structural_256_clip"
AE_SAVE_PATH="/work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_VAE_32x32_structural_256_clip"
AE_CHECKPOINT="${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt"

# Shared embedding location
SHARED_EMBEDDING_DIR="/work3/s233249/ImgiNav/experiments/shared_embeddings"
INPUT_MANIFEST="${SHARED_EMBEDDING_DIR}/manifest_with_embeddings.csv"
OUTPUT_MANIFEST="${SHARED_EMBEDDING_DIR}/manifest_with_embeddings.csv"

# Validate files exist
if [ ! -f "${AE_CONFIG}" ]; then
  echo "ERROR: Autoencoder config not found: ${AE_CONFIG}" >&2
  exit 1
fi

if [ ! -f "${AE_CHECKPOINT}" ]; then
  echo "ERROR: Autoencoder checkpoint not found: ${AE_CHECKPOINT}" >&2
  exit 1
fi

if [ ! -f "${INPUT_MANIFEST}" ]; then
  echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
  exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
# ----------------------------------------------------------------------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || {
        echo "[ERROR] Failed to activate conda env 'imginav'" >&2
        exit 1
    }
fi

cd "${BASE_DIR}"

# ----------------------------------------------------------------------
# Embed Layouts
# ----------------------------------------------------------------------
echo "=============================================================="
echo " Embedding Layouts with Non-Spatial CLIP Autoencoder"
echo "=============================================================="
echo " Autoencoder config: ${AE_CONFIG}"
echo " Autoencoder checkpoint: ${AE_CHECKPOINT}"
echo " Input manifest: ${INPUT_MANIFEST}"
echo " Output manifest: ${OUTPUT_MANIFEST}"
echo "=============================================================="

python training/embed_controlnet_dataset.py \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --ae-config "${AE_CONFIG}" \
    --input-manifest "${INPUT_MANIFEST}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --update-existing \
    --batch-size 32 \
    --num-workers 8

echo ""
echo "=============================================================="
echo "✓ Layout embedding completed"
echo "=============================================================="

