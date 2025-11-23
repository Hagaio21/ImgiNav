#!/bin/bash
#BSUB -J embed_clip_vaes_shared
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/embed_clip_vaes_shared.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/embed_clip_vaes_shared.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/embed_controlnet_dataset.py"

# Shared embeddings manifest (input and output are the same file)
SHARED_EMBEDDINGS_DIR="/work3/s233249/ImgiNav/experiments/shared_embeddings"
SHARED_MANIFEST="${SHARED_EMBEDDINGS_DIR}/manifest_with_embeddings.csv"

# VAE configs and checkpoints
VAE_CLIP_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml"
VAE_CLIP_SPATIAL_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip_spatial.yaml"

LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate shared manifest exists
if [ ! -f "${SHARED_MANIFEST}" ]; then
  echo "ERROR: Shared manifest not found: ${SHARED_MANIFEST}" >&2
  exit 1
fi

# Find VAE checkpoints
# Non-spatial CLIP VAE
VAE_CLIP_EXP_NAME=$(python3 -c "
import yaml
with open('${VAE_CLIP_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('experiment', {}).get('name', 'vae_clip'))
" 2>/dev/null || echo "vae_clip")

VAE_CLIP_SAVE_PATH=$(python3 -c "
import yaml
with open('${VAE_CLIP_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    save_path = config.get('experiment', {}).get('save_path', '')
    if save_path:
        print(save_path)
    else:
        print('outputs')
" 2>/dev/null || echo "outputs")

if [ -f "${VAE_CLIP_SAVE_PATH}/checkpoints/${VAE_CLIP_EXP_NAME}_checkpoint_best.pt" ]; then
  VAE_CLIP_CHECKPOINT="${VAE_CLIP_SAVE_PATH}/checkpoints/${VAE_CLIP_EXP_NAME}_checkpoint_best.pt"
elif [ -f "${VAE_CLIP_SAVE_PATH}/${VAE_CLIP_EXP_NAME}_checkpoint_best.pt" ]; then
  VAE_CLIP_CHECKPOINT="${VAE_CLIP_SAVE_PATH}/${VAE_CLIP_EXP_NAME}_checkpoint_best.pt"
else
  echo "WARNING: Non-spatial CLIP VAE checkpoint not found, skipping..."
  VAE_CLIP_CHECKPOINT=""
fi

# Spatial CLIP VAE
VAE_CLIP_SPATIAL_EXP_NAME=$(python3 -c "
import yaml
with open('${VAE_CLIP_SPATIAL_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    print(config.get('experiment', {}).get('name', 'vae_clip_spatial'))
" 2>/dev/null || echo "vae_clip_spatial")

VAE_CLIP_SPATIAL_SAVE_PATH=$(python3 -c "
import yaml
with open('${VAE_CLIP_SPATIAL_CONFIG}', 'r') as f:
    config = yaml.safe_load(f)
    save_path = config.get('experiment', {}).get('save_path', '')
    if save_path:
        print(save_path)
    else:
        print('outputs')
" 2>/dev/null || echo "outputs")

if [ -f "${VAE_CLIP_SPATIAL_SAVE_PATH}/checkpoints/${VAE_CLIP_SPATIAL_EXP_NAME}_checkpoint_best.pt" ]; then
  VAE_CLIP_SPATIAL_CHECKPOINT="${VAE_CLIP_SPATIAL_SAVE_PATH}/checkpoints/${VAE_CLIP_SPATIAL_EXP_NAME}_checkpoint_best.pt"
elif [ -f "${VAE_CLIP_SPATIAL_SAVE_PATH}/${VAE_CLIP_SPATIAL_EXP_NAME}_checkpoint_best.pt" ]; then
  VAE_CLIP_SPATIAL_CHECKPOINT="${VAE_CLIP_SPATIAL_SAVE_PATH}/${VAE_CLIP_SPATIAL_EXP_NAME}_checkpoint_best.pt"
else
  echo "WARNING: Spatial CLIP VAE checkpoint not found, skipping..."
  VAE_CLIP_SPATIAL_CHECKPOINT=""
fi

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
# RUN
# =============================================================================
echo "=========================================="
echo "Embedding Latents with CLIP VAEs (Shared)"
echo "=========================================="
echo "Shared manifest: ${SHARED_MANIFEST}"
echo "Output directory: ${SHARED_EMBEDDINGS_DIR}"
if [ -n "${VAE_CLIP_CHECKPOINT}" ]; then
  echo "Non-spatial CLIP VAE: ${VAE_CLIP_CHECKPOINT}"
fi
if [ -n "${VAE_CLIP_SPATIAL_CHECKPOINT}" ]; then
  echo "Spatial CLIP VAE: ${VAE_CLIP_SPATIAL_CHECKPOINT}"
fi
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Create shared embeddings directory
mkdir -p "${SHARED_EMBEDDINGS_DIR}"

# Run embedding for non-spatial CLIP VAE
if [ -n "${VAE_CLIP_CHECKPOINT}" ]; then
  echo ""
  echo "Embedding with non-spatial CLIP VAE..."
  echo "=========================================="
  # Update shared manifest with latent_path_vae_clip column
  python "${PYTHON_SCRIPT}" \
    --ae-checkpoint "${VAE_CLIP_CHECKPOINT}" \
    --input-manifest "${SHARED_MANIFEST}" \
    --output-manifest "${SHARED_MANIFEST}" \
    --layout-only \
    --batch-size 32 \
    --num-workers 8
fi

# Run embedding for spatial CLIP VAE
if [ -n "${VAE_CLIP_SPATIAL_CHECKPOINT}" ]; then
  echo ""
  echo "Embedding with spatial CLIP VAE..."
  echo "=========================================="
  # Update the same manifest with latent_path_vae_clip_spatial column
  python "${PYTHON_SCRIPT}" \
    --ae-checkpoint "${VAE_CLIP_SPATIAL_CHECKPOINT}" \
    --input-manifest "${SHARED_MANIFEST}" \
    --output-manifest "${SHARED_MANIFEST}" \
    --layout-only \
    --batch-size 32 \
    --num-workers 8
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Embedding COMPLETE - SUCCESS"
  echo "=========================================="
  echo "Manifest updated: ${SHARED_MANIFEST}"
  echo "Latents saved in: ${SHARED_EMBEDDINGS_DIR}/latents/"
  echo "Columns added: latent_path_vae_clip, latent_path_vae_clip_spatial"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Embedding FAILED with exit code: ${EXIT_CODE}"
  echo "=========================================="
  exit $EXIT_CODE
fi

