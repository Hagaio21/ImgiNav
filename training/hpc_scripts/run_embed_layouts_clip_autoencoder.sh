#!/bin/bash
#BSUB -J embed_layouts_clip
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/run_embed_layouts_clip_autoencoder.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/run_embed_layouts_clip_autoencoder.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/embed_controlnet_dataset.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Non-spatial CLIP autoencoder config and checkpoint
AE_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256_clip.yaml"
AE_EXP_NAME="new_layouts_VAE_32x32_structural_256_clip"
AE_SAVE_PATH="/work3/s233249/ImgiNav/experiments/new_layouts/new_layouts_VAE_32x32_structural_256_clip"
AE_CHECKPOINT="${AE_SAVE_PATH}/${AE_EXP_NAME}_checkpoint_best.pt"

# Shared embedding location
SHARED_EMBEDDING_DIR="/work3/s233249/ImgiNav/experiments/shared_embeddings"
INPUT_MANIFEST="${SHARED_EMBEDDING_DIR}/manifest_with_embeddings.csv"
OUTPUT_MANIFEST="${SHARED_EMBEDDING_DIR}/manifest_with_embeddings.csv"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

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
echo "Embedding Layouts with Non-Spatial CLIP Autoencoder"
echo "=========================================="
echo "Autoencoder: ${AE_EXP_NAME}"
echo "Config: ${AE_CONFIG}"
echo "Checkpoint: ${AE_CHECKPOINT}"
echo "Input manifest: ${INPUT_MANIFEST}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Load non-spatial CLIP autoencoder (${AE_EXP_NAME})"
echo "  2. Embed layouts from manifest (using recolored layouts if available)"
echo "  3. Save latents to: ${SHARED_EMBEDDING_DIR}/latents/${AE_EXP_NAME}/"
echo "  4. Update manifest with column: latent_path_${AE_EXP_NAME}"
echo "  5. Add autoencoder name column: latent_ae_${AE_EXP_NAME}"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run embedding
python "${PYTHON_SCRIPT}" \
    --ae-checkpoint "${AE_CHECKPOINT}" \
    --ae-config "${AE_CONFIG}" \
    --input-manifest "${INPUT_MANIFEST}" \
    --output-manifest "${OUTPUT_MANIFEST}" \
    --layout-only \
    --batch-size 32 \
    --num-workers 8

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Layout Embedding COMPLETE - SUCCESS"
  echo "Autoencoder: ${AE_EXP_NAME}"
  echo "End: $(date)"
  echo "=========================================="
  echo ""
  echo "Next steps:"
  echo "  1. Verify latent column created: latent_path_${AE_EXP_NAME}"
  echo "  2. Update diffusion config to use this column (if not already done)"
  echo "  3. Run diffusion training: bsub < run_conditional_crossattention_diffusion_clip_downs_bottleneck_small.sh"
else
  echo ""
  echo "=========================================="
  echo "Layout Embedding FAILED with exit code: ${EXIT_CODE}"
  echo "Autoencoder: ${AE_EXP_NAME}"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

