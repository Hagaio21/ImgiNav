#!/bin/bash
#BSUB -J conditional_crossattention_pipeline
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_pipeline.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/conditional_crossattention_pipeline.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_pipeline_conditional_crossattention.py"
AE_CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256.yaml"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/new_layouts/conditional_crossattention_diffusion.yaml"
CONTROLNET_MANIFEST="/work3/s233249/ImgiNav/experiments/controlnet/new_layouts/controlnet_unet48_d4_new_layouts_seg/manifest_with_embeddings.csv"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config files exist
if [ ! -f "${AE_CONFIG}" ]; then
  echo "ERROR: VAE config file not found: ${AE_CONFIG}" >&2
  exit 1
fi

if [ ! -f "${DIFFUSION_CONFIG}" ]; then
  echo "ERROR: Diffusion config file not found: ${DIFFUSION_CONFIG}" >&2
  exit 1
fi

if [ ! -f "${CONTROLNET_MANIFEST}" ]; then
  echo "ERROR: ControlNet manifest not found: ${CONTROLNET_MANIFEST}" >&2
  exit 1
fi

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64
# PyTorch memory management to avoid fragmentation
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
echo "Conditional Cross-Attention Diffusion Pipeline"
echo "Complete Training Pipeline (VAE + Embedding + Diffusion)"
echo "=========================================="
echo "VAE config: ${AE_CONFIG}"
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "ControlNet manifest: ${CONTROLNET_MANIFEST}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Pipeline Overview:"
echo "  1. Train 32x32 VAE (or use existing checkpoint)"
echo "  2. Embed ControlNet dataset with 32x32 VAE"
echo "     - Creates latent_path from VAE"
echo "     - Preserves graph_embedding_path and pov_embedding_path"
echo "  3. Calculate scale_factor from embedded latents"
echo "  4. Update diffusion config with:"
echo "     - VAE checkpoint path"
echo "     - Embedded manifest path"
echo "     - Calculated scale_factor"
echo "  5. Train conditional cross-attention diffusion model"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run pipeline
# Pipeline will automatically detect if VAE checkpoint exists and skip training if found
python "${PYTHON_SCRIPT}" \
  --ae-config "${AE_CONFIG}" \
  --diffusion-config "${DIFFUSION_CONFIG}" \
  --controlnet-manifest "${CONTROLNET_MANIFEST}" \
  --batch-size 32 \
  --num-workers 8

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Pipeline COMPLETE - SUCCESS"
  echo "VAE: $(basename ${AE_CONFIG})"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "Pipeline FAILED with exit code: ${EXIT_CODE}"
  echo "VAE: $(basename ${AE_CONFIG})"
  echo "Diffusion: $(basename ${DIFFUSION_CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

