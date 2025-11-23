#!/bin/bash
#BSUB -J train_diff_clip
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_diff_clip.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 72:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
CONFIG="${BASE_DIR}/experiments/diffusion/clip/diff_clip.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Diffusion config file not found: ${CONFIG}" >&2
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
echo "Training Diffusion Model with Non-Spatial CLIP VAE"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Experiment Overview:"
echo "  - Model: Conditional Diffusion with CLIP-aligned embeddings"
echo "  - VAE: Non-spatial CLIP VAE (vae_clip)"
echo "  - Conditioning: Text embeddings (graph) + POV embeddings"
echo "  - UNet: UnetWithAttention (base_channels=48, depth=3)"
echo "  - Cross-attention: Enabled in downs path"
echo "  - Embedding projection: CLIPEmbeddingToSpatial (uses CLIP projections from VAE)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training
# The script will automatically resume from latest checkpoint if available
echo ""
echo "Starting diffusion training..."
echo "=========================================="

python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "Training COMPLETE - SUCCESS"
  echo "=========================================="
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "Training FAILED with exit code: ${EXIT_CODE}"
  echo "=========================================="
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

