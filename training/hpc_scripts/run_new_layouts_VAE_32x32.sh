#!/bin/bash
#BSUB -J new_layouts_VAE_32x32
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/new_layouts_VAE_32x32.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/new_layouts_VAE_32x32.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 24:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train.py"
CONFIG="${BASE_DIR}/experiments/autoencoders/new_layouts/new_layouts_VAE_32x32_structural_256.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "ERROR: Config file not found: ${CONFIG}" >&2
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
echo "New Layouts: VAE Training (32×32×4 Latents)"
echo "256×256 input → 32×32×4 latents"
echo "=========================================="
echo "Config: ${CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "VAE Overview:"
echo "  - Input: 256×256 RGB images"
echo "  - Latent: 32×32×4 (4,096 dimensions)"
echo "  - Architecture: 3 downsampling steps (256→128→64→32)"
echo "  - Base channels: 48"
echo "  - Variational: Yes (KL divergence regularization)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training
python "${PYTHON_SCRIPT}" "${CONFIG}" --resume

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "VAE Training COMPLETE - SUCCESS"
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
else
  echo ""
  echo "=========================================="
  echo "VAE Training FAILED with exit code: ${EXIT_CODE}"
  echo "Config: $(basename ${CONFIG})"
  echo "End: $(date)"
  echo "=========================================="
  exit $EXIT_CODE
fi

