#!/bin/bash
#BSUB -J train_vae_tex_256
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_vae_tex_256.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/train_vae_tex_256.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=4000]"
#BSUB -gpu "num=1"
#BSUB -W 6:00
#BSUB -q gpul40s

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train.py"
VAE_CONFIG="${BASE_DIR}/experiments/autoencoders/v2/vae_tex_256_clip.yaml"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Validate config file exists
if [ ! -f "${VAE_CONFIG}" ]; then
  echo "ERROR: VAE config file not found: ${VAE_CONFIG}" >&2
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
echo "Training VAE - Textured Layouts (256x256)"
echo "=========================================="
echo "VAE config: ${VAE_CONFIG}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "This will train a VAE on textured layouts (256x256) with CLIP loss."
echo "Input: 256x256 textured layout images"
echo "Latent: 16x16x4 (after 4 downsampling steps)"
echo "=========================================="

cd "${BASE_DIR}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
  exit 1
fi

# Run training
echo ""
echo "Starting VAE training for textured layouts..."
echo "=========================================="

python "${PYTHON_SCRIPT}" "${VAE_CONFIG}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "VAE Training COMPLETE - SUCCESS"
  echo "=========================================="
  echo "Config: ${VAE_CONFIG}"
  echo "Output: outputs/autoencoders/v2/vae_tex_256_clip/"
  echo "End: $(date)"
  echo "=========================================="
  exit 0
else
  echo ""
  echo "=========================================="
  echo "VAE Training FAILED with exit code: ${EXIT_CODE}"
  echo "=========================================="
  exit $EXIT_CODE
fi

