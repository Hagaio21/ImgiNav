#!/bin/bash
#BSUB -J vae_stable
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/vae_stable.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/vae_stable.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 20:00
#BSUB -q gpuv100

set -euo pipefail

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
PYTHON_SCRIPT="${BASE_DIR}/training/train_ae.py"
CONFIG_FILE="${BASE_DIR}/config/architecture/autoencoders/VAE_stable.yml"

# =============================================================================
# MODULES
# =============================================================================
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

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
echo "Training Baseline VAE (Stable Diffusion-style)"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

python "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}"

echo "=========================================="
echo "Training COMPLETE"
echo "End: $(date)"
echo "=========================================="
